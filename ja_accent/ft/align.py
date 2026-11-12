"""Mora-level forced alignment and per-token duration targets.

Each mora's romanization is aligned with torchaudio's MMS_FA (a multilingual
CTC aligner over a-z). Within a run of speech a mora lasts from its first
token's onset to the next mora's onset (CTC onsets are reliable, vowel peaks
fire early); the mora's frames are split across its phoneme chars in
proportion to the pretrained duration predictor, and pauses go to spaces,
punctuation and BOS/EOS. Accent marks sit between moras and end up with the
1-frame minimum Kokoro's inference applies to every token.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

from ..g2p import ACCENT_MARKS, Annotation
from .constants import FRAMES_PER_SEC, SAMPLE_RATE, SAMPLES_PER_FRAME

# Seconds a mora may extend into an adjacent pause. CTC emissions are peaky,
# so raw spans cover only part of each sound.
MAX_ABSORB = 0.08
LEAD_ABSORB = 0.02
# Kokoro's decoder output runs ahead of its own alignment: token boundaries
# sit this much after the matching acoustic onsets (measured on stock
# synthesis; boundary error drops from 75 to 25 ms median).
DECODER_LEAD = 0.05


class MoraAligner:
    def __init__(self, device='cpu'):
        bundle = torchaudio.pipelines.MMS_FA
        self.device = device
        self.model = bundle.get_model(with_star=False).to(device).eval()
        self.dict = bundle.get_dict(star=None)
        self.resample = torchaudio.transforms.Resample(SAMPLE_RATE, bundle.sample_rate).to(device)
        self.sample_rate = bundle.sample_rate

    @torch.inference_mode()
    def align(self, wav24: torch.Tensor, ann: Annotation) -> Tuple[List[Optional[Tuple[float, float]]], float]:
        """Per-mora (start, end) in seconds, None for moras without own sound."""
        ids, owner = [], []
        for mi, m in enumerate(ann.moras):
            for c in m.roman:
                if c in self.dict:
                    ids.append(self.dict[c])
                    owner.append(mi)
        spans: List[Optional[Tuple[float, float]]] = [None] * len(ann.moras)
        if not ids:
            return spans, 0.0
        wav16 = self.resample(wav24.to(self.device)).unsqueeze(0)
        emission, _ = self.model(wav16)
        if emission.shape[1] < len(ids):
            return spans, 0.0
        targets = torch.tensor([ids], dtype=torch.int32, device=self.device)
        aligned, scores = torchaudio.functional.forced_align(
            torch.log_softmax(emission, dim=-1), targets, blank=0)
        tok = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
        sec = wav16.shape[1] / emission.shape[1] / self.sample_rate
        for t, mi in zip(tok, owner):
            a, b = t.start * sec, t.end * sec
            spans[mi] = (a, b) if spans[mi] is None else (spans[mi][0], b)
        return spans, float(np.mean([t.score for t in tok]))


def _spread(out: np.ndarray, positions: Sequence[int], a: float, b: float, weights: np.ndarray):
    if not positions:
        return
    w = np.clip(weights[list(positions)], 1e-3, None)
    out[list(positions)] = max(b - a, 0.0) * w / w.sum()


def speech_end_frame(wav: torch.Tensor, floor_db: float = -35.0) -> int:
    """Duration frame after the last one whose RMS is within ``floor_db`` of the loudest."""
    n = wav.shape[0] // SAMPLES_PER_FRAME
    rms = wav[: n * SAMPLES_PER_FRAME].reshape(n, -1).pow(2).mean(dim=1).sqrt()
    loud = torch.nonzero(rms > rms.max() * 10 ** (floor_db / 20)).flatten()
    return int(loud[-1]) + 1 if len(loud) else n


def token_durations(ann: Annotation, spans, weights: np.ndarray, total_frames: int,
                    speech_end: Optional[int] = None) -> Optional[np.ndarray]:
    """Integer frames for BOS + ps + EOS summing exactly to ``total_frames``.

    ``weights`` are the pretrained duration head's continuous outputs for the
    same token sequence; only their ratios within a segment matter.
    ``speech_end`` (from ``speech_end_frame``) lets the last mora keep a
    lengthened final vowel that CTC stops emitting for early.
    """
    spans = [None if s is None else (s[0] + DECODER_LEAD, s[1] + DECODER_LEAD) for s in spans]
    n_tok = len(ann.ps) + 2
    if total_frames < n_tok:
        return None
    mora_pos = [[] for _ in ann.moras]
    for i, m in enumerate(ann.char_mora):
        if m >= 0:
            mora_pos[m].append(i + 1)
    aligned = [mi for mi, s in enumerate(spans) if s is not None and mora_pos[mi]]
    if not aligned:
        return None

    iv = {mi: [spans[mi][0] * FRAMES_PER_SEC, spans[mi][1] * FRAMES_PER_SEC] for mi in aligned}
    first, last = aligned[0], aligned[-1]
    absorb = MAX_ABSORB * FRAMES_PER_SEC
    iv[first][0] = max(0.0, iv[first][0] - LEAD_ABSORB * FRAMES_PER_SEC)
    iv[last][1] = min(float(total_frames), max(iv[last][1] + absorb, speech_end or 0))
    for k, nk in zip(aligned, aligned[1:]):
        between = range(mora_pos[k][-1] + 1, mora_pos[nk][0])
        gap = max(iv[nk][0] - iv[k][1], 0.0)
        pause = any(ann.ps[p - 1] not in ACCENT_MARKS for p in between)
        if pause:
            take = min(gap / 2, absorb)
            iv[k][1] += take
            iv[nk][0] -= take
        else:
            iv[k][1] = iv[nk][0]

    fdur = np.zeros(n_tok)
    _spread(fdur, range(0, mora_pos[first][0]), 0.0, iv[first][0], weights)
    for k, nk in zip(aligned, aligned[1:]):
        _spread(fdur, mora_pos[k], *iv[k], weights)
        _spread(fdur, range(mora_pos[k][-1] + 1, mora_pos[nk][0]), iv[k][1], iv[nk][0], weights)
    _spread(fdur, mora_pos[last], *iv[last], weights)
    _spread(fdur, range(mora_pos[last][-1] + 1, n_tok), iv[last][1], float(total_frames), weights)

    cum = np.cumsum(fdur)
    if cum[-1] <= 0:
        return None
    cum *= total_frames / cum[-1]
    dur = np.diff(np.round(np.concatenate([[0.0], cum])).astype(np.int64))
    return _enforce_min_one(dur)


def _enforce_min_one(dur: np.ndarray) -> np.ndarray:
    """Inference clamps every token to >= 1 frame; borrow from the nearest
    token that can spare one so targets stay locally aligned."""
    dur = dur.copy()
    for i in np.flatnonzero(dur == 0):
        for r in range(1, len(dur)):
            j = next((j for j in (i - r, i + r) if 0 <= j < len(dur) and dur[j] > 1), None)
            if j is not None:
                dur[j] -= 1
                dur[i] += 1
                break
    assert (dur >= 1).all()
    return dur
