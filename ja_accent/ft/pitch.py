"""F0 / energy targets and the pitch-accent agreement score.

Curve convention (StyleTTS2, which Kokoro inherits): index k of the F0 and
energy curves is the frame centred at k * 12.5 ms, two per duration frame;
F0 is in Hz with 0 for unvoiced frames.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

from ..g2p import DOWNSTEP, UPSTEP, Annotation
from .constants import CURVE_HOP, MEL_N_FFT, MEL_WIN, N_MELS, SAMPLE_RATE


def extract_f0(wav: np.ndarray, n_curve: int, floor: float = 60.0, ceil: float = 700.0) -> np.ndarray:
    """WORLD harvest F0 at 80 fps, padded/trimmed to ``n_curve`` values."""
    import pyworld

    x = np.ascontiguousarray(wav, dtype=np.float64)
    f0, t = pyworld.harvest(x, SAMPLE_RATE, f0_floor=floor, f0_ceil=ceil,
                            frame_period=1000 * CURVE_HOP / SAMPLE_RATE)
    f0 = pyworld.stonemask(x, f0, t, SAMPLE_RATE)
    out = np.zeros(n_curve, dtype=np.float32)
    n = min(n_curve, len(f0))
    out[:n] = f0[:n]
    return out


class EnergyCurve(torch.nn.Module):
    """StyleTTS2 energy: log L2 norm over bins of the power mel spectrogram."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=MEL_N_FFT, win_length=MEL_WIN,
            hop_length=CURVE_HOP, n_mels=N_MELS, power=2.0)

    def forward(self, wav: torch.Tensor, n_curve: int) -> torch.Tensor:
        e = torch.log((self.mel(wav) + 1e-5).norm(dim=0))
        if e.shape[0] < n_curve:
            e = torch.nn.functional.pad(e, (0, n_curve - e.shape[0]), value=float(e[-1]))
        return e[:n_curve]


def mora_log_f0(ann: Annotation, dur: Sequence[int], f0: np.ndarray) -> np.ndarray:
    """Mean log-F0 over the voiced curve frames of each mora (nan if none).

    ``dur`` covers BOS + ps + EOS in duration frames; ``f0`` is the 80 fps curve.
    """
    starts = np.concatenate([[0], np.cumsum(dur)])
    vals = [[] for _ in ann.moras]
    for i, m in enumerate(ann.char_mora):
        if m < 0:
            continue
        a, b = 2 * starts[i + 1], 2 * starts[i + 2]
        seg = f0[a:b]
        vals[m].extend(seg[seg > 0].tolist())
    return np.array([np.log(v).mean() if v else np.nan for v in vals])


def accent_agreement(ann: Annotation, dur: Sequence[int], f0: np.ndarray,
                     margin: float = 0.0) -> Tuple[int, int]:
    """(correct, checked) accent marks whose pitch moves the labelled way.

    Each mark is checked against the voiced moras on either side within the
    same accent phrase: UPSTEP expects a rise, DOWNSTEP a fall. Marks next
    to fully devoiced moras are skipped.
    """
    lf0 = mora_log_f0(ann, dur, f0)
    correct = checked = 0
    for i, c in enumerate(ann.ps):
        if c not in (UPSTEP, DOWNSTEP):
            continue
        before = _neighbour_mora(ann, i, -1)
        after = _neighbour_mora(ann, i, +1)
        if before is None or after is None:
            continue
        if ann.moras[before].phrase != ann.moras[after].phrase:
            continue
        a, b = lf0[before], lf0[after]
        if np.isnan(a) or np.isnan(b):
            continue
        checked += 1
        delta = b - a
        correct += int(delta > margin if c == UPSTEP else delta < -margin)
    return correct, checked


def _neighbour_mora(ann: Annotation, i: int, step: int) -> Optional[int]:
    j = i + step
    while 0 <= j < len(ann.ps):
        if ann.ps[j] == ' ':
            return None
        if ann.char_mora[j] >= 0:
            return ann.char_mora[j]
        j += step
    return None
