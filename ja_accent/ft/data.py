import json
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .constants import SAMPLE_RATE, SAMPLES_PER_FRAME


def load_wav(path, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Mono float32 at ``sr``."""
    x, file_sr = sf.read(str(path), dtype='float32', always_2d=True)
    wav = torch.from_numpy(x.mean(axis=1))
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    return wav


def read_manifest(path) -> List[Dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def is_val(uid: str, val_pct: float) -> bool:
    return zlib.crc32(uid.encode()) % 10_000 < val_pct * 100


class ManifestDataset(torch.utils.data.Dataset):
    """One utterance per item; training uses gradient accumulation instead
    of padding so the forward stays identical to Kokoro's inference graph."""

    def __init__(self, prep_dir, vocab, split='train', val_pct=5.0, max_frames=800,
                 min_align=0.0, min_accent=0.0):
        self.root = Path(prep_dir)
        self.vocab = vocab
        rows = read_manifest(self.root / 'manifest.jsonl')
        keep, skipped = [], {}
        for r in rows:
            reason = None
            if r['frames'] > max_frames:
                reason = 'too_long'
            elif any(c not in vocab for c in r['ps']):
                reason = 'oov'
            elif r['align_score'] < min_align:
                reason = 'align'
            elif r['accent_checked'] and r['accent_correct'] / r['accent_checked'] < min_accent:
                reason = 'accent'
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
            elif (split == 'val') == is_val(r['id'], val_pct):
                keep.append(r)
        self.rows = keep
        self.speakers = sorted({r['speaker'] for r in rows}) or ['spk0']
        print(f'[data] {split}: {len(keep)} utterances' + (f', skipped {skipped}' if skipped else ''))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav = load_wav(self.root / 'wavs' / r['wav'])[: r['frames'] * SAMPLES_PER_FRAME]
        ids = torch.LongTensor([[0, *(self.vocab[c] for c in r['ps']), 0]])
        dur = torch.LongTensor(r['dur'])
        f0 = torch.from_numpy(np.load(self.root / 'f0' / f"{r['id']}.npy"))
        assert ids.shape[1] == dur.shape[0] and f0.shape[0] == 2 * r['frames'], r['id']
        return {'row': r, 'ids': ids, 'dur': dur, 'wav': wav, 'f0': f0}
