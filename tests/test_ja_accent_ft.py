import numpy as np
import pytest

pytest.importorskip('pyopenjtalk')
pytest.importorskip('torchaudio')
pytest.importorskip('pyworld')

from ja_accent import DOWNSTEP, UPSTEP, JAAccentG2P
from ja_accent.ft.align import _enforce_min_one, token_durations
from ja_accent.ft.pitch import accent_agreement


@pytest.fixture(scope='module')
def ann():
    return JAAccentG2P().annotate('橋が見える。')


def _even_spans(ann, sec_per_mora=0.15, lead=0.3):
    return [(lead + i * sec_per_mora + 0.03, lead + (i + 1) * sec_per_mora - 0.03)
            for i in range(len(ann.moras))]


def test_token_durations_cover_clip_and_respect_minimum(ann):
    total = 80
    dur = token_durations(ann, _even_spans(ann), np.ones(len(ann.ps) + 2), total)
    assert dur.sum() == total
    assert len(dur) == len(ann.ps) + 2
    assert (dur >= 1).all()
    assert dur[0] >= 10  # 0.3 s leading silence (minus absorb) goes to BOS


def test_marks_get_one_frame_between_adjacent_moras(ann):
    dur = token_durations(ann, _even_spans(ann), np.ones(len(ann.ps) + 2), 80)
    for i, c in enumerate(ann.ps):
        if c in (UPSTEP, DOWNSTEP):
            assert dur[i + 1] == 1


def test_token_durations_rejects_too_short(ann):
    assert token_durations(ann, _even_spans(ann), np.ones(len(ann.ps) + 2), len(ann.ps)) is None


def test_enforce_min_one_borrows_locally():
    out = _enforce_min_one(np.array([5, 0, 3, 1, 9]))
    assert out.tolist() == [4, 1, 3, 1, 9]


def _curve_for(ann, dur, flip=False):
    """Synthetic F0 that follows (or inverts) the labelled H/L pattern."""
    starts = np.concatenate([[0], np.cumsum(dur)])
    f0 = np.zeros(2 * int(sum(dur)), dtype=np.float32)
    for i, m in enumerate(ann.char_mora):
        if m >= 0:
            high = ann.moras[m].high != flip
            f0[2 * starts[i + 1]: 2 * starts[i + 2]] = 220.0 if high else 150.0
    return f0


def test_accent_agreement_matches_labels(ann):
    dur = token_durations(ann, _even_spans(ann), np.ones(len(ann.ps) + 2), 80)
    c, t = accent_agreement(ann, dur, _curve_for(ann, dur))
    assert t == sum(ann.ps.count(m) for m in (UPSTEP, DOWNSTEP)) and c == t
    c, t = accent_agreement(ann, dur, _curve_for(ann, dur, flip=True))
    assert c == 0 and t > 0


def test_accent_agreement_skips_unvoiced(ann):
    dur = token_durations(ann, _even_spans(ann), np.ones(len(ann.ps) + 2), 80)
    assert accent_agreement(ann, dur, np.zeros(160, dtype=np.float32)) == (0, 0)
