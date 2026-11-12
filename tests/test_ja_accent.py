import pytest

pytest.importorskip('pyopenjtalk')
pytest.importorskip('misaki.cutlet')

from ja_accent import ACCENT_IDS, DOWNSTEP, UPSTEP, JAAccentG2P, build_config, strip_accents
from ja_accent.g2p import _high_pattern


@pytest.fixture(scope='module')
def g2p():
    return JAAccentG2P()


@pytest.mark.parametrize('text, expected', [
    ('箸が', 'haꜜɕiɡa'),
    ('橋が', 'haꜛɕiꜜɡa'),
    ('端が', 'haꜛɕiɡa'),
])
def test_hashi_minimal_pairs(g2p, text, expected):
    assert g2p(text)[0] == expected


def test_minimal_pairs_are_distinct(g2p):
    outs = {g2p(t)[0] for t in ('箸が', '橋が', '端が')}
    assert len(outs) == 3
    assert len({strip_accents(o) for o in outs}) == 1


@pytest.mark.parametrize('n, acc, pattern', [
    (4, 0, 'LHHH'),
    (4, 1, 'HLLL'),
    (4, 2, 'LHLL'),
    (4, 4, 'LHHH'),
    (1, 0, 'L'),
    (1, 1, 'H'),
])
def test_high_pattern(n, acc, pattern):
    assert ''.join('H' if h else 'L' for h in _high_pattern(n, acc)) == pattern


def test_marks_off_matches_stripped(g2p):
    plain = JAAccentG2P(marks=False)
    for text in ('東京へ行きました、こんにちは！', '「ちょっと待って」と言った。', '新聞を読んでいますか？'):
        assert plain(text)[0] == strip_accents(g2p(text)[0])


def test_chain_merges_particles_into_phrase(g2p):
    items = [i for i in g2p.phrases('行きました') if not isinstance(i, str)]
    assert len(items) == 1
    assert items[0].acc == 3


def test_punctuation(g2p):
    ps, _ = g2p('はい、そうです。本当？')
    assert ps.count(',') == 1 and ps.endswith('?') and '. ' in ps


def test_moraic_nasal_assimilation(g2p):
    ps = strip_accents(g2p('新聞')[0])
    assert ps.startswith('ɕimbɯ')


def test_build_config_uses_free_slots():
    base = {'n_token': 178, 'vocab': {'a': 1, 'b': 2}}
    config = build_config(base)
    assert config['vocab'][UPSTEP] == ACCENT_IDS[UPSTEP]
    assert config['vocab'][DOWNSTEP] == ACCENT_IDS[DOWNSTEP]
    assert config['n_token'] == 178
    assert UPSTEP not in base['vocab']
    assert build_config(config) == config


def test_build_config_rejects_taken_slot():
    with pytest.raises(AssertionError):
        build_config({'n_token': 178, 'vocab': {'x': ACCENT_IDS[UPSTEP]}})
