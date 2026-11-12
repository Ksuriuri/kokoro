"""Recording scripts: build, check readings, and report coverage."""

import difflib
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import jaconv
from misaki.cutlet import HEPBURN

from .g2p import JAAccentG2P, _mora_ipa, _Phrase, _split_moras

CORPUS_DIR = Path(__file__).resolve().parent / 'corpus'
ITA_FILES = ('ita_emotion.txt', 'ita_recitation.txt')
ACCENT_FILE = 'accent_sentences.txt'

# Read-speech rate used for duration estimates.
MORAS_PER_SEC = 7.5
PAUSE_SEC = 0.35

_LONG = [('ɨɯ', 'ɨː'), ('ɯɯ', 'ɯː'), ('oɯ', 'oː'), ('oo', 'oː'), ('ei', 'eː'),
         ('ee', 'eː'), ('aa', 'aː'), ('ii', 'iː')]


def read_sentences(path) -> List[Tuple[str, str, Optional[str]]]:
    """(id, text, kana) from ITA (``ID:text,kana``), TSV (``id<TAB>text``) or plain lines."""
    path = Path(path)
    out = []
    for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'^([A-Z]+\d+_\d+):(.*?),([^,]*)$', line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
        elif '\t' in line:
            uid, text = line.split('\t')[:2]
            out.append((Path(uid).stem, text, None))
        else:
            out.append((f'{path.stem}_{n:04d}', line, None))
    return out


def _kana_ipa(kana: str) -> str:
    moras = _split_moras(jaconv.kata2hira(re.sub(r'[、。？！，．,.?!「」『』（）\s・]', '', kana)))
    return ''.join(_mora_ipa(m, moras[i + 1] if i + 1 < len(moras) else None) for i, m in enumerate(moras))


def _norm(ipa: str) -> str:
    for a, b in _LONG:
        ipa = ipa.replace(a, b)
    return ipa


def reading_similarity(g2p: JAAccentG2P, text: str, kana: str) -> Tuple[float, str, str]:
    ours = _norm(''.join(m.ipa for m in g2p.annotate(text).moras))
    ref = _norm(_kana_ipa(kana))
    return difflib.SequenceMatcher(None, ours, ref).ratio(), ours, ref


def accent_type(phrase: _Phrase) -> str:
    n, acc = len(phrase.moras), phrase.acc
    if acc == 0:
        return 'heiban'
    if acc == 1:
        return 'atamadaka'
    return 'odaka' if acc == n else 'nakadaka'


def stats(sentences, g2p: JAAccentG2P) -> dict:
    types, kana, lengths = Counter(), Counter(), []
    total_moras = pauses = 0
    for _, text, _ in sentences:
        ann = g2p.annotate(text)
        lengths.append(len(ann.ps))
        total_moras += len(ann.moras)
        pauses += 1 + sum(ann.ps.count(c) for c in ',.?!')
        for item in g2p.phrases(text):
            if isinstance(item, _Phrase):
                types[accent_type(item)] += 1
                kana.update(item.moras)
    inventory = {k for k in HEPBURN if all('\u3041' <= c <= '\u3096' for c in k)} | {'ん', 'っ', 'ー'}
    return {
        'sentences': len(sentences),
        'minutes': (total_moras / MORAS_PER_SEC + pauses * PAUSE_SEC) / 60,
        'ps_len': lengths,
        'accent_types': types,
        'mora_coverage': (len(inventory & set(kana)), len(inventory)),
        'missing_moras': sorted(inventory - set(kana)),
    }


def default_sentences():
    items = []
    for name in ITA_FILES:
        items += read_sentences(CORPUS_DIR / name)
    items += [(f'accent_{i:04d}', t, k) for i, (_, t, k) in enumerate(read_sentences(CORPUS_DIR / ACCENT_FILE), 1)]
    return items
