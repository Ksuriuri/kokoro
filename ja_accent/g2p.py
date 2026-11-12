"""Japanese G2P with pitch-accent marks for Kokoro.

Readings and accent come from pyopenjtalk (OpenJTalk NJD features); kana are
mapped to phonemes with the same table as misaki's cutlet G2P, which is what
Kokoro-82M v1.0 was trained on. Two marks are inserted on top of that:

    UPSTEP   'ꜛ'  before the first high mora after a low one (pitch rises)
    DOWNSTEP 'ꜜ'  right after the accent nucleus (pitch falls next)

    箸が  hashi+ga, acc=1 (atamadaka)  ->  haꜜɕiɡa
    橋が  hashi+ga, acc=2 (odaka)      ->  haꜛɕiꜜɡa
    端が  hashi+ga, acc=0 (heiban)     ->  haꜛɕiɡa

Accent phrases are separated by spaces. Stock Kokoro drops the marks because
they are not in its vocab, so the output also works with an unmodified model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import jaconv
import pyopenjtalk
from misaki.cutlet import HEPBURN, SUTEGANA

UPSTEP = 'ꜛ'
DOWNSTEP = 'ꜜ'
ACCENT_MARKS = (UPSTEP, DOWNSTEP)

OPENERS = frozenset('“«')
PUNCT = {
    '、': ',', '，': ',', ',': ',',
    '。': '.', '．': '.', '.': '.',
    '？': '?', '?': '?',
    '！': '!', '!': '!',
    '：': ':', ':': ':',
    '；': ';', ';': ';',
    '…': '…',
    '「': '“', '」': '”', '『': '“', '』': '”', '“': '“', '”': '”',
    '（': '«', '）': '»', '(': '«', ')': '»', '《': '«', '》': '»',
    '～': '—', '〜': '—', '—': '—',
}

# IPA -> MMS aligner alphabet (a-z), longest keys first.
_ROMAN = [
    ('ʨ', 'ch'), ('ʥ', 'j'), ('ʦ', 'ts'), ('ʣ', 'z'), ('ɕ', 'sh'), ('ç', 'h'),
    ('ɸ', 'f'), ('β', 'w'), ('ɾ', 'r'), ('ɡ', 'g'), ('ɯ', 'u'), ('ɨ', 'u'),
    ('ɲ', 'ny'), ('ŋ', 'n'), ('ɴ', 'n'), ('ʲ', 'y'), ('ᵝ', 'w'),
]
_VOWELS = frozenset('aeiou')


@dataclass
class _Phrase:
    moras: List[str]
    acc: int
    words: List[str] = field(default_factory=list)


@dataclass
class Mora:
    kana: str
    ipa: str
    roman: str      # aligner transcription, '' when the mora has no own sound (some ッ)
    high: bool
    phrase: int


@dataclass
class Annotation:
    ps: str
    char_mora: List[int]   # per ps char: index into moras, -1 for marks/spaces/punct
    moras: List[Mora]


def _split_moras(hira: str) -> List[str]:
    moras = []
    for ch in hira:
        if moras and ch in SUTEGANA and moras[-1] not in ('ー', 'っ', 'ん'):
            moras[-1] += ch
        else:
            moras.append(ch)
    return moras


def _moraic_nasal(next_mora: Optional[str]) -> str:
    tnk = HEPBURN.get(next_mora[0]) if next_mora else None
    if tnk:
        if tnk[0] in 'mpb':
            return 'm'
        if tnk[0] in 'kɡ':
            return 'ŋ'
        if tnk.startswith(('ɲ', 'ʨ', 'ʥ')):
            return 'ɲ'
        if tnk[0] in 'ntdɾz':
            return 'n'
    return 'ɴ'


def _mora_ipa(mora: str, next_mora: Optional[str]) -> str:
    if mora == 'ー':
        return 'ː'
    if mora == 'っ':
        return 'ʔ'
    if mora == 'ん':
        return _moraic_nasal(next_mora)
    if mora in HEPBURN:
        return HEPBURN[mora]
    if len(mora) == 2 and mora[0] in HEPBURN and mora[1] in HEPBURN:
        return HEPBURN[mora[0]][:-1] + HEPBURN[mora[1]]
    return ''


def _roman(ipa: str) -> str:
    for k, v in _ROMAN:
        ipa = ipa.replace(k, v)
    return ipa.replace('yi', 'i')


def _fill_special_romans(moras: List[Mora]) -> None:
    """Long vowels repeat the previous vowel; sokuon takes the next consonant."""
    for i, m in enumerate(moras):
        if m.ipa == 'ː':
            prev = moras[i - 1].roman if i > 0 else ''
            m.roman = prev[-1] if prev and prev[-1] in _VOWELS else ''
        elif m.ipa == 'ʔ':
            nxt = moras[i + 1].roman if i + 1 < len(moras) else ''
            m.roman = nxt[0] if nxt and nxt[0] not in _VOWELS else ''


def _high_pattern(n: int, acc: int) -> List[bool]:
    if acc == 0:
        return [i > 0 for i in range(n)]
    if acc == 1:
        return [i == 0 for i in range(n)]
    return [0 < i < acc for i in range(n)]


def strip_accents(ps: str) -> str:
    return ''.join(c for c in ps if c not in ACCENT_MARKS)


class JAAccentG2P:
    """Drop-in replacement for misaki's ``ja.JAG2P`` in ``KPipeline``.

    ``pipeline.g2p = JAAccentG2P()`` on a ``KPipeline(lang_code='j')``.
    Returns ``(phonemes, None)`` like the cutlet version.
    """

    def __init__(self, marks: bool = True):
        self.marks = marks

    def phrases(self, text: str) -> List[object]:
        """Accent phrases (``_Phrase``) interleaved with punctuation strings."""
        items: List[object] = []
        current: Optional[_Phrase] = None
        for w in pyopenjtalk.run_frontend(text):
            if w['mora_size'] == 0:
                p = ''.join(PUNCT.get(c, '') for c in w['string'])
                if p:
                    items.append(p)
                    current = None
                continue
            moras = _split_moras(jaconv.kata2hira(w['pron'].replace('’', '')))
            if current is not None and (w['chain_flag'] == 1 or moras[0] == 'ー'):
                current.moras += moras
                current.words.append(w['string'])
            else:
                current = _Phrase(moras, w['acc'], [w['string']])
                items.append(current)
        for item in items:
            if isinstance(item, _Phrase):
                item.acc = min(max(item.acc, 0), len(item.moras))
        return items

    def annotate(self, text: str) -> Annotation:
        items = self.phrases(text)
        chars: List[Tuple[str, int]] = []  # (char, mora index or -1)
        moras: List[Mora] = []

        def ends_with_space():
            return chars and chars[-1][0] == ' '

        phrase_idx = 0
        for idx, item in enumerate(items):
            if isinstance(item, _Phrase):
                nxt = items[idx + 1] if idx + 1 < len(items) else None
                next_mora = nxt.moras[0] if isinstance(nxt, _Phrase) else None
                if chars and not ends_with_space() and chars[-1][0] not in OPENERS:
                    chars.append((' ', -1))
                high = _high_pattern(len(item.moras), item.acc)
                for i, kana in enumerate(item.moras):
                    if self.marks and i > 0 and high[i] and not high[i - 1]:
                        chars.append((UPSTEP, -1))
                    ipa = _mora_ipa(kana, item.moras[i + 1] if i + 1 < len(item.moras) else next_mora)
                    if ipa:
                        moras.append(Mora(kana, ipa, _roman(ipa), high[i], phrase_idx))
                        chars.extend((c, len(moras) - 1) for c in ipa)
                    if self.marks and item.acc > 0 and i == item.acc - 1:
                        chars.append((DOWNSTEP, -1))
                phrase_idx += 1
                continue
            for c in item:
                if c in OPENERS:
                    if chars and not ends_with_space():
                        chars.append((' ', -1))
                    chars.append((c, -1))
                else:
                    while ends_with_space():
                        chars.pop()
                    chars.extend([(c, -1), (' ', -1)])

        # Collapse spaces, then cutlet's rule of gluing sokuon to its neighbours.
        out: List[Tuple[str, int]] = []
        for c in chars:
            if c[0] == ' ' and (not out or out[-1][0] == ' '):
                continue
            out.append(c)
        while out and out[-1][0] == ' ':
            out.pop()
        glued = []
        for i, c in enumerate(out):
            if c[0] == ' ':
                prev = out[i - 1][0] if i > 0 else ''
                nxt = out[i + 1][0] if i + 1 < len(out) else ''
                if nxt == 'ʔ' and prev not in '!",.:;?»—…”':
                    continue
                if prev == 'ʔ' and nxt not in '"«“':
                    continue
            glued.append(c)

        _fill_special_romans(moras)
        return Annotation(''.join(c for c, _ in glued), [m for _, m in glued], moras)

    def __call__(self, text: str) -> Tuple[str, None]:
        return self.annotate(text).ps, None
