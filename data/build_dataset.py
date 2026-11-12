"""Build data/metadata.tsv for ja_accent.ft.prepare from both recording sets.

    new_data/        ITA corpus, texts in ja_accent/corpus/recording_script.tsv
    raw_wav/         SparkTalk recordings (ffmpeg from raw_data/*.m4a),
                     texts in raw_script.tsv

Clips longer than --max-sec are cut at punctuation where the speaker pauses,
located by aligning the whole clip; pieces go to raw_split/.

    uv run python data/build_dataset.py
    uv run python -m ja_accent.ft.prepare --metadata data/metadata.tsv --wav-dir data --out data/prep
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ja_accent.ft.align import MoraAligner  # noqa: E402
from ja_accent.ft.data import load_wav
from ja_accent.ft.prepare import read_metadata
from ja_accent.g2p import JAAccentG2P

ROOT = Path(__file__).resolve().parent
ITA_SCRIPT = ROOT.parent / 'ja_accent' / 'corpus' / 'recording_script.tsv'
OPEN, CLOSE = '「（', '」）'


def split_points(text):
    """Text pieces ending at punctuation outside brackets."""
    pieces, depth, start = [], 0, 0
    for i, c in enumerate(text):
        depth += (c in OPEN) - (c in CLOSE)
        prev, nxt = text[i - 1] if i else '', text[i + 1] if i + 1 < len(text) else ''
        if c == '、' and prev.isdigit() and nxt.isdigit():  # 2、3分
            continue
        if depth == 0 and (c in '。？！、' or (c in CLOSE and nxt in OPEN)):
            pieces.append(text[start:i + 1])
            start = i + 1
    if text[start:]:
        pieces.append(text[start:])
    return pieces


def piece_spans(aligner, g2p, wav24, pieces):
    """(start, end) seconds of each piece's speech from one whole-clip alignment."""
    anns = [g2p.annotate(p) for p in pieces]

    class Joined:
        moras = [m for a in anns for m in a.moras]

    spans, score = aligner.align(wav24, Joined)
    out, k = [], 0
    for a in anns:
        s = [x for x in spans[k:k + len(a.moras)] if x is not None]
        k += len(a.moras)
        out.append((s[0][0], s[-1][1]) if s else None)
    return out, score


def split_clip(aligner, g2p, path, text, max_sec, min_pause):
    """[(start_sec, end_sec, text)] covering the clip, each <= max_sec where possible."""
    wav24 = load_wav(path)
    total = wav24.shape[0] / 24_000
    pieces = split_points(text)
    spans, score = piece_spans(aligner, g2p, wav24, pieces)
    # cut[i]: time between piece i and i+1 if the speaker pauses there
    cuts = {}
    for i in range(len(pieces) - 1):
        a, b = spans[i], spans[i + 1]
        if a and b and b[0] - a[1] >= min_pause:
            cuts[i] = (a[1] + b[0]) / 2
    segs, seg_start, first, last_cut = [], 0.0, 0, None
    for i in range(len(pieces)):
        end = spans[i][1] if spans[i] and i < len(pieces) - 1 else total
        if end - seg_start > max_sec and last_cut is not None:
            segs.append((seg_start, cuts[last_cut], ''.join(pieces[first:last_cut + 1])))
            seg_start, first, last_cut = cuts[last_cut], last_cut + 1, None
        if i in cuts:
            last_cut = i
    segs.append((seg_start, total, ''.join(pieces[first:])))
    return segs, score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-sec', type=float, default=19.5, help='training skips clips over --max-frames (20 s)')
    ap.add_argument('--min-pause', type=float, default=0.15)
    args = ap.parse_args()

    lines = [f'new_data/{r["file"]}\t{r["text"]}' for r in read_metadata(ITA_SCRIPT)]
    g2p, aligner = JAAccentG2P(), MoraAligner('cpu')
    split_dir = ROOT / 'raw_split'
    split_dir.mkdir(exist_ok=True)
    for r in read_metadata(ROOT / 'raw_script.tsv'):
        src = ROOT / 'raw_wav' / r['file']
        info = sf.info(str(src))
        if info.duration <= args.max_sec:
            lines.append(f'raw_wav/{r["file"]}\t{r["text"]}')
            continue
        segs, score = split_clip(aligner, g2p, src, r['text'], args.max_sec, args.min_pause)
        x, sr = sf.read(str(src), dtype='int16')
        stem = Path(r['file']).stem
        print(f'{stem} {info.duration:.1f}s align={score:.3f} -> {len(segs)} pieces')
        for k, (a, b, t) in enumerate(segs):
            name = f'{stem}_{k + 1}.wav'
            sf.write(split_dir / name, x[int(a * sr):int(b * sr)], sr, subtype='PCM_16')
            lines.append(f'raw_split/{name}\t{t}')
            flag = '  (over max-sec)' if b - a > args.max_sec else ''
            print(f'  {name} {b - a:5.1f}s  {t}{flag}')
    (ROOT / 'metadata.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {len(lines)} rows to {ROOT / "metadata.tsv"}')


if __name__ == '__main__':
    main()
