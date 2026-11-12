"""CLI for the accent G2P.

    uv run python -m ja_accent config --out ja_accent/config.json
    uv run python -m ja_accent show 箸が欲しい。 橋が見える。
    uv run python -m ja_accent show --file script.txt
    uv run python -m ja_accent compare --file script.txt
    uv run python -m ja_accent script --out recording_script.tsv
    uv run python -m ja_accent stats --file recording_script.tsv
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

from .g2p import DOWNSTEP, UPSTEP, JAAccentG2P, _high_pattern, _Phrase, strip_accents
from .vocab import build_config, load_base_config


def _read_lines(args):
    if args.file:
        return [l.strip() for l in Path(args.file).read_text(encoding='utf-8').splitlines() if l.strip()]
    return args.text


def cmd_config(args):
    config = build_config(load_base_config(args.repo_id))
    Path(args.out).write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {args.out}: {UPSTEP}={config["vocab"][UPSTEP]} {DOWNSTEP}={config["vocab"][DOWNSTEP]}')


def cmd_show(args):
    g2p = JAAccentG2P()
    for text in _read_lines(args):
        ps, _ = g2p(text)
        print(text)
        print(f'  ps: {ps}')
        if args.verbose:
            for item in g2p.phrases(text):
                if isinstance(item, _Phrase):
                    hl = ''.join('H' if h else 'L' for h in _high_pattern(len(item.moras), item.acc))
                    print(f'    {"".join(item.words)}  {"".join(item.moras)}  acc={item.acc}  {hl}')


def cmd_compare(args):
    from misaki import ja

    ours, ref = JAAccentG2P(), ja.JAG2P()
    lines = _read_lines(args)
    total = 0.0
    worst = []
    for text in lines:
        a = strip_accents(ours(text)[0]).replace(' ', '')
        b = ref(text)[0].replace(' ', '')
        r = difflib.SequenceMatcher(None, a, b).ratio()
        total += r
        worst.append((r, text, a, b))
    worst.sort()
    print(f'{len(lines)} lines, mean phoneme similarity to cutlet: {total / max(len(lines), 1):.4f}')
    for r, text, a, b in worst[:args.top]:
        if r >= 1.0:
            break
        print(f'\n{r:.3f}  {text}\n  ours:   {a}\n  cutlet: {b}')


def cmd_script(args):
    from .script import default_sentences, reading_similarity

    g2p = JAAccentG2P()
    keep, flagged = [], []
    for uid, text, kana in default_sentences():
        r = reading_similarity(g2p, text, kana) if kana else (1.0, '', '')
        (keep if r[0] >= args.min_reading else flagged).append((uid, text, kana, *r))
    Path(args.out).write_text(''.join(f'{u}.wav\t{t}\n' for u, t, *_ in keep), encoding='utf-8')
    print(f'wrote {args.out}: {len(keep)} sentences')
    if flagged:
        side = Path(args.out).with_name(Path(args.out).stem + '_misread.tsv')
        side.write_text('# OpenJTalk reads these differently from the corpus kana; the recording would\n'
                        '# not match its labels. Rewrite the misread words in kana to use them.\n'
                        + ''.join(f'{u}.wav\t{t}\t{k}\n' for u, t, k, *_ in flagged), encoding='utf-8')
        print(f'{len(flagged)} sentences OpenJTalk misreads were moved to {side}:')
        for uid, text, kana, r, ours, ref in sorted(flagged, key=lambda x: x[3]):
            print(f'  {r:.2f} {uid} {text}\n       openjtalk: {ours}\n       kana:      {ref}')


def cmd_stats(args):
    import numpy as np

    from .script import read_sentences, stats

    items = [s for path in args.file for s in read_sentences(path)]
    s = stats(items, JAAccentG2P())
    lens = np.array(s['ps_len'])
    print(f"{s['sentences']} sentences, ~{s['minutes']:.0f} min of read speech")
    print(f"phoneme length: median {np.median(lens):.0f}, p10 {np.percentile(lens, 10):.0f}, "
          f"p90 {np.percentile(lens, 90):.0f}, max {lens.max()}")
    hist = np.histogram(lens, bins=[0, 20, 40, 60, 80, 120, 510])[0]
    print('  length buckets 0-20/20-40/40-60/60-80/80-120/120+: ' + ' / '.join(map(str, hist)))
    total = sum(s['accent_types'].values())
    print('accent phrases: ' + ', '.join(f'{k} {v} ({v / total:.0%})' for k, v in s['accent_types'].most_common()))
    have, inv = s['mora_coverage']
    print(f'mora coverage: {have}/{inv}' + (f"; missing: {' '.join(s['missing_moras'])}" if s['missing_moras'] else ''))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m ja_accent', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('config', help='write Kokoro config.json with the accent marks added')
    p.add_argument('--out', default='ja_accent/config.json')
    p.add_argument('--repo-id', default='hexgrad/Kokoro-82M')
    p.set_defaults(fn=cmd_config)

    for name, fn, help_ in (('show', cmd_show, 'print accented phonemes'),
                            ('compare', cmd_compare, 'diff phonemes (marks stripped) against cutlet')):
        p = sub.add_parser(name, help=help_)
        p.add_argument('text', nargs='*')
        p.add_argument('--file')
        p.set_defaults(fn=fn)
    sub.choices['show'].add_argument('-v', '--verbose', action='store_true')
    sub.choices['compare'].add_argument('--top', type=int, default=15)

    p = sub.add_parser('script', help='write the recording script (ITA corpus + accent sentences) as TSV')
    p.add_argument('--out', default='recording_script.tsv')
    p.add_argument('--min-reading', type=float, default=0.95,
                   help='flag sentences whose OpenJTalk reading differs from the corpus kana')
    p.set_defaults(fn=cmd_script, text=None, file=True)

    p = sub.add_parser('stats', help='length, accent-type and mora coverage of sentence files')
    p.add_argument('--file', action='append', required=True)
    p.set_defaults(fn=cmd_stats, text=None)

    args = ap.parse_args(argv)
    if args.cmd in ('show', 'compare') and not (args.text or args.file):
        ap.error('give text or --file')
    args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
