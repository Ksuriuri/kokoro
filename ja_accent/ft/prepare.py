"""Prepare a Japanese dataset for finetuning.

Input: a directory of recordings and a TSV with one line per clip:

    clip_0001.wav<TAB>箸を落としてしまいました。
    clip_0002.wav<TAB>橋の向こうに駅があります。<TAB>speaker_a   (optional 3rd column)

Output (``--out``):
    wavs/<id>.wav      24 kHz mono, trimmed to whole duration frames
    f0/<id>.npy        WORLD F0 at 80 fps
    manifest.jsonl     ps, per-token durations, alignment / accent scores
    dropped.jsonl      clips that could not be used, with the reason
    accent_review.tsv  clips sorted by how well the recorded pitch matches
                       the accent labels; check the top of this file by ear

    uv run python -m ja_accent.ft.prepare --metadata data/metadata.tsv \\
        --wav-dir data/wavs --out data/prep
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from ..g2p import JAAccentG2P, _high_pattern
from ..vocab import build_config, init_accent_embeddings, load_base_config
from .align import MoraAligner, speech_end_frame, token_durations
from .constants import MAX_PHONEMES, REPO_ID, SAMPLE_RATE, SAMPLES_PER_FRAME
from .data import load_wav
from .forward import predict_durations, to_input_ids
from .pitch import accent_agreement, extract_f0


def read_metadata(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            raise SystemExit(f'{path}:{n}: expected "file<TAB>text[<TAB>speaker]"')
        rows.append({'file': parts[0].strip(), 'text': parts[1].strip(),
                     'speaker': parts[2].strip() if len(parts) > 2 and parts[2].strip() else 'spk0'})
    return rows


def load_model(device):
    from kokoro import KModel
    model = KModel(repo_id=REPO_ID, config=build_config(load_base_config(REPO_ID))).to(device).eval()
    init_accent_embeddings(model)
    return model


def phrase_summary(g2p, text):
    out = []
    for item in g2p.phrases(text):
        if not isinstance(item, str):
            hl = ''.join('H' if h else 'L' for h in _high_pattern(len(item.moras), item.acc))
            out.append(f'{"".join(item.words)}[{hl}]')
    return ' '.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--wav-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--init-voice', default='jf_alpha', help='stock voice for relative duration weights')
    ap.add_argument('--min-align-score', type=float, default=0.2)
    ap.add_argument('--f0-floor', type=float, default=60.0)
    ap.add_argument('--f0-ceil', type=float, default=700.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / 'wavs').mkdir(parents=True, exist_ok=True)
    (out / 'f0').mkdir(exist_ok=True)
    manifest_path = out / 'manifest.jsonl'
    done = set()
    if manifest_path.exists():
        done = {json.loads(l)['id'] for l in manifest_path.read_text(encoding='utf-8').splitlines() if l.strip()}

    rows = read_metadata(args.metadata)
    g2p = JAAccentG2P()
    model = load_model(args.device)
    aligner = MoraAligner(args.device)
    pack = torch.load(hf_hub_download(REPO_ID, f'voices/{args.init_voice}.pt'), weights_only=True)

    kept = dropped = 0
    with open(manifest_path, 'a', encoding='utf-8') as fm, open(out / 'dropped.jsonl', 'a', encoding='utf-8') as fd:
        def drop(uid, reason, **extra):
            nonlocal dropped
            dropped += 1
            fd.write(json.dumps({'id': uid, 'reason': reason, **extra}, ensure_ascii=False) + '\n')

        for n, row in enumerate(rows, 1):
            uid = Path(row['file']).stem
            if uid in done:
                continue
            if n % 50 == 0:
                print(f'[prepare] {n}/{len(rows)} kept={kept} dropped={dropped}')
            src = Path(args.wav_dir) / row['file']
            if not src.exists():
                drop(uid, 'missing_wav')
                continue
            wav = load_wav(src)
            frames = wav.shape[0] // SAMPLES_PER_FRAME
            wav = wav[: frames * SAMPLES_PER_FRAME]
            if frames < 12:
                drop(uid, 'too_short')
                continue
            peak = float(wav.abs().max())
            if peak >= 0.999:
                drop(uid, 'clipping', peak=peak)
                continue

            ann = g2p.annotate(row['text'])
            if not ann.ps or len(ann.ps) > MAX_PHONEMES:
                drop(uid, 'bad_ps_len', n=len(ann.ps))
                continue
            oov = sorted({c for c in ann.ps if c not in model.vocab})
            if oov:
                drop(uid, 'oov_phoneme', chars=oov, ps=ann.ps)
                continue

            spans, score = aligner.align(wav, ann)
            if score < args.min_align_score:
                drop(uid, 'low_align_score', score=round(score, 4), text=row['text'])
                continue
            ids = to_input_ids(model.vocab, ann.ps).to(args.device)
            ref = pack[len(ann.ps) - 1].to(args.device)
            weights = predict_durations(model, ids, ref).clamp_min(0.25).cpu().numpy()
            dur = token_durations(ann, spans, weights, frames, speech_end_frame(wav))
            if dur is None:
                drop(uid, 'no_durations')
                continue

            f0 = extract_f0(wav.numpy(), 2 * frames, args.f0_floor, args.f0_ceil)
            correct, checked = accent_agreement(ann, dur, f0)

            sf.write(out / 'wavs' / f'{uid}.wav', wav.numpy(), SAMPLE_RATE, subtype='PCM_16')
            np.save(out / 'f0' / f'{uid}.npy', f0)
            fm.write(json.dumps({
                'id': uid, 'wav': f'{uid}.wav', 'text': row['text'], 'ps': ann.ps,
                'dur': dur.tolist(), 'frames': int(frames), 'align_score': round(score, 4),
                'accent_correct': correct, 'accent_checked': checked, 'speaker': row['speaker'],
            }, ensure_ascii=False) + '\n')
            fm.flush()
            kept += 1

    manifest = [json.loads(l) for l in manifest_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    hours = sum(r['frames'] for r in manifest) * SAMPLES_PER_FRAME / SAMPLE_RATE / 3600
    c = sum(r['accent_correct'] for r in manifest)
    t = sum(r['accent_checked'] for r in manifest)
    print(f'[prepare] done: {len(manifest)} clips ({hours:.2f} h) kept, {dropped} dropped this run')
    print(f'[prepare] recorded pitch agrees with accent labels on {c}/{t} marks ({c / max(t, 1):.1%})')

    review = sorted(manifest, key=lambda r: (r['accent_correct'] / r['accent_checked']) if r['accent_checked'] else 1.0)
    with open(out / 'accent_review.tsv', 'w', encoding='utf-8') as f:
        f.write('agreement\tid\ttext\tphrases\tps\n')
        for r in review:
            agree = f"{r['accent_correct']}/{r['accent_checked']}"
            f.write(f"{agree}\t{r['id']}\t{r['text']}\t{phrase_summary(g2p, r['text'])}\t{r['ps']}\n")


if __name__ == '__main__':
    main()
