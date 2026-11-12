"""Synthesize Japanese with accent marks.

    # exported model
    uv run python -m ja_accent.ft.infer --model export/kokoro-ja-accent.pth \\
        --config export/config.json --voice export/spk0.pt --text 箸と橋。 --out out.wav
    # straight from a training checkpoint
    uv run python -m ja_accent.ft.infer --ckpt runs/exp1/ckpt_best.pt --text 箸と橋。 --out out.wav
    # stock Kokoro for comparison (marks are dropped)
    uv run python -m ja_accent.ft.infer --voice jf_alpha --text 箸と橋。 --out stock.wav
"""

import argparse

import numpy as np
import soundfile as sf
import torch

from ..g2p import JAAccentG2P
from .constants import REPO_ID, SAMPLE_RATE
from .forward import load_checkpoint_into


def build_pipeline(args):
    from kokoro import KModel, KPipeline

    voice = args.voice
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model = KModel(repo_id=ck['repo_id'], config=ck['config'])
        vec = load_checkpoint_into(model, ck, args.speaker)
        voice = vec.reshape(1, 1, -1).repeat(510, 1, 1)
    else:
        model = KModel(repo_id=REPO_ID, config=args.config, model=args.model)
    model = model.to(args.device).eval()
    pipe = KPipeline(lang_code='j', repo_id=REPO_ID, model=model)
    pipe.g2p = JAAccentG2P()
    return pipe, voice


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text', required=True)
    ap.add_argument('--out', default='out.wav')
    ap.add_argument('--ckpt')
    ap.add_argument('--speaker', default=None, help='speaker key in --ckpt (default: first)')
    ap.add_argument('--model', default=None, help='exported .pth (default: stock weights)')
    ap.add_argument('--config', default=None, help='exported config.json (default: stock)')
    ap.add_argument('--voice', default='jf_alpha', help='voicepack .pt or stock voice name')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    pipe, voice = build_pipeline(args)
    chunks = []
    for r in pipe(args.text, voice=voice, speed=args.speed):
        print(r.phonemes)
        chunks.append(r.audio.cpu().numpy())
    sf.write(args.out, np.concatenate(chunks), SAMPLE_RATE)
    print(f'[infer] wrote {args.out}')


if __name__ == '__main__':
    main()
