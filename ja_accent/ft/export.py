"""Export a training checkpoint for the stock ``kokoro`` library.

Writes to --out:
    config.json            Kokoro config with the accent marks in the vocab
    kokoro-ja-accent.pth   full weights in kokoro-v1_0.pth format (marks folded in)
    <speaker>.pt           [510, 1, 256] voicepack per speaker

    uv run python -m ja_accent.ft.export --ckpt runs/exp1/ckpt_best.pt --out export/

Use with ``KModel(config=..., model=...)`` and ``pipeline.g2p = JAAccentG2P()``.
"""

import argparse
import json
from pathlib import Path

import torch

from .constants import MAX_PHONEMES
from .forward import load_checkpoint_into

WEIGHT_MODULES = ('bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='export')
    args = ap.parse_args(argv)

    from kokoro import KModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = KModel(repo_id=ck['repo_id'], config=ck['config'])
    load_checkpoint_into(model, ck)

    (out / 'config.json').write_text(json.dumps(ck['config'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    torch.save({m: getattr(model, m).state_dict() for m in WEIGHT_MODULES}, out / 'kokoro-ja-accent.pth')
    for spk, vec in ck['voices'].items():
        torch.save(vec.reshape(1, 1, -1).repeat(MAX_PHONEMES, 1, 1), out / f'{spk}.pt')
    print(f"[export] step {ck['step']} -> {out}: config.json, kokoro-ja-accent.pth, "
          f"{', '.join(f'{s}.pt' for s in ck['voices'])}")


if __name__ == '__main__':
    main()
