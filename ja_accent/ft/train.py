"""Finetune Kokoro-82M on a Japanese voice with pitch-accent marks.

Trained:
  voice     a 256-dim style vector per speaker (first 128 dims condition the
            decoder, last 128 the prosody predictor), init from --init-voice
  marks     offsets for the accent-mark embeddings (own learning rate)
  text      bert, bert_encoder, text_encoder: must learn what the marks mean
  acoustic  predictor and decoder body; the vocoder stays frozen by default

The first --warm-steps steps train only voice + marks.

Losses (teacher-forced with aligned durations; the decoder sees real F0/energy):
  mel L1, multi-resolution STFT, duration L1, F0 (Hz smooth-L1 and voiced
  log-F0 L1), energy smooth-L1

Every --eval-every steps the held-out minimal-pair sentences are synthesized
on the real inference path and scored for accent agreement (eval/ has wavs).

    uv run python -m ja_accent.ft.train --data data/prep --out runs/exp1
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from ..g2p import JAAccentG2P
from ..vocab import build_config, init_accent_embeddings, load_base_config
from .constants import REPO_ID, SAMPLE_RATE
from .data import ManifestDataset
from .forward import MarkEmbeddings, canonical_modules, set_trainable, teacher_forced
from .losses import MelLoss, MultiResolutionSTFTLoss, duration_loss, energy_loss, f0_losses
from .pitch import EnergyCurve, accent_agreement, extract_f0

DEFAULT_EVAL = Path(__file__).resolve().parent.parent / 'eval_minimal_pairs.txt'
TEXT_MODULES = ('bert', 'bert_encoder', 'text_encoder')


def build_model(device):
    from kokoro import KModel
    config = build_config(load_base_config(REPO_ID))
    model = KModel(repo_id=REPO_ID, config=config).to(device)
    init_accent_embeddings(model)
    return model, config


def lr_factor(step, args):
    warm = min(1.0, (step + 1) / max(1, args.lr_warmup))
    return warm * 0.5 * (1 + math.cos(math.pi * step / args.steps))


@torch.no_grad()
def evaluate_accent(model, voice, texts, g2p, out_dir=None, tag=''):
    """Accent agreement of synthesized speech. Works with models whose vocab
    lacks the marks too (stock Kokoro): dropped chars get zero duration."""
    correct = checked = 0
    for i, text in enumerate(texts):
        ann = g2p.annotate(text)
        o = model(ann.ps, voice, return_output=True)
        audio = o.audio.cpu().numpy()
        pred = iter(o.pred_dur.cpu().tolist())
        dur = [next(pred)] + [next(pred) if c in model.vocab else 0 for c in ann.ps] + [next(pred)]
        f0 = extract_f0(audio, 2 * int(sum(dur)))
        c, t = accent_agreement(ann, dur, f0)
        correct, checked = correct + c, checked + t
        if out_dir is not None:
            sf.write(out_dir / f'{tag}_{i:02d}.wav', audio, SAMPLE_RATE)
    return correct, checked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='output dir of ja_accent.ft.prepare')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=6000, help='optimizer steps')
    ap.add_argument('--accum', type=int, default=8, help='utterances per optimizer step')
    ap.add_argument('--warm-steps', type=int, default=500, help='steps training only voice + marks')
    ap.add_argument('--train-modules', default='bert,bert_encoder,text_encoder,predictor,decoder_body')
    ap.add_argument('--init-voice', default='jf_alpha')
    ap.add_argument('--lr-voice', type=float, default=2e-3)
    ap.add_argument('--lr-marks', type=float, default=1e-3)
    ap.add_argument('--lr-text', type=float, default=1e-5)
    ap.add_argument('--lr-model', type=float, default=1.5e-5)
    ap.add_argument('--lr-warmup', type=int, default=100)
    ap.add_argument('--w-mel', type=float, default=5.0)
    ap.add_argument('--w-stft', type=float, default=1.0)
    ap.add_argument('--w-dur', type=float, default=1.0)
    ap.add_argument('--w-f0', type=float, default=1.0, help='Hz smooth-L1 / 10 (StyleTTS2)')
    ap.add_argument('--w-logf0', type=float, default=2.0, help='voiced log-F0 L1, carries the accent')
    ap.add_argument('--w-energy', type=float, default=1.0)
    ap.add_argument('--w-voice-reg', type=float, default=0.0, help='L2 pull of the voice toward its init')
    ap.add_argument('--max-frames', type=int, default=800)
    ap.add_argument('--min-align', type=float, default=0.0)
    ap.add_argument('--min-accent', type=float, default=0.0,
                    help='drop clips whose recorded pitch matches fewer than this fraction of marks')
    ap.add_argument('--val-pct', type=float, default=5.0)
    ap.add_argument('--eval-texts', default=str(DEFAULT_EVAL))
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--ckpt-every', type=int, default=500)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device
    out_dir = Path(args.out)
    (out_dir / 'eval').mkdir(parents=True, exist_ok=True)
    (out_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))

    model, config = build_model(device)
    (out_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    modules = [m for m in args.train_modules.split(',') if m]
    by_module = set_trainable(model, modules)
    text_params = [p for m, ps in by_module.items() if m in TEXT_MODULES for p in ps]
    acoustic_params = [p for m, ps in by_module.items() if m not in TEXT_MODULES for p in ps]
    marks = MarkEmbeddings(model).attach()
    print(f'[train] modules {modules}: text {sum(p.numel() for p in text_params) / 1e6:.1f}M, '
          f'acoustic {sum(p.numel() for p in acoustic_params) / 1e6:.1f}M params')

    common = dict(prep_dir=args.data, vocab=model.vocab, val_pct=args.val_pct, max_frames=args.max_frames,
                  min_align=args.min_align, min_accent=args.min_accent)
    ds_train = ManifestDataset(split='train', **common)
    ds_val = ManifestDataset(split='val', **common)
    assert len(ds_train) > 0, 'no training clips'

    pack = torch.load(hf_hub_download(REPO_ID, f'voices/{args.init_voice}.pt'), weights_only=True)
    init_vec = pack.mean(dim=0).to(device)
    voices = torch.nn.ParameterDict({s: torch.nn.Parameter(init_vec.clone()) for s in ds_train.speakers})

    groups = [
        {'name': 'voice', 'params': list(voices.parameters()), 'lr': args.lr_voice, 'weight_decay': 0.0},
        {'name': 'marks', 'params': list(marks.parameters()), 'lr': args.lr_marks, 'weight_decay': 0.0},
        {'name': 'text', 'params': text_params, 'lr': args.lr_text, 'weight_decay': 1e-4},
        {'name': 'acoustic', 'params': acoustic_params, 'lr': args.lr_model, 'weight_decay': 1e-4},
    ]
    opt = torch.optim.AdamW([g for g in groups if g['params']])
    base_lr = {g['name']: g['lr'] for g in opt.param_groups}

    mel_loss = MelLoss().to(device)
    stft_loss = MultiResolutionSTFTLoss().to(device)
    energy = EnergyCurve().to(device)
    g2p = JAAccentG2P()
    eval_texts = [l.strip() for l in Path(args.eval_texts).read_text(encoding='utf-8').splitlines() if l.strip()]

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        for name, sd in ck['modules'].items():
            getattr(model, name).load_state_dict(sd)
        for k, v in ck['marks'].items():
            marks.delta[k].data.copy_(v)
        for s, v in ck['voices'].items():
            voices[s].data.copy_(v.to(device))
        opt.load_state_dict(ck['opt'])
        start = ck['step']
        print(f'[train] resumed at step {start}')

    saved = canonical_modules(modules)

    def save(path, step, extra=None):
        torch.save({
            'step': step, 'repo_id': REPO_ID, 'config': config, 'train_modules': saved,
            'modules': {m: getattr(model, m).state_dict() for m in saved},
            'marks': {k: v.detach().cpu() for k, v in marks.delta.items()},
            'voices': {k: v.detach().cpu() for k, v in voices.items()},
            'opt': opt.state_dict(), 'args': vars(args), **(extra or {}),
        }, path)

    log = open(out_dir / 'log.jsonl', 'a')

    def emit(msg):
        print(msg)
        log.write(json.dumps(msg) + '\n')
        log.flush()

    if start == 0:
        c, t = evaluate_accent(model, voices[ds_train.speakers[0]].detach(), eval_texts, g2p,
                               out_dir / 'eval', 'step000000')
        emit({'step': 0, 'eval_accent': round(c / max(t, 1), 4), 'eval_accent_n': t})

    def step_loss(item, ref_s):
        ids, dur, wav = item['ids'].to(device), item['dur'].to(device), item['wav'].to(device)
        n_curve = 2 * int(dur.sum())
        f0_real = item['f0'].to(device).unsqueeze(0)
        with torch.no_grad():
            n_real = energy(wav, n_curve).unsqueeze(0)
        o = teacher_forced(model, ids, ref_s, dur, f0_real, n_real)
        l_hz, l_log = f0_losses(o.f0_pred, f0_real)
        parts = {
            'mel': mel_loss(o.audio, wav), 'stft': stft_loss(o.audio, wav),
            'dur': duration_loss(o.dur_pred, dur), 'f0': l_hz, 'logf0': l_log,
            'energy': energy_loss(o.n_pred, n_real),
        }
        w = {'mel': args.w_mel, 'stft': args.w_stft, 'dur': args.w_dur, 'f0': args.w_f0,
             'logf0': args.w_logf0, 'energy': args.w_energy}
        total = sum(w[k] * v for k, v in parts.items())
        return total, parts

    order, cursor = list(range(len(ds_train))), len(ds_train)
    best_val = float('inf')
    t0 = time.time()
    for step in range(start, args.steps):
        f = lr_factor(step, args)
        for g in opt.param_groups:
            frozen = g['name'] in ('text', 'acoustic') and step < args.warm_steps
            g['lr'] = 0.0 if frozen else base_lr[g['name']] * f

        opt.zero_grad(set_to_none=True)
        acc = {}
        for _ in range(args.accum):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            item = ds_train[order[cursor]]
            cursor += 1
            ref_s = voices[item['row']['speaker']]
            try:
                total, parts = step_loss(item, ref_s)
            except torch.OutOfMemoryError:
                print(f"[train] OOM on {item['row']['id']}, skipping")
                torch.cuda.empty_cache()
                continue
            if args.w_voice_reg > 0:
                total = total + args.w_voice_reg * (ref_s - init_vec).pow(2).sum()
            (total / args.accum).backward()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v.item() / args.accum
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
        opt.step()

        if (step + 1) % args.log_every == 0 or step == start:
            lrs = {f"lr_{g['name']}": g['lr'] for g in opt.param_groups}
            emit({'step': step + 1, **{k: round(v, 4) for k, v in acc.items()}, **lrs,
                  'sec': round(time.time() - t0, 1)})

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vl, n = {}, 0
            with torch.no_grad():
                for i in range(min(len(ds_val), 32)):
                    item = ds_val[i]
                    _, parts = step_loss(item, voices[item['row']['speaker']])
                    for k, v in parts.items():
                        vl[k] = vl.get(k, 0.0) + v.item()
                    n += 1
            msg = {'step': step + 1}
            if n:
                vl = {k: v / n for k, v in vl.items()}
                val_total = args.w_mel * vl['mel'] + args.w_stft * vl['stft'] + args.w_dur * vl['dur']
                msg.update(val_loss=round(val_total, 4), **{f'val_{k}': round(v, 4) for k, v in vl.items()})
            c, t = evaluate_accent(model, voices[ds_train.speakers[0]].detach(), eval_texts, g2p,
                                   out_dir / 'eval', f'step{step + 1:06d}')
            emit({**msg, 'eval_accent': round(c / max(t, 1), 4), 'eval_accent_n': t})
            if n and val_total < best_val:
                best_val = val_total
                save(out_dir / 'ckpt_best.pt', step + 1)

        if (step + 1) % args.ckpt_every == 0:
            save(out_dir / 'ckpt_last.pt', step + 1)
    save(out_dir / 'ckpt_last.pt', args.steps)
    print(f'[train] done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
