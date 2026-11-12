"""Sanity checks for the finetuning code; CPU is enough.

  1 parity   teacher-forced forward with predicted durations == stock inference
  2 curves   harvest F0 / energy of stock audio line up with the model's own
             F0/energy curves at lag 0 (validates the 80 fps convention)
  3 grads    one backward reaches voice, mark offsets, text and acoustic modules
  4 fit      a few steps on a synthetic target reduce the loss
  5 e2e      (--e2e) synthesize a tiny dataset with a stock voice, then run
             prepare -> train -> export -> infer on it

    uv run python -m ja_accent.ft.smoke [--e2e] [--out smoke_out]
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from ..g2p import DOWNSTEP, UPSTEP, JAAccentG2P
from .constants import REPO_ID, SAMPLE_RATE
from .forward import MarkEmbeddings, set_trainable, teacher_forced, to_input_ids
from .losses import MelLoss, MultiResolutionSTFTLoss, duration_loss, f0_losses
from .pitch import EnergyCurve, extract_f0
from .train import build_model

TEXT = '橋の上で箸を落としました。'
E2E_TEXTS = [
    '箸を落としてしまいました。', '橋の向こうに駅があります。', '端の席に座ってください。',
    '雨が降ってきたので傘をさした。', '飴を一つもらえますか？', '今日はとても良い天気ですね。',
    '新しい本を三冊買いました。', '駅前のカフェで友達に会った。', '柿が赤く色づいてきた。',
    '牡蠣のフライが食べたい。', '明日の会議は十時からです。', 'ちょっと待ってください。',
]


def _pack(voice):
    return torch.load(hf_hub_download(REPO_ID, f'voices/{voice}.pt'), weights_only=True)


def _corr_at_lag(a, b, lag):
    if lag > 0:
        a, b = a[lag:], b[:-lag]
    elif lag < 0:
        a, b = a[:lag], b[-lag:]
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def check_parity(model, ps, ref):
    ids = to_input_ids(model.vocab, ps)
    torch.manual_seed(0)
    with torch.no_grad():
        o = model(ps, ref, return_output=True)
    torch.manual_seed(0)
    with torch.no_grad():
        tf = teacher_forced(model, ids, ref, o.pred_dur)
    diff = float((tf.audio - o.audio).abs().max())
    assert tf.audio.shape == o.audio.shape and diff < 1e-3, diff
    print(f'PASS 1 parity: {o.audio.shape[0] / SAMPLE_RATE:.2f}s, max |diff| {diff:.1e}')
    return ids, o, tf


def check_curves(o, tf):
    audio = o.audio.numpy()
    n_curve = 2 * int(o.pred_dur.sum())
    f0 = extract_f0(audio, n_curve)
    pred = tf.f0_pred.squeeze(0).numpy()
    voiced = (f0 > 0) & (pred > 0)
    lags = {lag: _corr_at_lag(np.where(voiced, f0, 0), np.where(voiced, pred, 0), lag) for lag in range(-3, 4)}
    best = max(lags, key=lags.get)
    ratio = float(np.median(f0[voiced] / pred[voiced]))
    e = EnergyCurve()(o.audio, n_curve).numpy()
    e_corr = _corr_at_lag(e, tf.n_pred.squeeze(0).numpy(), 0)
    assert best == 0 and lags[0] > 0.8 and 0.8 < ratio < 1.25 and e_corr > 0.8, (lags, ratio, e_corr)
    print(f'PASS 2 curves: F0 corr {lags[0]:.3f} at lag 0 (best lag {best}), Hz ratio {ratio:.3f}; '
          f'energy corr {e_corr:.3f}')


def check_grads_and_fit(model, ids, ref, target):
    by_module = set_trainable(model, ['bert', 'bert_encoder', 'text_encoder', 'predictor', 'decoder_body'])
    marks = MarkEmbeddings(model).attach()
    voice = torch.nn.Parameter(ref.clone())
    mel, stft = MelLoss(), MultiResolutionSTFTLoss()
    params = [voice, *marks.parameters(), *(p for ps in by_module.values() for p in ps)]
    opt = torch.optim.AdamW([{'params': [voice, *marks.parameters()], 'lr': 3e-3},
                             {'params': [p for ps in by_module.values() for p in ps], 'lr': 1e-5}])
    f0_real = torch.from_numpy(extract_f0(target['audio'].numpy(), 2 * int(target['dur'].sum()))).unsqueeze(0)
    losses = []
    for i in range(6):
        opt.zero_grad()
        o = teacher_forced(model, ids, voice, target['dur'], f0_real)
        l_hz, l_log = f0_losses(o.f0_pred, f0_real)
        loss = 5 * mel(o.audio, target['audio']) + stft(o.audio, target['audio']) \
            + duration_loss(o.dur_pred, target['dur']) + l_hz + 2 * l_log
        loss.backward()
        if i == 0:
            g = {'voice': voice.grad, 'marks.bert': marks.delta['bert'].grad,
                 'marks.text': marks.delta['text'].grad,
                 'bert': model.bert.encoder.albert_layer_groups[0].albert_layers[0].ffn.weight.grad,
                 'text_encoder': model.text_encoder.embedding.weight.grad,
                 'predictor': model.predictor.F0_proj.weight.grad}
            missing = [k for k, v in g.items() if v is None or v.abs().sum() == 0]
            assert not missing, f'no gradient: {missing}'
            print(f'PASS 3 grads: {", ".join(g)}')
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    print(f'PASS 4 fit: loss {losses[0]:.3f} -> {losses[-1]:.3f}')
    marks.detach_hooks()


def run_e2e(out: Path, device):
    from kokoro import KModel

    from . import export, infer, prepare, train

    wav_dir = out / 'e2e_wavs'
    wav_dir.mkdir(parents=True, exist_ok=True)
    stock = KModel(repo_id=REPO_ID).eval()
    g2p, pack = JAAccentG2P(), _pack('jf_alpha')
    lines = []
    for i, text in enumerate(E2E_TEXTS):
        ps, _ = g2p(text)
        with torch.no_grad():
            audio = stock(ps, pack[len(ps) - 1]).numpy()
        sf.write(wav_dir / f'clip{i:03d}.wav', np.pad(audio, (2400, 2400)), SAMPLE_RATE)
        lines.append(f'clip{i:03d}.wav\t{text}')
    (out / 'e2e_metadata.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    prep = out / 'e2e_prep'
    prepare.main(['--metadata', str(out / 'e2e_metadata.tsv'), '--wav-dir', str(wav_dir),
                  '--out', str(prep), '--device', device])
    run = out / 'e2e_run'
    train.main(['--data', str(prep), '--out', str(run), '--steps', '4', '--accum', '2',
                '--warm-steps', '2', '--eval-every', '4', '--ckpt-every', '4', '--log-every', '1',
                '--val-pct', '20', '--device', device])
    export.main(['--ckpt', str(run / 'ckpt_last.pt'), '--out', str(out / 'e2e_export')])
    infer.main(['--model', str(out / 'e2e_export' / 'kokoro-ja-accent.pth'),
                '--config', str(out / 'e2e_export' / 'config.json'),
                '--voice', str(out / 'e2e_export' / 'spk0.pt'),
                '--text', TEXT, '--out', str(out / 'e2e_infer.wav'), '--device', device])
    print('PASS 5 e2e: prepare -> train -> export -> infer')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='smoke_out')
    ap.add_argument('--e2e', action='store_true')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, _ = build_model('cpu')
    model.eval()
    ps, _ = JAAccentG2P()(TEXT)
    assert UPSTEP in ps and DOWNSTEP in ps and all(c in model.vocab for c in ps), ps
    ref = _pack('jf_alpha')[len(ps) - 1]

    ids, o, tf = check_parity(model, ps, ref)
    check_curves(o, tf)
    with torch.no_grad():
        other = model(ps, _pack('jf_nezumi')[len(ps) - 1], return_output=True)
    target = {'audio': other.audio, 'dur': other.pred_dur}
    check_grads_and_fit(model, to_input_ids(model.vocab, ps), ref, target)

    if args.e2e:
        run_e2e(out, args.device)
    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
