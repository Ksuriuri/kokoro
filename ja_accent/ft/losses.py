import torch
import torch.nn.functional as F
import torchaudio

from .constants import MEL_HOP, MEL_N_FFT, MEL_WIN, N_MELS, SAMPLE_RATE


class MelLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=MEL_N_FFT, win_length=MEL_WIN,
            hop_length=MEL_HOP, n_mels=N_MELS, power=1.0)

    def forward(self, x, y):
        n = min(x.shape[-1], y.shape[-1])
        return F.l1_loss(torch.log(self.mel(x[..., :n]).clamp_min(1e-5)),
                         torch.log(self.mel(y[..., :n]).clamp_min(1e-5)))


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Spectral convergence + log-magnitude L1 at three resolutions."""

    RESOLUTIONS = ((512, 50, 240), (1024, 120, 600), (2048, 240, 1200))

    def __init__(self):
        super().__init__()
        for i, (_, _, win) in enumerate(self.RESOLUTIONS):
            self.register_buffer(f'win{i}', torch.hann_window(win), persistent=False)

    def forward(self, x, y):
        n = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :n], y[..., :n]
        total = x.new_zeros(())
        for i, (n_fft, hop, win) in enumerate(self.RESOLUTIONS):
            w = getattr(self, f'win{i}')
            X = torch.stft(x, n_fft, hop, win, window=w, return_complex=True).abs().clamp_min(1e-7)
            Y = torch.stft(y, n_fft, hop, win, window=w, return_complex=True).abs().clamp_min(1e-7)
            total = total + torch.norm(Y - X) / torch.norm(Y).clamp_min(1e-7) + F.l1_loss(X.log(), Y.log())
        return total / len(self.RESOLUTIONS)


def duration_loss(dur_pred, dur_tgt):
    return F.l1_loss(dur_pred, dur_tgt.float())


def f0_losses(f0_pred, f0_real):
    """(StyleTTS2's Hz smooth-L1 / 10 over all frames, log-F0 L1 on voiced frames).

    The log term weights relative pitch movement, which is what carries the
    accent, independently of the speaker's register.
    """
    hz = F.smooth_l1_loss(f0_pred, f0_real) / 10
    voiced = f0_real > 0
    if voiced.any():
        log = F.l1_loss(torch.log(f0_pred[voiced].clamp_min(10.0)), torch.log(f0_real[voiced]))
    else:
        log = f0_pred.new_zeros(())
    return hz, log


def energy_loss(n_pred, n_real):
    return F.smooth_l1_loss(n_pred, n_real)
