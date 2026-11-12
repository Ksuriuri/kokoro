"""Training-time forward pass for KModel.

``teacher_forced`` is ``KModel.forward_with_tokens`` with the predicted
alignment replaced by one built from ground-truth durations, so the output
waveform has exactly the target's length and reconstruction losses apply.
Following StyleTTS2, the decoder can be fed the real F0/energy curves while the
predictor's F0/energy heads are supervised separately.

The duration head gets no gradient through the hard alignment; it is
supervised directly on its continuous output.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from ..g2p import ACCENT_MARKS


@dataclass
class TFOutput:
    audio: torch.Tensor      # [samples]
    dur_pred: torch.Tensor   # [T] continuous frames
    f0_pred: torch.Tensor    # [1, 2F] Hz
    n_pred: torch.Tensor     # [1, 2F]


def to_input_ids(vocab: Dict[str, int], ps: str) -> torch.LongTensor:
    return torch.LongTensor([[0, *(vocab[c] for c in ps), 0]])


def alignment(dur: torch.LongTensor) -> torch.Tensor:
    """[T] frame counts -> [1, T, sum(dur)] hard monotonic alignment."""
    T = dur.shape[0]
    idx = torch.repeat_interleave(torch.arange(T, device=dur.device), dur)
    aln = torch.zeros((T, idx.shape[0]), device=dur.device)
    aln[idx, torch.arange(idx.shape[0], device=dur.device)] = 1
    return aln.unsqueeze(0)


def _encode(model, input_ids: torch.LongTensor, ref_s: torch.Tensor):
    T = input_ids.shape[1]
    lengths = torch.full((1,), T, device=input_ids.device, dtype=torch.long)
    mask = torch.zeros((1, T), dtype=torch.bool, device=input_ids.device)
    bert_dur = model.bert(input_ids, attention_mask=(~mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, lengths, mask)
    x, _ = model.predictor.lstm(d)
    dur = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
    return d, dur.squeeze(0), lengths, mask, s


def teacher_forced(model, input_ids, ref_s, dur_tgt, f0_real=None, n_real=None) -> TFOutput:
    d, dur_pred, lengths, mask, s = _encode(model, input_ids, ref_s)
    aln = alignment(dur_tgt)
    en = d.transpose(-1, -2) @ aln
    f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
    asr = model.text_encoder(input_ids, lengths, mask) @ aln
    audio = model.decoder(
        asr,
        f0_real if f0_real is not None else f0_pred,
        n_real if n_real is not None else n_pred,
        ref_s[:, :128],
    )
    return TFOutput(audio.squeeze(), dur_pred, f0_pred, n_pred)


@torch.no_grad()
def predict_durations(model, input_ids, ref_s) -> torch.Tensor:
    """Continuous per-token durations from the duration head, [T]."""
    return _encode(model, input_ids, ref_s)[1]


def enable_rnn_backward(model) -> None:
    """cuDNN only backprops LSTMs in train mode. Kokoro's LSTMs are single
    layer with no dropout, so train mode is numerically identical."""
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.train()


class MarkEmbeddings(torch.nn.Module):
    """Trainable offsets for the accent-mark rows of both embedding tables.

    The marks need to move much further than the pretrained rows, and Adam's
    per-element normalization means a gradient scale cannot do that; a
    separate parameter gets its own learning rate. ``fold`` writes the
    offsets into the model weights for export.
    """

    def __init__(self, model, marks: Iterable[str] = ACCENT_MARKS):
        super().__init__()
        self.ids = [model.vocab[m] for m in marks]
        self.tables = {
            'bert': model.bert.embeddings.word_embeddings,
            'text': model.text_encoder.embedding,
        }
        self.delta = torch.nn.ParameterDict({
            k: torch.nn.Parameter(torch.zeros(len(self.ids), t.embedding_dim, device=t.weight.device))
            for k, t in self.tables.items()
        })
        self._handles: List = []

    def _full(self, key):
        t = self.tables[key]
        full = torch.zeros(t.num_embeddings, t.embedding_dim, device=t.weight.device, dtype=t.weight.dtype)
        return full.index_put((torch.tensor(self.ids, device=full.device),), self.delta[key])

    def attach(self) -> 'MarkEmbeddings':
        for key, table in self.tables.items():
            def hook(module, inputs, output, key=key):
                return output + F.embedding(inputs[0], self._full(key))
            self._handles.append(table.register_forward_hook(hook))
        return self

    def detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def fold(self) -> None:
        self.detach_hooks()
        for key, table in self.tables.items():
            table.weight[self.ids] += self.delta[key].to(table.weight.dtype)
            self.delta[key].zero_()


def set_trainable(model, names: Iterable[str]) -> Dict[str, List[torch.nn.Parameter]]:
    """Unfreeze the named modules; returns their parameters by name.

    ``decoder_body`` trains the decoder but keeps its vocoder (generator)
    frozen: reconstruction-only losses degrade a GAN-trained generator.
    Everything stays in eval mode (no dropout) so the graph matches inference.
    """
    model.requires_grad_(False)
    model.eval()
    params = {}
    for name in names:
        if name == 'decoder_body':
            model.decoder.requires_grad_(True)
            model.decoder.generator.requires_grad_(False)
            params[name] = [p for n, p in model.decoder.named_parameters() if not n.startswith('generator.')]
        else:
            mod = getattr(model, name)
            mod.requires_grad_(True)
            params[name] = list(mod.parameters())
    enable_rnn_backward(model)
    return params


def canonical_modules(names: Iterable[str]) -> List[str]:
    return sorted({'decoder' if n == 'decoder_body' else n for n in names})


def load_checkpoint_into(model, ckpt: Dict, voice_key: Optional[str] = None):
    """Apply a training checkpoint to ``model`` (marks folded). Returns the voice [1, 256]."""
    for name, sd in ckpt['modules'].items():
        getattr(model, name).load_state_dict(sd)
    marks = MarkEmbeddings(model)
    for key, val in ckpt['marks'].items():
        marks.delta[key].data.copy_(val.to(marks.delta[key].device))
    marks.fold()
    voices = ckpt['voices']
    return voices[voice_key or next(iter(voices))]
