"""Vocab extension for the accent marks.

Kokoro-82M v1.0 has n_token=178 but only 114 ids in its vocab, so the marks
take two unused ids and every weight keeps its shape: the same checkpoint
loads with either config.
"""

import json
from typing import Dict, Optional

import torch
from huggingface_hub import hf_hub_download

from .g2p import DOWNSTEP, UPSTEP

ACCENT_IDS = {UPSTEP: 7, DOWNSTEP: 8}

# Mandarin rising / falling tone marks, the closest pitch semantics the
# pretrained embeddings already have.
INIT_SOURCES = {UPSTEP: '↗', DOWNSTEP: '↘'}


def load_base_config(repo_id: str = 'hexgrad/Kokoro-82M') -> Dict:
    with open(hf_hub_download(repo_id=repo_id, filename='config.json'), encoding='utf-8') as f:
        return json.load(f)


def build_config(base: Dict) -> Dict:
    config = json.loads(json.dumps(base))
    vocab = config['vocab']
    used = set(vocab.values())
    for mark, idx in ACCENT_IDS.items():
        if vocab.get(mark) == idx:
            continue
        assert mark not in vocab, f'{mark!r} already mapped to {vocab[mark]}'
        assert idx not in used, f'id {idx} already used'
        assert 0 < idx < config['n_token'], idx
        vocab[mark] = idx
    return config


@torch.no_grad()
def init_accent_embeddings(model, sources: Optional[Dict[str, str]] = None, noise: float = 0.0):
    """Copy pretrained rows into the mark ids of both embedding tables.

    The rows at unused ids were never trained, so starting from related tokens
    gives the finetune a better initial point. Call once before training.
    """
    sources = sources or INIT_SOURCES
    tables = [model.bert.embeddings.word_embeddings.weight, model.text_encoder.embedding.weight]
    for mark, src in sources.items():
        dst_id, src_id = model.vocab[mark], model.vocab[src]
        for w in tables:
            w[dst_id] = w[src_id] + noise * torch.randn_like(w[src_id])
