#!/usr/bin/env bash
# Finetune Kokoro-82M with pitch-accent marks on an NVIDIA GPU, in the background.
# Everything is written to runs/$RUN inside this project; rerunning with the
# same RUN resumes from runs/$RUN/ckpt_last.pt.
#
#   bash scripts/train_cuda.sh                      # runs/exp1, default hyperparameters
#   RUN=exp2 GPU=1 bash scripts/train_cuda.sh --steps 4000 --w-logf0 3
#
# Env: RUN (exp1)  DATA (data/prep)  GPU (0)  NO_SYNC=1 skips `uv sync`
#      (keeps a manually installed torch)  SKIP_SMOKE=1 skips the GPU self-check
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-exp1}
DATA=${DATA:-data/prep}
OUT=runs/$RUN
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1

if [ ! -f "$DATA/manifest.jsonl" ]; then
    echo "missing $DATA/manifest.jsonl: copy data/prep from the prepare machine first" >&2
    exit 1
fi
command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

[ -n "${NO_SYNC:-}" ] || uv sync --extra ja --extra ja-train
# the venv's python directly, so train.pid is the trainer itself
PY=.venv/bin/python

$PY - <<'EOF'
import torch
assert torch.cuda.is_available(), 'CUDA is not available to torch'
cap = torch.cuda.get_device_capability(0)
print(f'torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}, sm_{cap[0]}{cap[1]}')
try:
    (torch.ones(8, device='cuda') @ torch.ones(8, device='cuda')).item()
except RuntimeError as e:
    raise SystemExit(f'this torch build cannot run on sm_{cap[0]}{cap[1]} ({e}); see ja_accent/TRAINING.md')
EOF

mkdir -p "$OUT"
if [ -z "${SKIP_SMOKE:-}" ] && [ ! -f "$OUT/smoke.ok" ]; then
    $PY -m ja_accent.ft.smoke --device cuda --out "$OUT/smoke" && touch "$OUT/smoke.ok"
fi

RESUME=()
if [ -f "$OUT/ckpt_last.pt" ]; then
    echo "resuming from $OUT/ckpt_last.pt"
    RESUME=(--resume "$OUT/ckpt_last.pt")
fi

# own session: survives the ssh/terminal closing, not just SIGHUP
DETACH=(nohup)
command -v setsid >/dev/null && DETACH=(setsid nohup)
"${DETACH[@]}" $PY -m ja_accent.ft.train --data "$DATA" --out "$OUT" ${RESUME[@]+"${RESUME[@]}"} "$@" >> "$OUT/train.out" 2>&1 &
echo $! > "$OUT/train.pid"
echo "training in background, pid $(cat "$OUT/train.pid")"
echo "  log:  tail -f $OUT/train.out"
echo "  stop: kill \$(cat $OUT/train.pid)"
