#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pool-dir> [checkpoint-dir] [fresh|resume] [max-steps]"
  exit 2
fi

POOL_DIR="$1"
CHECKPOINT_DIR="${2:-./model/v2_exp1_shuffle}"
MODE="${3:-fresh}"
MAX_STEPS="${4:-0}"

if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
  echo "Mode must be fresh or resume" >&2
  exit 2
fi

MODE_FLAG="--$MODE"
PYTHON="./venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

"$PYTHON" train_mc_parameter_pool_transformer_loss.py \
  --pool-dir "$POOL_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --model-type transformer \
  --sampling-mode shuffle \
  --shuffle-block-size 200000 \
  --active-block-size 200000 \
  --precompute-chunk-size 4096 \
  --integration-points 128 \
  --input-points 100 \
  --output-points 100 \
  --batch-size 64 \
  --noise-level 0.09 \
  --loss-profile pinn \
  --loss-normalization relative \
  --physics-target clean \
  --lambda-grad 0.1 \
  --lambda-physics 0.1 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --max-hours 23 \
  --max-steps "$MAX_STEPS" \
  --checkpoint-every-steps 2000 \
  --log-every-steps 100 \
  --amp \
  --require-complete-pool \
  "$MODE_FLAG"
