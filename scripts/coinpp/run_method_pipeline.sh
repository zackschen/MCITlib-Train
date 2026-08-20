#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

METHOD="${METHOD:-${1:-LoRA-FT}}"
FACTOR="${FACTOR:-evidence_complexity}"
PROFILE="${PROFILE:-strict}"
MODE="${MODE:-all}"
PYTHON="${PYTHON:-python3}"
GPU_NUM="${GPU_NUM:-4}"
EVAL_GPU="${EVAL_GPU:-0}"
EPOCHS="${EPOCHS:-1}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-0}"
EVAL_SCOPE="${EVAL_SCOPE:-all}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
SKIP_MISSING_CHECKPOINTS="${SKIP_MISSING_CHECKPOINTS:-0}"
SMOLORA_EMB="${SMOLORA_EMB:-}"

train_args=(
  --method "$METHOD"
  --factor "$FACTOR"
  --profile "$PROFILE"
  --gpu-num "$GPU_NUM"
  --epochs "$EPOCHS"
  --start-stage "$START_STAGE"
  --stop-stage "$STOP_STAGE"
)
eval_args=(
  --method "$METHOD"
  --factor "$FACTOR"
  --profile "$PROFILE"
  --gpu "$EVAL_GPU"
  --start-stage "$START_STAGE"
  --stop-stage "$STOP_STAGE"
  --eval-scope "$EVAL_SCOPE"
)

if [[ "$FORCE" == "1" ]]; then
  train_args+=(--force)
  eval_args+=(--force)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  train_args+=(--dry-run)
  eval_args+=(--dry-run)
fi
if [[ "$SKIP_MISSING_CHECKPOINTS" == "1" ]]; then
  eval_args+=(--skip-missing-checkpoints)
fi
if [[ -n "$LIMIT" ]]; then
  eval_args+=(--limit "$LIMIT")
fi
if [[ "$METHOD" == "SMoLoRA" ]]; then
  if [[ -z "$SMOLORA_EMB" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] SMOLORA_EMB is not set; the launcher will use a placeholder."
    else
      echo "SMoLoRA requires SMOLORA_EMB=/path/to/factor_embeddings.pkl" >&2
      exit 2
    fi
  else
    train_args+=(--smolora-emb "$SMOLORA_EMB")
  fi
fi

echo "CoIN++ method pipeline"
echo "  method/profile/factor: $METHOD / $PROFILE / $FACTOR"
echo "  mode:                  $MODE"
echo "  stages:                $START_STAGE-$STOP_STAGE"

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
  "$PYTHON" scripts/coinpp/run_method.py "${train_args[@]}"
fi

if [[ "$METHOD" == "MR-LoRA" && "$MODE" != "train" ]]; then
  echo "MR-LoRA expert/router training is supported, but generic matrix evaluation is not."
  echo "Run with MODE=train, then configure a factor-specific router mapping."
  exit 0
fi

if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
  "$PYTHON" scripts/coinpp/eval_method.py "${eval_args[@]}"
fi

if [[ "$MODE" == "score" || "$MODE" == "all" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] Skip scoring because no predictions were generated."
  else
    "$PYTHON" scripts/coinpp/score_coinpp.py \
      --method "$METHOD" \
      --factor "$FACTOR" \
      --profile "$PROFILE"
  fi
fi
