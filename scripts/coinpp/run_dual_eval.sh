#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

METHOD="${METHOD:-${1:-LoRA-FT}}"
FACTOR="${FACTOR:-evidence_complexity}"
PROFILE="${PROFILE:-strict}"
PYTHON="${PYTHON:-python3}"
EASY_ROOT="${EASY_ROOT:-/home/chencheng/data/Code/Easy_Train_MLLM}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8001/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-8}"
LIMIT="${LIMIT:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
ANNOTATION_CACHE="${ANNOTATION_CACHE:-}"

if [[ -z "$JUDGE_MODEL" ]]; then
  echo "Set JUDGE_MODEL to the served OpenAI-compatible judge model." >&2
  exit 2
fi
if [[ ! -f "$EASY_ROOT/scripts/evaluate_coinpp_dual.py" ]]; then
  echo "CoIN++ dual evaluator not found under EASY_ROOT=$EASY_ROOT" >&2
  exit 2
fi

model_slug=$(printf '%s' "$JUDGE_MODEL" | tr '/ :' '___')
EVAL_ROOT="$ROOT/results/CoIN++/$PROFILE/$FACTOR/$METHOD/eval"
ORDER_JSON="$ROOT/datasets/CoIN++/$PROFILE/$FACTOR/manifest.json"
JUDGE_CACHE="${JUDGE_CACHE:-$ROOT/results/CoIN++/judge_cache/${model_slug}.jsonl}"
SUMMARY_DIR="${SUMMARY_DIR:-$EVAL_ROOT/summary_dual}"
if [[ -z "$ANNOTATION_CACHE" ]]; then
  if [[ "$PROFILE" == "clean_5k" ]]; then
    ANNOTATION_CACHE="$EASY_ROOT/cl_dataset/coin_factor1_final_clean/evaluation_annotations.jsonl"
  else
    ANNOTATION_CACHE="$EASY_ROOT/cl_dataset/coin_factor1_final_textvqa_clean/evaluation_annotations.jsonl"
  fi
fi

args=(
  --result-root "$EVAL_ROOT"
  --project-root "$EASY_ROOT"
  --judge-model "$JUDGE_MODEL"
  --judge-base-url "$JUDGE_BASE_URL"
  --judge-api-key "$JUDGE_API_KEY"
  --judge-cache "$JUDGE_CACHE"
  --judge-workers "$JUDGE_WORKERS"
  --judge-batch-size "$JUDGE_BATCH_SIZE"
)
if [[ "$PROFILE" == "clean_5k" && ! -f "$ANNOTATION_CACHE" ]]; then
  echo "clean_5k annotation cache not found: $ANNOTATION_CACHE" >&2
  exit 2
fi
if [[ -f "$ANNOTATION_CACHE" ]]; then
  args+=(--annotation-cache "$ANNOTATION_CACHE")
fi
if [[ "$SKIP_COMPLETED" == "1" ]]; then
  args+=(--skip-completed)
fi
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

echo "CoIN++ independent standard + all-sample LLM-Judge evaluation"
echo "  method/profile/factor: $METHOD / $PROFILE / $FACTOR"
echo "  result root:           $EVAL_ROOT"
echo "  judge model:           $JUDGE_MODEL"
echo "  judge API:             $JUDGE_BASE_URL"
echo "  judge cache:           $JUDGE_CACHE"
echo "  annotations:           $ANNOTATION_CACHE"

PYTHONPATH="$EASY_ROOT:$EASY_ROOT/scripts:${PYTHONPATH:-}" \
  "$PYTHON" "$EASY_ROOT/scripts/evaluate_coinpp_dual.py" "${args[@]}"

PYTHONPATH="$EASY_ROOT:$EASY_ROOT/scripts:${PYTHONPATH:-}" \
  "$PYTHON" "$EASY_ROOT/scripts/summarize_coinpp_dual_eval.py" \
    --result-root "$EVAL_ROOT" \
    --order-json "$ORDER_JSON" \
    --output-dir "$SUMMARY_DIR" \
    --factor-name "$FACTOR"

echo "Dual evaluation summary: $SUMMARY_DIR"
