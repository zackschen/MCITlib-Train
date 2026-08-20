#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

METHODS="${METHODS:-LoRA-FT Replay OLoRA MoELoRA CL-MoE HiDe DISCO}"
FACTORS="${FACTORS:-evidence_complexity}"
PROFILE="${PROFILE:-strict}"
MODE="${MODE:-all}"

read -r -a method_list <<< "$METHODS"
read -r -a factor_list <<< "$FACTORS"

echo "CoIN++ multi-method pipeline"
echo "  methods:  $METHODS"
echo "  factors:  $FACTORS"
echo "  profile:  $PROFILE"
echo "  mode:     $MODE"

for factor in "${factor_list[@]}"; do
  for method in "${method_list[@]}"; do
    echo
    echo "===== $method | $factor ====="
    METHOD="$method" \
    FACTOR="$factor" \
    PROFILE="$PROFILE" \
    MODE="$MODE" \
      bash scripts/coinpp/run_method_pipeline.sh
  done
done
