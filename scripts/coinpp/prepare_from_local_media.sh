#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
PROFILE="${PROFILE:-clean_5k}"
MEDIA_ROOT="${MEDIA_ROOT:-$ROOT/datasets/CoIN++/media}"
SOURCE_ROOT="${SOURCE_ROOT:-$MEDIA_ROOT/coin_factor1_final_clean}"

required=(
  "$SOURCE_ROOT/train.json"
  "$SOURCE_ROOT/splits"
  "$SOURCE_ROOT/images"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Required clean-final data is missing: $path" >&2
    exit 2
  fi
done

echo "Prepare CoIN++ from local media"
echo "  repository:  $ROOT"
echo "  profile:     $PROFILE"
echo "  source root: $SOURCE_ROOT"
echo "  media root:  $MEDIA_ROOT"

"$PYTHON" scripts/coinpp/prepare_coinpp.py \
  --profile "$PROFILE" \
  --source-root "$SOURCE_ROOT" \
  --media-root "$MEDIA_ROOT" \
  --force-links

echo "Generated training data: $ROOT/datasets/CoIN++/$PROFILE"
echo "Generated data configs:  $ROOT/configs/data_configs/CoIN++/$PROFILE"
