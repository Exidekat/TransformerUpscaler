#!/usr/bin/env bash
# run_eval_all.sh
#
# Execute eval_metrics.py over all models in the models/ directory.
# Usage: bash tools/run_eval_all.sh --data_dir DIR --scale SCALE [--max_images N] [--log_interval N] [--compile] [--quantize]

#set -euo pipefail

function show_help() {
  cat <<EOF
Usage: $0 --data_dir DIR --scale SCALE [--max_images N] [--log_interval N] [--compile] [--quantize]

Options:
  --data_dir DIR       Directory of high-res images (passed to eval_metrics.py)
  --scale SCALE        Upscale factor (2,3,4,6).
  --max_images N       (Optional) Maximum number of images to evaluate per model.
  --log_interval N     (Optional) Log progress every N images (default: 10).
  --compile            (Optional) Enable torch.compile in eval_metrics.py.
  --quantize           (Optional) Enable quantization in eval_metrics.py.
  -h|--help            Show this help message.
EOF
  exit 1
}

# Default values
DATA_DIR=""
SCALE=""
MAX_IMAGES=""
LOG_INTERVAL=10
COMPILE_FLAG=""
QUANTIZE_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="$2"; shift 2;;
    --scale)
      SCALE="$2"; shift 2;;
    --max_images)
      MAX_IMAGES="$2"; shift 2;;
    --log_interval)
      LOG_INTERVAL="$2"; shift 2;;
    --compile)
      COMPILE_FLAG="--compile"; shift;;
    --quantize)
      QUANTIZE_FLAG="--quantize"; shift;;
    -h|--help)
      show_help;;
    *)
      echo "Unknown argument: $1" >&2; show_help;;
  esac
done

# Validate required arguments
if [[ -z "$DATA_DIR" || -z "$SCALE" ]]; then
  echo "Error: --data_dir and --scale are required." >&2
  show_help
fi

# Iterate over every directory containing a model.py, at any depth under models/.
# (A one-level glob silently skipped the models/SwinBased/* variants.)
while IFS= read -r model_file; do
    model_path=$(dirname "$model_file")
    # Name relative to models/, e.g. "Fastv2" or "SwinBased/base_hai"
    model_name=${model_path#models/}
    printf '\n=== Evaluating model: %s ===\n' "$model_name"
    # Build eval_metrics.py arguments
    ARGS=(
      --data_dir "$DATA_DIR"
      --model "$model_name"
      --scale "$SCALE"
      --log_interval "$LOG_INTERVAL"
    )
    if [[ -n "$MAX_IMAGES" ]]; then
      ARGS+=(--max_images "$MAX_IMAGES")
    fi
    if [[ -n "$COMPILE_FLAG" ]]; then
      ARGS+=(--compile)
    fi
    if [[ -n "$QUANTIZE_FLAG" ]]; then
      ARGS+=(--quantize)
    fi
    # Run evaluation
    python eval_metrics.py "${ARGS[@]}"
done < <(find models -mindepth 1 -name model.py -type f | sort)

printf '\nAll models evaluated.\n'