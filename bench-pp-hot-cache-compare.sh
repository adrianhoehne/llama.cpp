#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/bin/llama-bench}"
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL}"
PROMPT_FILE="${PROMPT_FILE:-pp-bench-conversation-code.txt}"
HOT_CACHE="${HOT_CACHE:-qwen36}"
DEVICE="${DEVICE:-CUDA0}"
N_PROMPT="${N_PROMPT:-3070}"
REPS="${REPS:-5}"
RESERVE_MIB="${RESERVE_MIB:-3000}"
LAYER_CURVE="${LAYER_CURVE:-0.7}"
WEIGHTING="${WEIGHTING:-flat}"
OUT_DIR="${OUT_DIR:-bench-pp-hot-cache-results}"

mkdir -p "$OUT_DIR"

common_args=(
  -hf "$MODEL"
  --prompt-file "$PROMPT_FILE"
  -p "$N_PROMPT"
  -n 0
  -r "$REPS"
  --device "$DEVICE"
  --progress
)

hot_cache_args=(
  --cpu-moe
  --moe-hot-cache "$HOT_CACHE"
  --moe-hot-cache-max-mib -1
  --moe-hot-cache-auto-reserve-mib "$RESERVE_MIB"
  --moe-hot-cache-weighting "$WEIGHTING"
  --moe-hot-cache-layer-curve "$LAYER_CURVE"
)

run_case() {
  local name="$1"
  shift

  local output="$OUT_DIR/${name}.md"
  echo
  echo "== $name =="
  echo "Writing: $output"
  "$@" | tee "$output"
}

run_case baseline-no-hot-cache \
  env -u LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE \
      -u LLAMA_MOE_HOT_CACHE_PP_WORKLIST_ORDER \
      -u LLAMA_MOE_HOT_CACHE_PP_COMPACT_COLD_REDUCE \
      -u LLAMA_MOE_HOT_CACHE_PP_BYPASS \
      -u LLAMA_MOE_HOT_CACHE_PP_BYPASS_MIN_TOKENS \
      -u LLAMA_MOE_HOT_CACHE_PP_MIN_HOT_EXPERT_RATIO \
      "$BIN" \
      "${common_args[@]}" \
      -ncmoe 31

run_case hot-cache-pp-reduce-off \
  "$BIN" \
  "${common_args[@]}" \
  "${hot_cache_args[@]}" \
  --moe-hot-cache-pp-reduce-merge off

run_case hot-cache-pp-reduce-on \
  "$BIN" \
  "${common_args[@]}" \
  "${hot_cache_args[@]}" \
  --moe-hot-cache-pp-reduce-merge on

echo
echo "Done. Results written to: $OUT_DIR"
