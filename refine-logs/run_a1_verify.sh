#!/usr/bin/env bash
# A1 verification run. Reruns Block 0 against the fixed code and prints the
# verdict directly, so nothing has to be shipped back to be read.
#
#   DATA_PATH=/path/clip.mp4 SRC_PROMPT=... TRG_PROMPT=... \
#   SRC_WORD=... TRG_WORD=... bash refine-logs/run_a1_verify.sh
#
# Set NO_BANK=1 for the second condition (--bank_prev_context_mode no_bank),
# which measures what the bank contributes through the prev blending path.
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

: "${DATA_PATH:?set DATA_PATH to the source clip}"
SRC_PROMPT="${SRC_PROMPT:?set SRC_PROMPT}"
TRG_PROMPT="${TRG_PROMPT:?set TRG_PROMPT}"
SRC_WORD="${SRC_WORD:?set SRC_WORD}"
TRG_WORD="${TRG_WORD:?set TRG_WORD}"

GPUS="${GPUS:-0}"
NPROC="${NPROC:-1}"
PORT="${PORT:-29501}"
NO_BANK="${NO_BANK:-0}"

if [ "$NO_BANK" = "1" ]; then
  OUT_DIR="refine-logs/block0_out_nobank"
  EXTRA=(--bank_prev_context_mode no_bank)
else
  OUT_DIR="refine-logs/block0_out_fixed"
  EXTRA=()
fi
mkdir -p "$OUT_DIR"

# Record the configuration next to the logs. The original Block 0 run did not,
# which is why its numbers cannot be reproduced exactly.
cat > "$OUT_DIR/run_config.txt" <<EOF
DATA_PATH=$DATA_PATH
SRC_PROMPT=$SRC_PROMPT
TRG_PROMPT=$TRG_PROMPT
SRC_WORD=$SRC_WORD
TRG_WORD=$TRG_WORD
NO_BANK=$NO_BANK
commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
date=$(date -Is)
EOF

echo "[a1] out      -> $OUT_DIR"
echo "[a1] data     -> $DATA_PATH"
echo "[a1] no_bank  -> $NO_BANK"
echo "[a1] commit   -> $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

LAUNCH=(python)
if [ "$NPROC" -gt 1 ]; then
  LAUNCH=(torchrun --nproc_per_node="$NPROC" --master_port="$PORT")
fi

CUDA_VISIBLE_DEVICES="$GPUS" "${LAUNCH[@]}" \
  inference_edit_streamgve.py \
  --data_path "$DATA_PATH" \
  --save_path "$OUT_DIR/edited.mp4" \
  --src_prompt "$SRC_PROMPT" \
  --trg_prompt "$TRG_PROMPT" \
  --src_word "$SRC_WORD" \
  --trg_word "$TRG_WORD" \
  --bridge_mode soft_fg_target \
  --bridge_fg_target_floor 0.65 \
  --fg_boost_factor 4.0 \
  --bank_slot_align_log "$OUT_DIR/bank_slot_align.log" \
  --bank_debug_log      "$OUT_DIR/bank_debug.log" \
  --attn_mass_log       "$OUT_DIR/attn_mass.log" \
  --debug_mask_dir      "$OUT_DIR/cross_mask" \
  --seed 0 \
  "${EXTRA[@]}"

echo
python refine-logs/a1_verdict.py --out_dir "$OUT_DIR"
