#!/usr/bin/env bash
# Block 0 — Instrumentation run (NO new method). Just turn on existing debug logs.
# Goal: confirm the diagnosed root cause (cuts corrupt vel_mask / bank alignment / identity).
#
# 用法: 改下面 USER CONFIG 段, 在 GPU 机器上跑.
# 跑完把 refine-logs/block0_out/ 整个目录拿回来, 用 block0_analyze.py 出 G0 判定.
set -euo pipefail

# ============== USER CONFIG (改这里) ==============
# 多镜头测试片 (3-4 个镜头拼接, 已知切点). 单个视频文件或帧目录, 按你现有 --data_path 的格式.
DATA_PATH="${DATA_PATH:-/path/to/your/multishot_clip}"
# 源/目标 prompt + 触发词 (描述整段视频里要编辑的主体)
SRC_PROMPT="${SRC_PROMPT:-a video of an orange car}"
TRG_PROMPT="${TRG_PROMPT:-a video of a black car}"
SRC_WORD="${SRC_WORD:-orange car}"
TRG_WORD="${TRG_WORD:-black car}"
# GPU
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
PORT="${PORT:-29501}"
# =================================================

OUT_DIR="$(cd "$(dirname "$0")" && pwd)/block0_out"
mkdir -p "$OUT_DIR"

echo "[block0] output -> $OUT_DIR"
echo "[block0] data    -> $DATA_PATH"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$PORT" \
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
  --bank_slot_align_log   "$OUT_DIR/bank_slot_align.log" \
  --bank_debug_log        "$OUT_DIR/bank_debug.log" \
  --attn_mass_log         "$OUT_DIR/attn_mass.log" \
  --debug_velocity_mask_dir "$OUT_DIR/vel_mask" \
  --debug_mask_dir          "$OUT_DIR/cross_mask" \
  --seed 0

echo "[block0] DONE. 把整个 $OUT_DIR 拿回来跑 block0_analyze.py"
echo "[block0] 记下你的切点像素帧号 (例如 0 起算的 cut at frame 30, 60), 分析时要传 --cut_frames"
