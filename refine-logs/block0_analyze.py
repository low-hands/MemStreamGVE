#!/usr/bin/env python3
"""Block 0 analyzer — G0 gate decision.

Parses the debug logs produced by block0_run.sh and tests whether hard cuts
visibly corrupt (a) bank slot src/trg alignment and (b) target self-attention
mass, relative to non-cut blocks. Velocity-mask images are flagged for manual
inspection (they are overlays, not scalars).

G0 PASS  = cuts DO degrade the metrics (root-cause diagnosis confirmed -> proceed to Block 1).
G0 FAIL  = cuts do NOT degrade -> re-diagnose before building anything.

Usage:
  python block0_analyze.py --out_dir refine-logs/block0_out \
      --cut_frames 30,60 \
      [--num_frame_per_block 3] [--vae_time_down 4] [--first_frame_offset 1] \
      [--window 2]

--cut_frames are PIXEL frame indices (0-based) where hard cuts occur in the
source video. They are converted to latent-frame block indices to match the
`block=` field in the logs (= current_start_frame, a latent index).
"""
import argparse
import os
import re
import sys
from collections import defaultdict

BANK_RE = re.compile(
    r"\[BANK_SLOT\] block=(?P<block>\d+) layer=(?P<layer>\d+) "
    r"valid=(?P<valid>\S+) cos_mean=(?P<cos_mean>\S+) "
    r"cos_p10=(?P<cos_p10>\S+) cos_p50=(?P<cos_p50>\S+) cos_p90=(?P<cos_p90>\S+) "
    r"fg_match=(?P<fg_match>\S+) fg_iou=(?P<fg_iou>\S+)"
)
ATTN_RE = re.compile(
    r"\[ATTN_MASS\] step=(?P<step>-?\d+) bank=(?P<bank>\S+) "
    r"sink_local=(?P<sink_local>\S+) current=(?P<current>\S+) "
    r"prev_len=(?P<prev_len>\d+) cur_len=(?P<cur_len>\d+)"
)


def _f(x):
    try:
        v = float(x)
        return v
    except (ValueError, TypeError):
        return float("nan")


def pixel_to_latent_block(cut_px, npb, vae_down, first_offset):
    """Pixel frame -> latent frame -> block_start (= current_start_frame).

    Latent frames are produced by VAE temporal downsampling. With the standard
    Wan/causal layout the first latent frame covers 1 pixel frame and each
    subsequent latent frame covers `vae_down` pixel frames.
    block_start values in the log advance by npb per block: 0, npb, 2*npb, ...
    so we map a latent frame to the nearest lower multiple of npb.
    """
    if cut_px <= first_offset:
        lat = 0
    else:
        lat = first_offset + (cut_px - first_offset) // vae_down
    block = (lat // npb) * npb
    return lat, block


def parse_bank(path):
    """-> dict[block] = dict[layer] = metrics; averaged over layers -> per-block."""
    per_block_layer = defaultdict(dict)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for line in f:
            m = BANK_RE.search(line)
            if not m:
                continue
            b = int(m.group("block"))
            l = int(m.group("layer"))
            per_block_layer[b][l] = {
                "cos_mean": _f(m.group("cos_mean")),
                "cos_p50": _f(m.group("cos_p50")),
                "fg_match": _f(m.group("fg_match")),
                "fg_iou": _f(m.group("fg_iou")),
            }
    # average across layers per block
    per_block = {}
    for b, layers in per_block_layer.items():
        keys = ["cos_mean", "cos_p50", "fg_match", "fg_iou"]
        agg = {}
        for k in keys:
            vals = [d[k] for d in layers.values() if d[k] == d[k]]  # drop nan
            agg[k] = sum(vals) / len(vals) if vals else float("nan")
        per_block[b] = agg
    return per_block


def parse_attn(path):
    """ATTN_MASS has no block field, only step. We aggregate sequentially and
    cannot align to blocks reliably, so we just report global stats + flag."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            m = ATTN_RE.search(line)
            if not m:
                continue
            rows.append({
                "step": int(m.group("step")),
                "bank": _f(m.group("bank")),
                "sink_local": _f(m.group("sink_local")),
                "current": _f(m.group("current")),
            })
    return rows


def nearmean(per_block, blocks_set, key, want_cut):
    vals = []
    for b, m in per_block.items():
        is_cut = b in blocks_set
        if is_cut == want_cut and m[key] == m[key]:
            vals.append(m[key])
    return (sum(vals) / len(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cut_frames", required=True,
                    help="comma-separated PIXEL frame indices of hard cuts, e.g. 30,60")
    ap.add_argument("--num_frame_per_block", type=int, default=3)
    ap.add_argument("--vae_time_down", type=int, default=4)
    ap.add_argument("--first_frame_offset", type=int, default=1)
    ap.add_argument("--window", type=int, default=2,
                    help="+/- blocks around a cut block to treat as 'cut-affected'")
    args = ap.parse_args()

    npb = args.num_frame_per_block
    cut_px = [int(x) for x in args.cut_frames.split(",") if x.strip() != ""]

    # cut pixel -> cut block, expand by window
    cut_blocks = set()
    cut_map = []
    for cp in cut_px:
        lat, blk = pixel_to_latent_block(cp, npb, args.vae_time_down, args.first_frame_offset)
        cut_map.append((cp, lat, blk))
        for w in range(-args.window, args.window + 1):
            cut_blocks.add(blk + w * npb)

    print("=" * 64)
    print("BLOCK 0 — G0 GATE ANALYSIS")
    print("=" * 64)
    print(f"num_frame_per_block={npb} vae_time_down={args.vae_time_down} "
          f"first_frame_offset={args.first_frame_offset} window=±{args.window} blocks")
    print("\ncut pixel -> latent frame -> block_start:")
    for cp, lat, blk in cut_map:
        print(f"  px {cp:>4d} -> lat {lat:>4d} -> block {blk:>4d} "
              f"(cut-affected blocks: {sorted(blk + w*npb for w in range(-args.window, args.window+1))})")

    bank_log = os.path.join(args.out_dir, "bank_slot_align.log")
    attn_log = os.path.join(args.out_dir, "attn_mass.log")
    vel_dir = os.path.join(args.out_dir, "vel_mask")

    per_block = parse_bank(bank_log)
    if not per_block:
        print(f"\n[ERROR] no [BANK_SLOT] rows parsed from {bank_log}")
        print("  Did the run actually write --bank_slot_align_log ? Aborting.")
        sys.exit(2)

    print(f"\nparsed {len(per_block)} blocks from bank_slot_align.log")
    all_blocks = sorted(per_block.keys())
    print(f"block range: {all_blocks[0]}..{all_blocks[-1]}")

    # ---- Metric 1: bank slot alignment, cut vs non-cut ----
    print("\n" + "-" * 64)
    print("METRIC 1: bank slot src/trg alignment (lower at cuts = leakage signal)")
    print("-" * 64)
    verdicts = []
    for key, label, degrade_is_lower in [
        ("cos_mean", "bank cos_mean", True),
        ("cos_p50", "bank cos_p50", True),
        ("fg_match", "fg_match", True),
        ("fg_iou", "fg_iou", True),
    ]:
        cut_v = nearmean(per_block, cut_blocks, key, want_cut=True)
        non_v = nearmean(per_block, cut_blocks, key, want_cut=False)
        if cut_v != cut_v or non_v != non_v:
            print(f"  {label:14s}: cut={cut_v:.4f} non-cut={non_v:.4f}  (insufficient data)")
            continue
        delta = cut_v - non_v
        degraded = (delta < -0.02) if degrade_is_lower else (delta > 0.02)
        verdicts.append(degraded)
        flag = "DEGRADED@cut" if degraded else "no clear change"
        print(f"  {label:14s}: cut={cut_v:.4f} non-cut={non_v:.4f} Δ={delta:+.4f}  -> {flag}")

    # ---- Metric 2: attn mass (no block alignment; report distribution) ----
    print("\n" + "-" * 64)
    print("METRIC 2: target self-attn mass (bank/sink_local/current)")
    print("  NOTE: ATTN_MASS log carries only `step`, not `block`. Cannot align")
    print("  to cuts automatically. Reported as global means for manual review.")
    print("-" * 64)
    attn = parse_attn(attn_log)
    if attn:
        for key in ["bank", "sink_local", "current"]:
            vals = [r[key] for r in attn if r[key] == r[key]]
            if vals:
                print(f"  {key:11s}: mean={sum(vals)/len(vals):.4f} "
                      f"min={min(vals):.4f} max={max(vals):.4f} n={len(vals)}")
    else:
        print("  (no ATTN_MASS rows; skip)")

    # ---- Metric 3: velocity mask overlays (manual) ----
    print("\n" + "-" * 64)
    print("METRIC 3: velocity-diff edit mask overlays (MANUAL inspection)")
    print("-" * 64)
    if os.path.isdir(vel_dir):
        imgs = sorted(os.listdir(vel_dir))
        print(f"  {len(imgs)} overlays in {vel_dir}")
        print(f"  -> manually inspect overlays AT the cut blocks above:")
        for cp, lat, blk in cut_map:
            print(f"     cut px {cp} -> look around block {blk} overlays for diffuse/garbage mask")
    else:
        print(f"  (no vel_mask dir at {vel_dir}; did you pass --debug_velocity_mask_dir?)")

    # ---- G0 verdict ----
    print("\n" + "=" * 64)
    print("G0 GATE VERDICT")
    print("=" * 64)
    if verdicts and sum(verdicts) >= 1:
        print("  AUTO (bank-alignment): cuts DEGRADE >=1 alignment metric -> consistent with diagnosis.")
        print("  STATUS: G0 likely PASS (confirm with vel_mask overlays + decoded frames at cuts).")
        print("  NEXT: proceed to Block 1 (ΔKV transport probe).")
    else:
        print("  AUTO (bank-alignment): NO clear degradation at cuts.")
        print("  STATUS: G0 INCONCLUSIVE from bank metrics alone.")
        print("  ACTION: inspect vel_mask overlays + decoded frames at cut blocks before deciding.")
        print("          If those also show no cut-induced damage -> RE-DIAGNOSE (do not build Block 1).")
    print("\n  (Auto-check only covers bank alignment. vel_mask + decoded identity at")
    print("   cuts are the primary visual evidence and must be eyeballed.)")


if __name__ == "__main__":
    main()
