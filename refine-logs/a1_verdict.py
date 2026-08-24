#!/usr/bin/env python3
"""A1 verdict + bank read-pattern report.

Parses bank_debug.log and attn_mass.log from a Block 0 run and prints, against
the pre-fix baseline, whether the target-bank double-write is gone.

Usage:
  python refine-logs/a1_verdict.py --out_dir refine-logs/block0_out
"""
import argparse
import os
import re
import statistics
from collections import defaultdict

# Steady state only: the bank is not full before this (pointer advances every
# record_interval blocks, capacity is bank_size slots), and partially filled
# slots drag the averages down.
STEADY_FROM = 30

# Measured on the pre-fix run, block >= 30.
BASELINE = {
    "id_match_mean": 0.781,
    "id_match_frac1": 0.39,
    "per_layer": {0: (2.80, 2.03), 15: (2.13, 2.10), 29: (2.90, 2.10)},  # (src_uniq, trg_uniq)
}

ALIGN_RE = re.compile(
    r"block=(\d+) layer=(\d+) id_match=([\d.]+) fg_iou=([-\d.]+) "
    r"src_blocks=\[([^\]]*)\] trg_blocks=\[([^\]]*)\]"
)
MASS_RE = re.compile(
    r"\[ATTN_MASS\] step=(-?\d+) bank=(\S+) sink_local=(\S+) current=(\S+)"
    r"(?: bank_fg=(\S+) bank_bg=(\S+) bank_fg_frac=(\S+))? prev_len=(\d+)"
)


def read(path):
    if not os.path.exists(path):
        return ""
    # The logger writes a literal backslash-n rather than a newline.
    return open(path, encoding="utf-8", errors="replace").read().replace("\\n", "\n")


def parse_align(path):
    rows = []
    for line in read(path).split("\n"):
        m = ALIGN_RE.search(line)
        if not m:
            continue
        b, ly, idm, _iou, s, t = m.groups()
        su = [x for x in s.split(",") if x.strip()]
        tu = [x for x in t.split(",") if x.strip()]
        if su:
            rows.append((int(b), int(ly), float(idm), len(su), len(tu)))
    return rows


def parse_mass(path):
    rows = []
    for line in read(path).split("\n"):
        m = MASS_RE.search(line)
        if not m:
            continue
        rows.append({
            "bank": float(m.group(2)),
            "sink_local": float(m.group(3)),
            "current": float(m.group(4)),
            "bank_fg": float(m.group(5)) if m.group(5) else None,
            "bank_bg": float(m.group(6)) if m.group(6) else None,
            "fg_frac": float(m.group(7)) if m.group(7) else None,
            "prev_len": int(m.group(8)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="refine-logs/block0_out")
    args = ap.parse_args()

    align = parse_align(os.path.join(args.out_dir, "bank_debug.log"))
    steady = [r for r in align if r[0] >= STEADY_FROM]

    print("=" * 68)
    print("A1 VERDICT  (target bank double-write)")
    print("=" * 68)
    if not steady:
        print(f"  no [BANK_ALIGN] rows with block >= {STEADY_FROM}; ran too few blocks?")
        return 1

    idm = [r[2] for r in steady]
    mean_idm = statistics.mean(idm)
    frac1 = sum(1 for x in idm if x > 0.999) / len(idm)

    print(f"\n  {'metric':<26}{'before':>10}{'now':>10}")
    print(f"  {'-'*46}")
    print(f"  {'id_match mean':<26}{BASELINE['id_match_mean']:>10.3f}{mean_idm:>10.3f}")
    print(f"  {'id_match == 1.0 frac':<26}{BASELINE['id_match_frac1']:>10.2f}{frac1:>10.2f}")

    print(f"\n  per-layer distinct blocks in the 3 slots (src / trg):")
    print(f"  {'layer':>6}{'before':>16}{'now':>16}{'gap now':>10}")
    gaps = []
    by_layer = defaultdict(list)
    for r in steady:
        by_layer[r[1]].append(r)
    for ly in sorted(by_layer):
        rs = by_layer[ly]
        s_now = statistics.mean(r[3] for r in rs)
        t_now = statistics.mean(r[4] for r in rs)
        gap = t_now - s_now
        gaps.append(gap)
        b = BASELINE["per_layer"].get(ly)
        before = f"{b[0]:.2f} / {b[1]:.2f}" if b else "-"
        print(f"  {ly:>6}{before:>16}{f'{s_now:.2f} / {t_now:.2f}':>16}{gap:>10.2f}")

    worst_gap = min(gaps)
    print()
    # The A1-specific signature is the src/trg asymmetry: the double-write held
    # trg down to ~2 distinct blocks at every layer while src varied by layer.
    # Absolute counts are contaminated by block_id aliasing (ids repeat inside a
    # record_interval epoch), but the aliasing hits both branches identically, so
    # the gap cancels it out.
    if worst_gap > -0.15:
        print("  ==> A1 FIXED. The src/trg asymmetry is gone (worst gap "
              f"{worst_gap:+.2f}, was -0.80).")
    elif worst_gap > -0.45:
        print(f"  ==> PARTIAL. Asymmetry shrank but is not gone (worst gap {worst_gap:+.2f},"
              " was -0.80). Something still writes the target bank twice.")
    else:
        print(f"  ==> NOT FIXED. Asymmetry unchanged (worst gap {worst_gap:+.2f}).")

    print()
    if mean_idm > 0.999:
        print("  id_match == 1.0: source and target selected the same history on this")
        print("  clip. Note this is not a guarantee -- the two branches still score")
        print("  independently, so another clip/prompt pair can diverge again.")
    else:
        print(f"  id_match {mean_idm:.3f} < 1.0: the residual misalignment is the net")
        print("  contribution of independent per-branch top-k selection. Sharing one")
        print("  index set (source-driven) is what makes id_match == 1 structural.")

    # ---- bank read pattern -------------------------------------------------
    mass = parse_mass(os.path.join(args.out_dir, "attn_mass.log"))
    mass = [m for m in mass if m["prev_len"] > 0]
    print()
    print("=" * 68)
    print("BANK READ PATTERN  (what long-term memory is actually used for)")
    print("=" * 68)
    if not mass:
        print("  no [ATTN_MASS] rows.")
        return 0

    print(f"\n  attention mass: bank={statistics.mean(m['bank'] for m in mass):.4f}"
          f"  sink_local={statistics.mean(m['sink_local'] for m in mass):.4f}"
          f"  current={statistics.mean(m['current'] for m in mass):.4f}")

    withfg = [m for m in mass if m["bank_fg"] is not None]
    if not withfg:
        print("\n  bank_fg / bank_bg absent -- running older code without the split.")
        return 0

    fg = statistics.mean(m["bank_fg"] for m in withfg)
    bg = statistics.mean(m["bank_bg"] for m in withfg)
    frac = statistics.mean(m["fg_frac"] for m in withfg)
    share = fg / (fg + bg) if (fg + bg) > 0 else 0.0

    print(f"  of which  bank_fg={fg:.4f}   bank_bg={bg:.4f}")
    print(f"\n  foreground is {frac*100:.1f}% of bank tokens "
          f"but takes {share*100:.1f}% of the bank's attention")
    ratio = share / frac if frac > 0 else 0.0
    print(f"  over-read ratio: {ratio:.1f}x")

    print()
    if ratio >= 2.0:
        print("  ==> The bank is read mainly for the edited subject. Storing two full")
        print("      banks wastes the duplicated background: one source bank plus a")
        print("      foreground residual keeps the same information.")
    elif ratio >= 1.2:
        print("  ==> Foreground is over-read, but not dominantly. A residual store")
        print("      still saves memory; the background copy is not pure waste.")
    else:
        print("  ==> Foreground is NOT preferentially read: the bank supplies generic")
        print("      scene context, not edited identity. Then carrying target content")
        print("      in the bank buys little, and a single source bank may suffice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
