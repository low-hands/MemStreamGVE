#!/usr/bin/env python3
"""
probe_transport.py — Block 1 ΔKV Transport Probe orchestration + G1 verdict.

EXPERIMENT_PLAN Block 1 / FINAL_PROPOSAL §4.

This is the GO/NO-GO harness. For each two-shot pair (same subject, changed
pose/view/light), it:
  1. DUMP : run editing on shot A with --probe_mode dump -> capsules.pt
  2. C0   : run editing on shot B with no injection (vanilla baseline)
  3. C1   : run editing on shot B injecting the DIRECT target KV capsule
  4. C2   : run editing on shot B injecting the SOURCE-ANCHORED residual (proposed)

Then it computes, on foreground crops (oracle/SAM masks) vs background separately:
  S_id    = cos(DINOv2(edit_B_fg), DINOv2(edit_A_fg))      cross-shot identity
  S_txt   = CLIP-T(edit_B_fg, trg_prompt)                   edit fidelity
  S_struct= cos(struct(edit_B_fg), struct(src_B_fg))        current-shot structure
  L_bg    = LPIPS(edit_B_bg, src_B_bg)                       background leakage

PASS thresholds (proceed to Block 2 only if ALL hold) — FINAL_PROPOSAL §4:
  1. S_id(C2) - S_id(C0)        >= +0.03  (or +10% relative)
  2. C2 beats C1 on id/struct tradeoff in >= 70% of pairs
  3. S_struct(C2)              >= 0.95 * S_struct(C0)
  4. L_bg(C2) - L_bg(C0)        <  0.02
  5. catastrophic artifacts     <  15% of pairs

This file is the ORCHESTRATOR + METRICS, kept separate from the model code.
It expects a manifest JSON describing the pairs. Heavy metric deps (DINOv2,
CLIP, LPIPS) are imported lazily so --dry-run works without them.

Manifest schema (list of pairs):
[
  {
    "id": "person_reverse_01",
    "category": "shot_reverse",            # one of the §4 / Block-4 categories
    "edit_type": "color",                  # color|material|texture (low-level only for probe)
    "shotA_video": "/abs/path/shotA.mp4",
    "shotB_video": "/abs/path/shotB.mp4",
    "src_prompt": "a man in a black sweatshirt rides a bike",
    "trg_prompt": "a man in a RED sweatshirt rides a bike",
    "src_word": "sweatshirt",
    "trg_word": "sweatshirt",
    "shotA_fg_mask": "/abs/path/shotA_fg.mp4",   # oracle/SAM fg mask (optional; bg metric skipped if absent)
    "shotB_fg_mask": "/abs/path/shotB_fg.mp4"
  },
  ...
]

Usage (on GPU box):
  python refine-logs/probe_transport.py \
      --manifest refine-logs/probe_pairs.json \
      --work_dir refine-logs/block1_out \
      --config_path configs/inference.yaml \
      --probe_layers 12-26

Add --dry-run to only print the command plan (no torch, no GPU) for inspection.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ----------------------------------------------------------------------------- 
# Command construction
# -----------------------------------------------------------------------------

def _edit_cmd(args, *, data_path, save_path, src_prompt, trg_prompt,
              src_word, trg_word, probe_mode, probe_save=None, probe_capsule=None):
    """Build the inference_edit_streamgve.py command for one run."""
    cmd = [
        sys.executable, "inference_edit_streamgve.py",
        "--data_path", data_path,
        "--save_path", save_path,
        "--src_prompt", src_prompt,
        "--trg_prompt", trg_prompt,
        "--src_word", src_word,
        "--trg_word", trg_word,
        "--config_path", args.config_path,
        "--step", str(args.step),
        "--seed", str(args.seed),
        # keep the current best editing config fixed across conditions
        "--bridge_mode", args.bridge_mode,
        "--bridge_fg_target_floor", str(args.bridge_fg_target_floor),
        "--fg_boost_factor", str(args.fg_boost_factor),
        # probe
        "--probe_mode", probe_mode,
        "--probe_topk", str(args.probe_topk),
        "--probe_temp", str(args.probe_temp),
        "--probe_max_per_key", str(args.probe_max_per_key),
    ]
    if args.probe_layers:
        cmd += ["--probe_layers", args.probe_layers]
    if probe_save:
        cmd += ["--probe_save_path", probe_save]
    if probe_capsule:
        cmd += ["--probe_capsule_path", probe_capsule]
    return cmd


def _run(cmd, dry_run, log_path=None):
    print("  $ " + " ".join(cmd))
    if dry_run:
        return 0
    with open(log_path, "w", encoding="utf-8") if log_path else _NullCtx() as logf:
        proc = subprocess.run(
            cmd, cwd=os.getcwd(),
            stdout=(logf if log_path else None),
            stderr=subprocess.STDOUT if log_path else None,
        )
    return proc.returncode


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------------- 
# Metrics (lazy, GPU)
# -----------------------------------------------------------------------------

def _load_metric_models(device):
    """Lazily load DINOv2, CLIP, LPIPS. Returns a dict of callables."""
    import torch
    import torchvision.transforms as T
    models = {}

    # DINOv2 for identity / structure features
    try:
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        models["dino"] = dino
    except Exception as e:
        print(f"[warn] DINOv2 load failed: {e}; S_id/S_struct will be skipped.")
        models["dino"] = None

    # CLIP for text fidelity
    try:
        import open_clip
        clip_model, _, clip_pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        clip_model = clip_model.to(device).eval()
        clip_tok = open_clip.get_tokenizer("ViT-B-32")
        models["clip"] = (clip_model, clip_pre, clip_tok)
    except Exception as e:
        print(f"[warn] open_clip load failed: {e}; S_txt will be skipped.")
        models["clip"] = None

    # LPIPS for background leakage
    try:
        import lpips
        models["lpips"] = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as e:
        print(f"[warn] lpips load failed: {e}; L_bg will be skipped.")
        models["lpips"] = None

    models["_T"] = T
    models["_device"] = device
    return models


def _read_video_frames(path):
    import imageio.v3 as iio
    import numpy as np
    vid = iio.imread(path, plugin="pyav")  # [T,H,W,C] uint8
    return np.asarray(vid)


def _read_mask_frames(path):
    import numpy as np
    m = _read_video_frames(path)
    if m.ndim == 4:
        m = m[..., 0]
    return (m > 127).astype("float32")  # [T,H,W]


def _fg_bg_crops(frames, masks):
    """Return mean-pooled fg pixels stack and bg pixels stack per frame.
    Simplest robust reduction: masked frames (fg) and inverse-masked (bg)."""
    import numpy as np
    T = min(frames.shape[0], masks.shape[0])
    frames = frames[:T].astype("float32") / 255.0
    masks = masks[:T]
    if masks.shape[1:] != frames.shape[1:3]:
        # nearest resize mask to frame size
        import torch
        import torch.nn.functional as F
        mm = torch.from_numpy(masks)[:, None]
        mm = F.interpolate(mm, size=frames.shape[1:3], mode="nearest")[:, 0]
        masks = mm.numpy()
    fg = frames * masks[..., None]
    bg = frames * (1.0 - masks[..., None])
    return fg, bg, masks


def _dino_feat(models, frames_fg):
    import torch
    dino = models["dino"]
    if dino is None:
        return None
    T = models["_T"]
    dev = models["_device"]
    tf = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), antialias=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for f in frames_fg:
            x = tf((f * 255).astype("uint8")).unsqueeze(0).to(dev)
            feats.append(dino(x).squeeze(0))
    return torch.stack(feats).mean(0)  # [D]


def _clip_txt(models, frames_fg, prompt):
    import torch
    if models["clip"] is None:
        return None
    clip_model, clip_pre, clip_tok = models["clip"]
    dev = models["_device"]
    from PIL import Image
    sims = []
    with torch.no_grad():
        txt = clip_tok([prompt]).to(dev)
        tfeat = clip_model.encode_text(txt)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        for f in frames_fg:
            img = Image.fromarray((f * 255).astype("uint8"))
            x = clip_pre(img).unsqueeze(0).to(dev)
            ifeat = clip_model.encode_image(x)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            sims.append((ifeat @ tfeat.T).item())
    return float(sum(sims) / max(len(sims), 1))


def _lpips_bg(models, bg_a, bg_b):
    import torch
    if models["lpips"] is None:
        return None
    dev = models["_device"]
    lp = models["lpips"]
    T = min(bg_a.shape[0], bg_b.shape[0])
    vals = []
    with torch.no_grad():
        for i in range(T):
            a = torch.from_numpy(bg_a[i]).permute(2, 0, 1)[None].to(dev) * 2 - 1
            b = torch.from_numpy(bg_b[i]).permute(2, 0, 1)[None].to(dev) * 2 - 1
            vals.append(lp(a, b).item())
    return float(sum(vals) / max(len(vals), 1))


def _cos(a, b):
    import torch
    if a is None or b is None:
        return None
    return float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())


# ----------------------------------------------------------------------------- 
# Per-pair evaluation
# -----------------------------------------------------------------------------

def evaluate_pair(models, pair, paths):
    """Compute S_id, S_txt, S_struct, L_bg for C0/C1/C2 of one pair."""
    out = {}
    have_mask = "shotB_fg_mask" in pair and os.path.exists(pair.get("shotB_fg_mask", ""))

    framesA = _read_video_frames(paths["editA"])  # edited shot A (from dump run)
    maskA = _read_mask_frames(pair["shotA_fg_mask"]) if pair.get("shotA_fg_mask") else None
    srcB = _read_video_frames(pair["shotB_video"])
    maskB = _read_mask_frames(pair["shotB_fg_mask"]) if have_mask else None

    fgA = _dino_feat(models, _fg_bg_crops(framesA, maskA)[0]) if maskA is not None else None
    src_fgB = None
    if maskB is not None:
        src_fg, _, _ = _fg_bg_crops(srcB, maskB)
        src_fgB = _dino_feat(models, src_fg)

    for cond in ("C0", "C1", "C2"):
        editB = _read_video_frames(paths[cond])
        if maskB is not None:
            fgB, bgB, _ = _fg_bg_crops(editB, maskB)
            _, src_bgB, _ = _fg_bg_crops(srcB, maskB)
        else:
            fgB = editB.astype("float32") / 255.0
            bgB = None
            src_bgB = None
        feat_fgB = _dino_feat(models, fgB)
        out[cond] = {
            "S_id": _cos(feat_fgB, fgA),
            "S_txt": _clip_txt(models, fgB, pair["trg_prompt"]),
            "S_struct": _cos(feat_fgB, src_fgB),
            "L_bg": _lpips_bg(models, src_bgB, bgB) if bgB is not None else None,
        }
    return out


# ----------------------------------------------------------------------------- 
# G1 verdict
# -----------------------------------------------------------------------------

def g1_verdict(per_pair):
    """Apply the 5 FINAL_PROPOSAL §4 PASS thresholds across all pairs."""
    import statistics as st

    def _mean(key, cond):
        vals = [p[cond][key] for p in per_pair if p[cond].get(key) is not None]
        return st.mean(vals) if vals else None

    s_id_c0 = _mean("S_id", "C0")
    s_id_c2 = _mean("S_id", "C2")
    s_struct_c0 = _mean("S_struct", "C0")
    s_struct_c2 = _mean("S_struct", "C2")
    lbg_c0 = _mean("L_bg", "C0")
    lbg_c2 = _mean("L_bg", "C2")

    # threshold 1: S_id absolute +0.03 or relative +10%
    t1 = None
    if s_id_c0 is not None and s_id_c2 is not None:
        abs_gain = s_id_c2 - s_id_c0
        rel_gain = abs_gain / abs(s_id_c0) if s_id_c0 != 0 else 0
        t1 = (abs_gain >= 0.03) or (rel_gain >= 0.10)

    # threshold 2: C2 beats C1 on id/struct tradeoff in >=70% of pairs
    wins = 0
    counted = 0
    for p in per_pair:
        c1, c2 = p["C1"], p["C2"]
        if None in (c1.get("S_id"), c2.get("S_id"), c1.get("S_struct"), c2.get("S_struct")):
            continue
        counted += 1
        c2_score = c2["S_id"] + c2["S_struct"]
        c1_score = c1["S_id"] + c1["S_struct"]
        if c2_score >= c1_score:
            wins += 1
    t2 = (counted > 0) and (wins / counted >= 0.70)

    # threshold 3: S_struct(C2) >= 0.95 * S_struct(C0)
    t3 = None
    if s_struct_c0 is not None and s_struct_c2 is not None:
        t3 = s_struct_c2 >= 0.95 * s_struct_c0

    # threshold 4: L_bg(C2) - L_bg(C0) < 0.02
    t4 = None
    if lbg_c0 is not None and lbg_c2 is not None:
        t4 = (lbg_c2 - lbg_c0) < 0.02

    # threshold 5: catastrophic artifacts <15% — needs manual flag; report ratio if provided
    cat = [p.get("catastrophic", None) for p in per_pair]
    cat = [c for c in cat if c is not None]
    t5 = (sum(cat) / len(cat) < 0.15) if cat else None

    checks = {
        "T1_S_id_gain>=0.03/+10%": t1,
        "T2_C2>C1_in>=70%": t2,
        "T3_S_struct(C2)>=0.95*C0": t3,
        "T4_dL_bg<0.02": t4,
        "T5_catastrophic<15%": t5,
    }
    decided = [v for v in checks.values() if v is not None]
    overall = "PASS" if (decided and all(decided)) else (
        "FAIL" if any(v is False for v in checks.values()) else "INCOMPLETE")
    summary = {
        "S_id": {"C0": s_id_c0, "C2": s_id_c2},
        "S_struct": {"C0": s_struct_c0, "C2": s_struct_c2},
        "L_bg": {"C0": lbg_c0, "C2": lbg_c2},
        "C2_beats_C1_ratio": (wins / counted) if counted else None,
    }
    return checks, overall, summary


# ----------------------------------------------------------------------------- 
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON list of two-shot pairs.")
    ap.add_argument("--work_dir", default="refine-logs/block1_out")
    ap.add_argument("--config_path", default="configs/inference.yaml")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bridge_mode", default="soft_fg_target")
    ap.add_argument("--bridge_fg_target_floor", type=float, default=0.65)
    ap.add_argument("--fg_boost_factor", type=float, default=4.0)
    ap.add_argument("--probe_layers", default="12-26")
    ap.add_argument("--probe_topk", type=int, default=4)
    ap.add_argument("--probe_temp", type=float, default=0.07)
    ap.add_argument("--probe_max_per_key", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print the command plan; no GPU, no metrics.")
    ap.add_argument("--skip_run", action="store_true",
                    help="Skip editing runs (reuse existing outputs); only compute metrics.")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(Path(args.manifest).read_text())

    # ---- Stage 1: run dump + C0/C1/C2 for every pair ----
    run_paths = {}
    for pair in pairs:
        pid = pair["id"]
        pdir = work / pid
        pdir.mkdir(parents=True, exist_ok=True)
        caps = str(pdir / "capsule.pt")
        paths = {
            "capsule": caps,
            "editA": str(pdir / "editA_dump.mp4"),
            "C0": str(pdir / "editB_C0.mp4"),
            "C1": str(pdir / "editB_C1.mp4"),
            "C2": str(pdir / "editB_C2.mp4"),
        }
        run_paths[pid] = paths
        print(f"\n=== pair {pid} ({pair.get('category')}, {pair.get('edit_type')}) ===")

        if not args.skip_run:
            # DUMP on shot A
            rc = _run(_edit_cmd(
                args, data_path=pair["shotA_video"], save_path=paths["editA"],
                src_prompt=pair["src_prompt"], trg_prompt=pair["trg_prompt"],
                src_word=pair["src_word"], trg_word=pair["trg_word"],
                probe_mode="dump", probe_save=caps,
            ), args.dry_run, log_path=str(pdir / "dump.log") if not args.dry_run else None)
            if rc != 0 and not args.dry_run:
                print(f"  [error] dump run failed for {pid} (rc={rc}); skipping pair.")
                continue
            # C0 / C1 / C2 on shot B
            for cond in ("C0", "C1", "C2"):
                rc = _run(_edit_cmd(
                    args, data_path=pair["shotB_video"], save_path=paths[cond],
                    src_prompt=pair["src_prompt"], trg_prompt=pair["trg_prompt"],
                    src_word=pair["src_word"], trg_word=pair["trg_word"],
                    probe_mode=cond,
                    probe_capsule=(caps if cond in ("C1", "C2") else None),
                ), args.dry_run, log_path=str(pdir / f"{cond}.log") if not args.dry_run else None)
                if rc != 0 and not args.dry_run:
                    print(f"  [error] {cond} run failed for {pid} (rc={rc}).")

    if args.dry_run:
        print("\n[dry-run] command plan printed above. No metrics computed.")
        return

    # ---- Stage 2: metrics ----
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = _load_metric_models(device)

    per_pair = []
    for pair in pairs:
        pid = pair["id"]
        paths = run_paths[pid]
        if not all(os.path.exists(paths[c]) for c in ("editA", "C0", "C1", "C2")):
            print(f"[warn] missing outputs for {pid}; skipping in metrics.")
            continue
        m = evaluate_pair(models, pair, paths)
        m["id"] = pid
        m["category"] = pair.get("category")
        m["edit_type"] = pair.get("edit_type")
        if "catastrophic" in pair:
            m["catastrophic"] = pair["catastrophic"]
        per_pair.append(m)
        print(f"[metrics] {pid}: "
              f"S_id C0={m['C0']['S_id']} C2={m['C2']['S_id']} | "
              f"S_struct C0={m['C0']['S_struct']} C2={m['C2']['S_struct']}")

    checks, overall, summary = g1_verdict(per_pair)

    report = {
        "n_pairs": len(per_pair),
        "g1_checks": checks,
        "g1_overall": overall,
        "summary": summary,
        "per_pair": per_pair,
    }
    out_json = work / "G1_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 60)
    print("G1 GATE VERDICT")
    print("=" * 60)
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\n  OVERALL: {overall}")
    print(f"  summary: {json.dumps(summary, indent=2)}")
    print(f"\n  full report -> {out_json}")
    print("\n  Reminder (FINAL_PROPOSAL §6): the paper stands or falls on this probe.")
    print("  PASS  -> build Block 2+ (reset-only, then full method).")
    print("  FAIL  -> STOP. Reframe per NO-GO interpretations. Do NOT add heuristics to rescue.")


if __name__ == "__main__":
    main()
