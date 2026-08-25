#!/usr/bin/env python3
"""Mask diagnosis: where the masks and the actual edit disagree.

Compares, per pixel, what each mask permitted against what the output actually
changed. Three outcomes matter:

  hit    mask marked it and it changed      -- working as intended
  FP     mask marked it and nothing changed -- permission that was not used
  LEAK   mask did not mark it but it changed -- the edit escaped the mask

LEAK is the one that indicates a real failure: something changed that no mask
authorised. FP is wasted permission, which is only dangerous where the model
does want to edit (a semantically similar neighbour).
"""
import argparse
import glob
import os
import re

import numpy as np


def load_video(path):
    import imageio.v3 as iio
    return iio.imread(path).astype(np.float32)


def latent_to_pixel_range(f_lat, n_pix):
    """Wan VAE temporal layout: latent 0 covers pixel 0, latent k>=1 covers 4."""
    if f_lat == 0:
        return 0, 1
    start = 1 + (f_lat - 1) * 4
    return min(start, n_pix), min(start + 4, n_pix)


def upsample(mask_2d, out_h, out_w):
    h, w = mask_2d.shape
    yi = (np.arange(out_h) * h // out_h).clip(0, h - 1)
    xi = (np.arange(out_w) * w // out_w).clip(0, w - 1)
    return mask_2d[yi][:, xi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis_dir", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--edited", default=None)
    ap.add_argument("--thr", type=float, default=12.0, help="8-bit change threshold")
    ap.add_argument("--vel_thr", type=float, default=0.5, help="latent mask binarisation")
    ap.add_argument("--out", default="mask_diag.png")
    args = ap.parse_args()

    edited = args.edited or os.path.join(args.vis_dir, "edited.mp4")
    src = load_video(args.source)
    edt = load_video(edited)
    n = min(len(src), len(edt))
    src, edt = src[:n], edt[:n]
    changed = (np.abs(edt - src).mean(axis=3) > args.thr)          # [T,H,W] bool
    H, W = changed.shape[1:]

    vel_files = sorted(glob.glob(os.path.join(args.vis_dir, "vel_mask", "*_velmask.npy")))
    cross_files = sorted(glob.glob(os.path.join(args.vis_dir, "cross_mask", "*_srcmask.npy")))
    if not vel_files:
        raise SystemExit("no velmask .npy found -- rerun with the raw-dump patch")

    # keep only the final denoising step of each block: that is the mask in force
    # when the block's latent stops being updated
    by_block = {}
    for f in vel_files:
        m = re.search(r"block_(\d+)_step_(\d+)", os.path.basename(f))
        b, s = int(m.group(1)), int(m.group(2))
        if b not in by_block or s > by_block[b][0]:
            by_block[b] = (s, f)
    cross_by_block = {}
    for f in cross_files:
        b = int(re.search(r"block_(\d+)", os.path.basename(f)).group(1))
        cross_by_block[b] = f

    rows = []
    tot = {"hit": 0, "fp": 0, "leak": 0, "mask": 0, "chg": 0}
    ctot = {"hit": 0, "fp": 0, "leak": 0, "mask": 0}

    for b in sorted(by_block):
        vel = np.load(by_block[b][1]).astype(np.float32)          # [1,F,1,h,w]
        vel = vel[0, :, 0]                                        # [F,h,w]
        cross = None
        if b in cross_by_block:
            c = np.load(cross_by_block[b]).astype(np.float32)[0]  # [F*1560]
            nf = vel.shape[0]
            cross = c.reshape(nf, 30, 52)                         # patch grid per frame

        for fi in range(vel.shape[0]):
            p0, p1 = latent_to_pixel_range(b + fi, n)
            if p0 >= p1:
                continue
            vm = upsample(vel[fi] > args.vel_thr, H, W)
            ch = changed[p0:p1].any(axis=0)
            hit = int((vm & ch).sum()); fp = int((vm & ~ch).sum()); leak = int((~vm & ch).sum())
            tot["hit"] += hit; tot["fp"] += fp; tot["leak"] += leak
            tot["mask"] += int(vm.sum()); tot["chg"] += int(ch.sum())
            if cross is not None:
                cm = upsample(cross[fi] > 0.5, H, W)
                ctot["hit"] += int((cm & ch).sum()); ctot["fp"] += int((cm & ~ch).sum())
                ctot["leak"] += int((~cm & ch).sum()); ctot["mask"] += int(cm.sum())
            rows.append((b, fi, p0, vm, ch, cross[fi] if cross is not None else None))

    def pr(name, d, chg):
        prec = d["hit"] / max(d["hit"] + d["fp"], 1)
        rec = d["hit"] / max(chg, 1)
        print(f"  {name:<22} precision={prec*100:5.1f}%  recall={rec*100:5.1f}%  "
              f"mask={d['mask']/1e6:.2f}M  hit={d['hit']/1e6:.2f}M  FP={d['fp']/1e6:.2f}M  LEAK={d['leak']/1e6:.2f}M")

    print(f"\n帧数 {n}   实际改动像素 {tot['chg']/1e6:.2f}M ({tot['chg']/(n*H*W)*100:.1f}%)")
    print("\n各 mask 对「实际改动」的吻合度:")
    pr("Latent (velocity)", tot, tot["chg"])
    if ctot["mask"]:
        pr("Attention (cross-attn)", ctot, tot["chg"])

    print("\n逐 block 的 LEAK 占比(实际改了但 latent mask 没标):")
    per = {}
    for b, fi, p0, vm, ch, _ in rows:
        d = per.setdefault(b, [0, 0])
        d[0] += int((~vm & ch).sum()); d[1] += int(ch.sum())
    for b in sorted(per):
        lk, c = per[b]
        bar = "#" * int(lk / max(c, 1) * 40)
        print(f"  block {b:3d}  LEAK {lk/max(c,1)*100:5.1f}%  {bar}")

    # ---- picture ----------------------------------------------------------
    from PIL import Image, ImageDraw
    picks = rows[len(rows) // 6], rows[len(rows) // 2], rows[-len(rows) // 6]
    sc = 300 / W
    tw, th = int(W * sc), int(H * sc)
    pad, top = 8, 24
    canvas = Image.new("RGB", (tw * 3 + pad * 4, (th + top) * len(picks) + pad), (18, 18, 20))
    dr = ImageDraw.Draw(canvas)
    for i, (b, fi, p0, vm, ch, cr) in enumerate(picks):
        y = pad + i * (th + top)
        base = src[p0].astype(np.uint8)
        # panel 1: source
        # panel 2: latent mask vs change
        cmp_ = np.zeros((H, W, 3), np.uint8)
        cmp_[vm & ch] = (60, 220, 60)      # hit
        cmp_[vm & ~ch] = (230, 60, 60)     # false positive
        cmp_[~vm & ch] = (70, 130, 255)    # leak
        ov = (base * 0.4 + cmp_ * 0.6).astype(np.uint8)
        # panel 3: cross-attn mask
        cm3 = np.zeros((H, W, 3), np.uint8)
        if cr is not None:
            cm = upsample(cr > 0.5, H, W)
            cm3[cm & ch] = (60, 220, 60)
            cm3[cm & ~ch] = (230, 60, 60)
            cm3[~cm & ch] = (70, 130, 255)
        ov3 = (base * 0.4 + cm3 * 0.6).astype(np.uint8)
        for j, (img, t) in enumerate(zip(
                [base, ov, ov3],
                [f"blk{b} f{fi} source",
                 "Latent mask  绿=命中 红=空放 蓝=漏出",
                 "Attention mask  同上"])):
            x = pad + j * (tw + pad)
            dr.text((x, y), t, fill=(235, 235, 235))
            canvas.paste(Image.fromarray(img).resize((tw, th)), (x, y + top))
    canvas.save(args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
