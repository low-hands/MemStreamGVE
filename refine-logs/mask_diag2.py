import glob, os, re, sys
import numpy as np, imageio.v3 as iio

vis, srcv = sys.argv[1], sys.argv[2]
s = iio.imread(srcv).astype(np.float32)
e = iio.imread(os.path.join(vis, "edited.mp4")).astype(np.float32)
n = min(len(s), len(e)); s, e = s[:n], e[:n]
D = np.abs(e - s).mean(axis=3)                      # [T,H,W] 连续幅度
H, W = D.shape[1:]

def up(m, h, w):
    yi = (np.arange(h) * m.shape[0] // h).clip(0, m.shape[0]-1)
    xi = (np.arange(w) * m.shape[1] // w).clip(0, m.shape[1]-1)
    return m[yi][:, xi]

def lat2pix(f, n):
    return (0, 1) if f == 0 else (min(1+(f-1)*4, n), min(5+(f-1)*4, n))

by = {}
for f in sorted(glob.glob(os.path.join(vis, "vel_mask", "*_velmask.npy"))):
    b, st = map(int, re.search(r"block_(\d+)_step_(\d+)", os.path.basename(f)).groups())
    if b not in by or st > by[b][0]: by[b] = (st, f)
cby = {int(re.search(r"block_(\d+)", os.path.basename(f)).group(1)): f
       for f in sorted(glob.glob(os.path.join(vis, "cross_mask", "*_srcmask.npy")))}

acc = {"vel_soft":0.0, "vel_hard":0.0, "cross":0.0, "tot":0.0,
       "vel_area":0.0, "cross_area":0.0, "area":0.0,
       "hi_in_vel":0.0, "hi_in_cross":0.0, "hi_tot":0.0}
for b in sorted(by):
    v = np.load(by[b][1]).astype(np.float32)[0,:,0]        # [F,h,w]
    c = None
    if b in cby:
        c = np.load(cby[b]).astype(np.float32)[0].reshape(v.shape[0], 30, 52)
    for fi in range(v.shape[0]):
        p0, p1 = lat2pix(b+fi, n)
        if p0 >= p1: continue
        d = D[p0:p1].max(axis=0)
        vs = up(v[fi], H, W)                    # 软 mask 0..1
        vh = vs > 0.5
        acc["tot"] += d.sum(); acc["area"] += d.size
        acc["vel_soft"] += (d*vs).sum(); acc["vel_hard"] += d[vh].sum()
        acc["vel_area"] += vh.sum()
        hi = d > 32
        acc["hi_tot"] += d[hi].sum(); acc["hi_in_vel"] += d[hi & vh].sum()
        if c is not None:
            cm = up(c[fi] > 0.5, H, W)
            acc["cross"] += d[cm].sum(); acc["cross_area"] += cm.sum()
            acc["hi_in_cross"] += d[hi & cm].sum()

T = acc["tot"]; HI = acc["hi_tot"]
print(f"总 |Δ| 能量 {T/1e6:.1f}M    其中高幅(>32)部分 {HI/1e6:.1f}M ({HI/T*100:.1f}%)\n")
print(f"{'':<26}{'面积占比':>10}{'捕获总能量':>12}{'捕获高幅能量':>14}")
print(f"{'Latent mask (硬>0.5)':<26}{acc['vel_area']/acc['area']*100:>9.1f}%{acc['vel_hard']/T*100:>11.1f}%{acc['hi_in_vel']/HI*100:>13.1f}%")
print(f"{'Latent mask (软加权)':<26}{'—':>10}{acc['vel_soft']/T*100:>11.1f}%{'—':>14}")
print(f"{'Attention mask':<26}{acc['cross_area']/acc['area']*100:>9.1f}%{acc['cross']/T*100:>11.1f}%{acc['hi_in_cross']/HI*100:>13.1f}%")
