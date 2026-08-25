import glob, os, re, sys
import numpy as np, imageio.v3 as iio
from PIL import Image, ImageDraw

vis, srcv, out = sys.argv[1], sys.argv[2], sys.argv[3]
s = iio.imread(srcv).astype(np.float32)
e = iio.imread(os.path.join(vis, "edited.mp4")).astype(np.float32)
n = min(len(s), len(e)); s, e = s[:n], e[:n]
D = np.abs(e - s).mean(axis=3); H, W = D.shape[1:]

def up(m,h,w):
    yi=(np.arange(h)*m.shape[0]//h).clip(0,m.shape[0]-1); xi=(np.arange(w)*m.shape[1]//w).clip(0,m.shape[1]-1)
    return m[yi][:,xi]
def lat2pix(f,n): return (0,1) if f==0 else (min(1+(f-1)*4,n), min(5+(f-1)*4,n))

by={}
for f in sorted(glob.glob(os.path.join(vis,"vel_mask","*_velmask.npy"))):
    b,st=map(int,re.search(r"block_(\d+)_step_(\d+)",os.path.basename(f)).groups())
    if b not in by or st>by[b][0]: by[b]=(st,f)
cby={int(re.search(r"block_(\d+)",os.path.basename(f)).group(1)):f
     for f in sorted(glob.glob(os.path.join(vis,"cross_mask","*_srcmask.npy")))}

picks=[9,27,45]; rows=[]
for b in picks:
    v=np.load(by[b][1]).astype(np.float32)[0,:,0]
    c=np.load(cby[b]).astype(np.float32)[0].reshape(v.shape[0],30,52)
    fi=1; p0,p1=lat2pix(b+fi,n)
    d=D[p0:p1].max(axis=0)
    cm=up(c[fi]>0.5,H,W); hi=d>32
    rows.append((b,s[p0].astype(np.uint8),d,cm,hi))

sc=300/W; tw,th=int(W*sc),int(H*sc); pad,top=8,24
cv=Image.new("RGB",(tw*3+pad*4,(th+top)*len(rows)+pad),(18,18,20)); dr=ImageDraw.Draw(cv)
for i,(b,base,d,cm,hi) in enumerate(rows):
    y=pad+i*(th+top)
    amp=np.clip(d/64,0,1); heat=np.zeros((H,W,3),np.uint8); heat[...,0]=(amp*255).astype(np.uint8)
    p_amp=(base*0.45+heat*0.55).astype(np.uint8)
    cls=np.zeros((H,W,3),np.uint8)
    cls[hi & cm]=(60,220,60); cls[hi & ~cm]=(70,130,255); cls[~hi & cm]=(230,60,60)
    p_cls=(base*0.4+cls*0.6).astype(np.uint8)
    frac_out = d[hi & ~cm].sum()/max(d[hi].sum(),1e-6)*100
    for j,(img,t) in enumerate(zip([base,p_amp,p_cls],
        [f"blk{b} source", "改动幅度 |Δ| (红越亮越大)",
         f"高幅>32: 绿=mask内 蓝=mask外({frac_out:.0f}%) 红=mask内无改动"])):
        x=pad+j*(tw+pad); dr.text((x,y),t,fill=(235,235,235))
        cv.paste(Image.fromarray(img).resize((tw,th)),(x,y+top))
cv.save(out); print("saved",out)
