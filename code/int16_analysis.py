import numpy as np
from mx_experiment_marmousi import Harness, quantize_dequantize, rel_l2

H=Harness(dtype=np.float32)
rec_ref,_,_=H.run(quant=None)
rng=np.random.default_rng(0)
_,_,hist=H.run(quant=None, store_wavefield=True)
peak_global=float(np.abs(np.asarray(hist)).max())

def store_int16_blocked(x, block_size, peak=None):
    """int16 fixed-point, but with one scale per block instead of one for the whole grid.
    block_size = whole grid reproduces the global int16 scheme; smaller blocks give each region
    its own scale, which is exactly what microscaling does."""
    shp=x.shape; xf=np.asarray(x).reshape(-1).astype(np.float64); n=xf.size
    if block_size>=n:                      # global: one scale (peak of the whole run)
        scale=(peak if peak else np.abs(xf).max())/32767.0 + 1e-30
        q=np.clip(np.round(xf/scale),-32768,32767); return (q*scale).reshape(shp).astype(x.dtype)
    pad=(-n)%block_size
    if pad: xf=np.concatenate([xf,np.zeros(pad)])
    blk=xf.reshape(-1,block_size)
    amax=np.max(np.abs(blk),axis=1,keepdims=True); nz=amax[:,0]>0
    scale=np.ones_like(amax); scale[nz,0]=amax[nz,0]/32767.0
    q=np.clip(np.round(blk/scale),-32768,32767)
    return (q*scale).reshape(-1)[:n].reshape(shp).astype(x.dtype)

print("scheme                                   error")
# global int16 (current) — scale from run peak
e=rel_l2(H.run(quant=lambda a: store_int16_blocked(a, 10**9, peak=peak_global))[0], rec_ref)
print(f"int16, ONE global scale (current)        {e:.3e}")
# int16 with per-block scaling, shrinking blocks
for bs in [4096,1024,256,64,32]:
    e=rel_l2(H.run(quant=lambda a,bs=bs: store_int16_blocked(a,bs))[0], rec_ref)
    print(f"int16, per-block scale, block {bs:5d}      {e:.3e}")
# references
e_fp16=rel_l2(H.run(quant=lambda a: a.astype(np.float16).astype(np.float32))[0], rec_ref)
e_mx=rel_l2(H.run(quant=lambda a: quantize_dequantize(a,32,12,"nearest",rng))[0], rec_ref)
print(f"FP16 (reference point)                   {e_fp16:.3e}")
print(f"MX block-32, 12 mantissa (reference)     {e_mx:.3e}")
