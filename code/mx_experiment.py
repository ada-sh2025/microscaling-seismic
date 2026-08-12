#!/usr/bin/env python3
"""
Weeks 3-4 skeleton — microscaling (MX) accuracy experiment, built on m1_baseline.py.

Idea
----
Re-run the SAME acoustic forward model as M1, but store the wavefield in a
microscaling (MX) block-float format between time steps: every `quantize_every`
steps the freshly computed wavefield slice is packed to MX and unpacked back to
float32 (this emulates "the wavefield lives in MX in memory; the stencil still
computes in float32"). We then measure how far the MX run drifts from a
full-precision reference, and how that error GROWS over the long time-stepping.

Experimental control
--------------------
The reference here is produced by the *same* stepping driver with quantisation
switched OFF (`quant=None`). Comparing MX-on vs MX-off through one code path
isolates the error caused purely by quantisation. (The M1 packaged FP32/FP64
files remain a useful external cross-check.)

What is real vs. what is a stub
-------------------------------
  * quantize_dequantize(...)  -> WORKING, but a *simplified* MX emulation
    (shared exponent per block + N mantissa bits + nearest/stochastic rounding).
    TODO: swap in exact OCP MX formats (MXFP e2m1/e3m2/e2m3, MXINT8, scale=E8M0).
  * run_forward_mx(...)       -> WORKING in-loop injection + error-growth capture.
  * metrics + sweep           -> WORKING.
  * bandwidth / bytes / runtime savings are NOT measured here — this file is the
    ACCURACY half only. Cost comes from a separate bandwidth model + micro-kernel.

Run:  python3 mx_experiment.py     (needs m1_baseline.py in the same folder)
"""

import os
import numpy as np

from devito import TimeFunction, Eq, Operator, solve
from examples.seismic import Model, AcquisitionGeometry, Receiver

# Reuse the exact M1 configuration + true model so the two stages stay in sync.
from m1_baseline import (EXTENT, SHAPE, ORIGIN, NBL, SPACE_ORDER,
                         T0, TN, F0, NREC, SRC_DEPTH, REC_DEPTH,
                         build_true_velocity)

OUTDIR = "mx_reference"
SHOT_X = EXTENT[0] / 2.0          # single representative shot for the accuracy study


# ---------------------------------------------------------------------------
# 1) MX core  (WORKING — simplified emulation; swap in exact OCP MX later)
# ---------------------------------------------------------------------------
def quantize_dequantize(x, block_size=32, mantissa_bits=3, rounding="nearest", rng=None):
    """Pack `x` to microscaling block-float and unpack back to x.dtype.

    Each contiguous block of `block_size` values shares one exponent (scale);
    every value then keeps `mantissa_bits` mantissa bits relative to that scale.

    rounding: "nearest" or "stochastic" (stochastic reduces the systematic bias
    that otherwise builds up over thousands of steps).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    shp = x.shape
    xf = x.reshape(-1).astype(np.float64)
    n = xf.size
    pad = (-n) % block_size
    if pad:
        xf = np.concatenate([xf, np.zeros(pad)])
    blk = xf.reshape(-1, block_size)

    amax = np.max(np.abs(blk), axis=1, keepdims=True)
    nz = amax[:, 0] > 0
    scale = np.zeros_like(amax)
    scale[nz, 0] = 2.0 ** np.floor(np.log2(amax[nz, 0]))     # shared block exponent
    step = 2.0 ** (-mantissa_bits)                            # mantissa resolution

    y = np.zeros_like(blk)
    y[nz] = blk[nz] / scale[nz]                               # ~ [-2, 2)
    t = y / step
    if rounding == "nearest":
        q = np.round(t)
    elif rounding == "stochastic":
        q = np.floor(t + rng.random(t.shape))
    else:
        raise ValueError(f"unknown rounding: {rounding}")

    out = np.zeros_like(blk)
    out[nz] = q[nz] * step * scale[nz]
    out = out.reshape(-1)[:n].reshape(shp)
    return out.astype(x.dtype)


# ---------------------------------------------------------------------------
# 2) Forward harness  (built once, reused for every sweep config)
# ---------------------------------------------------------------------------
class Harness:
    """A manually-stepped acoustic forward solve we can interrupt each step."""

    def __init__(self, dtype=np.float32):
        spacing = (EXTENT[0] / (SHAPE[0] - 1), EXTENT[1] / (SHAPE[1] - 1))
        vp = build_true_velocity(SHAPE, EXTENT, ORIGIN).astype(dtype)
        self.model = Model(vp=vp, origin=ORIGIN, shape=SHAPE, spacing=spacing,
                           nbl=NBL, space_order=SPACE_ORDER, bcs="damp", dtype=dtype)

        rec_c = np.empty((NREC, 2)); rec_c[:, 0] = np.linspace(0, EXTENT[0], NREC); rec_c[:, 1] = REC_DEPTH
        self.geometry = AcquisitionGeometry(self.model, rec_c,
                                            np.array([SHOT_X, SRC_DEPTH], dtype=dtype),
                                            T0, TN, f0=F0, src_type="Ricker")
        self.dt = self.model.critical_dt
        self.nt = self.geometry.time_axis.num

        self.u = TimeFunction(name="u", grid=self.model.grid,
                              time_order=2, space_order=SPACE_ORDER)
        pde = self.model.m * self.u.dt2 - self.u.laplace + self.model.damp * self.u.dt
        stencil = Eq(self.u.forward, solve(pde, self.u.forward))
        src = self.geometry.src
        self.rec = Receiver(name="rec", grid=self.model.grid,
                            time_range=self.geometry.time_axis,
                            coordinates=self.geometry.rec_positions)
        self.op = Operator([stencil]
                           + src.inject(field=self.u.forward, expr=src * self.dt ** 2 / self.model.m)
                           + self.rec.interpolate(expr=self.u))

    def run(self, quant=None, quantize_every=1, store_wavefield=True):
        """Step the solve; if `quant` is given, pack/unpack the live wavefield.

        Returns (shot_record, wavefield_history[nt,nx,nz] or None).
        """
        self.u.data[:] = 0.0
        hist = np.zeros((self.nt, SHAPE[0], SHAPE[1]), np.float64) if store_wavefield else None
        for i in range(self.nt - 1):
            self.op.apply(time_m=i, time_M=i, dt=self.dt)
            buf = (i + 1) % 3                             # freshly written wavefield slice
            if quant is not None and (i + 1) % quantize_every == 0:
                # Store the *physical* wavefield in MX (skip the absorbing layer, whose
                # large damped values otherwise destabilise per-step re-quantisation).
                self.u.data[buf][NBL:-NBL, NBL:-NBL] = quant(self.u.data[buf][NBL:-NBL, NBL:-NBL])
            if store_wavefield:
                hist[i + 1] = self.u.data[buf][NBL:-NBL, NBL:-NBL]
        return self.rec.data.copy(), hist


# ---------------------------------------------------------------------------
# 3) Metrics
# ---------------------------------------------------------------------------
def rel_l2(a, b):
    b = np.asarray(b, np.float64); a = np.asarray(a, np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def error_growth(hist_mx, hist_ref):
    """Per-timestep relative wavefield error -> shows accumulation over time."""
    num = np.linalg.norm((hist_mx - hist_ref).reshape(hist_ref.shape[0], -1), axis=1)
    den = np.linalg.norm(hist_ref.reshape(hist_ref.shape[0], -1), axis=1)
    out = np.zeros_like(num); good = den > 0
    out[good] = num[good] / den[good]
    return out


# ---------------------------------------------------------------------------
# 4) Experiment driver
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    H = Harness(dtype=np.float32)
    print(f"grid={SHAPE}  nt={H.nt}  dt={H.dt:.4f} ms  (single shot at x={SHOT_X:.0f} m)")

    # (a) Full-precision reference through the SAME driver (quant off)
    rec_ref, hist_ref = H.run(quant=None)
    print(f"reference shot-record norm = {np.linalg.norm(rec_ref):.6e}")
    np.save(os.path.join(OUTDIR, "hist_ref.npy"), hist_ref)

    # (b) Sweep block size / mantissa bits / rounding
    BLOCKS   = [32]
    BITS     = [4, 6, 8, 10]           # 10 mantissa bits ~ FP16-like (stable); 4 bits ~ boundary of usable
    ROUNDING = ["nearest", "stochastic"]
    QEVERY   = 1                       # store wavefield in MX every step

    print("\n block bits  rounding     rec_relerr   final_wavefield_relerr")
    print(" " + "-" * 58)
    rows = []
    for bs in BLOCKS:
        for mb in BITS:
            for rd in ROUNDING:
                quant = (lambda a, bs=bs, mb=mb, rd=rd:
                         quantize_dequantize(a, bs, mb, rd, rng))
                rec_mx, hist_mx = H.run(quant=quant, quantize_every=QEVERY)
                e_rec = rel_l2(rec_mx, rec_ref)
                e_wf  = rel_l2(hist_mx[-1], hist_ref[-1])
                rows.append((bs, mb, rd, e_rec, e_wf))
                print(f" {bs:5d} {mb:4d}  {rd:11s}  {e_rec:.3e}    {e_wf:.3e}")

    # (c) Error growth over time: nearest vs stochastic at the same bit-width.
    #     (Whether stochastic's known benefit for the parabolic heat equation carries
    #      over to this hyperbolic wave equation is an OPEN question for the project.)
    mb = 6
    _, h_near = H.run(quant=lambda a: quantize_dequantize(a, 32, mb, "nearest"))
    _, h_stoc = H.run(quant=lambda a: quantize_dequantize(a, 32, mb, "stochastic", rng))
    g_near = error_growth(h_near, hist_ref)
    g_stoc = error_growth(h_stoc, hist_ref)
    np.save(os.path.join(OUTDIR, "growth_nearest.npy"),   g_near)
    np.save(os.path.join(OUTDIR, "growth_stochastic.npy"), g_stoc)
    idx = np.linspace(1, H.nt - 1, 6).astype(int)
    print(f"\nerror growth at {mb} mantissa bits (relative wavefield error vs time):")
    print("   step:    " + "  ".join(f"{i:6d}" for i in idx))
    print("   nearest: " + "  ".join(f"{g_near[i]:.4f}" for i in idx))
    print("   stochas: " + "  ".join(f"{g_stoc[i]:.4f}" for i in idx))

    print(f"\nSaved curves + reference wavefield to ./{OUTDIR}/")
    print("NOTE: this is the ACCURACY study only; bytes/bandwidth/runtime savings "
          "come from the separate bandwidth model + micro-kernel.")


if __name__ == "__main__":
    main()
