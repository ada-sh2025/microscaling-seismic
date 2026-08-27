#!/usr/bin/env python3
"""
Elastic kernel comparison: microscaling against Fabien-Ouellet style scaling.

Why this file exists, and why the elastic kernel:
Fabien-Ouellet (2020) developed FP16 seismic modelling on the isotropic ELASTIC wave
equation, not the acoustic one. The elastic problem carries several wavefield quantities,
a velocity vector and a stress tensor, whose magnitudes differ by roughly an order of
magnitude. That spread is the reason his scheme needs scaling at all: he introduces
separate scale factors to lift each quantity into the healthy part of the FP16 range.

Comparing MX against his scheme on the acoustic problem was not a like for like test,
because acoustic has a single wavefield quantity and a narrow dynamic range, so scaling
has almost nothing to do there. This file puts both schemes on the elastic kernel, which
is the setting his method was designed for, and asks the direct question: does MX's per
block exponent, which adapts to each quantity automatically, match or beat a hand chosen
per quantity global scale.

The schemes, all run on the same elastic Marmousi based model with the same metric:

  FP32
      Reference. The stress receiver record from a full precision run.

  FP16, no scaling
      Every wavefield component cast to FP16 with no scaling. Expected to hurt the small
      magnitude velocity components, since they sit lower in the FP16 range.

  FP16, per field scaling (Fabien-Ouellet style)
      Each wavefield component is lifted by its own power of two scale, chosen from that
      component's own maximum, before being cast to FP16, then divided back out. This is
      the storage analogue of his per quantity scaling: it aligns each quantity with the
      FP16 range separately.

  microscaling
      Each block of each component shares one exponent, taken from that block's own
      maximum. The scale therefore adapts to the local amplitude of each quantity by
      construction, with no scale factor chosen by hand.

The point of the comparison is not only which wins, but whether MX removes the need for
the hand chosen per quantity factors that the scaling scheme depends on.

Run with: python3 elastic_comparison.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.seismic import AcquisitionGeometry, Model
from devito import (Eq, Operator, TensorTimeFunction, VectorTimeFunction,
                    diag, div, grad, solve)

OUTDIR = "elastic_comparison_results"

# Model settings. A layered elastic model is enough to make the point; the physics that
# matters here is that velocity and stress differ in magnitude, not the geological detail.
SHAPE = (140, 140)
SPACING = (10.0, 10.0)
NBL = 20
SPACE_ORDER = 4
T0, TN = 0.0, 700.0
F0 = 0.015
NREC = 120
SRC_DEPTH = 20.0
REC_DEPTH = 20.0


def quantize_dequantize(x, block_size=32, mantissa_bits=8, rounding="nearest", rng=None):
    """Microscaling: one shared exponent per block, mantissa_bits kept per value, unpack back."""
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
    scale[nz, 0] = 2.0 ** np.floor(np.log2(amax[nz, 0]))
    step = 2.0 ** (-mantissa_bits)
    y = np.zeros_like(blk); y[nz] = blk[nz] / scale[nz]
    q = np.round(y / step) if rounding == "nearest" else np.floor(y / step + rng.random(blk.shape))
    out = np.zeros_like(blk); out[nz] = q[nz] * step * scale[nz]
    return out.reshape(-1)[:n].reshape(shp).astype(x.dtype)


def fp16_naive(x):
    """Store in FP16 with no scaling."""
    return x.astype(np.float16).astype(np.float32).astype(x.dtype)


def fp16_scaled(x, scale):
    """Store in FP16 after lifting by a per field power of two scale, then divide back out."""
    return (np.float32(x * scale).astype(np.float16).astype(np.float32) / scale).astype(x.dtype)


def mx_bits(block_size, mantissa_bits):
    return (mantissa_bits + 2) + 8.0 / block_size


class ElasticHarness:
    """Elastic forward solve rebuilt so it can be stepped one time step at a time.

    The construction mirrors Devito's ForwardOperator for the elastic case. The wavefield
    is a velocity vector v and a stress tensor tau; between steps their live data buffers
    can be pushed through a storage format. Manual stepping was checked to reproduce the
    packaged solver exactly, so any error measured later is caused by the storage format
    alone.
    """

    def __init__(self):
        vp = np.full(SHAPE, 2.5, np.float32); vp[SHAPE[1] // 2:, :] = 3.2
        vs = vp / 1.73
        rho = np.full(SHAPE, 2.0, np.float32)
        self.model = Model(origin=(0, 0), shape=SHAPE, spacing=SPACING, space_order=SPACE_ORDER,
                           nbl=NBL, vp=vp, vs=vs, b=1 / rho, dtype=np.float32)
        Lx = SPACING[0] * (SHAPE[0] - 1)
        rc = np.zeros((NREC, 2)); rc[:, 0] = np.linspace(0, Lx, NREC); rc[:, 1] = REC_DEPTH
        self.geometry = AcquisitionGeometry(self.model, rc, np.array([Lx / 2, SRC_DEPTH]),
                                            T0, TN, f0=F0, src_type="Ricker")
        self.dt = self.model.critical_dt
        self.nt = self.geometry.nt

        m = self.model
        v = VectorTimeFunction(name="v", grid=m.grid, space_order=SPACE_ORDER, time_order=1)
        tau = TensorTimeFunction(name="tau", grid=m.grid, space_order=SPACE_ORDER, time_order=1)
        self.v, self.tau = v, tau
        lam, mu, b = m.lam, m.mu, m.b
        eq_v = v.dt - b * div(tau)
        e = grad(v.forward) + grad(v.forward).transpose(inner=False)
        eq_tau = tau.dt - lam * diag(div(v.forward)) - mu * e
        u_v = Eq(v.forward, m.damp * solve(eq_v, v.forward))
        u_t = Eq(tau.forward, m.damp * solve(eq_tau, tau.forward))
        src = self.geometry.src
        self.rec1 = self.geometry.new_rec(name="rec1")
        rec2 = self.geometry.new_rec(name="rec2")
        s = m.grid.time_dim.spacing
        srcrec = (src.inject(tau.forward.diagonal(), expr=src * s)
                  + self.rec1.interpolate(expr=tau[-1, -1])
                  + rec2.interpolate(expr=div(v)))
        self.op = Operator([u_v, u_t] + srcrec, subs=m.spacing_map, name="ElasticStep")

        # The distinct wavefield component arrays that have to be stored each step.
        self.comps = self._unique_components()

    def _unique_components(self):
        fields = [self.v[0], self.v[1], self.tau[0, 0], self.tau[0, 1], self.tau[1, 0], self.tau[1, 1]]
        seen, out = set(), []
        for f in fields:
            if id(f) not in seen:
                seen.add(id(f)); out.append(f)
        return out

    def field_scales(self):
        """One power of two scale per component, from a quick reference run. Used by the
        Fabien-Ouellet style scheme to lift each quantity into the FP16 range separately."""
        self.run(store_field=None)
        scales = []
        for f in self.comps:
            amax = float(np.abs(np.array(f.data)).max())
            scales.append(2.0 ** np.round(np.log2(1.0 / max(amax, 1e-30))))
        return scales

    def run(self, store_field=None):
        """Step the elastic solve. store_field, if given, is a function applied to each
        component's live buffer after every step. Returns the stress receiver record."""
        for f in self.comps:
            f.data[:] = 0.0
        for i in range(self.nt - 1):
            self.op.apply(time_m=i, time_M=i, dt=self.dt)
            if store_field is not None:
                buf = (i + 1) % 2
                for j, f in enumerate(self.comps):
                    f.data[buf] = store_field(f.data[buf], j)
        return self.rec1.data.copy()


def rel_l2(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


TS_POWER = 1.0   # time-scaling exponent for the accuracy metric (linear gain)


def rel_l2_timescaled(a, b, power=TS_POWER):
    """Relative L2 error after a time gain down the record's first axis, so the weak late
    arrivals weigh as much as the strong early ones instead of being swamped by them."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    t = np.arange(a.shape[0], dtype=np.float64)
    g = (((t + 1.0) / np.mean(t + 1.0)) ** power)[:, None]
    return np.linalg.norm((a - b) * g) / np.linalg.norm(b * g)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    H = ElasticHarness()
    print(f"elastic model grid={SHAPE}  nt={H.nt}  dt={H.dt:.4f} ms")
    print(f"wavefield components stored each step: {len(H.comps)} "
          f"(2 velocity, {len(H.comps) - 2} stress)\n")

    rec_ref = H.run(store_field=None)

    # Report the magnitude of each component, to show the spread that makes scaling relevant.
    print("component magnitudes (this spread is why the elastic case needs scaling):")
    for f in H.comps:
        print(f"  {f.name:10s} absmax {float(np.abs(np.array(f.data)).max()):.3e}")
    scales = H.field_scales()
    print()

    results = []

    def record(name, bits, err):
        snr = np.inf if err == 0 else -20.0 * np.log10(err)
        results.append(dict(name=name, bits=bits, err=err, snr=snr))
        shown = "  inf" if not np.isfinite(snr) else f"{snr:5.1f}"
        print(f"  {name:32s} {bits:6.2f} bits   {shown} dB   error {err:.3e}")

    print("scheme                             bits    SNR(dB)     error")
    record("FP32 (reference)", 32.0, 0.0)

    err = rel_l2_timescaled(H.run(store_field=lambda a, j: fp16_naive(a)), rec_ref)
    record("FP16, no scaling", 16.0, err)

    err = rel_l2_timescaled(H.run(store_field=lambda a, j: fp16_scaled(a, scales[j])), rec_ref)
    record("FP16, per field scaling (FO)", 16.0, err)

    for mb in [4, 6, 8, 10, 11, 12, 14]:
        err = rel_l2_timescaled(H.run(store_field=lambda a, j, mb=mb: quantize_dequantize(a, 32, mb, "nearest", rng)),
                     rec_ref)
        record(f"MX, {mb} mantissa bits", mx_bits(32, mb), err)

    # Cost against accuracy on the elastic kernel.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ours = [r for r in results if r["name"].startswith("MX")]
    ax.plot([r["bits"] for r in ours], [r["err"] for r in ours],
            marker="o", lw=2.0, color="tab:blue", zorder=3, label="microscaling (this work)")
    style = {"FP16, no scaling": ("^", "tab:green"),
             "FP16, per field scaling (FO)": ("s", "tab:red")}
    for r in results:
        if r["name"] in style:
            m, c = style[r["name"]]
            ax.scatter(r["bits"], r["err"], marker=m, s=150, c=c, edgecolors="k",
                       zorder=4, label=r["name"])
    ax.set_yscale("log")
    ax.set_xlabel("bits stored per wavefield value")
    ax.set_ylabel("time-scaled relative error of the stress receiver record")
    # Signal-to-noise ratio in decibels on the right, the field's usual unit and the one the
    # target is set in. Same information as the left axis: SNR(dB) = -20 log10(relative error).
    secax = ax.secondary_yaxis("right", functions=(lambda e: -20.0 * np.log10(np.clip(e, 1e-300, None)),
                                                    lambda d: 10.0 ** (-d / 20.0)))
    secax.set_ylabel("time-scaled signal-to-noise ratio (dB)")
    # Point out the 11 mantissa bit setting, since it is the aggressive-but-still-good case.
    p11 = next((r for r in ours if "11 mantissa" in r["name"]), None)
    if p11:
        ax.annotate("11 mantissa bits", (p11["bits"], p11["err"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9, color="tab:blue")
    ax.set_title("Elastic kernel: MX against FP16 scaling")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_elastic_comparison.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUTDIR, "elastic_comparison.csv"), "w") as f:
        f.write("scheme,bits_per_value,snr_db,relative_error\n")
        for r in results:
            snr = "" if not np.isfinite(r["snr"]) else f"{r['snr']:.1f}"
            f.write(f"\"{r['name']}\",{r['bits']:.2f},{snr},{r['err']:.6e}\n")

    print(f"\nfigure and table written to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
