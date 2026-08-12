#!/usr/bin/env python3
"""
Full reproduction of Fabien-Ouellet (2020), and an equal-conditions comparison with MX.

Why this file is separate from the storage comparison:
Everything earlier in the project compressed the wavefield in storage while the arithmetic
stayed at FP32. That is not what Fabien-Ouellet does. His method runs the finite difference
arithmetic itself in FP16, and scales the equation so that the 16-bit computation stays in
range. Reproducing him therefore requires a solver that genuinely computes in FP16, including
the real overflow and underflow of the format.

Devito cannot do this: its code generator rejects a float16 grid ("Converting float16 to a
ctypes type"). So the reproduction is a small, self-contained 2nd-order acoustic stepper
written directly in numpy, where every array is the chosen dtype and every operation rounds
to it. numpy float16 arithmetic overflows to inf at 65504 and loses precision below its
smallest normal near 6.1e-5, so the format's real behaviour is present, not emulated.

What is compared, all on the same cropped Marmousi velocities, same source, same receiver:

  FP32 arithmetic
      Reference.

  FP16 arithmetic, no scaling
      The naive attempt. Exposes whatever overflow or underflow the format suffers.

  FP16 arithmetic, with scaling (Fabien-Ouellet)
      His method. A power-of-two factor multiplies the field into the healthy part of the
      FP16 range before the 16-bit arithmetic, and is divided back out of the recorded trace.

  MX storage, FP32 arithmetic
      Our scheme in its natural role. The stencil is computed in FP32; only the stored
      wavefield is narrowed to MX between steps.

The point the comparison makes:
FP16 and MX are not the same kind of object. FP16 is a *compute* precision: it narrows the
arithmetic, which cuts compute cost and memory together but risks stability, and is what
needs scaling. MX is a *storage* precision: the arithmetic stays FP32, so it never has the
range problem, but it only cuts memory, not compute. A fair comparison has to say which axis
it is on. This file places both on one accuracy axis and states the difference in kind
explicitly, rather than declaring one the winner as if they were interchangeable.

Run with: python3 fabien_reproduction.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(over="ignore", invalid="ignore")   # FP16 overflow is expected and handled explicitly

OUTDIR = "fabien_reproduction_results"
DATA_FILE = "data/Simple2D/vp_marmousi_bi"


def load_velocity():
    """A small cropped Marmousi velocity patch, coarsened so the pure-numpy stepper is quick."""
    import urllib.request
    if not os.path.exists(DATA_FILE):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/devitocodes/data/master/Simple2D/vp_marmousi_bi", DATA_FILE)
    v = np.fromfile(DATA_FILE, dtype="float32").reshape(1601, 401)[301:-300, :]
    return v[400:640:2, 40:240:2].astype(np.float32)   # ~120 x 100, real geology


def laplacian(u):
    """Five-point second-order Laplacian, in the dtype of u."""
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
                       - np.array(4, u.dtype) * u[1:-1, 1:-1])
    return lap


def mx_store(x, block_size=32, mantissa_bits=10, rng=None):
    """Microscaling pack/unpack, applied to the stored wavefield (arithmetic stays FP32)."""
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
    q = np.round(y / step)
    out = np.zeros_like(blk); out[nz] = q[nz] * step * scale[nz]
    return out.reshape(-1)[:n].reshape(shp).astype(x.dtype)


def run(vp, arith_dtype, scale=1.0, store_fn=None, nt=900):
    """One acoustic simulation.

    arith_dtype sets the precision of the arithmetic (this is what FP16 changes).
    scale is Fabien-Ouellet's factor applied to the field for the FP16 arithmetic.
    store_fn, if given, transforms the stored wavefield each step (this is what MX uses,
    with the arithmetic left at FP32).
    Returns the receiver trace in physical units, and the step it diverged at, or None.
    """
    nx, nz = vp.shape
    dx = 20.0
    dt = 0.4 * dx / float(vp.max())
    c2 = (vp.astype(np.float64) * dt / dx) ** 2
    c2 = (c2 * scale if False else c2)                     # velocity map is not scaled, the field is
    c2a = c2.astype(arith_dtype)
    two = np.array(2, arith_dtype)

    up = np.zeros((nx, nz), arith_dtype)
    uc = np.zeros((nx, nz), arith_dtype)

    t = np.arange(nt) * dt
    f0 = 0.020e3; t0 = 1.0 / f0
    ric = (1 - 2 * (np.pi * f0 * (t - t0)) ** 2) * np.exp(-(np.pi * f0 * (t - t0)) ** 2)
    ric = (ric * scale).astype(arith_dtype)
    sx, sz = nx // 2, 4
    rx, rz = nx // 2, 3
    rec = np.zeros(nt)

    for it in range(nt):
        lap = laplacian(uc)
        un = (two * uc - up + c2a * lap).astype(arith_dtype)
        un[sx, sz] = (un[sx, sz] + ric[it]).astype(arith_dtype)
        if store_fn is not None:
            un = store_fn(un).astype(arith_dtype)          # narrow the stored wavefield
        up, uc = uc, un
        rec[it] = np.float64(uc[rx, rz]) / scale           # record, undoing the scale
        if not np.all(np.isfinite(uc)):
            return rec, it
    return rec, None


def mx_bits(block_size, mantissa_bits):
    return (mantissa_bits + 2) + 8.0 / block_size


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    vp = load_velocity()
    print(f"velocity patch {vp.shape}, vp range {vp.min():.2f} to {vp.max():.2f} km/s\n")

    ref, _ = run(vp, np.float32)
    print(f"FP32 reference trace norm {np.linalg.norm(ref):.4e}\n")

    def err(rec):
        return np.linalg.norm(rec - ref) / np.linalg.norm(ref)

    rows = []

    # Fabien-Ouellet: FP16 arithmetic, without and with scaling.
    rec, blew = run(vp, np.float16, scale=1.0)
    e = "diverged" if blew is not None else f"{err(rec):.3e}"
    rows.append(("FP16 arithmetic, no scaling", "FP16", "16", e))
    print(f"FP16 arithmetic, no scaling            {e}" + (f" (step {blew})" if blew is not None else ""))

    # choose the scale that lifts the field into the FP16 sweet spot
    amax = float(np.abs(ref).max())
    scale = 2.0 ** np.round(np.log2(1.0 / max(amax, 1e-30)))
    rec, blew = run(vp, np.float16, scale=scale)
    e_fo = None if blew is not None else err(rec)
    e = "diverged" if blew is not None else f"{e_fo:.3e}"
    rows.append((f"FP16 arithmetic, scaling 2^{int(np.log2(scale))} (Fabien-Ouellet)", "FP16", "16", e))
    print(f"FP16 arithmetic, scaling 2^{int(np.log2(scale))} (FO)      {e}\n")

    # MX as a storage format, FP32 arithmetic.
    mx_pts = []
    for mb in [6, 8, 10, 12]:
        rec, _ = run(vp, np.float32, store_fn=lambda a, mb=mb: mx_store(a, 32, mb, rng))
        e = err(rec); b = mx_bits(32, mb)
        rows.append((f"MX storage, {mb} mantissa bits", "FP32", f"{b:.2f}", f"{e:.3e}"))
        mx_pts.append((b, e))
        print(f"MX storage {mb:2d} mantissa bits ({b:.2f} bits)   {e:.3e}")

    # Table
    with open(os.path.join(OUTDIR, "reproduction.csv"), "w") as f:
        f.write("scheme,arithmetic,storage_bits,relative_error\n")
        for r in rows:
            f.write(f"\"{r[0]}\",{r[1]},{r[2]},{r[3]}\n")

    # Figure: accuracy of each scheme, with compute precision shown by colour.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(len(rows))
    for i, r in enumerate(rows):
        val = r[3]
        if val == "diverged":
            ax.scatter(i, 1.0, marker="x", s=140, c="k", zorder=4)
            ax.annotate("diverged", (i, 1.0), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
        else:
            col = "tab:red" if r[1] == "FP16" else "tab:blue"
            ax.scatter(i, float(val), s=110, c=col, edgecolors="k", zorder=4)
    if e_fo is not None:
        ax.axhline(e_fo, color="tab:red", ls="--", lw=1.0, alpha=0.7)
        ax.text(len(rows) - 1, e_fo * 1.25, "Fabien-Ouellet FP16", ha="right",
                fontsize=8, color="tab:red")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0].replace(" arithmetic", "").replace(" storage", "").replace(", ", "\n")
                        for r in rows], fontsize=7, rotation=0)
    ax.set_ylabel("relative error of the receiver trace")
    ax.set_title("FP16 arithmetic (red) against MX storage (blue), same numpy acoustic stepper")
    ax.grid(alpha=0.3, axis="y", which="both")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",
                              markeredgecolor="k", label="FP16: 16-bit arithmetic (cuts compute + memory)"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
                              markeredgecolor="k", label="MX: FP32 arithmetic, narrow storage (cuts memory only)")],
              fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_fabien_reproduction.png"), dpi=130)
    plt.close(fig)

    print(f"\ntable and figure written to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
