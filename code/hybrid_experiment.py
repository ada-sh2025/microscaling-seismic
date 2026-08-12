#!/usr/bin/env python3
"""
Hybrid experiment: FP16 arithmetic combined with MX storage.

The question:
FP16 and MX act on different resources. FP16 narrows the arithmetic, cutting compute and
memory but limited to about 1e-2 accuracy and needing a scale factor. MX narrows only the
stored field, cutting memory alone but reaching much lower error because the arithmetic stays
exact. The natural thing to ask is whether they combine: run the arithmetic in FP16, to save
compute, and store the wavefield in MX below 16 bits, to save more memory than FP16 alone.

The risk is that two lossy layers in the same feedback loop compound. FP16 rounding enters
every step, MX rounding enters every step, and the wavefield is fed back each time, so the
errors could reinforce and either blow up or land well above the sum of the two on their own.
This file measures whether that happens.

Setup:
The same self-contained numpy acoustic stepper as the reproduction, on the same cropped
Marmousi velocities. Four cases:

  FP32 arithmetic
      Reference.

  FP16 arithmetic, scaled
      Pure FP16 (Fabien-Ouellet). Compute in 16 bits, no extra storage compression.

  FP32 arithmetic, MX storage
      Pure MX. Exact arithmetic, wavefield stored narrow.

  FP16 arithmetic, scaled, plus MX storage
      The hybrid. Compute in 16 bits, and the stored wavefield further narrowed to MX.

For the hybrid and the pure MX runs the storage bit width is swept, so the two can be read
against each other at equal storage cost, and against the pure FP16 line.

Run with: python3 hybrid_experiment.py
Requires fabien_reproduction.py in the same folder (the stepper is reused unchanged).
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(over="ignore", invalid="ignore")

# Reuse the verified FP16-capable stepper and the MX routine, unchanged.
from fabien_reproduction import run, mx_store, load_velocity, mx_bits

OUTDIR = "hybrid_results"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    vp = load_velocity()
    print(f"velocity patch {vp.shape}, vp {vp.min():.2f} to {vp.max():.2f} km/s\n")

    ref, _ = run(vp, np.float32)

    def err(rec):
        return np.linalg.norm(rec - ref) / np.linalg.norm(ref)

    # Pure FP16 (Fabien-Ouellet), the reference line for "compute saving only".
    amax = float(np.abs(ref).max())
    scale = 2.0 ** np.round(np.log2(1.0 / max(amax, 1e-30)))
    rec, blew = run(vp, np.float16, scale=scale)
    fp16_err = None if blew is not None else err(rec)
    print(f"pure FP16 + scaling:            "
          f"{'diverged' if blew is not None else f'{fp16_err:.3e}'}   (compute + memory)")
    print()

    BITS = [6, 8, 10, 12]
    pure_mx, hybrid = [], []

    print(f"{'mantissa':>8s} {'storage bits':>12s} {'pure MX (FP32)':>16s} {'hybrid (FP16+MX)':>18s}")
    for mb in BITS:
        b = mx_bits(32, mb)

        # Pure MX: exact arithmetic, MX storage.
        rec, blew = run(vp, np.float32, store_fn=lambda a, mb=mb: mx_store(a, 32, mb, rng))
        e_mx = None if blew is not None else err(rec)
        pure_mx.append((b, e_mx))

        # Hybrid: FP16 arithmetic (scaled) plus MX storage on top.
        rec, blew = run(vp, np.float16, scale=scale,
                        store_fn=lambda a, mb=mb: mx_store(a, 32, mb, rng))
        e_hy = None if blew is not None else err(rec)
        hybrid.append((b, e_hy))

        s_mx = "diverged" if e_mx is None else f"{e_mx:.3e}"
        s_hy = "diverged" if e_hy is None else f"{e_hy:.3e}"
        print(f"{mb:8d} {b:12.2f} {s_mx:>16s} {s_hy:>18s}")

    # Save a small table.
    with open(os.path.join(OUTDIR, "hybrid.csv"), "w") as f:
        f.write("storage_bits,pure_mx_error,hybrid_error,pure_fp16_error\n")
        for (b, emx), (_, ehy) in zip(pure_mx, hybrid):
            f.write(f"{b:.2f},{'' if emx is None else f'{emx:.6e}'},"
                    f"{'' if ehy is None else f'{ehy:.6e}'},{fp16_err:.6e}\n")

    # Figure: error against storage bits, three curves.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bx = [b for b, e in pure_mx if e is not None]
    ax.plot([b for b, e in pure_mx if e is not None], [e for b, e in pure_mx if e is not None],
            marker="o", lw=1.9, color="tab:blue", label="pure MX (FP32 arithmetic)")
    ax.plot([b for b, e in hybrid if e is not None], [e for b, e in hybrid if e is not None],
            marker="s", lw=1.9, color="tab:purple", label="hybrid (FP16 arithmetic + MX storage)")
    if fp16_err is not None:
        ax.axhline(fp16_err, color="tab:red", ls="--", lw=1.3)
        ax.text(max(bx), fp16_err * 1.18, "pure FP16 (compute + memory)",
                ha="right", fontsize=9, color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("storage bits per wavefield value")
    ax.set_ylabel("relative error of the receiver trace")
    ax.set_title("Do FP16 arithmetic and MX storage combine, or compound?")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_hybrid.png"), dpi=130)
    plt.close(fig)

    # Plain-language verdict from the numbers.
    print()
    ok = [(b, ehy, emx) for (b, ehy), (_, emx) in zip(hybrid, pure_mx)
          if ehy is not None and emx is not None]
    if ok and fp16_err is not None:
        # at the storage width closest to FP16-limited accuracy
        b, ehy, emx = min(ok, key=lambda t: abs(t[1] - fp16_err))
        floor = max(fp16_err, emx)
        ratio = ehy / floor
        print(f"At {b:.2f} storage bits: hybrid {ehy:.3e}, pure MX {emx:.3e}, pure FP16 {fp16_err:.3e}")
        print(f"Hybrid error is {ratio:.2f}x the larger of the two component errors "
              f"({'compounds badly' if ratio > 2 else 'does not compound badly'}).")
    print(f"\ntable and figure written to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
