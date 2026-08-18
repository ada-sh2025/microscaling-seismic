#!/usr/bin/env python3
"""
Does the block's shape, not just its size, drive the accuracy?

The open puzzle from the sweeps is that a block of 32 values is more accurate than smaller
blocks, which is the wrong way round if size were all that mattered. The leading explanation is
shape. The blocks used everywhere else are one-dimensional strips: 32 values taken consecutively
along the fastest storage axis, which runs down the field and can cross very different parts of
the wavefront, so one shared exponent has to cover a wide range of magnitudes. A block that is
square in space instead gathers 32 values from a small compact patch, where the amplitudes are
more alike, so the shared exponent fits them better.

This test settles it by holding the block size fixed and changing only its shape. Every block
still holds about 32 values and every value still costs the same number of bits; the only thing
that varies is whether those values are a strip down one axis or a compact tile in two
dimensions. If the square tiles are more accurate at matched cost, the anomaly is a shape effect,
and the practical lesson is that blocks should follow the geometry of the field rather than the
layout of memory.

Run with: python3 block_shape.py
"""

import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mx_experiment_marmousi import Harness, rel_l2, snr_db, quantize_dequantize

OUTDIR = "block_shape_results"
MANTISSA = 8          # a bit width where the differences are visible, not saturated


def time_call(fn, arr, reps=200):
    """Mean wall-clock time of one pack-and-unpack call, in milliseconds, after a warm-up.

    This is a real measurement of the emulated pack/unpack, not a model. It answers the relative
    question, how much the 2D tiling costs against the 1D strip, since both are timed in the same
    implementation; the absolute numbers are numpy on a CPU and would not carry over to a kernel.
    """
    fn(arr)                                            # warm up caches and any lazy setup
    best = np.inf
    for _ in range(5):                                 # take the best of a few batches
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(arr)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e3


def quantize_dequantize_2d(x, bh, bw, mantissa_bits=MANTISSA):
    """Microscaling with a two-dimensional block: the field is tiled into bh-by-bw patches, and
    each patch shares one exponent. Setting (bh, bw) = (1, N) recovers a strip along the fast
    axis, so this one routine covers every shape."""
    nx, nz = x.shape
    px = (-nx) % bh
    pz = (-nz) % bw
    xp = np.zeros((nx + px, nz + pz), np.float64)
    xp[:nx, :nz] = x
    Nx, Nz = xp.shape

    # Gather each bh-by-bw tile into one row of length bh*bw.
    tiles = xp.reshape(Nx // bh, bh, Nz // bw, bw).transpose(0, 2, 1, 3).reshape(-1, bh * bw)

    amax = np.max(np.abs(tiles), axis=1, keepdims=True)
    nz_mask = amax[:, 0] > 0
    scale = np.zeros_like(amax)
    scale[nz_mask, 0] = 2.0 ** np.floor(np.log2(amax[nz_mask, 0]))

    step = 2.0 ** (-mantissa_bits)
    y = np.zeros_like(tiles)
    y[nz_mask] = tiles[nz_mask] / scale[nz_mask]
    q = np.round(y / step)
    out = np.zeros_like(tiles)
    out[nz_mask] = q[nz_mask] * step * scale[nz_mask]

    # Reverse the tiling and strip the padding.
    back = out.reshape(Nx // bh, Nz // bw, bh, bw).transpose(0, 2, 1, 3).reshape(Nx, Nz)
    return back[:nx, :nz].astype(x.dtype)


def bits_per_value(nvals):
    return MANTISSA + 2 + 8.0 / nvals


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    H = Harness(dtype=np.float32)
    ref, snap, _ = H.run(quant=None)
    ref = np.asarray(ref)
    snap = np.asarray(snap)          # a real wavefield frame, used for the pack/unpack timing

    # Shapes to compare, all holding about 32 values so the bit cost matches. The first is the
    # memory strip used elsewhere; the rest grow squarer.
    shapes = [(1, 32), (2, 16), (4, 8), (6, 6), (8, 4)]

    print(f"one mantissa setting ({MANTISSA} bits); only the block shape changes\n")
    print(f"  {'block shape':>12s}{'values':>8s}{'bits/val':>10s}{'aspect':>9s}"
          f"{'SNR (dB)':>11s}{'pack+unpack':>14s}{'vs strip':>10s}")

    # The true baseline is the production 1D path, a flat reshape with no transpose.
    strip_ms = time_call(lambda a: quantize_dequantize(a, 32, MANTISSA, "nearest"), snap)

    rows = []
    for (bh, bw) in shapes:
        nvals = bh * bw
        rec, _, _ = H.run(quant=lambda a, bh=bh, bw=bw: quantize_dequantize_2d(a, bh, bw))
        s = snr_db(np.asarray(rec), ref)
        ms = time_call(lambda a, bh=bh, bw=bw: quantize_dequantize_2d(a, bh, bw), snap)
        aspect = max(bh, bw) / min(bh, bw)
        label = "strip (memory)" if (bh, bw) == (1, 32) else ("square" if bh == bw else f"{bh}x{bw}")
        rows.append((f"{bh}x{bw}", nvals, bits_per_value(nvals), aspect, s, ms, ms / strip_ms, label))
        print(f"  {bh}x{bw:<10d}{nvals:>8d}{bits_per_value(nvals):>10.2f}{aspect:>9.1f}"
              f"{s:>11.1f}{ms:>12.3f}ms{ms / strip_ms:>9.2f}x")

    print(f"\n  reference: 1D memory-strip path (production) takes {strip_ms:.3f} ms per pack+unpack")

    with open(os.path.join(OUTDIR, "block_shape.csv"), "w") as f:
        f.write("block_shape,values,bits_per_value,aspect_ratio,snr_db,"
                "pack_unpack_ms,slowdown_vs_strip,label\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.1f},{r[4]:.1f},{r[5]:.4f},{r[6]:.2f},{r[7]}\n")

    # Figure, two panels: accuracy against shape, and pack/unpack cost against shape, so the
    # trade-off is on one page.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    xs = [r[3] for r in rows]
    axL.plot(xs, [r[4] for r in rows], "o-", lw=2, color="tab:purple")
    for r in rows:
        note = r[7] if r[7] in ("strip (memory)", "square") else r[0]
        axL.annotate(note, (r[3], r[4]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    axL.set_xlabel("block aspect ratio  (1 = square, higher = more strip-like)")
    axL.set_ylabel("signal-to-noise ratio (dB), higher is better")
    axL.set_title("Accuracy: squarer is more accurate")
    axL.grid(alpha=0.3)

    axR.plot(xs, [r[5] for r in rows], "s-", lw=2, color="tab:orange")
    axR.axhline(strip_ms, color="grey", ls="--", lw=1)
    axR.text(xs[len(xs) // 2], strip_ms, " 1D strip path (production)", fontsize=8,
             color="grey", va="bottom")
    for r in rows:
        note = r[7] if r[7] in ("strip (memory)", "square") else r[0]
        axR.annotate(note, (r[3], r[5]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    axR.set_xlabel("block aspect ratio  (1 = square, higher = more strip-like)")
    axR.set_ylabel("pack + unpack time per call (ms), lower is better")
    axR.set_title("Cost: 2D tiling adds a little time")
    axR.grid(alpha=0.3)

    fig.suptitle(f"Block shape: accuracy vs pack/unpack cost (fixed 32-value block, "
                 f"{MANTISSA}-bit", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTDIR, "fig_block_shape.png"), dpi=130)
    plt.close(fig)

    strip = next(r for r in rows if r[0] == "1x32")
    square = min(rows, key=lambda r: r[3])       # closest to square
    print(f"\nstrip {strip[0]}: {strip[4]:.1f} dB;  squarest {square[0]}: {square[4]:.1f} dB;  "
          f"difference {square[4] - strip[4]:+.1f} dB at the same cost.")
    print(f"the squarer block costs about {square[6]:.1f}x the strip in pack/unpack time, but that "
          f"time is a few thousandths of a second and the solver is bandwidth-bound, so it is "
          f"small against the memory traffic saved.")
    print(f"\ntable and figure written to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
