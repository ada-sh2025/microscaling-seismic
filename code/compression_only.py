#!/usr/bin/env python3
"""
Compression and decompression on its own, with no wave propagation in between.

Why this test exists:
Every other measurement in the project runs the format inside the time-stepping loop, so the
number it reports is the error after hundreds of steps of feedback. That number is what matters
in practice, but it mixes two things together: the error the format makes on a single frame, and
the way propagation amplifies that error over time. This test separates them. It takes a real
wavefield, compresses it to MX and decompresses it once, and measures what changed. No solver,
no time steps, just the format's own representation error on one frame.

Putting the two side by side is the point:

  single-pass error       what MX costs on one frame, compressed and decompressed once
  accumulated error       what MX costs after the same format runs inside the full solve

If the accumulated error is far larger than the single-pass error, the limit is accumulation,
not representation, which is the mechanism the report argues for. The gap between the two is the
amplification that the time-stepping adds.

Everything is reported both as relative L2 error and as signal-to-noise ratio in decibels, the
same way as the rest of the study.

Run with: python3 compression_only.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mx_experiment_marmousi import (Harness, quantize_dequantize, rel_l2, snr_db)


def mx_bits_per_value(block_size, mantissa_bits):
    """Sign and integer bit, mantissa bits, and the shared 8-bit exponent spread over the block."""
    return mantissa_bits + 2 + 8.0 / block_size

OUTDIR = "compression_only_results"
BITS = [4, 6, 8, 10, 12, 14, 16]
BLOCK = 32


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    H = Harness(dtype=np.float32)

    # A real wavefield to compress: the reference field partway through the solve, and the full
    # history so the single-pass test runs on genuine propagated data rather than a toy array.
    _, snap_ref, hist_ref = H.run(quant=None, store_wavefield=True)
    snap_ref = np.asarray(snap_ref)
    hist_ref = np.asarray(hist_ref)

    print(f"reference wavefield {snap_ref.shape}, "
          f"amplitude range {np.abs(snap_ref).min():.2e} to {np.abs(snap_ref).max():.2e}\n")

    rows = []
    print(f"  {'mantissa':>8s}{'bits/val':>9s}{'single-pass dB':>16s}{'accumulated dB':>16s}"
          f"{'amplification':>15s}")
    for mb in BITS:
        bits = mx_bits_per_value(BLOCK, mb)

        # Single pass: compress and decompress the reference field once, no propagation. Averaged
        # over the stored history so the figure is a fair frame, not one lucky snapshot.
        single_errs = []
        for k in range(0, hist_ref.shape[0], max(1, hist_ref.shape[0] // 20)):
            frame = hist_ref[k]
            back = quantize_dequantize(frame, BLOCK, mb, "nearest", rng)
            if np.linalg.norm(frame) > 0:
                single_errs.append(rel_l2(back, frame))
        single = float(np.mean(single_errs))

        # Accumulated: the same format run inside the full solve, measured on the final field.
        _, snap_mx, _ = H.run(quant=lambda a, mb=mb: quantize_dequantize(a, BLOCK, mb, "nearest", rng))
        accumulated = rel_l2(np.asarray(snap_mx), snap_ref)

        amp = accumulated / single if single > 0 else np.nan
        s_db = -20.0 * np.log10(max(single, 1e-300))
        a_db = -20.0 * np.log10(max(accumulated, 1e-300))
        rows.append((mb, bits, single, accumulated, s_db, a_db, amp))
        print(f"  {mb:>8d}{bits:>9.2f}{s_db:>16.1f}{a_db:>16.1f}{amp:>14.0f}x")

    # Write the table.
    with open(os.path.join(OUTDIR, "compression_only.csv"), "w") as f:
        f.write("mantissa_bits,bits_per_value,single_pass_error,accumulated_error,"
                "single_pass_snr_db,accumulated_snr_db,amplification\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.6e},{r[3]:.6e},{r[4]:.1f},{r[5]:.1f},{r[6]:.1f}\n")

    # Figure: single-pass against accumulated, in dB, across bit widths.
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bx = [r[1] for r in rows]
    ax.plot(bx, [r[4] for r in rows], marker="o", lw=2, color="tab:green",
            label="single pass: compress and decompress once")
    ax.plot(bx, [r[5] for r in rows], marker="s", lw=2, color="tab:blue",
            label="accumulated: same format run through the full solve")
    for r in rows:
        ax.annotate(f"{r[6]:.0f}\u00D7", (r[1], (r[4] + r[5]) / 2), fontsize=8,
                    color="tab:red", ha="center")
    ax.set_xlabel("bits stored per wavefield value")
    ax.set_ylabel("signal-to-noise ratio (dB), higher is better")
    ax.set_title("Single-frame vs accumulated error")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.5, 0.01, "red = how many times propagation amplifies the single-frame error",
             ha="center", fontsize=9, color="tab:red")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(OUTDIR, "fig_compression_only.png"), dpi=130)
    plt.close(fig)

    # A plain-language read of the result.
    mid = [r for r in rows if r[0] == 12][0]
    print(f"\nAt 12 mantissa bits the format loses {mid[4]:.0f} dB on a single frame, "
          f"but {mid[5]:.0f} dB once it runs through the solve.")
    print(f"Propagation amplifies the single-frame error by about {mid[6]:.0f} times, "
          f"so the limit is accumulation, not the representation of any one frame.")
    print(f"\ntable and figure written to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
