#!/usr/bin/env python3
"""
Isolating the reflections with a water model, and scoring the schemes on those reflections.

The problem this solves:
The global error, whether written as relative L2 or as SNR in decibels, is dominated by the
strong direct arrival that runs along the top of the record. That arrival travels through the
near-surface water layer and never enters the geology, so it carries none of the image and is
the easy part for any format to reproduce. A single number built mostly from it tells us little
about the weak reflections that the image is actually made of.

The method, as proposed in the review meeting:
Run a second model that is water everywhere, at the near-surface velocity, with the source,
receivers, geometry and time-stepping kept exactly the same as the real run. With no velocity
contrasts, that model produces only the direct arrival and no reflections, and it computes it
very accurately. Subtracting it removes the direct arrival from the real record and leaves the
reflected and refracted energy on its own:

    reflections_ref = full_ref - water_ref
    reflections_MX  = full_MX  - water_MX

The score is then the signal-to-noise ratio in decibels computed on the reflections alone,
which is the quantity that corresponds to what the imaging actually cares about. The MX run is
put through the water model too, so the direct-arrival part it removes is the one MX itself
produced, not the full-precision one.

Keeping everything identical except the velocity is essential: the water model reuses the same
harness, so dt, the number of steps, the source wavelet and the operator are unchanged, and only
model.vp is overwritten. The Marmousi time step is smaller than water would require on its own,
so the water run is comfortably stable.

Run with: python3 water_subtraction.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mx_experiment_marmousi import (Harness, quantize_dequantize, rel_l2, snr_db, snr_db_timescaled,
                                    load_cropped_vp, TN, SPACING, NREC)

OUTDIR = "water_subtraction_results"
BLOCK = 32
BITS = [8, 10, 12, 14]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)

    H = Harness(dtype=np.float32)
    vp = load_cropped_vp()
    water_v = float(vp[:, 0].mean())          # near-surface velocity the direct arrival travels at
    print(f"velocity model {vp.shape}, {vp.min():.2f} to {vp.max():.2f} km/s; "
          f"water model set to {water_v:.2f} km/s (the surface velocity)\n")

    def mx(mb):
        return lambda a, mb=mb: quantize_dequantize(a, BLOCK, mb, "nearest", rng)

    # --- full model: reference and MX at each bit width ---
    full_ref, _, _ = H.run(quant=None)
    full_ref = np.asarray(full_ref)
    full_mx = {mb: np.asarray(H.run(quant=mx(mb))[0]) for mb in BITS}

    # --- swap the whole model to water and rerun, everything else identical ---
    vp_full_backup = np.array(H.model.vp.data)
    H.model.vp.data[:] = water_v
    water_ref, _, _ = H.run(quant=None)
    water_ref = np.asarray(water_ref)
    water_mx = {mb: np.asarray(H.run(quant=mx(mb))[0]) for mb in BITS}
    H.model.vp.data[:] = vp_full_backup       # restore, in case the harness is reused

    # --- subtract to isolate reflections ---
    refl_ref = full_ref - water_ref
    frac = np.linalg.norm(refl_ref) / np.linalg.norm(full_ref)
    print(f"the reflections are {100 * frac:.1f}% of the full record's energy; "
          f"the direct arrival is the other {100 * (1 - frac):.0f}%\n")

    # --- score: global SNR vs reflection-only SNR, per bit width ---
    rows = []
    print(f"  {'mantissa':>8s}{'bits/val':>9s}{'global SNR dB':>15s}{'reflection SNR dB':>19s}")
    for mb in BITS:
        g = snr_db_timescaled(full_mx[mb], full_ref)            # time-scaled whole-record metric
        refl_mx = full_mx[mb] - water_mx[mb]                    # MX reflections
        r = snr_db_timescaled(refl_mx, refl_ref)                # reflections only, time-scaled
        bits = mb + 2 + 8.0 / BLOCK
        rows.append((mb, bits, g, r))
        print(f"  {mb:>8d}{bits:>9.2f}{g:>15.1f}{r:>19.1f}")

    with open(os.path.join(OUTDIR, "water_subtraction.csv"), "w") as f:
        f.write("mantissa_bits,bits_per_value,global_snr_db,reflection_snr_db\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.1f},{r[3]:.1f}\n")

    # --- figure 1: the two SNRs against bit width ---
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bx = [r[1] for r in rows]
    ax.plot(bx, [r[2] for r in rows], marker="o", lw=2, color="tab:blue",
            label="time-scaled SNR (whole record)")
    ax.plot(bx, [r[3] for r in rows], marker="s", lw=2, color="tab:red",
            label="reflection-only SNR (water model removed)")
    ax.set_xlabel("bits stored per wavefield value")
    ax.set_ylabel("signal-to-noise ratio (dB), higher is better")
    ax.set_title("Time-scaled whole-record vs reflection-only SNR")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_water_snr.png"), dpi=130)
    plt.close(fig)

    # --- figure 2: the subtraction itself, as wiggle gathers ---
    t = np.linspace(0, TN / 1000.0, full_ref.shape[0])
    band = slice(NREC // 2 - 30, NREC // 2 + 30, 2)
    from mx_experiment_marmousi import _wiggle, WIGGLE_SATURATION
    # Time-scaling gain, linear in t, a gentle correction for geometric spreading. It lifts the
    # weak deep reflections so they are visible alongside the shallow ones without over-amplifying
    # the noise the way a t-squared gain would. Applied to the reflection panel, where it matters;
    # the full and water panels keep the saturated direct arrival.
    tgain = ((t + 0.02) / np.mean(t + 0.02)) ** 1.0
    common = (np.max(np.abs(full_ref[:, band])) + 1e-30) / WIGGLE_SATURATION
    refl_scaled = refl_ref[:, band].astype(np.float64) * tgain[:, None]
    common_r = (np.max(np.abs(refl_scaled)) + 1e-30) / 3.0
    fig, axes = plt.subplots(1, 3, figsize=(11, 6), sharey=True)
    _wiggle(axes[0], full_ref[:, band].astype(np.float64), t, norm=common)
    axes[0].set_title("full record"); axes[0].set_ylabel("time (s)")
    _wiggle(axes[1], water_ref[:, band].astype(np.float64), t, norm=common)
    axes[1].set_title("water (direct only)")
    _wiggle(axes[2], refl_scaled, t, norm=common_r)
    axes[2].set_title("reflections (time-scaled)")
    for a in axes:
        a.set_xlabel("trace")
    fig.suptitle("Water subtraction: full record minus water leaves the reflections",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_water_gather.png"), dpi=130)
    plt.close(fig)

    # plain-language read
    mid = [r for r in rows if r[0] == 12][0]
    print(f"\nAt 12 mantissa bits the global SNR is {mid[2]:.0f} dB, but on the reflections alone "
          f"it is {mid[3]:.0f} dB.")
    print("The reflection SNR is the honest figure: it drops the easy direct arrival and scores "
          "the weak energy the image is built from.")
    print(f"\ntable and figures written to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
