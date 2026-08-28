#!/usr/bin/env python3
"""
Head to head comparison of microscaling against the established reduced precision schemes.

Why this file exists:
There is no published work applying microscaling to seismic wave propagation, so there is no
number in the literature that our result can simply be placed next to. Quoting figures from
other papers would not help either, because they were measured on different velocity models,
different grids and different error metrics, so the comparison would not be fair and would not
survive review.

The way round this is to reimplement the competing schemes inside our own harness, and run
every one of them on the same cropped Marmousi model, with the same acquisition, the same time
stepping and the same error metric. Only then is the comparison genuinely head to head.

This turns out to be natural, because every scheme here is just a different way of storing the
wavefield between time steps. Each is a function that takes a float32 wavefield, forces it
through a narrower representation, and hands back float32. They all plug into exactly the same
slot in the solver loop, so the only thing that differs between runs is the storage format.

The schemes compared:

  float32
      The working precision of the industry, and the reference that everything else is
      measured against.

  float16 with no scaling
      The naive thing to try. It is included to test whether the scaling trick below is
      actually needed, rather than assuming it is.

  float16 with global scaling
      The scheme of Fabien-Ouellet (2020), the strongest published baseline for reduced
      precision seismic modelling. float16 has a narrow exponent range, so on some problems
      the smaller wavefield values underflow and the weak arrivals are lost. One constant is
      chosen by hand, multiplies the whole field to lift it into the healthy part of the
      float16 range before storing, and divides back out afterwards. It is a single scale for
      the entire grid and the entire simulation, which is what makes it global.

  bfloat16
      The other sixteen bit format in wide use, and the one most AI hardware supports. It
      keeps all of float32's exponent range so it can never underflow and needs no scaling,
      but it pays for that range by keeping only seven mantissa bits.

  microscaling, our scheme
      Every block of values shares one exponent, so the scale follows the local amplitude of
      the wavefield instead of being fixed once for the whole grid. The question this file
      answers is whether that local adaptation buys enough accuracy to beat a single global
      scale at the same, or at a smaller, number of bits per value.

Why bits per value is the right measure of cost:
The solver is limited by memory bandwidth, not by arithmetic. Its speed depends on how fast
the wavefield arrays move through the memory hierarchy, not on how fast the floating point
units run. So the number of bits stored per value is what the memory traffic scales with, and
cutting it is what makes the solver cheaper and faster. That makes bits per value an honest
cost axis, and it has the advantage of not depending on which GPU happens to be available.

Run with: python3 compare_schemes.py
"""

import os
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The solver harness and the microscaling routine are imported unchanged from the main
# experiment, so that every scheme really is being tested under identical conditions.
from mx_experiment_marmousi import (Harness, quantize_dequantize, rel_l2, snr_db,
                                    rel_l2_timescaled, snr_db_timescaled, TN, _wiggle, WIGGLE_SATURATION)

TS_POWER = 1.0   # time-scaling exponent for the accuracy metric (linear gain)

OUTDIR = "comparison_results"


def store_float16_naive(x):
    """Store the wavefield in plain float16, with no scaling of any kind.

    This is the obvious thing to try, and it is here to be tested rather than assumed to fail.
    float16 can only represent normal numbers down to about 6e-5. Whether that matters depends
    entirely on the amplitudes the wavefield actually reaches, so the honest thing is to run it
    and look, instead of asserting in advance that it must break.
    """
    return x.astype(np.float16).astype(np.float32).astype(x.dtype)


def store_float16_global(x, scale):
    """Store the wavefield in float16 after lifting it with one fixed global scale.

    This is the published approach. Multiplying by a constant first moves the whole field into
    the healthy middle of the float16 range, and dividing by the same constant afterwards undoes
    it. The scale is a power of two, so the multiply and the divide are exact and introduce no
    rounding error of their own. It is chosen once, from the amplitude of the reference run, and
    then held fixed for the entire simulation, which is precisely what makes it global rather
    than per block.
    """
    return (np.float32(x * scale).astype(np.float16).astype(np.float32) / scale).astype(x.dtype)


def store_bfloat16(x):
    """Store the wavefield in bfloat16, emulated by rounding a float32 to seven mantissa bits.

    bfloat16 keeps all eight of float32's exponent bits, so it spans the same enormous range of
    magnitudes and never needs a scaling factor. The price is that only seven mantissa bits
    survive, so each individual value is coarse.

    numpy has no bfloat16 type, so the format is emulated on the bit pattern directly. Take the
    float32, add a rounding term so that the discarded half rounds to nearest even rather than
    always downwards, then clear the bottom sixteen bits of the mantissa.
    """
    u = np.asarray(x, np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32).astype(x.dtype)


def store_int16_scaled(x, scale):
    """Store the wavefield as 16-bit signed integers relative to one global scale.

    This is fixed point, not floating point. A single scale maps the field onto the integer
    range minus 32768 to 32767, every value is rounded to the nearest integer, and the scale is
    divided back out. It is the large block limit of microscaling: one shared exponent for the
    whole grid instead of one per block. Because it is fixed point, a value far below the field
    maximum keeps almost none of its bits, so it is expected to trail floating point formats on a
    wavefield, whose amplitude spans a wide range.
    """
    q = np.clip(np.round(x.astype(np.float64) * scale), -32768, 32767)
    return (q / scale).astype(x.dtype)


def mx_bits_per_value(block_size, mantissa_bits):
    """How many bits one wavefield value costs under our microscaling format.

    Each value needs one sign bit, one bit for the integer part of the normalised value, and the
    mantissa bits themselves. On top of that the block's shared exponent has to be paid for, but
    it is shared across the whole block, so spread over block_size values its cost is small.
    That sharing is the entire point of the format, and it is also why a bigger block looks
    cheaper on this axis while being less able to follow the local amplitude.
    """
    return (mantissa_bits + 2) + 8.0 / block_size


def plot_cost_vs_accuracy(results, fp32_own_error, outdir):
    """The headline figure: what each scheme costs, and what it costs you in accuracy.

    Cost runs along the bottom and error up the side, both on scales where lower is better, so a
    scheme is better the further it sits towards the bottom left. If one of our microscaling
    points sits below and to the left of a published scheme, that scheme has been beaten on both
    counts at once: fewer bits stored and less error.

    The dashed line is the error float32 itself carries against float64 on this same model. It
    is the level at which a scheme stops being distinguishable from float32 in practice.
    """
    fig, ax = plt.subplots(figsize=(9, 5.8))

    # Our scheme, drawn as a curve because it is a family of settings rather than one point.
    ours = sorted([r for r in results if r["family"] == "ours"], key=lambda r: r["bits"])
    ax.plot([r["bits"] for r in ours], [r["err"] for r in ours],
            marker="o", lw=2.0, color="tab:blue", zorder=3,
            label="microscaling, this work (block 32)")

    # The competing schemes, each a single fixed format and so a single point.
    style = {"float32":                  ("*", "k",          210),
             "float16, no scaling":      ("^", "tab:green",  130),
             "float16 + global scaling": ("s", "tab:red",    130),
             "bfloat16":                 ("D", "tab:orange", 120),
             "int16 + scaling":          ("P", "tab:brown",  130)}
    for r in results:
        if r["family"] == "ours":
            continue
        m, c, s = style[r["name"]]
        ax.scatter(r["bits"], r["err"], marker=m, s=s, c=c,
                   edgecolors="k", zorder=4, label=r["name"])

    ax.axhline(fp32_own_error, color="k", ls="--", lw=1.2)
    ax.text(ax.get_xlim()[1], fp32_own_error * 1.35, "float32 own error",
            ha="right", fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("bits stored per wavefield value, which is what the memory traffic scales with")
    ax.set_ylabel("time-scaled relative error of the shot record")
    # Signal-to-noise ratio in decibels is the field's usual way of reporting this, so it goes on
    # the right as the primary reading. It is the same information as the left axis, since
    # SNR(dB) = -20 log10(relative error); the two axes stay locked together.
    secax = ax.secondary_yaxis("right", functions=(lambda e: -20.0 * np.log10(np.clip(e, 1e-300, None)),
                                                    lambda d: 10.0 ** (-d / 20.0)))
    secax.set_ylabel("time-scaled signal-to-noise ratio (dB)")
    secax.set_yticks([20, 40, 60, 80])          # clean, evenly spaced dB ticks
    secax.minorticks_off()                       # drop the crowded auto minor ticks at the top
    ax.set_title("Cost against accuracy (cropped Marmousi)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_cost_vs_accuracy.png"), dpi=130)
    plt.close(fig)


def plot_block_size(block_results, outdir):
    """Does it pay to share the exponent over fewer values?

    A small block follows the local amplitude of the wavefield closely, so its shared exponent
    fits the values it covers and little precision is wasted. A large block has to stretch one
    exponent across a wider range of magnitudes, so the smaller values in it lose out. Against
    that, a small block pays for its exponent more often, so it costs more bits per value.

    This figure puts both effects on the same picture: the horizontal axis is what the block
    actually costs once the shared exponent is counted, and the vertical axis is what it buys.
    It is the plot that says whether fine grained scaling is worth its overhead.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bits = [r["bits"] for r in block_results]
    errs = [r["err"] for r in block_results]
    ax.plot(bits, errs, marker="o", lw=1.8, color="tab:purple")
    for r in block_results:
        ax.annotate(f"block {r['block']}", (r["bits"], r["err"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("bits stored per value, including the share of the block exponent")
    ax.set_ylabel("time-scaled relative error of the shot record")
    ax.set_title(f"Effect of block size ({BLOCK_SWEEP_BITS} mantissa bits)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_block_size.png"), dpi=130)
    plt.close(fig)


def plot_traces(rec_ref, traces, outdir):
    """How far each scheme departs from full precision, shown as a time-scaled difference gather.

    Drawing the gathers themselves is not very telling here, because at a usable setting MX and
    FP16 both reproduce the record closely and the panels look alike. What matters is the
    difference from full precision, so this plots exactly that: each scheme's gather minus the
    reference, on one common amplitude scale, so more residual means a worse scheme. A time gain
    proportional to time (a linear gain) is applied, the standard correction for geometric spreading,
    which lifts the weak late arrivals so the residual is visible over the whole record rather
    than only near the top. Read this way the order is plain: bfloat16 leaves a large residual, MX
    the smallest, FP16 in between.
    """
    nt, nrec = rec_ref.shape
    band = slice(nrec // 2 - 30, nrec // 2 + 30, 2)        # about 30 central traces
    t = np.linspace(0, TN / 1000.0, nt)
    ref = np.asarray(rec_ref)[:, band].astype(np.float64)

    # Time-scaling gain, linear in t, normalised to about one in the middle of the record so the
    # numbers stay in a sensible range.
    tgain = ((t + 0.02) / np.mean(t + 0.02)) ** 1.0

    # Residual of each scheme against full precision, with the time gain applied.
    resid = [(name, (np.asarray(rec)[:, band].astype(np.float64) - ref) * tgain[:, None])
             for (name, rec, _c) in traces]
    allmax = max(np.max(np.abs(r)) for _, r in resid) + 1e-30
    common = allmax / 6.0                                   # common scale, biggest residual saturates

    short = {"float16 + global scaling": "FP16 + scaling", "bfloat16": "bfloat16",
             "microscaling, 12 mantissa bits": "MX, 12 bits"}
    n = len(resid)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 6.4), sharey=True)
    for ax, (name, data) in zip(axes, resid):
        _wiggle(ax, data, t, norm=common)
        ax.set_title(short.get(name, name), fontsize=10)
        ax.set_xlabel("trace")
    axes[0].set_ylabel("time (s)")
    fig.suptitle("Difference from full precision (time-scaled)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_trace_errors.png"), dpi=130)
    plt.close(fig)


# What gets swept. The bit widths are for our own format at a fixed block size, and the block
# sizes are swept separately at one fixed bit width so the two effects stay separable.
MX_BLOCK       = 32
MX_BITS        = [4, 6, 8, 10, 12, 14]
BLOCK_SWEEP    = [8, 16, 32, 64]
BLOCK_SWEEP_BITS = 10


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)

    # Build the solver once at float32. That is the precision the stencil arithmetic runs in for
    # every single scheme here. The only thing that changes between runs is how the wavefield is
    # stored between the steps.
    H = Harness(dtype=np.float32)
    print(f"cropped Marmousi grid={H.shape}  nt={H.nt}  dt={H.dt:.4f} ms\n")

    # The float32 run with no storage compression at all. Everything is measured against this.
    rec_ref, snap_ref, _ = H.run(quant=None)

    # The same simulation in float64, used only to find out how much error float32 already
    # carries by itself. That is the yardstick: a scheme that reaches this level is, for all
    # practical purposes, as accurate as float32.
    H64 = Harness(dtype=np.float64)
    rec_64, _, _ = H64.run(quant=None)
    fp32_own_error = rel_l2_timescaled(rec_ref, rec_64, TS_POWER)
    print(f"float32 own error against float64: {fp32_own_error:.3e}")
    print("that is the level a scheme must reach to count as being as good as float32\n")

    # Choose the global scale for the float16 scheme. It lifts the largest value the wavefield
    # reaches into a comfortable part of the float16 range rather than up against its ceiling,
    # and it is rounded to a power of two so that applying and removing it is exact.
    amax = float(np.abs(snap_ref).max())
    scale = 2.0 ** np.round(np.log2(1e3 / max(amax, 1e-30)))
    print(f"float16 global scale chosen as 2^{int(np.log2(scale))}\n")

    results = []
    traces  = []      # a few full shot records, kept so the error panels can be drawn

    # float32 itself, listed at its true cost so it appears on the plot.
    results.append(dict(name="float32", family="baseline", bits=32.0, err=fp32_own_error))

    print("scheme                          bits/value    error")
    print(f"  float32                          32.00     {fp32_own_error:.3e}")

    # float16 with no scaling. Included to test the assumption, not to confirm it.
    rec, _, _ = H.run(quant=store_float16_naive)
    e = rel_l2_timescaled(rec, rec_ref, TS_POWER)
    results.append(dict(name="float16, no scaling", family="published", bits=16.0, err=e))
    print(f"  float16, no scaling              16.00     {e:.3e}")

    # float16 with the published global scaling.
    rec_f16, _, _ = H.run(quant=lambda a: store_float16_global(a, scale))
    e = rel_l2_timescaled(rec_f16, rec_ref, TS_POWER)
    results.append(dict(name="float16 + global scaling", family="published", bits=16.0, err=e))
    traces.append(("float16 + global scaling", rec_f16, "tab:red"))
    print(f"  float16 + global scaling         16.00     {e:.3e}")

    # bfloat16.
    rec_bf, _, _ = H.run(quant=store_bfloat16)
    e = rel_l2_timescaled(rec_bf, rec_ref, TS_POWER)
    results.append(dict(name="bfloat16", family="published", bits=16.0, err=e))
    traces.append(("bfloat16", rec_bf, "tab:orange"))
    print(f"  bfloat16                         16.00     {e:.3e}")

    # int16 with global scaling (fixed point). To give it a fair chance, the scale maps the
    # largest amplitude the wavefield reaches at any time, not just the final frame, to the int16
    # ceiling, so the strong early arrivals are not clipped.
    _, _, hist_ref = H.run(quant=None, store_wavefield=True)
    peak = float(np.abs(hist_ref).max())
    int16_scale = 32767.0 / max(peak, 1e-30)
    rec_i16, _, _ = H.run(quant=lambda a: store_int16_scaled(a, int16_scale))
    e = rel_l2_timescaled(rec_i16, rec_ref, TS_POWER)
    results.append(dict(name="int16 + scaling", family="published", bits=16.0, err=e))
    print(f"  int16 + scaling                  16.00     {e:.3e}")

    # Our microscaling scheme across bit widths, at a fixed block size.
    print()
    for mb in MX_BITS:
        rec, _, _ = H.run(quant=lambda a, mb=mb: quantize_dequantize(a, MX_BLOCK, mb, "nearest", rng))
        e = rel_l2_timescaled(rec, rec_ref, TS_POWER)
        b = mx_bits_per_value(MX_BLOCK, mb)
        results.append(dict(name=f"MX block {MX_BLOCK}, {mb} mantissa bits",
                            family="ours", bits=b, err=e))
        print(f"  MX {mb:2d} mantissa bits             {b:5.2f}     {e:.3e}")
        # Keep the setting that costs about the same as the sixteen bit formats, so the trace
        # plot compares schemes at roughly equal cost rather than comparing apples with oranges.
        if mb == 12:
            traces.append((f"microscaling, {mb} mantissa bits", rec, "tab:blue"))

    # How much the block size matters, at one fixed bit width.
    print()
    block_results = []
    for bsz in BLOCK_SWEEP:
        rec, _, _ = H.run(quant=lambda a, bsz=bsz: quantize_dequantize(a, bsz, BLOCK_SWEEP_BITS,
                                                                       "nearest", rng))
        e = rel_l2_timescaled(rec, rec_ref, TS_POWER)
        b = mx_bits_per_value(bsz, BLOCK_SWEEP_BITS)
        block_results.append(dict(block=bsz, bits=b, err=e))
        print(f"  MX block {bsz:2d}, {BLOCK_SWEEP_BITS} mantissa bits    {b:5.2f}     {e:.3e}")

    # Write the numbers out so they can go straight into the report. The scheme names contain
    # commas, so they are quoted, otherwise they would split into extra columns when read back.
    with open(os.path.join(OUTDIR, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "family", "bits_per_value", "relative_error", "snr_db",
                    "compression_vs_float32"])
        for r in results:
            snr = -20.0 * np.log10(max(r["err"], 1e-300))
            w.writerow([r["name"], r["family"], f"{r['bits']:.2f}",
                        f"{r['err']:.6e}", f"{snr:.1f}", f"{32.0 / r['bits']:.2f}"])

    # Summary with signal-to-noise ratio as the leading column, since that is the metric the
    # report leads on. Higher dB is better; the float32-against-float64 line is the pass mark.
    print("\nsummary, signal-to-noise ratio first (higher is better):")
    print(f"  {'scheme':<32s}{'bits':>7s}{'SNR (dB)':>11s}{'rel. error':>13s}")
    for r in sorted(results, key=lambda r: r["err"]):
        snr = -20.0 * np.log10(max(r["err"], 1e-300))
        print(f"  {r['name']:<32s}{r['bits']:>7.2f}{snr:>11.1f}{r['err']:>13.2e}")
    print(f"  {'float32 vs float64 (pass mark)':<32s}{'':>7s}{snr_db_timescaled(rec_ref, rec_64, TS_POWER):>11.1f}"
          f"{fp32_own_error:>13.2e}")

    plot_cost_vs_accuracy(results, fp32_own_error, OUTDIR)
    plot_block_size(block_results, OUTDIR)
    plot_traces(rec_ref, traces, OUTDIR)

    print(f"\ntable and three figures written to ./{OUTDIR}/")
    print("reading the main plot: a scheme is better the further it sits towards the bottom "
          "left, meaning it loses less accuracy while storing fewer bits per value. The right "
          "axis reads the same thing as signal-to-noise ratio in decibels.")


if __name__ == "__main__":
    main()
