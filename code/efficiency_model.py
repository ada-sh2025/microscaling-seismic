#!/usr/bin/env python3
"""
Efficiency against error, the other half of the comparison.

The problem this file solves:
compare_schemes.py measures how much accuracy each storage format costs. It does not say what
each format buys, and a trade off cannot be judged from only one of its two sides. This file
supplies the other side.

Why efficiency is modelled rather than timed:
The obvious thing would be to time the schemes and report the speedup. That would be
meaningless here, for two reasons. Our microscaling is emulated in numpy, so packing and
unpacking adds work rather than removing it. And Devito's generated stencil still reads and
writes float32 whatever we do to the array in between, so the memory traffic inside the kernel
never actually falls. A wall clock number from this setup would measure the emulation, not the
format.

The honest route, and the one the project plan already commits to as its fallback, is to model
the saving instead. The solver is limited by memory bandwidth: its speed is set by how fast the
wavefield and model arrays move through the memory hierarchy, not by how fast the floating point
units run. That gives a direct chain of reasoning. Storing fewer bits per value means fewer bytes
moved every step, which means less pressure on the bottleneck, which means a shorter runtime. If
the solver is purely bandwidth bound, the speedup is simply how much the memory traffic shrank.
That is a roofline argument, and it has the useful property of not depending on which GPU happens
to be sitting in the machine.

The one thing the model must not gloss over:
Only the wavefield is stored in microscaling. The model arrays, the velocity and whatever else
the kernel needs, stay in float32. So the traffic saving is never as large as the compression
ratio of a single array, and how much smaller it is depends on how the kernel's arrays are split
between model and wavefield. For the acoustic kernel the wavefield is only part of the traffic,
so the saving is diluted. For the anisotropic kernels the wavefield dominates completely, and the
saving is close to the full compression ratio. Modelling that split is the point of the array
counts below, and it is what turns a per array compression number into an honest speedup estimate.

Input:
comparison.csv, written by compare_schemes.py. Run that first.

Run with: python3 efficiency_model.py
"""

import os
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INFILE = "comparison_results/comparison.csv"
OUTDIR = "comparison_results"


# How many arrays each kind of finite difference kernel has to move every time step, split into
# the model arrays, which stay in float32, and the wavefield arrays, which are the ones we can
# store in a narrow format. The counts follow Table 1 of the project plan. The pattern is the
# important part: as the physics gets more realistic, anisotropy multiplies the number of
# wavefield variables, so the wavefield comes to dominate the traffic almost entirely. That is
# exactly the regime in which compressing the wavefield pays off most.
KERNELS = {
    "acoustic":            dict(model=2,  field=4),
    "elastic isotropic":   dict(model=3,  field=9),
    "elastic TTI":         dict(model=22, field=36),
    "viscoelastic TTI":    dict(model=30, field=84),
}


def traffic_bytes(kernel, field_bits, compress_model=False):
    """Bytes moved per grid point per time step, for one kernel and one storage format.

    Every array the kernel touches has to be read or written each step, so the traffic is simply
    the number of arrays times the size of one value in each. The wavefield arrays are stored at
    field_bits, and the model arrays stay at float32 unless we choose to compress them too.
    """
    k = KERNELS[kernel]
    model_bits = field_bits if compress_model else 32.0
    return (k["model"] * model_bits + k["field"] * field_bits) / 8.0


def predicted_speedup(kernel, field_bits, compress_model=False):
    """How much faster the solver should run, if it is purely limited by memory bandwidth.

    Under that assumption the runtime is proportional to the bytes moved, so the speedup is just
    the ratio of the float32 traffic to the traffic of the narrower format. This is an upper
    bound rather than a promise: a real kernel never reaches its bandwidth ceiling exactly, and
    packing and unpacking costs a few extra instructions per value. But it is the right quantity
    to compare formats with, because it isolates the effect of the format from the accidents of
    a particular machine.
    """
    base = traffic_bytes(kernel, 32.0, compress_model=False)
    return base / traffic_bytes(kernel, field_bits, compress_model)


def load_schemes(path):
    """Read back the accuracy results that compare_schemes.py measured."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(name=r["scheme"], family=r["family"],
                             bits=float(r["bits_per_value"]),
                             err=float(r["relative_error"])))
    return rows


def plot_efficiency_vs_error(rows, kernel, outdir):
    """The trade off that actually matters: what you pay in accuracy, and what you get in speed.

    Error runs along the bottom and the predicted speedup up the side, so the desirable corner is
    the top left: fast and accurate. A format that sits above and to the left of another one beats
    it outright. This is the plot that answers whether microscaling is worth using instead of the
    sixteen bit formats already in the literature, because it puts both of them on the same two
    axes at once.
    """
    fig, ax = plt.subplots(figsize=(9, 5.8))

    ours = sorted([r for r in rows if r["family"] == "ours"], key=lambda r: r["bits"])
    ax.plot([r["err"] for r in ours],
            [predicted_speedup(kernel, r["bits"]) for r in ours],
            marker="o", lw=2.0, color="tab:blue", zorder=3,
            label="microscaling, wavefield only")

    # The same format, but with the model arrays stored narrow as well. The gap between the two
    # curves is the speedup that is currently being left on the table simply because the model
    # arrays are still float32, and it widens as the format gets more aggressive.
    ax.plot([r["err"] for r in ours],
            [predicted_speedup(kernel, r["bits"], compress_model=True) for r in ours],
            marker="o", ms=4, lw=1.6, ls=":", color="tab:blue", alpha=0.6, zorder=3,
            label="microscaling, wavefield and model (the ceiling)")

    style = {"float32":                  ("*", "k",          210),
             "float16, no scaling":      ("^", "tab:green",  130),
             "float16 + global scaling": ("s", "tab:red",    130),
             "bfloat16":                 ("D", "tab:orange", 120)}
    for r in rows:
        if r["family"] == "ours":
            continue
        m, c, s = style[r["name"]]
        ax.scatter(r["err"], predicted_speedup(kernel, r["bits"]),
                   marker=m, s=s, c=c, edgecolors="k", zorder=4, label=r["name"])

    ax.set_xscale("log")
    ax.set_xlabel("relative error of the shot record, lower is better")
    ax.set_ylabel("predicted speedup if the solver is bandwidth bound")
    ax.set_title(f"Efficiency against error, {kernel} kernel")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.invert_xaxis()          # so that better accuracy is to the right, and best is top right
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_efficiency_vs_error.png"), dpi=130)
    plt.close(fig)


def plot_by_kernel(rows, outdir):
    """How the saving changes as the physics gets more realistic.

    The acoustic kernel spends a fair share of its traffic on model arrays that we are not
    compressing, so the wavefield saving gets diluted. The anisotropic kernels carry so many
    wavefield variables that the model arrays barely register, and there the saving approaches the
    full compression ratio of the format. This figure is the argument for why the work matters:
    the harder and more expensive the problem, the more a narrow wavefield format buys.
    """
    # Pick our best setting that is still at least as accurate as the published float16, and the
    # float16 point itself, so the two can be compared on equal accuracy footing.
    f16 = next(r for r in rows if r["name"] == "float16 + global scaling")
    ours_ok = [r for r in rows if r["family"] == "ours" and r["err"] <= f16["err"]]
    best = min(ours_ok, key=lambda r: r["bits"]) if ours_ok else None

    labels = list(KERNELS.keys())
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - w/2, [predicted_speedup(k, f16["bits"]) for k in labels], w,
           color="tab:red", edgecolor="k", label=f"float16, error {f16['err']:.1e}")
    if best:
        ax.bar(x + w/2, [predicted_speedup(k, best["bits"]) for k in labels], w,
               color="tab:blue", edgecolor="k",
               label=f"microscaling {best['bits']:.2f} bits, error {best['err']:.1e}")

    ax.axhline(1.0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("predicted speedup over float32")
    ax.set_title("Where the saving comes from: the more wavefield the kernel carries, the more it gains")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_speedup_by_kernel.png"), dpi=130)
    plt.close(fig)
    return f16, best


def main():
    if not os.path.exists(INFILE):
        raise SystemExit(f"{INFILE} not found. Run compare_schemes.py first.")
    rows = load_schemes(INFILE)

    # Report the memory traffic and the predicted speedup for every scheme, on every kernel.
    print("Predicted speedup if the solver is bandwidth bound.")
    print("Only the wavefield is stored narrow; the model arrays stay in float32.\n")
    header = f"{'scheme':32s} {'bits':>6s} {'error':>10s}  " + "  ".join(f"{k:>18s}" for k in KERNELS)
    print(header)
    print("." * len(header))
    for r in sorted(rows, key=lambda r: r["bits"]):
        sp = "  ".join(f"{predicted_speedup(k, r['bits']):18.2f}" for k in KERNELS)
        print(f"{r['name']:32s} {r['bits']:6.2f} {r['err']:10.2e}  {sp}")

    # The ceiling. Compressing only the wavefield leaves the model arrays sitting in float32, and
    # they go on being moved every step whatever we do to the wavefield. That puts a hard cap on
    # the speedup, and the cap is low: even a very aggressive format cannot get far past two
    # times, because beyond a point the model traffic is all that is left. The second table shows
    # what would be reachable if the model arrays were stored narrow as well, which is the obvious
    # next thing to try and, for the anisotropic kernels with their many parameter arrays, by far
    # the bigger prize.
    print("\nThe ceiling. Left: wavefield only, what we do now. Right: wavefield and model both.\n")
    print(f"{'scheme':32s} {'bits':>6s}   " +
          "  ".join(f"{k:>26s}" for k in KERNELS))
    print("." * (32 + 8 + 28 * len(KERNELS)))
    for r in sorted(rows, key=lambda r: r["bits"]):
        cells = []
        for k in KERNELS:
            a = predicted_speedup(k, r["bits"], compress_model=False)
            b = predicted_speedup(k, r["bits"], compress_model=True)
            cells.append(f"{a:11.2f} / {b:-6.2f}     ")
        print(f"{r['name']:32s} {r['bits']:6.2f}   " + "".join(cells))

    plot_efficiency_vs_error(rows, "acoustic", OUTDIR)
    f16, best = plot_by_kernel(rows, OUTDIR)

    # State the headline comparison in words, since it is the point of the whole exercise.
    if best:
        print(f"\nAt an accuracy at least as good as the published float16 scheme:")
        print(f"  float16       {f16['bits']:.2f} bits, error {f16['err']:.2e}")
        print(f"  microscaling  {best['bits']:.2f} bits, error {best['err']:.2e}")
        for k in KERNELS:
            a = predicted_speedup(k, f16["bits"])
            b = predicted_speedup(k, best["bits"])
            print(f"  {k:20s} float16 {a:.2f}x   microscaling {b:.2f}x")

    print(f"\nfigures written to ./{OUTDIR}/")
    print("Caveat: these are bandwidth bound upper bounds. A real kernel never quite reaches its "
          "bandwidth ceiling, and packing costs a few instructions per value. They compare formats "
          "fairly, but they are not a promise of wall clock speed.")


if __name__ == "__main__":
    main()
