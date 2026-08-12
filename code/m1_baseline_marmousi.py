#!/usr/bin/env python3
"""
M1 baseline on the Marmousi model.

Purpose of this file:
It produces the full precision reference results that every later microscaling (MX)
experiment is measured against. There is no MX anywhere in this file. It simply runs
an ordinary acoustic forward simulation on the Marmousi velocity model, once in
float32 and once in float64, and saves the results to disk.

Why Marmousi:
Marmousi is the standard synthetic 2D velocity model in seismic imaging. It is about
7.5 km wide and 3 km deep and contains dipping layers, faults and a complex reservoir
zone, so it produces a realistic multipath wavefield. That makes it a far more
convincing baseline than a toy layered model.

What it gives the project:
First, a set of golden reference arrays. Any MX run can be compared against these to
say how much accuracy was lost. Second, the float32 versus float64 error printed at
the end. That number is the natural yardstick: it says how far float32 already sits
from a more exact answer, so an MX setting whose error is around that level is
effectively as good as float32.

Run with: python3 m1_baseline_marmousi.py
The Marmousi velocity file is downloaded automatically on the first run.
"""

import os
import urllib.request
import numpy as np

# Agg is a backend that writes image files without needing a screen, so the plots work
# the same whether this runs on a laptop or on a headless cluster node.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver


# Location of the Marmousi velocity model. It is a raw binary file holding 1601 by 401
# float32 values, hosted in the devitocodes/data repository.
DATA_PATH = "data"
_MARM_REL = "Simple2D/vp_marmousi_bi"
_MARM_URL = "https://raw.githubusercontent.com/devitocodes/data/master/Simple2D/vp_marmousi_bi"
_MARM_SHAPE   = (1601, 401)
_MARM_SPACING = (7.5, 7.5)          # grid spacing in metres, the same in x and in z


def ensure_marmousi_data():
    """Download the Marmousi velocity binary the first time it is needed.

    Returns the path to the local copy. On later runs the file is already there,
    so nothing is downloaded.
    """
    dst = os.path.join(DATA_PATH, _MARM_REL)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print("downloading Marmousi vp ...")
        urllib.request.urlretrieve(_MARM_URL, dst)
    return dst


# Simulation settings. These are the knobs to turn when running experiments.
NBL         = 20          # thickness of the absorbing boundary, in grid points
SPACE_ORDER = 4           # order of the finite difference stencil in space
T0, TN      = 0.0, 3000.0 # start and end of the recording window, in milliseconds
F0          = 0.025       # peak frequency of the Ricker source, 25 Hz written in kHz
NSHOTS      = 5           # number of sources, spread evenly across the surface
NREC        = 500         # number of receivers, spread evenly across the surface
SRC_DEPTH   = 20.0        # depth of the sources, in metres
REC_DEPTH   = 20.0        # depth of the receivers, in metres
REF_SHOT    = NSHOTS // 2 # the shot whose wavefield snapshot we keep

OUTDIR      = "m1_marmousi_reference"


def build_model(dtype):
    """Load the Marmousi velocity field and wrap it in a Devito Model at the given precision.

    The binary is read here directly rather than through Devito's demo_model helper.
    That helper hardcodes float32 for Marmousi, so asking it for float64 would quietly
    hand back a float32 model and the float64 reference would be fake. Reading the file
    ourselves means the requested dtype is genuinely honoured.

    The crop v[301:-300, :] reproduces exactly what the Devito preset does. It trims the
    edges of the raw 1601 by 401 section down to 1000 by 401, which keeps the interesting
    geology while making the simulation cheaper.
    """
    path = ensure_marmousi_data()
    v = np.fromfile(path, dtype="float32").reshape(_MARM_SHAPE)
    v = v[301:-300, :].astype(dtype)
    return Model(vp=v, origin=(0.0, 0.0), shape=v.shape, spacing=_MARM_SPACING,
                 nbl=NBL, space_order=SPACE_ORDER, bcs="damp", dtype=dtype)


def run_reference(dtype):
    """Run the forward simulation at one precision and collect the reference data.

    For every shot it fires the source, propagates the wave through the model, and
    records what arrives at the receivers. The shot records are the main output. One
    wavefield snapshot is also kept as a sanity image. The full time history is
    deliberately not stored, because on Marmousi it would run to several gigabytes.
    """
    model = build_model(dtype)
    Lx = model.spacing[0] * (model.shape[0] - 1)   # physical width of the model in metres

    # Lay the sources and the receivers out along the surface, evenly spaced.
    src = np.empty((NSHOTS, 2)); src[:, 0] = np.linspace(0, Lx, NSHOTS); src[:, 1] = SRC_DEPTH
    rec = np.empty((NREC, 2));   rec[:, 0] = np.linspace(0, Lx, NREC);   rec[:, 1] = REC_DEPTH

    # The geometry object holds the source and receiver positions plus the time axis.
    # The solver turns the wave equation into a compiled finite difference kernel.
    geometry = AcquisitionGeometry(model, rec, src[0, :], T0, TN, f0=F0, src_type="Ricker")
    solver = AcousticWaveSolver(model, geometry, space_order=SPACE_ORDER)

    nt = geometry.time_axis.num                    # how many time steps Devito will take
    shot_records = np.zeros((NSHOTS, nt, NREC), dtype=dtype)
    snapshot = None

    for s in range(NSHOTS):
        # Move the source to the next position, then propagate the wave for the whole
        # recording window. rec_out holds what the receivers picked up.
        geometry.src_positions[0, :] = src[s, :]
        rec_out, u, _ = solver.forward(vp=model.vp)
        shot_records[s] = rec_out.data[:]

        if s == REF_SHOT:
            # Devito keeps only the last few time slices in a small rotating buffer, so
            # we take the slice with the largest norm as the final wavefield, then strip
            # off the absorbing boundary so that only the physical region is left.
            bufs = np.array(u.data)
            k = int(np.argmax([np.linalg.norm(b) for b in bufs]))
            snapshot = np.array(bufs[k][NBL:-NBL, NBL:-NBL])

    # The velocity field and the source and receiver positions are returned as well, purely
    # so that the plots below can show what the simulation actually saw.
    vp_plot = np.array(model.vp.data[NBL:-NBL, NBL:-NBL])

    return dict(shape=model.shape, dt=float(model.critical_dt), nt=nt,
                Lx=Lx, Lz=model.spacing[1] * (model.shape[1] - 1),
                shot_records=shot_records, snapshot=snapshot,
                vp=vp_plot, src=src, rec=rec)


def make_plots(r32, r64, rel):
    """Draw the four figures that describe the baseline.

    All of the arrays are stored as (x, z), meaning the first axis runs across the model
    and the second runs downwards. Images are therefore transposed before plotting so that
    depth appears on the vertical axis, which is how seismic sections are normally shown.
    """
    Lx_km, Lz_km = r32["Lx"] / 1000.0, r32["Lz"] / 1000.0
    tmax = TN / 1000.0                      # recording length in seconds

    # Figure 1. The velocity model, with the sources and receivers drawn on top. This is the
    # ground truth of the subsurface: everything the simulation produces comes from it.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(r32["vp"].T, cmap="viridis", aspect="auto",
                   extent=[0, Lx_km, Lz_km, 0])
    ax.scatter(r32["src"][:, 0] / 1000.0, r32["src"][:, 1] / 1000.0,
               marker="*", s=180, c="red", edgecolors="k", label="sources", zorder=3)
    ax.scatter(r32["rec"][::25, 0] / 1000.0, r32["rec"][::25, 1] / 1000.0,
               marker="v", s=22, c="white", edgecolors="k", label="receivers", zorder=3)
    ax.set_xlabel("distance (km)"); ax.set_ylabel("depth (km)")
    ax.set_title("Marmousi velocity model with acquisition geometry")
    ax.legend(loc="lower right")
    fig.colorbar(im, ax=ax, label="velocity (km/s)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig1_velocity_model.png"), dpi=130)
    plt.close(fig)

    # Figure 2. The shot record for the middle shot. Each column is one receiver and time runs
    # downwards, so the curved bands are the wave arriving at different receivers at different
    # times. This is the data an imaging algorithm would actually be given.
    shot = r32["shot_records"][REF_SHOT]
    clip = 0.1 * np.abs(shot).max()         # clip the colour scale so weak later arrivals show up
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(shot, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip,
              extent=[0, Lx_km, tmax, 0])
    ax.set_xlabel("receiver position (km)"); ax.set_ylabel("time (s)")
    ax.set_title(f"Shot record, float32 (shot {REF_SHOT})")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig2_shot_record.png"), dpi=130)
    plt.close(fig)

    # Figure 3. A snapshot of the wavefield near the end of the simulation. It shows the wave
    # spread through the model, scattered by the layers and faults.
    snap = r32["snapshot"]
    clip = 0.15 * np.abs(snap).max()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(snap.T, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip,
              extent=[0, Lx_km, Lz_km, 0])
    ax.set_xlabel("distance (km)"); ax.set_ylabel("depth (km)")
    ax.set_title("Wavefield snapshot, float32")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig3_wavefield_snapshot.png"), dpi=130)
    plt.close(fig)

    # Figure 4. How far float32 already sits from float64. The two traces lie on top of each
    # other, and the difference below them is tiny. That tiny difference is the yardstick: an
    # MX setting only counts as being as good as float32 once its own error reaches this level.
    itrace = NREC // 2
    t = np.linspace(0, tmax, r32["nt"])
    a = r32["shot_records"][REF_SHOT][:, itrace].astype(np.float64)
    b = r64["shot_records"][REF_SHOT][:, itrace]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, b, lw=1.6, c="k", label="float64")
    axes[0].plot(t, a, lw=0.9, c="tab:red", ls="--", label="float32")
    axes[0].set_ylabel("amplitude"); axes[0].legend(loc="upper right")
    axes[0].set_title(f"float32 against float64, single trace (receiver {itrace})")
    axes[1].plot(t, a - b, lw=0.9, c="tab:blue")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("difference")
    axes[1].set_title(f"difference, relative error over the whole record {rel:.2e}")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig4_fp32_vs_fp64.png"), dpi=130)
    plt.close(fig)

    print(f"figures written to ./{OUTDIR}/")


def main():
    """Produce the float32 and float64 references and report how far apart they are."""
    os.makedirs(OUTDIR, exist_ok=True)
    results = {}

    # Run the same simulation twice, once in each precision, and save both to disk.
    for tag, dtype in [("fp32", np.float32), ("fp64", np.float64)]:
        print(f"\n{tag.upper()} reference")
        r = run_reference(dtype)
        results[tag] = r
        np.save(os.path.join(OUTDIR, f"shot_records_{tag}.npy"), r["shot_records"])
        np.save(os.path.join(OUTDIR, f"snapshot_{tag}.npy"), r["snapshot"])
        print(f"  grid {r['shape']}  domain {r['Lx']/1000:.2f} by {r['Lz']/1000:.2f} km")
        print(f"  dt {r['dt']:.4f} ms  nt {r['nt']}")
        print(f"  shot_records {r['shot_records'].shape}  norm {np.linalg.norm(r['shot_records']):.6e}")

    # The gap between float32 and float64 is the yardstick for the whole project. It says
    # how much error float32 already carries on its own, which is the level an MX setting
    # must reach before we can fairly call it as accurate as float32.
    a = results["fp32"]["shot_records"].astype(np.float64)
    b = results["fp64"]["shot_records"]
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"\nfloat32 versus float64 relative shot record error {rel:.3e}")
    print(f"reference arrays saved to ./{OUTDIR}/")

    # Draw the figures that describe the baseline.
    make_plots(results["fp32"], results["fp64"], rel)


if __name__ == "__main__":
    main()
