#!/usr/bin/env python3
"""
M1 baseline — reference wavefield + shot records for the MX floating-point project.

What this combines
------------------
  * the coursework velocity model (IO_coursework_1.ipynb, Question 2): a 2D
    layered background with two circular anomalies, plus a smoothed start model;
  * the Devito *acoustic example* forward machinery
    (examples.seismic  Model / AcquisitionGeometry  +  AcousticWaveSolver).

What it produces
----------------
  A forward acoustic solve on the TRUE model, saved at FP32 *and* FP64:
    - shot_records_<prec>.npy   shape (nshots, nt, nrec)  -> data at the receivers
    - ref_wavefield_<prec>.npy  shape (nt, nx, nz)        -> full wavefield, one shot
  These are the "golden" references. Later, MX-quantised runs are compared
  against them (waveform error at receivers + wavefield error in the volume).
  It also prints the FP32-vs-FP64 relative error, which sets the natural scale
  for "how much MX error is acceptable".

This file *is* your test harness: in weeks 3-4 you insert the MX pack/unpack
layer on the wavefield arrays and re-run, then diff against these arrays.
"""

import os
import numpy as np

from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

# ---------------------------------------------------------------------------
# Configuration  (defaults match the coursework; change freely for experiments)
# ---------------------------------------------------------------------------
EXTENT      = (1000.0, 1000.0)   # physical domain size in metres (Lx, Lz)
SHAPE       = (101, 101)         # grid points (nx, nz)  -- fixed extent, so this only changes sampling
ORIGIN      = (0.0, 0.0)
NBL         = 40                 # absorbing-boundary thickness (grid points)
SPACE_ORDER = 4                  # finite-difference spatial order

T0, TN      = 0.0, 1000.0        # time window in milliseconds
F0          = 0.010              # Ricker peak frequency (kHz)
NSHOTS      = 7                  # sources evenly spaced along the surface
NREC        = 101                # receivers along the surface
SRC_DEPTH   = 20.0               # source depth (m)
REC_DEPTH   = 20.0               # receiver depth (m)
REF_SHOT    = NSHOTS // 2        # which shot's full wavefield to store as the reference volume

OUTDIR      = "m1_reference"


def build_true_velocity(shape, extent, origin=(0.0, 0.0)):
    """Coursework TRUE model (km/s): layered background + two circular anomalies."""
    nx, nz = shape
    Lx, Lz = extent
    ox, oz = origin
    x = np.linspace(ox, ox + Lx, nx, dtype=np.float64)
    z = np.linspace(oz, oz + Lz, nz, dtype=np.float64)
    X, Z = np.meshgrid(x, z, indexing="ij")

    vp = np.full((nx, nz), 2.0)                       # background
    vp[:, Z[0, :] >= oz + 0.35 * Lz] = 2.6            # layer 2
    vp[:, Z[0, :] >= oz + 0.70 * Lz] = 3.2            # layer 3
    # low-velocity lens
    xl, zl, rl = ox + 0.65 * Lx, oz + 0.45 * Lz, 0.12 * Lx
    vp[(X - xl) ** 2 + (Z - zl) ** 2 <= rl ** 2] = 1.8
    # high-velocity inclusion
    xi, zi, ri = ox + 0.35 * Lx, oz + 0.80 * Lz, 0.10 * Lx
    vp[(X - xi) ** 2 + (Z - zi) ** 2 <= ri ** 2] = 3.6
    return vp


def _source_locations():
    src = np.empty((NSHOTS, 2))
    src[:, 0] = np.linspace(0.0, EXTENT[0], NSHOTS)
    src[:, 1] = SRC_DEPTH
    return src


def _receiver_locations():
    rec = np.empty((NREC, 2))
    rec[:, 0] = np.linspace(0.0, EXTENT[0], NREC)
    rec[:, 1] = REC_DEPTH
    return rec


def run_reference(dtype):
    """Forward-model the TRUE model at `dtype`; return shot records + one full wavefield."""
    spacing = (EXTENT[0] / (SHAPE[0] - 1), EXTENT[1] / (SHAPE[1] - 1))
    vp_true = build_true_velocity(SHAPE, EXTENT, ORIGIN).astype(dtype)

    model = Model(vp=vp_true, origin=ORIGIN, shape=SHAPE, spacing=spacing,
                  nbl=NBL, space_order=SPACE_ORDER, bcs="damp", dtype=dtype)

    src = _source_locations()
    rec = _receiver_locations()
    geometry = AcquisitionGeometry(model, rec, src[0, :], T0, TN,
                                   f0=F0, src_type="Ricker")
    solver = AcousticWaveSolver(model, geometry, space_order=SPACE_ORDER)

    nt = geometry.time_axis.num
    shot_records = np.zeros((NSHOTS, nt, NREC), dtype=dtype)
    ref_wavefield = None

    for s in range(NSHOTS):
        geometry.src_positions[0, :] = src[s, :]
        if s == REF_SHOT:
            rec_out, u, _ = solver.forward(vp=model.vp, save=True)   # keep full time history
            ref_wavefield = np.array(u.data[:, NBL:-NBL, NBL:-NBL])  # strip absorbing layer
        else:
            rec_out, _, _ = solver.forward(vp=model.vp)
        shot_records[s] = rec_out.data[:]

    return dict(dt=float(model.critical_dt), nt=nt,
                shot_records=shot_records, ref_wavefield=ref_wavefield)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    results = {}
    for tag, dtype in [("fp32", np.float32), ("fp64", np.float64)]:
        print(f"\n=== {tag.upper()} reference ===")
        r = run_reference(dtype)
        results[tag] = r
        np.save(os.path.join(OUTDIR, f"shot_records_{tag}.npy"), r["shot_records"])
        np.save(os.path.join(OUTDIR, f"ref_wavefield_{tag}.npy"), r["ref_wavefield"])
        print(f"  dt = {r['dt']:.4f} ms,  nt = {r['nt']}")
        print(f"  shot_records  {r['shot_records'].shape}  ||.||2 = {np.linalg.norm(r['shot_records']):.6e}")
        print(f"  ref_wavefield {r['ref_wavefield'].shape}")

    # How far FP32 already sits from FP64 -> the natural yardstick for MX error.
    a = results["fp32"]["shot_records"].astype(np.float64)
    b = results["fp64"]["shot_records"]
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"\nFP32-vs-FP64 relative shot-record error = {rel:.3e}")
    print(f"Saved reference arrays to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
