#!/usr/bin/env python3
"""
Microscaling (MX) accuracy experiment on a cropped Marmousi model.

Purpose of this file:
It answers the central question of the project, which is how far the wavefield can be
squeezed below float32 before the result stops being trustworthy. It does this by
running the same acoustic simulation as the baseline, but storing the wavefield in an
MX block format between time steps, and then measuring how far the result drifts away
from a full precision run.

How MX is applied here:
Every time step the ordinary Devito kernel advances the wave in float32. The freshly
computed wavefield is then packed into MX and unpacked straight back to float32, and
that unpacked wavefield is fed into the next step. So the arithmetic always stays in
native precision and only the storage is narrowed, which is exactly the setting the
project is about. Because the unpacked field goes straight back into the next step,
quantisation error can build up over time, and capturing that build up is one of the
main goals here.

Why the model is cropped:
The MX harness has to step the solver one time step at a time from Python so that it
can pack and unpack in between. On full Marmousi that Python loop is far too slow for
a parameter sweep. So a smaller window is cut out of the real Marmousi section, at the
same 7.5 m sampling and with the same real geology, and the recording is shortened.
The sweep then finishes in well under a minute while still exercising a realistic
multipath wavefield. The workflow is to develop and sweep settings here, then confirm
the chosen settings on the full Marmousi baseline.

Scope:
This file measures accuracy only. Savings in bytes, bandwidth and runtime are not
measured here; they come from a separate bandwidth model and a micro kernel.

Run with: python3 mx_experiment_marmousi.py
The Marmousi velocity file is downloaded automatically on the first run.
"""

import os
import urllib.request
import numpy as np

# Agg is a backend that writes image files without needing a screen, so the plots work the
# same whether this runs on a laptop or on a headless cluster node.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from devito import TimeFunction, Eq, Operator, solve
from examples.seismic import Model, AcquisitionGeometry, Receiver


# Where the Marmousi velocity binary lives, and which window of it we cut out.
# The crop indices refer to the standard 1000 by 401 model, which is what you get after
# the usual edge trimming of the raw 1601 by 401 file.
DATA_FILE = "data/Simple2D/vp_marmousi_bi"
DATA_URL  = "https://raw.githubusercontent.com/devitocodes/data/master/Simple2D/vp_marmousi_bi"

CROP_X  = (340, 680)    # 340 grid points across, roughly 2.54 km wide
CROP_Z  = (20, 220)     # 200 grid points down, roughly 1.49 km deep
SPACING = (7.5, 7.5)    # grid spacing in metres, unchanged from real Marmousi
NBL = 20                # thickness of the absorbing boundary, in grid points
SPACE_ORDER = 4         # order of the finite difference stencil in space

# Acquisition settings. A single shot is enough for an accuracy study, and the short
# recording window is what keeps the Python step loop fast enough to sweep.
T0, TN = 0.0, 800.0     # start and end of the recording window, in milliseconds
F0     = 0.025          # peak frequency of the Ricker source, 25 Hz written in kHz
NREC   = 250            # number of receivers along the surface
SRC_DEPTH = 20.0        # depth of the source, in metres
REC_DEPTH = 20.0        # depth of the receivers, in metres

OUTDIR = "mx_marmousi_reference"


def load_cropped_vp(dtype=np.float32):
    """Fetch the Marmousi velocity file if needed, then return the cropped window.

    The file is read as raw float32, reshaped to the full 1601 by 401 section, trimmed
    at the edges the same way the standard Devito preset does, and finally cut down to
    the window set by CROP_X and CROP_Z.
    """
    if not os.path.exists(DATA_FILE):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        print("downloading Marmousi vp ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    v = np.fromfile(DATA_FILE, dtype="float32").reshape(1601, 401)[301:-300, :]
    return v[CROP_X[0]:CROP_X[1], CROP_Z[0]:CROP_Z[1]].astype(dtype)


def quantize_dequantize(x, block_size=32, mantissa_bits=3, rounding="nearest", rng=None):
    """Emulate microscaling storage: pack the array into MX, then unpack it straight back.

    This is the heart of the experiment. Both halves happen in one call, so what goes in
    is a float32 array and what comes out is a float32 array that has been forced through
    a much narrower representation and has therefore lost precision.

    How the format works. The array is split into blocks of block_size consecutive values.
    Every block gets one shared exponent, taken as the power of two just below the largest
    magnitude in that block. Each value in the block is then divided by that shared scale
    and rounded onto a grid of mantissa_bits bits. So a block stores one exponent plus a
    set of narrow codes, instead of a full exponent for every single value. That shared
    exponent is what makes the format compact, and it is also why block size matters: a
    block that spans a wide range of magnitudes cannot be represented well by one scale.

    Rounding. With nearest rounding each value goes to the closest grid point, which is
    accurate per value but leaves a small systematic bias that can accumulate over
    thousands of steps. With stochastic rounding a value lands on the neighbouring grid
    points at random, with probability set by how close it is to each, which removes that
    bias at the cost of extra noise. Whether this helps for the wave equation is one of
    the open questions of the project.

    Note the intermediate arithmetic is done in float64 so that the rounding itself does
    not add noise of its own. The value returned is cast back to the dtype of the input.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Flatten the field and pad it so that it divides evenly into blocks.
    shp = x.shape
    xf = x.reshape(-1).astype(np.float64)
    n = xf.size
    pad = (-n) % block_size
    if pad:
        xf = np.concatenate([xf, np.zeros(pad)])
    blk = xf.reshape(-1, block_size)

    # Give every block its shared exponent, taken from the largest magnitude it holds.
    # Blocks that are entirely zero are left alone, since they have no scale to speak of.
    amax = np.max(np.abs(blk), axis=1, keepdims=True)
    nz = amax[:, 0] > 0
    scale = np.zeros_like(amax)
    scale[nz, 0] = 2.0 ** np.floor(np.log2(amax[nz, 0]))

    # Normalise each value by its block scale, then express it on a grid whose spacing is
    # set by the number of mantissa bits we are allowed to keep.
    step = 2.0 ** (-mantissa_bits)
    y = np.zeros_like(blk)
    y[nz] = blk[nz] / scale[nz]
    t = y / step

    # Snap onto the grid, either to the closest point or randomly between the two
    # neighbouring points.
    if rounding == "nearest":
        q = np.round(t)
    elif rounding == "stochastic":
        q = np.floor(t + rng.random(t.shape))
    else:
        raise ValueError(rounding)

    # Unpack: rebuild an ordinary float from the narrow code and the shared block scale,
    # undo the padding, and restore the original shape and dtype.
    out = np.zeros_like(blk)
    out[nz] = q[nz] * step * scale[nz]
    return out.reshape(-1)[:n].reshape(shp).astype(x.dtype)


class Harness:
    """The forward solver, set up so that we can interrupt it after every single time step.

    Devito would normally run all the time steps inside one compiled kernel, which gives
    us no chance to touch the wavefield in between. Here the operator is built once and
    then applied one step at a time, which lets us pack and unpack the wavefield between
    steps. The setup cost is paid once and the same harness is reused for every setting in
    the sweep.
    """

    def __init__(self, dtype=np.float32):
        # Build the cropped Marmousi model with an absorbing boundary around it.
        vp = load_cropped_vp(dtype)
        self.shape = vp.shape
        self.model = Model(vp=vp, origin=(0.0, 0.0), shape=vp.shape, spacing=SPACING,
                           nbl=NBL, space_order=SPACE_ORDER, bcs="damp", dtype=dtype)

        # Put the receivers along the surface and the single source in the middle.
        Lx = SPACING[0] * (vp.shape[0] - 1)
        rec_c = np.empty((NREC, 2)); rec_c[:, 0] = np.linspace(0, Lx, NREC); rec_c[:, 1] = REC_DEPTH
        self.geometry = AcquisitionGeometry(self.model, rec_c,
                                            np.array([Lx / 2, SRC_DEPTH], dtype=dtype),
                                            T0, TN, f0=F0, src_type="Ricker")
        self.dt = self.model.critical_dt   # largest stable time step for this grid
        self.nt = self.geometry.time_axis.num

        # u is the wavefield. It is a TimeFunction, so Devito keeps a few time slices of it
        # in a small rotating buffer rather than the whole history.
        self.u = TimeFunction(name="u", grid=self.model.grid, time_order=2, space_order=SPACE_ORDER)

        # The acoustic wave equation, with the damping term that feeds the absorbing boundary.
        # solve() rearranges it into an explicit update rule for the next time slice.
        pde = self.model.m * self.u.dt2 - self.u.laplace + self.model.damp * self.u.dt
        src = self.geometry.src
        self.rec = Receiver(name="rec", grid=self.model.grid,
                            time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)

        # The operator bundles three jobs into one compiled kernel: advance the wavefield,
        # add the source into it, and read the wavefield out at the receiver positions.
        self.op = Operator([Eq(self.u.forward, solve(pde, self.u.forward))]
                           + src.inject(field=self.u.forward, expr=src * self.dt ** 2 / self.model.m)
                           + self.rec.interpolate(expr=self.u))

    def run(self, quant=None, quantize_every=1, store_wavefield=False):
        """Step the simulation through the whole recording window.

        If quant is None the run is a plain full precision simulation, which is how the
        reference is produced. If quant is given, the wavefield is pushed through MX
        storage after each step and the result is fed back into the next step.

        Returns the shot record, the final wavefield, and optionally the full time history
        of the wavefield, which is what the error growth curve is computed from.
        """
        self.u.data[:] = 0.0
        hist = np.zeros((self.nt, self.shape[0], self.shape[1]), np.float64) if store_wavefield else None

        for i in range(self.nt - 1):
            # Advance the wave by exactly one time step.
            self.op.apply(time_m=i, time_M=i, dt=self.dt)

            # Devito rotates through three time slices, so this picks out the one that has
            # just been written.
            buf = (i + 1) % 3

            if quant is not None and (i + 1) % quantize_every == 0:
                # This is where the wavefield is put into MX storage and taken back out.
                # Only the physical interior is quantised. The absorbing boundary is left
                # alone, because the large damped values living there make the per step
                # requantisation unstable and they are not part of the result anyway.
                self.u.data[buf][NBL:-NBL, NBL:-NBL] = quant(self.u.data[buf][NBL:-NBL, NBL:-NBL])

            if store_wavefield:
                hist[i + 1] = self.u.data[buf][NBL:-NBL, NBL:-NBL]

        final_slice = np.array(self.u.data[(self.nt - 1) % 3][NBL:-NBL, NBL:-NBL])
        return self.rec.data.copy(), final_slice, hist


def rel_l2(a, b):
    """Relative error of a against the reference b, measured in the L2 norm.

    A value of 1e-3 means the MX result differs from the reference by about a tenth of a
    percent of the reference's own size.
    """
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def snr_db(a, b):
    """Signal-to-noise ratio of a against reference b, in decibels.

    The reference is the signal and the departure of a from it is the noise, so this is just the
    relative L2 error expressed the way the seismic literature usually reports it:

        SNR(dB) = 20 * log10( ||b|| / ||a - b|| ) = -20 * log10( rel_l2(a, b) ).

    Every factor of ten reduction in relative error is 20 dB, which makes the gaps between
    schemes easy to read. A perfect match returns infinity.
    """
    e = rel_l2(a, b)
    return np.inf if e == 0 else -20.0 * np.log10(e)


def error_growth(hist_mx, hist_ref):
    """Relative wavefield error at each stored time step, so the accumulation can be plotted.

    Both arguments are stacks of wavefield snapshots taken at the same steps; this returns one
    relative L2 error per snapshot, which is the curve plot_error_growth draws.
    """
    a = np.asarray(hist_mx, np.float64)
    b = np.asarray(hist_ref, np.float64)
    n = min(a.shape[0], b.shape[0])
    out = np.zeros(n)
    for k in range(n):
        denom = np.linalg.norm(b[k])
        out[k] = np.linalg.norm(a[k] - b[k]) / denom if denom > 0 else 0.0
    return out
    """Relative wavefield error at every time step, which shows how the error accumulates.

    This is the important curve for the project. A single pass through MX only loses a
    small amount of accuracy, but because the quantised field is fed back into the next
    step, that small loss can grow over thousands of steps. This function makes that
    growth visible.
    """
    num = np.linalg.norm((hist_mx - hist_ref).reshape(hist_ref.shape[0], -1), axis=1)
    den = np.linalg.norm(hist_ref.reshape(hist_ref.shape[0], -1), axis=1)
    out = np.zeros_like(num)
    g = den > 0                      # skip the early steps where the field is still all zero
    out[g] = num[g] / den[g]
    return out


def plot_accuracy_vs_bits(results, fp32_own_error, outdir):
    """The headline figure: how the error changes as the stored values get narrower.

    Each point is one MX setting. The horizontal line is the error float32 itself already
    carries, so any point below that line is a setting whose accuracy loss is no worse than
    what float32 loses anyway. Where the curve crosses that line is the answer to the central
    question of the project, which is how far the wavefield can safely be squeezed.

    The float32 error is measured on this same cropped model, not borrowed from the full
    Marmousi baseline. The two are not interchangeable: the cropped run takes far fewer time
    steps, so float32 accumulates far less error of its own, and its bar sits much lower.
    Drawing the full model's bar on a cropped model's results would flatter the MX numbers.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for rd in ROUNDING:
        bits = [mb for (bs, mb, r) in results if r == rd]
        errs = [results[(bs, mb, r)]["rec"] for (bs, mb, r) in results if r == rd]
        order = np.argsort(bits)
        ax.plot(np.array(bits)[order], np.array(errs)[order],
                marker="o", lw=1.8, label=f"{rd} rounding")

    ax.axhline(fp32_own_error, color="k", ls="--", lw=1.2)
    ax.text(ax.get_xlim()[1], fp32_own_error * 1.25,
            "float32 own error", ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("mantissa bits kept per value")
    ax.set_ylabel("relative error of the shot record")
    ax.set_title("Accuracy against storage precision")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_accuracy_vs_bits.png"), dpi=130)
    plt.close(fig)


def plot_error_growth(growth, dt, outdir):
    """How the error builds up as the simulation runs on.

    A single pass through MX loses only a little accuracy, but the quantised wavefield is fed
    back into the next step, so that small loss compounds. This plot is what makes the
    accumulation visible, and it is also where the two rounding modes can be compared: nearest
    rounding leaves a systematic bias that can build up, while stochastic rounding trades that
    bias for extra noise.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for rd, curve in growth.items():
        t = np.arange(curve.size) * dt / 1000.0        # convert milliseconds to seconds
        ax.plot(t[1:], curve[1:], lw=1.6, label=f"{rd} rounding")
    ax.set_yscale("log")
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("relative error of the wavefield")
    ax.set_title(f"Error accumulation over time, {GROWTH_BITS} mantissa bits")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_error_growth.png"), dpi=130)
    plt.close(fig)


def plot_wavefield_comparison(snap_ref, snaps, outdir):
    """The reference wavefield, the MX wavefield at two bit widths, and what MX got wrong at each.

    The reference and the MX wavefields are shown on the same amplitude scale, where at a usable
    bit width they look the same. The difference panels are the interesting ones, and each is
    drawn on its own much finer scale, stated in its colour bar, rather than on the wavefield
    scale. On the wavefield scale the residual is invisible, which tells us only that it is small;
    on its own scale its structure shows, and whether that structure follows the wavefronts, which
    would mean coherent distortion, or is spread out, which would mean harmless grain. Two bit
    widths are shown so the residual can be watched growing and taking shape as the format is made
    more aggressive.
    """
    Lx_km = SPACING[0] * (snap_ref.shape[0] - 1) / 1000.0
    Lz_km = SPACING[1] * (snap_ref.shape[1] - 1) / 1000.0
    ext = [0, Lx_km, Lz_km, 0]
    clip = 0.15 * np.abs(snap_ref).max()
    bits_list = sorted(snaps.keys(), reverse=True)      # e.g. 14 then 10

    nrow = 1 + 2 * len(bits_list)
    fig, axes = plt.subplots(nrow, 1, figsize=(9, 3.0 * nrow))

    def show(ax, data, title, c, cmap="seismic", fmt=None):
        im = ax.imshow(np.asarray(data).T, cmap=cmap, aspect="auto", vmin=-c, vmax=c, extent=ext)
        ax.set_title(title, fontsize=11); ax.set_ylabel("depth (km)")
        cb = fig.colorbar(im, ax=ax)
        if fmt:
            cb.formatter.set_powerlimits((0, 0)); cb.update_ticks()

    show(axes[0], snap_ref, "reference wavefield, full precision", clip)
    row = 1
    for b in bits_list:
        snap_mx = snaps[b]
        diff = snap_mx.astype(np.float64) - snap_ref.astype(np.float64)
        dclip = np.abs(diff).max() + 1e-30              # each difference on its own fine scale
        show(axes[row], snap_mx, f"MX wavefield, {b} mantissa bits", clip)
        show(axes[row + 1], diff,
             f"difference at {b} bits (scale +/- {dclip:.1e}, {clip / dclip:.0f}x finer)",
             dclip, fmt=True)
        row += 2

    axes[-1].set_xlabel("distance (km)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_wavefield_comparison.png"), dpi=130)
    plt.close(fig)


WIGGLE_SATURATION = 40   # how far the strong direct arrival is driven past the display range,
                         # so it clips flat and the much weaker reflections underneath become
                         # visible; this is where the interesting differences between schemes are
WIGGLE_CLIP = 1.3        # maximum trace deflection, in trace-spacing units, at which a wiggle
                         # saturates


def _wiggle(ax, data, t, gain=1.9, color="k", norm=None, clip=WIGGLE_CLIP):
    """Draw a variable-area wiggle panel: one vertical trace per column, time increasing
    downward, positive lobes filled. This is the conventional way seismic data is displayed.

    Amplitudes are scaled by norm; if norm is None the panel's own maximum is used. Deflections
    beyond clip trace-spacings are saturated, which is deliberate: the direct arrival is far
    stronger than the reflections, so it is driven off scale and clipped flat, letting the weak
    reflected and refracted energy that actually carries the image show through. Pass clip=None
    to disable saturation."""
    ntr = data.shape[1]
    if norm is None:
        norm = np.max(np.abs(data)) + 1e-30
    for i in range(ntr):
        tr = data[:, i] / norm * gain
        if clip is not None:
            tr = np.clip(tr, -clip, clip)
        ax.plot(i + tr, t, color=color, lw=0.4)
        ax.fill_betweenx(t, i, i + tr, where=(tr > 0), color=color, lw=0)
    ax.set_ylim(t[-1], t[0])
    ax.set_xlim(-1.6, ntr + 0.6)
    ax.set_xticks([])


def plot_trace_comparison(rec_ref, rec_mx, bits, outdir):
    """A small receiver gather shown as wiggle traces, full precision against MX.

    The single overlaid trace is replaced by a wiggle display, the conventional seismic format:
    a band of adjacent traces, time increasing downward, positive lobes filled. The strong direct
    arrival is deliberately driven off scale and clipped flat, so the weak reflected and refracted
    energy underneath, which is what carries the image and what we actually care about, becomes
    visible and can be compared between full precision and MX.
    """
    nt, nrec = rec_ref.shape
    band = slice(nrec // 2 - 30, nrec // 2 + 30, 2)     # about 30 central traces
    t = np.linspace(0, TN / 1000.0, nt)
    ref = np.asarray(rec_ref)[:, band].astype(np.float64)
    mx = np.asarray(rec_mx)[:, band].astype(np.float64)
    # Normalise so the strong direct arrival is driven well past the display range and clips
    # flat, which brings out the weak reflections where the schemes actually differ.
    common = (np.max(np.abs(ref)) + 1e-30) / WIGGLE_SATURATION

    fig, axes = plt.subplots(1, 2, figsize=(9, 6.2), sharey=True)
    _wiggle(axes[0], ref, t, norm=common)
    axes[0].set_title("full precision"); axes[0].set_ylabel("time (s)")
    _wiggle(axes[1], mx, t, norm=common)
    axes[1].set_title(f"MX, {bits} mantissa bits")
    for ax in axes:
        ax.set_xlabel("trace")
    fig.suptitle("Receiver gather: full precision against MX")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig4_trace_comparison.png"), dpi=130)
    plt.close(fig)


# The sweep. These are the settings that get varied, and they are the main thing to change
# when exploring. Block size sets how many values share one exponent. Mantissa bits sets how
# much precision each value keeps and therefore how much storage is saved. QEVERY controls
# how often the wavefield is pushed through MX: 1 means every step, which is the most
# demanding case, and a larger number means fewer passes and so less accumulated error.
BLOCKS   = [32]
BITS     = [4, 6, 8, 10, 12, 14, 16]
ROUNDING = ["nearest", "stochastic"]
QEVERY   = 1
GROWTH_BITS = 8          # the bit width at which the error growth curve is recorded
WAVEFIELD_FIG_BITS = 14  # the bit width used for the wavefield and trace comparison figures;
                         # kept high so the MX and reference panels are genuinely hard to
                         # tell apart, which is the point those figures make
WAVEFIELD_DIFF_BITS = [14, 10]  # the wavefield comparison shows these bit widths side by side,
                                # each with its own difference panel on its own fine scale, so
                                # the residual structure is visible rather than washed out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)

    # Build the solver once. Every configuration in the sweep reuses it.
    H = Harness(dtype=np.float32)
    print(f"cropped Marmousi grid={H.shape}  domain="
          f"{SPACING[0]*(H.shape[0]-1)/1000:.2f}x{SPACING[1]*(H.shape[1]-1)/1000:.2f} km  "
          f"nt={H.nt}  dt={H.dt:.4f} ms")

    # The reference run, with quantisation switched off. It goes through exactly the same
    # stepping driver as the MX runs, so any difference we measure later comes purely from
    # quantisation and not from some other detail of how the loop is driven.
    rec_ref, snap_ref, hist_ref = H.run(quant=None, store_wavefield=True)
    np.save(os.path.join(OUTDIR, "snapshot_ref.npy"), snap_ref)
    print(f"reference shot-record norm = {np.linalg.norm(rec_ref):.6e}")

    # Work out how much error float32 itself already carries, by running the identical
    # simulation in float64 and comparing. This is the yardstick: an MX setting whose error
    # reaches this level is, in practice, as accurate as float32.
    # It has to be measured on this same cropped model. The full Marmousi baseline takes
    # several times as many time steps, so float32 accumulates much more error there and its
    # bar sits much higher. Borrowing that number here would make the MX results look better
    # than they are.
    H64 = Harness(dtype=np.float64)
    rec_64, _, _ = H64.run(quant=None)
    fp32_own_error = rel_l2(rec_ref, rec_64)
    print(f"float32 own error against float64 = {fp32_own_error:.3e}\n")

    # Now run every combination of block size, bit width and rounding mode, and report how
    # far each one drifts from the reference. rec_relerr is the error in what the receivers
    # would actually record, which is the quantity that matters for imaging.
    print(" block bits  rounding     rec_relerr   final_wavefield_relerr")
    print(" " + "." * 58)
    growth = {}
    results = {}          # every sweep result, keyed by setting, used to draw the accuracy plot
    example = {}          # one MX run kept in full, so the wavefield and trace can be compared
    wf_snaps = {}         # MX wavefield snapshots at each difference bit width

    for bs in BLOCKS:
        for mb in BITS:
            for rd in ROUNDING:
                # The full time history is only kept for the one bit width we want the
                # growth curve at, since storing it for every run would be wasteful.
                want_hist = (mb == GROWTH_BITS)
                quant = (lambda a, bs=bs, mb=mb, rd=rd: quantize_dequantize(a, bs, mb, rd, rng))
                rec_mx, snap_mx, hist_mx = H.run(quant=quant, quantize_every=QEVERY,
                                                 store_wavefield=want_hist)
                e_rec = rel_l2(rec_mx, rec_ref)
                e_wf  = rel_l2(snap_mx, snap_ref)
                print(f" {bs:5d} {mb:4d}  {rd:11s}  {e_rec:.3e}    {e_wf:.3e}")

                results[(bs, mb, rd)] = dict(rec=e_rec, wf=e_wf)
                if want_hist:
                    growth[rd] = error_growth(hist_mx, hist_ref)
                # Keep the trace-comparison run at the higher bit width, and keep the wavefield
                # snapshots at each of the difference bit widths so they can be shown side by side.
                if rd == "nearest":
                    if mb == WAVEFIELD_FIG_BITS:
                        example = dict(rec=rec_mx, snap=snap_mx, bits=mb)
                    if mb in WAVEFIELD_DIFF_BITS:
                        wf_snaps[mb] = np.asarray(snap_mx).copy()

    # Print how the error builds up over time, comparing the two rounding modes at the same
    # bit width. This is where it becomes visible that a tiny per step error can turn into a
    # large error by the end of the simulation.
    if growth:
        idx = np.linspace(1, H.nt - 1, 6).astype(int)
        print(f"\nerror growth at {GROWTH_BITS} mantissa bits (relative wavefield error vs time):")
        print("   step:    " + "  ".join(f"{i:6d}" for i in idx))
        for rd in ROUNDING:
            if rd in growth:
                np.save(os.path.join(OUTDIR, f"growth_{rd}.npy"), growth[rd])
                print(f"   {rd[:7]:7s}: " + "  ".join(f"{growth[rd][i]:.4f}" for i in idx))

    # Draw the figures. The first two are the ones that carry the findings: how accuracy falls
    # away as the values get narrower, and how the error compounds as the simulation runs on.
    # The last two show what that error actually looks like in the wavefield and in the data.
    plot_accuracy_vs_bits(results, fp32_own_error, OUTDIR)
    if growth:
        plot_error_growth(growth, H.dt, OUTDIR)
    if wf_snaps:
        plot_wavefield_comparison(snap_ref, wf_snaps, OUTDIR)
    if example:
        plot_trace_comparison(rec_ref, example["rec"], example["bits"], OUTDIR)

    print(f"\nSaved curves, reference snapshot and figures to ./{OUTDIR}/")
    print("NOTE: accuracy only. Bytes, bandwidth and runtime savings come from the "
          "separate bandwidth model and micro kernel. Confirm final settings on full Marmousi.")


if __name__ == "__main__":
    main()
