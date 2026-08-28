# Microscaling Storage Formats for Seismic Wave Propagation

An SLB × Imperial College London industry-sponsored research project (IRP).

This repository applies **microscaling (MX) block floating-point** — a storage format from
machine learning — to the wavefield of a finite-difference seismic solver, and measures what it
costs in accuracy and what it saves in memory. The headline result: as a storage format MX is
both cheaper and more accurate than FP16, on the acoustic and elastic kernels.

The interim report (`MX_Interim_Report_EN.docx`) is the write-up; this README is how to
reproduce the numbers and figures in it.

---

## What is being measured

- **Accuracy** — the wavefield is pushed through MX storage after every time step (packed to MX,
  unpacked to FP32, fed to the next step); the arithmetic stays FP32, only the storage is
  narrowed. Error is the relative L2 of the shot record against a full-precision run, reported
  as **signal-to-noise ratio in decibels**, `SNR(dB) = -20·log10(relative error)` — higher is
  better, and a factor of ten in error is 20 dB.
- **Cost** — bits stored per wavefield value. The solver is memory-bandwidth-bound, so fewer
  bits stored means fewer bytes moved per step.
- **Efficiency** — predicted from memory traffic, not wall-clock. MX is emulated in numpy, so
  timing would measure the emulation rather than the format; a bandwidth model is the honest
  quantity and does not depend on the GPU.

The velocity model throughout is **Marmousi** (a cropped window for the accuracy sweeps, since
they repeat many runs).

---

## Requirements

```bash
pip install devito pytest matplotlib numpy
```

`pytest` is required even though no tests are run here — `examples.seismic` fails to import
without it. Node.js is only needed if you want to rebuild the report from `build_final_report.js`.

The Marmousi velocity file is downloaded automatically on first run to `data/Simple2D/`. An
internet connection is needed for that one download.

---

## How the scripts fit together

All scripts live in `code/` and are run from there. Most import the shared solver harness from
`mx_experiment_marmousi.py`, so keep the files in the same directory.

```
mx_experiment_marmousi.py     the harness: solver, MX quantiser, rel_l2, snr_db, wiggle helper
      │
      ├── compare_schemes.py         writes comparison_results/comparison.csv  ← run this first
      │        │
      │        └── efficiency_model.py   reads comparison.csv                  ← run this second
      │
      ├── elastic_comparison.py      self-contained
      ├── compression_only.py        self-contained
      ├── water_subtraction.py       self-contained
      └── int16_analysis.py          self-contained (console output only)

fabien_reproduction.py         self-contained
      └── hybrid_experiment.py       imports fabien_reproduction.py            ← keep together
```

The only ordering constraint is **`compare_schemes.py` before `efficiency_model.py`** (the
second reads a CSV the first writes). Everything else can be run in any order.

Each script writes its figures and a CSV to its own `*_results/` folder (except
`int16_analysis.py`, which prints to the console).

---

## Running the tests

From inside `code/`:

### 1. Baseline — reference wavefield and shot records
```bash
python3 m1_baseline_marmousi.py
```
FP32/FP64 baseline on Marmousi: velocity model, shot record, wavefield snapshot, and the
FP32-vs-FP64 difference that sets the accuracy "pass mark". → `m1_marmousi_reference/`

### 2. MX accuracy sweep
```bash
python3 mx_experiment_marmousi.py
```
MX error against bit width, the error-growth curve, and the wavefield / wiggle comparison
figures. → `mx_marmousi_reference/`

### 3. Scheme comparison — MX vs FP16, bfloat16, int16  *(run before step 4)*
```bash
python3 compare_schemes.py
```
Cost-against-accuracy for every scheme on one plot, with an SNR(dB) axis; the block-size sweep;
and the per-scheme wiggle gathers with the direct arrival saturated. Writes
`comparison_results/comparison.csv`, which the efficiency model reads. → `comparison_results/`

### 4. Efficiency model  *(needs step 3 first)*
```bash
python3 efficiency_model.py
```
Predicted speed-up from memory traffic, per kernel, with the wavefield-only result and the
higher ceiling reachable if the model arrays were compressed too. → `comparison_results/`

### 5. Elastic kernel — MX (incl. 11-bit) vs per-field FP16 scaling
```bash
python3 elastic_comparison.py
```
The comparison on the elastic kernel, where Fabien-Ouellet's scaling was designed to help.
Reports SNR(dB) and includes the aggressive 11-mantissa-bit MX point. → `elastic_comparison_results/`

### 6. Compression on its own — single-frame vs accumulated error
```bash
python3 compression_only.py
```
Compresses and decompresses a real wavefield once, with no propagation, and compares the
single-frame error to the error the same format accumulates over the full solve. Shows that the
limit is accumulation, not the representation of one frame. → `compression_only_results/`

### 7. Water-model subtraction — reflection-only SNR
```bash
python3 water_subtraction.py
```
Runs a water-everywhere model (same source, geometry and time-stepping, velocity only changed)
to isolate the direct arrival, subtracts it to leave the reflections, and scores SNR(dB) on the
reflections alone — the energy the image is actually built from. → `water_subtraction_results/`

### 7b. Block shape — does a square block beat a memory strip?
```bash
python3 block_shape.py
```
Holds the block size and bit cost fixed and varies only the block shape, from a 1x32 memory
strip to a 6x6 square, and times the pack/unpack of each. Shows the block-size anomaly is a
shape effect (square blocks are ~4.6 dB more accurate at the same cost) and that the 2D tiling
adds only a small pack/unpack overhead, negligible in a bandwidth-bound solver. → `block_shape_results/`

### 8. int16 diagnostic — why it underperforms, and how it becomes MX
```bash
python3 int16_analysis.py
```
Shows that int16 with one global scale trails FP16 because a single scale can't span the
wavefield's dynamic range, and that giving int16 per-block scaling drops it below FP16 and
converges to MX. Console output only.

### 9. FP16-arithmetic reproduction of Fabien-Ouellet
```bash
python3 fabien_reproduction.py
```
A self-contained numpy stepper that genuinely computes in FP16 (Devito cannot build an FP16
grid), reproducing the overflow behaviour and the effect of scaling. → `fabien_reproduction_results/`

### 10. Hybrid — FP16 arithmetic + MX storage  *(keep with step 9's file)*
```bash
python3 hybrid_experiment.py
```
Combines FP16 arithmetic with MX storage and checks whether the two errors compound. Imports
`fabien_reproduction.py`, so run it from the same directory. → `hybrid_results/`

---

## Notes and limitations

- **Emulation, not hardware.** MX is emulated in numpy. Accuracy numbers are hardware-independent
  and exact; efficiency is modelled from bandwidth, not timed.
- **Cropped vs full Marmousi.** Sweeps use a cropped window for speed. The full model runs about
  four times as many steps, so error accumulation is worse there — final bit-widths should be
  confirmed on it, and the safe setting only goes higher.
- **Simplified MX.** The quantiser uses a shared exponent plus a fixed-point code (block 32,
  ~12 mantissa bits, 14.25 bits/value). The exact OCP MXFP standard gives each element its own
  small exponent; swapping to it is a next step.
- **Legacy scripts.** `m1_baseline.py` and `mx_experiment.py` are early versions on a simple
  constant/layered model, kept for reference. The `_marmousi` versions supersede them.
