# Fibre Phantoms — synthetic CT data with known ground truth

Builds a synthetic fibre-composite specimen (glass fibres in a resin matrix,
inside a cylindrical pipe), simulates a cone-beam X-ray CT scan of it with the
[ASTRA Toolbox](https://astra-toolbox.com/), and exports the **projection
images plus the scan geometry** so that reconstruction can be done by someone
else, with their own method.

**The point:** the specimen is built voxel-by-voxel *before* it is scanned, so
its true structure is known exactly. That makes it possible to put a number on
how good a reconstruction is — which a real scan can never do, because no
ground truth exists for it.

If you are here to reconstruct and score, go straight to
**[How to use this](#how-to-use-this)**.

---

## The datasets

| Dataset | Volume | Angles | Detector | Geometry | Fibre | Size |
|---|---|---|---|---|---|---|
| `data/full_realistic_820` | 820×600×600 | 832 over 360° | 1720×1070 | real scanner, mag 11.92 | 15.44 vol% | 27 GB |
| `data/full_generic_820` | 820×600×600 | 832 over 360° | 1200×1024 | round-number, mag 2 | 15.44 vol% | 19 GB |

**`full_realistic_820`** is the main deliverable: the real PA6GF30 scanner
geometry at the real specimen size, with fibre content matched to the PA6GF30
datasheet.

**The two fibre datasets are a matched pair.** They share a *bit-identical*
ground truth — same seed, same specimen, scanned two ways — so the only
difference between them is the scan geometry. `full_realistic_820` uses the
real scanner's source and detector distances; `full_generic_820` uses round
numbers (magnification exactly 2) with no relation to any real machine.

### Fibre-free reference phantom

| Dataset | Content | What it is for |
|---|---|---|
| `data/uniform_generic_820` | solid resin cylinder, 100.000 % resin, no fibres | Truth is one flat value, so any variation in a reconstruction is artefact and can be measured |

Measure it with `analysis/measure_artefacts.py`. Further reference phantoms —
a uniform one on the realistic geometry, or a resolution phantom of rods and
spheres at known radii — can be generated with `make_reference_phantom.py`.

Every dataset contains:

```
ground_truth.nii              the phantom as built: 0=air/void, 100=resin, 130=fibre
projections_clean.npy         noise-free projections
projections_noisy.npy         projections with Poisson photon noise
projections_clean_tiff/       same, one 32-bit TIFF per angle
projections_noisy_tiff/
angles.npy                    rotation angle per projection, radians
geometry.json                 full scan geometry
reconstruction_reference_clean.nii   ASTRA FDK from the clean projections
reconstruction_reference_noisy.nii   ASTRA FDK from the noisy projections
run_parameters.json           every parameter, plus the true phase fractions
```

---

## How to use this

Three steps: **reconstruct → score → read the numbers.**

| I want to... | Run |
|---|---|
| Reconstruct with the built-in ASTRA baseline | `analysis/reconstruct_from_projections.py` |
| Score any reconstruction against the truth | `analysis/score_reconstruction.py` |
| Characterise artefacts (cupping, noise, drift) | `analysis/measure_artefacts.py` |
| Build a new dataset | `generate_for_reconstruction.py` |

---

### Step 0 — Install

```bash
conda create -n fiber_phantom python=3.10
conda activate fiber_phantom
pip install -r requirements.txt
```

ASTRA is only needed if you reconstruct with the built-in script. Scoring
needs no GPU and no ASTRA.

```bash
conda install -c astra-toolbox astra-toolbox   # only for reconstruction
```

---

### Step 1 — Know which file is which

Inside `data/full_realistic_820/`:

| File | Role |
|---|---|
| `projections_noisy.npy` | **INPUT.** The X-ray images you reconstruct from. Realistic case. |
| `projections_clean.npy` | **INPUT.** Same scan without noise. Use first, as a sanity check. |
| `projections_*_tiff/` | The same images as one 32-bit TIFF per angle, if your software cannot read `.npy` |
| `angles.npy` | **INPUT.** Rotation angle of each projection, in radians |
| `geometry.json` | **INPUT.** Source/detector distances, detector size, pixel pitch |
| `ground_truth.nii` | **ANSWER KEY.** Do not use while reconstructing — only for scoring |
| `reconstruction_reference_noisy.nii` | **BASELINE.** Plain FDK on the same noisy input; the result to beat |
| `reconstruction_reference_clean.nii` | Plain FDK on the clean input |
| `run_parameters.json` | Every setting used, plus the true phase fractions |

---

### Step 2 — Reconstruct

**Option A — the built-in baseline** (needs ASTRA + GPU):

```bash
python analysis/reconstruct_from_projections.py \
    --dataset data/full_realistic_820 --which noisy
```

It also accepts `--algorithm SIRT3D_CUDA` and `--iterations N` if you just want
a different ASTRA algorithm.

**Option B — your own algorithm.** Read `geometry.json`, the projections and
`angles.npy`, reconstruct however you like, and save a `.nii` on the same voxel
grid. The scorer does not care what produced it. See
[Three things that will bite you](#three-things-that-will-bite-you) first.

---

### Step 3 — Score it

```bash
python analysis/score_reconstruction.py \
    --truth data/full_realistic_820/ground_truth.nii \
    --recon my_reconstruction.nii \
    --pipe-radius 290 --axial-margin 40
```

Two flags that matter:

- `--pipe-radius 290` — the specimen radius. Same for all shipped datasets.
- `--axial-margin 40` — excludes the unusable slices at each end of the
  rotation axis. **Use 40 for `full_realistic_820`, 60 for
  `full_generic_820`.** Those slices are not seen from every angle and
  reconstruct badly no matter what algorithm you use; including them makes any
  result look considerably worse than it is.

Your reconstruction can be on **any intensity scale** — the scorer fits it onto
the ground-truth scale by least squares first, so it measures structure, not
brightness.

---

### Step 4 — Read the output

```
--- Intensity fidelity ---
  RMSE   : 3.2211 attenuation units  (10.7% of the fibre-vs-resin contrast of 30)
  PSNR   : 32.12 dB  (referenced to the 0-130 attenuation range)
  SSIM   : 0.9398

--- Fibre volume fraction ---
  true            : 15.945 %      what is actually there
  Otsu threshold  : 15.823 %  (error -0.122 pp)    what you would measure
  linear unmixing : 17.383 %  (error +1.438 pp)

--- Phase separability ---
  matrix vs fibre d' = 7.04  (cleanly separable)
```

- **RMSE** — in **attenuation units, not a percentage.** The phantom is built
  with air 0, resin 100, fibre 130, so an RMSE of 3.22 means the typical voxel
  is off by 3.22 of those units. Judge it against the **fibre-vs-resin contrast
  of 30**, since telling those two apart is the entire task — dividing by 100
  or by the full 0–130 range makes any result look better than it is.
- **PSNR / SSIM** — overall fidelity. Higher is better.

**Fibre fraction error** — the practical number. The specimen is 15.945 % fibre
by volume, known exactly. You threshold your reconstruction, count the voxels
above the cut, and compare:

```
true 15.945 %  →  measured 16.724 %  →  error +0.78 pp
```

`pp` is percentage *points*, not percent — an error of 2.5 pp is a 16 %
relative error on the quantity itself. This is what propagates into any
stiffness or strength calculation downstream, and on a real scan there is no
way to know it is wrong.

**d′** — whether fibre and matrix can be told apart at all. After
reconstruction each phase occupies a *range* of intensities, and `d′` is how far
apart those ranges sit relative to their width:

```
matrix voxels reconstruct at  101.43 +/- 4.87
fibre  voxels reconstruct at  122.48 +/- 5.79

d' = gap / spread = 21.05 / 5.35 = 3.94
```

| `d′` | meaning |
|---|---|
| above 3 | cleanly separable, thresholding works |
| 2 to 3 | marginal, expect misclassification |
| below 2 | not separable — no threshold works, and any fibre fraction is an artefact |

**Read `d′` first.** It says whether the measurement is possible; the fraction
error says whether it is right. A good `d′` does not guarantee a good fraction:
SIRT reaches `d′` 4.96 on clean data, better separation than FDK's 3.94, yet
over-counts fibre by 2.54 pp because its smoothing shifts both phases.

---

## What the projection values are

The projections are **line integrals of attenuation, not raw photon counts.**
They are already log-corrected — **do not take a log of them again.** Feed them
straight into your reconstructor.

The noisy set was made by drawing Poisson photon counts and converting back:

```
counts  = Poisson( i0 * exp(-f * p) )
p_noisy = -log(counts / i0) / f
```

with `i0` and the absorption factor `f` both recorded in `geometry.json`. If
you would rather start from raw counts and apply your own correction, invert
that relation — or ask, and the counts can be exported directly.

## What `geometry.json` gives you

| Field | Meaning |
|---|---|
| `geometry_type` | `"cone"` for all shipped datasets |
| `volume_dimensions` | `[820, 600, 600]` — the grid your output must match |
| `source_origin`, `origin_det` | Source-to-object and object-to-detector distances |
| `det_width_u`, `det_width_v` | Detector pixel pitch |
| `det_count_x`, `det_count_y` | Detector rows and columns |
| `projection_axis_order` | `"(detector_row, angle, detector_column)"` |
| `magnification`, `voxel_size_at_isocenter` | Derived, for convenience |
| `astra_vol_geom_call` | The exact ASTRA call that matches this volume |

The distances are in a consistent arbitrary unit, not millimetres. Only their
ratios matter, and they are scaled so one detector pixel equals one voxel at
the isocentre.

## Three things that will bite you

**1. The angle axis is in the middle.** `projections_*.npy` has shape
`(detector_row, angle, detector_column)` — ASTRA's convention, and not what
most people assume. Transpose if your code expects angle-first.

**2. `nibabel` and `SimpleITK` disagree about axis order.** `ground_truth.nii`
loads as `(820, 600, 600)` with the fibre axis first under nibabel, but
`(600, 600, 820)` under SimpleITK. Mixing them without transposing does not
error — it silently misaligns your volumes and every metric comes out wrong.
Pick one loader and stay with it.

**3. The volume geometry argument order is not a simple swap.**
`astra.create_vol_geom(Y, X, Z)` expects its data shaped `(Z, Y, X)`. Getting
this wrong produces a reconstruction that looks almost right but is subtly
rotated. `geometry.json` records the exact call as `astra_vol_geom_call`.

## Suggested order

1. Reconstruct `projections_clean.npy` and score it. This is your ceiling. If
   it does not score well, the problem is geometry handling, not your
   algorithm.
2. Reconstruct `projections_noisy.npy` and score it. The gap between the two is
   your method's noise robustness.
3. Compare against `reconstruction_reference_noisy.nii` — plain FDK on the same
   input — to see whether you beat the baseline.

---

## Built-in algorithm results

ASTRA offers three usable cone-beam algorithms. All were run on
`full_realistic_820` and scored against the ground truth, so you have a
baseline to compare your own method against.

**Clean projections**

| | SSIM | *d′* | fibre fraction error |
|---|---|---|---|
| FDK | 0.940 | **7.04** | −0.12 pp |
| CGLS (50 iter) | **0.952** | 6.98 | **−0.10 pp** |
| SIRT (150 iter) | 0.902 | 4.96 | +2.54 pp |

**Noisy projections**

| | SSIM | *d′* | fibre fraction error |
|---|---|---|---|
| FDK | 0.776 | 3.94 | **+0.78 pp** |
| CGLS (50 iter) | 0.739 | 3.20 | +3.34 pp |
| SIRT (150 iter) | **0.849** | **4.47** | +3.50 pp |

True fibre fraction 15.945 %. Three things worth knowing:

- **The ranking reverses.** CGLS is best on clean data and worst on noisy;
  SIRT is the opposite. Judge an algorithm on the noise level you actually
  have.
- **SIRT over-counts fibre by ~2.5 pp even with no noise.** Its smoothing
  dilates fibres by about a voxel. Do not use it to measure volume fraction.
- **A better SSIM does not mean a better measurement.** SIRT wins SSIM on
  noisy data and still has the worst fibre fraction.

Full breakdown, including where each algorithm's errors sit, is in
`data/full_realistic_820/reconstructions/COMPARISON.md`.

---

## Generating new data

Needs ASTRA and a CUDA GPU:

```bash
conda install -c astra-toolbox astra-toolbox

python generate_for_reconstruction.py \
    --config configs/full_realistic_820.json --out data/my_run --tiff
```

Copy a config and edit it. The parameters worth reaching for first:

| Parameter | Meaning |
|---|---|
| `num_filaments` | Fibre count, and so the fibre volume fraction |
| `radius_range` | Fibre radius in voxels. Keep at 3 or above; 1-voxel fibres at realistic packing density are not separable in the reconstruction |
| `orientation_jitter_deg` | Skin–core misalignment: fibres near the wall stay aligned, fibres near the axis tilt more |
| `generator_mode` | `straight`, `c_curve`, `kink_curve`, `full_wave_curve`, `half_wave_curve` |
| `defect_type` | `none`, `hole`, `square_notch`, `double_square_notch`, `v_notch`, `double_v_notch`, `reduced` |
| `i0` | Photon count. Lower is noisier |
| `num_angles`, `angular_range_degrees` | Projection count and rotation range |
| `det_count_x`, `det_count_y` | Detector size. Too small truncates the specimen; generation checks and warns |

Runtime for the full-scale config is about 9 minutes on an L40, most of it
writing the 26 GB to disk.

## Repository layout

```
fiber_phantom/
  generate_filaments.py     fibre placement, resin fill, voids
  next_point_generator.py   fibre path shapes (straight, curves, kinks)
  defects.py                macro defects (holes, notches)
  config.py                 config -> generator objects
  scan_export.py            scan simulation, projection + geometry export
generate_for_reconstruction.py   CLI: config in, fibre dataset out
make_reference_phantom.py        CLI: config in, fibre-free phantom out
configs/                    specimen and scan parameter sets
analysis/
  reconstruct_from_projections.py  reconstruct from geometry.json alone
  score_reconstruction.py          score a reconstruction against truth
  measure_artefacts.py             cupping, noise, rings, axial drift
data/                       datasets (git-ignored)
```

`data/` is excluded from git — one projection stack alone is 6 GB. Transfer it
by copy, not by clone.

---

## Changes from upstream

Forked from [marychrisgo/Fiber-Phantoms](https://github.com/marychrisgo/Fiber-Phantoms)
(MIT). Substantive changes, roughly in order of how much they affect results:

- **Export projections and geometry.** Upstream reconstructed internally and
  discarded the projection data, so the scan could not be reconstructed by
  anyone else. Projections, per-angle rotation values and a complete geometry
  description are now the primary output.
- **Fixed the ASTRA volume geometry axis order.** `create_vol_geom(Y, X, Z)`
  needs its data as `(Z, Y, X)`, which is not a simple two-axis swap.
- **Removed silent default clustering.** Fibre clustering defaulted *on*, so
  every run forced 50 % of fibres into one tight clump without the caller
  asking. Placement is now uniform unless clustering is requested.
- **Lowered fibre attenuation 255 → 130.** The extreme contrast against resin
  was the main source of streak artefacts: background standard deviation fell
  from 11.877 to 3.264 in a controlled A/B test.
- **Made the attempt budget scale with fibre count** (was a fixed 10,000),
  which had made high fibre counts unreachable.
- **Vectorised the resin fill**, previously a 295-million-iteration Python loop.
- **Added `angular_range_degrees`**; upstream was fixed at 180°.
- **Added `orientation_jitter_deg`**, a skin–core orientation gradient.
- **Fixed defect parameter selection**: `hole_params` was passed regardless of
  which `defect_type` was chosen, so selecting a notch silently configured it
  with hole settings.
- **Added `fiber_phantom/__init__.py`**, absent upstream, which made
  `find_packages()` return nothing and `pip install .` produce an empty package.
- **Fixed Otsu segmentation**, which was computed without excluding the dark
  tail and so split dark-vs-material instead of matrix-vs-fibre, reporting
  ~96 % of the specimen as fibre.
- **Added a truncation check**, so a detector too small for the specimen is
  caught at generation time rather than discovered as a reconstruction artefact.
- **Chunked the Poisson noise**, which otherwise transiently allocated tens of
  gigabytes at full scale.
- **Replaced hardcoded paths with CLI arguments** throughout.
- **Added `score_reconstruction.py` and `reconstruct_from_projections.py`**,
  neither of which existed upstream.
- **Added fibre-free reference phantoms** (`make_reference_phantom.py`) and
  `measure_artefacts.py`. These are what revealed that cone-beam end slices
  reconstruct roughly 50 % low, which had been silently dominating every score.

## License

MIT — see [LICENSE](LICENSE), which carries the upstream copyright notice.
