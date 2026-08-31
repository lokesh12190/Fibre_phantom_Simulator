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

If you are here to reconstruct and score, read
**[docs/RECONSTRUCTION_GUIDE.md](docs/RECONSTRUCTION_GUIDE.md)**.

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

## Quick start

Scoring and reconstruction need no GPU unless you use ASTRA yourself:

```bash
conda create -n fiber_phantom python=3.10
conda activate fiber_phantom
pip install -r requirements.txt
```

Reconstruct from the exported projections and geometry:

```bash
python analysis/reconstruct_from_projections.py \
    --dataset data/full_realistic_820 --which noisy
```

Score any reconstruction against the truth:

```bash
python analysis/score_reconstruction.py \
    --truth data/full_realistic_820/ground_truth.nii \
    --recon your_reconstruction.nii \
    --pipe-radius 290 --axial-margin 40
```

`--axial-margin` excludes the unusable slices at each end of the rotation axis.
Use 40 for the realistic geometry and 60 for the generic one. Leaving it out
makes any reconstruction look considerably worse than it is.

Your reconstruction may be on any intensity scale — the scorer fits it onto
the ground-truth scale by least squares first, so it measures structure rather
than penalising a brightness offset.

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
generate_for_reconstruction.py   CLI: config in, dataset out
configs/                    specimen and scan parameter sets
make_reference_phantom.py        CLI: fibre-free calibration phantoms
analysis/
  reconstruct_from_projections.py  reconstruct from geometry.json alone
  score_reconstruction.py          score a reconstruction against truth
  measure_artefacts.py             cupping, noise, rings, axial drift
  measure_resolution.py            what feature size is resolvable
  diagnose_volume.py               intensity statistics and histogram
  segment_volume.py                void/matrix/fibre segmentation
data/                       datasets (git-ignored)
docs/RECONSTRUCTION_GUIDE.md     how to reconstruct and score
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
- **Added fibre-free reference phantoms** (`make_reference_phantom.py`) with
  `measure_artefacts.py` and `measure_resolution.py`. These are what revealed
  that cone-beam end slices reconstruct roughly 50 % low, which had been
  silently dominating every score.

## License

MIT — see [LICENSE](LICENSE), which carries the upstream copyright notice.
