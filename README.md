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

| Dataset | Volume | Angles | Detector | Fibre | Size |
|---|---|---|---|---|---|
| `data/full_realistic_820` | 820×600×600 | 832 over 360° | 1720×1070 | 15.46 vol% | 26 GB |
| `data/pilot_generic_256` | 256³ | 180 over 360° | 512×512 | 10.8 vol% | 850 MB |
| `data/pilot_realistic_256` | 256³ | 180 over 360° | 512×512 | 10.8 vol% | 489 MB |

**`full_realistic_820`** is the main deliverable. It uses the real PA6GF30
scanner geometry (magnification 11.92, so one detector pixel is one voxel at
isocentre) at the real specimen size, with fibre content matched to the
PA6GF30 datasheet.

The two pilots are small and fast to work with. `pilot_generic` uses
round-number geometry (magnification exactly 2) with no relation to any real
scanner; `pilot_realistic` uses the real scanner distances at reduced size.

Every dataset contains:

```
ground_truth.nii              the phantom as built: 0=air/void, 100=resin, 130=fibre
projections_clean.npy         noise-free projections
projections_noisy.npy         projections with Poisson photon noise
projections_clean_tiff/       same, one 32-bit TIFF per angle
projections_noisy_tiff/
angles.npy                    rotation angle per projection, radians
geometry.json                 full scan geometry
reconstruction_reference.nii  ASTRA FDK result, as a known-good comparison
run_parameters.json           every parameter, plus the true phase fractions
```

---

## Verified state of the shipped data

Measured, not estimated — reproduce with the commands below.

| | full_realistic_820 | pilot_realistic_256 | pilot_generic_256 |
|---|---|---|---|
| Reconstructed from `geometry.json` alone | **exact, diff = 0** | **exact, diff = 0** | **exact, diff = 0** |
| Specimen truncated on detector | no | no | no |
| True fibre fraction | 15.46 vol% | 10.80 vol% | 10.81 vol% |
| Fibre fraction measured by Otsu | 15.12 % (**−0.49 pp**) | 11.13 % (−0.11 pp) | 10.93 % (−0.38 pp) |
| matrix/fibre *d′* | **3.78** cleanly separable | **5.05** cleanly separable | **3.17** cleanly separable |
| SSIM / PSNR (FDK vs truth) | 0.916 / 25.8 dB | 0.909 / 28.1 dB | 0.896 / 25.2 dB |

The *d′* row is the one that matters: it says the fibres are genuinely
resolved, so a measured fibre fraction reflects the material rather than the
reconstruction. The earlier 1-voxel-fibre dataset scored *d′* = 2.27 with a
+3.83 pp Otsu error, which is what motivated the radius-3 fibres here.

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
    --pipe-radius 290
```

Your reconstruction may be on any intensity scale — the scorer fits it onto
the ground-truth scale by least squares first, so it measures structure rather
than penalising a brightness offset.

---

## Generating new data

Needs ASTRA and a CUDA GPU:

```bash
conda install -c astra-toolbox astra-toolbox

python generate_for_reconstruction.py \
    --config configs/pilot_generic_256.json --out data/my_run --tiff
```

Copy a config and edit it. The parameters worth reaching for first:

| Parameter | Meaning |
|---|---|
| `num_filaments` | Fibre count, and so the fibre volume fraction |
| `radius_range` | Fibre radius in voxels. Keep at 3 or above — see below |
| `orientation_jitter_deg` | Skin–core misalignment: fibres near the wall stay aligned, fibres near the axis tilt more |
| `generator_mode` | `straight`, `c_curve`, `kink_curve`, `full_wave_curve`, `half_wave_curve` |
| `defect_type` | `none`, `hole`, `square_notch`, `double_square_notch`, `v_notch`, `double_v_notch`, `reduced` |
| `i0` | Photon count. Lower is noisier |
| `num_angles`, `angular_range_degrees` | Projection count and rotation range |
| `det_count_x`, `det_count_y` | Detector size. Too small truncates the specimen; generation checks and warns |

Runtime for the full-scale config is about 9 minutes on an L40, most of it
writing the 26 GB to disk.

### Two things that were got wrong before, and are worth not repeating

**Fibre radius.** An earlier dataset used 1-voxel fibres, which the scan could
not resolve: reconstructed fibre and matrix intensities overlapped so heavily
(*d′* = 2.27) that no threshold could recover the true fibre fraction, and
Otsu over-counted by nearly 4 percentage points. At radius 3 the phases
separate cleanly. `score_reconstruction.py` reports *d′* so this is visible
immediately.

**Photon count.** An earlier config used `i0 = 1e8`, at which the "noisy"
reconstruction was numerically indistinguishable from the clean one, making
noise studies meaningless. These configs use `i0 = 1e5`, which degrades the
reconstruction measurably without destroying it.

---

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
analysis/
  reconstruct_from_projections.py  reconstruct from geometry.json alone
  score_reconstruction.py          score a reconstruction against truth
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

## License

MIT — see [LICENSE](LICENSE), which carries the upstream copyright notice.
