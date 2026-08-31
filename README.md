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
| `data/pilot_realistic_256` | 256³ | 180 over 360° | 512×512 | real scanner, mag 11.92 | 10.82 vol% | 914 MB |
| `data/pilot_generic_256` | 256³ | 180 over 360° | 512×512 | round-number, mag 2 | 10.82 vol% | 914 MB |

**`full_realistic_820`** is the main deliverable: the real PA6GF30 scanner
geometry at the real specimen size, with fibre content matched to the PA6GF30
datasheet.

**The datasets come in matched pairs.** Within each size, the realistic and
generic datasets share a *bit-identical* ground truth — same seed, same
specimen, scanned two ways. The only difference is the scan geometry, so
comparing results across a pair isolates the effect of geometry alone. The
`realistic` variants use the real scanner's source and detector distances; the
`generic` variants use round numbers (magnification exactly 2) with no
relation to any real machine.

The 256³ pilots are small and quick to iterate on; the 820 datasets are the
real thing.

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

## Verified state of the shipped data

Measured, not estimated — reproduce with the commands below.

All figures below are for the shipped FDK reference reconstruction, scored
inside the specimen mask eroded 20 voxels inward from the pipe wall.

| | full_realistic_820 | full_generic_820 | pilot_realistic_256 | pilot_generic_256 |
|---|---|---|---|---|
| Reconstructed from `geometry.json` alone | **exact, diff = 0** | **exact, diff = 0** | **exact, diff = 0** | **exact, diff = 0** |
| Specimen truncated on detector | no | no | no | no |
| Regenerates identically from its seed | yes | yes | yes | yes |
| True fibre fraction (eroded mask) | 15.62 % | 15.62 % | 11.09 % | 11.09 % |
| Otsu error, clean | **−0.49 pp** | −0.95 pp | **−0.10 pp** | −0.45 pp |
| Otsu error, noisy | +0.85 pp | +0.91 pp | +0.80 pp | +0.53 pp |
| matrix/fibre *d′*, clean | **3.78** | 3.04 | **5.07** | 3.09 |
| matrix/fibre *d′*, noisy | 3.00 | 2.59 | 3.78 | 2.80 |
| SSIM, clean → noisy | 0.916 → 0.782 | 0.897 → 0.783 | 0.908 → 0.819 | 0.896 → 0.848 |
| PSNR, clean | 25.8 dB | 24.7 dB | 28.1 dB | 25.2 dB |

Three things this table is saying:

- ***d′* is the row that matters.** It says the fibres are genuinely resolved,
  so a measured fibre fraction reflects the material rather than the
  reconstruction. Every clean reconstruction clears 3.0. An earlier
  1-voxel-fibre dataset scored 2.27 with a +3.83 pp Otsu error, which is what
  motivated the radius-3 fibres used here.
- **The noise is real.** SSIM drops by 0.05–0.13 and *d′* falls below 3 on the
  generic geometry, so the noisy sets are a genuine test rather than a copy of
  the clean ones.
- **The realistic geometry reconstructs better than the generic one** at the
  same specimen and photon count (*d′* 3.78 vs 3.04). Its far larger
  source-to-object distance gives a 4.5° cone half-angle against 7.3°, and
  less cone-beam distortion away from the central plane.

### Fibre content is not radially uniform

The true fibre fraction reads 15.62 % in the table above but 15.44 % in the
dataset table. Both are correct — they use different masks, and the fraction
genuinely varies with radius. Measured on `full_realistic_820`:

| radius (vox) | 0–100 | 100–200 | 200–250 | 250–270 | 270–280 | 280–290 |
|---|---|---|---|---|---|---|
| fibre | 14.16 % | 15.37 % | 16.27 % | 16.32 % | 16.12 % | 12.63 % |

Two separate effects, both consequences of how the phantom is built:

- **The outermost band is sparse** (12.63 %). A fibre is only placed if its
  full radius fits inside the pipe, so the last few voxels before the wall
  cannot be filled. Eroding the mask removes this band, which is why the
  eroded figure is the higher of the two.
- **The core is sparse too** (14.16 %). `orientation_jitter_deg` tilts fibres
  near the axis by up to 25° while leaving fibres near the wall aligned. Tilted
  fibres sweep a longer path and are rejected by the collision check more
  often, so fewer of them land.

This matters if you measure fibre fraction in a sub-volume rather than the
whole specimen: expect roughly 14 % near the axis and 16 % at mid-radius even
with a perfect reconstruction. `run_parameters.json` records the whole-pipe
value. Setting `orientation_jitter_deg` to 0 removes the core gradient.

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
