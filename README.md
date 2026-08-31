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

### Fibre-free reference phantoms

Four more datasets contain no fibres at all. They are calibration objects: the
content is exactly known and deliberately simple, so anything in a
reconstruction that should not be there can be measured rather than eyeballed.

| Dataset | Content | What it is for |
|---|---|---|
| `data/uniform_realistic_820` | solid resin cylinder, 100.000 % resin | Truth is one flat value, so any variation is artefact |
| `data/uniform_generic_820` | same, generic geometry | |
| `data/resolution_realistic_820` | rod pairs + spheres at radii 1,2,3,4,6,8 vox | What feature size the geometry can actually resolve |
| `data/resolution_generic_820` | same, generic geometry | |

Measure them with `analysis/measure_artefacts.py` and
`analysis/measure_resolution.py`. What they showed is in
**Reconstruction limits** below — including one result that changes how the
fibre datasets should be scored.

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
inside the specimen mask eroded 20 voxels inward from the pipe wall **and
trimmed axially** (`--axial-margin 40` for the realistic geometry, 60 for the
generic one). The axial trim is not cosmetic — see **Reconstruction limits**.

| | full_realistic_820 | full_generic_820 |
|---|---|---|
| Reconstructed from `geometry.json` alone | **exact, diff = 0** | **exact, diff = 0** |
| Specimen truncated on detector | no | no |
| Regenerates identically from its seed | yes | yes |
| True fibre fraction (scored region) | 15.95 % | 15.94 % |
| Otsu error, clean | **−0.12 pp** | **−0.10 pp** |
| Otsu error, noisy | +0.78 pp | +0.77 pp |
| matrix/fibre *d′*, clean | **7.04** | **7.05** |
| matrix/fibre *d′*, noisy | 3.94 | 3.92 |
| SSIM, clean → noisy | 0.940 → 0.776 | 0.940 → 0.775 |
| PSNR, clean | 32.1 dB | 32.1 dB |
| Usable fraction of the axial range | 90 % | 85 % |

Three things this table is saying:

- ***d′* is the row that matters.** It says the fibres are genuinely resolved,
  so a measured fibre fraction reflects the material rather than the
  reconstruction. Both clean reconstructions reach 7, and Otsu then recovers
  the true fibre fraction to about a tenth of a percentage point. An earlier
  1-voxel-fibre dataset scored 2.27 with a +3.83 pp Otsu error, which is what
  motivated the radius-3 fibres used here.
- **The noise is real.** SSIM falls from 0.94 to 0.78 and *d′* roughly halves,
  so the noisy sets are a genuine test rather than a copy of the clean ones.
- **The two geometries are equivalent where they can be trusted.** Scored over
  the usable region they are indistinguishable (*d′* 7.04 vs 7.05). The wider
  cone of the generic geometry does not degrade the interior; it only makes the
  unusable margin at each end larger, costing 5 % more of the volume. An
  earlier version of this table claimed the realistic geometry reconstructed
  better — that difference was entirely the axial artefact described below.

## Reconstruction limits

Measured on the fibre-free phantoms, where the true answer is exactly known.

### The ends of the rotation axis are unusable

This is the one that changes how you should score anything. In cone-beam
geometry the slices at each end of the rotation axis are not seen from every
angle, so FDK reconstructs them very badly. On the uniform phantom, where every
voxel should read 100:

| geometry | cone half-angle | slice 0 reads | unusable margin each end |
|---|---|---|---|
| realistic | 4.5° | 51.6 (**−49 %**) | 35 slices |
| generic | 7.3° | 51.2 (**−50 %**) | 56 slices |

Including that region in a score is not a small effect. On
`full_realistic_820`, trimming 40 slices from each end changes the result from
PSNR 25.8 dB / *d′* 3.78 to **PSNR 32.1 dB / *d′* 7.04** — the broken ends were
dominating the statistics of an otherwise good reconstruction.

So pass `--axial-margin` to `score_reconstruction.py` (40 for the realistic
geometry, 60 for the generic one), and ignore those slices in any measurement
you make. Everything between them is flat to within 1 %.

### What is resolvable

From the resolution phantom, isolated features in clean resin:

| radius (vox) | contrast recovered | pair separable? |
|---|---|---|
| 1 | 70 % | yes, but contrast down to 70 % |
| 2 | 96 % | yes |
| 3 and above | 98 % | yes |

**Isolated 1-voxel features resolve.** This is worth stating carefully, because
it is easy to conclude the opposite from the fibre data. The scan geometry is
not the limitation. What defeats 1-voxel fibres in a real specimen is
**density**: at 15 vol% every fibre has neighbours, their blur overlaps and
fills the gaps between them, and the fibre and matrix intensity distributions
merge. Treat the table above as a best case, and judge a real specimen by the
*d′* that `score_reconstruction.py` reports on an actual fibre dataset.

Note also that a radius-1 *sphere* recovers only 23 % of its contrast against
70 % for a radius-1 *rod*. Same radius, very different outcome: a rod is
extended along the beam and accumulates far more signal. Compact features such
as small voids are much harder to see than fibres of the same radius.

### Noise floor

| | clean | noisy |
|---|---|---|
| noise std inside the specimen | 0.63 | 5.14 |
| as a fraction of fibre contrast (30) | 2.1 % | **17 %** |

In the noisy datasets the reconstruction noise is 17 % of the entire signal a
fibre produces. That is a hard floor on what any segmentation can achieve.

### What this simulation does NOT reproduce

Both are worth knowing before treating these datasets as a stand-in for a real
scan:

- **No beam hardening, so no cupping.** Measured cupping is +0.003 across the
  radius, which is zero. ASTRA's forward projection is a monochromatic line
  integral; there is no X-ray spectrum. A real scan of this material will show
  cupping that this data cannot reproduce, so it will not exercise a
  beam-hardening correction.
- **No ring artefacts** (measured 0.03, i.e. none). Rings come from
  detector-element-to-element variation, which is not simulated.

---

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
