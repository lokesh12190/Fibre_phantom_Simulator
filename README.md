# Fibre Phantoms — synthetic CT data with known ground truth

Generates a synthetic unidirectional fibre-composite specimen (glass fibres in
a resin matrix, inside a cylindrical pipe), simulates a cone-beam X-ray CT scan
of it with the [ASTRA Toolbox](https://astra-toolbox.com/), and reconstructs it.

**The point:** because the specimen is built voxel-by-voxel *before* it is
scanned, the true structure is known exactly. That makes it possible to put a
number on how good a reconstruction or segmentation actually is — which is not
possible on a real scan, where no ground truth exists.

The geometry and scan parameters are matched to a real PA6GF30 scan
(820×600×600 voxels, cone beam, 832 projections over 360°), so results here
should transfer to that data.

---

## Quick start

The reference dataset is already generated and ships alongside this repo in
`data/`. You do **not** need a GPU or ASTRA to use it — only to generate new
data.

```bash
conda create -n fiber_phantom python=3.10
conda activate fiber_phantom
pip install -r requirements.txt
```

Score a reconstruction against the truth:

```bash
python analysis/score_reconstruction.py \
    --truth data/reference_206k_FDK/ground_truth.nii \
    --recon data/reference_206k_FDK/reconstruction_noisy.nii \
    --labels data/reference_206k_FDK/segmentation/labels.npy
```

To evaluate **your own** reconstruction, point `--recon` at your volume. It
must be on the same 820×600×600 voxel grid as the ground truth; the script
handles intensity scaling itself, so it does not matter what units your
reconstruction is in.

---

## The reference dataset

`data/reference_206k_FDK/` — 206,000 filaments, FDK reconstruction, seed 245.

| File | What it is |
|---|---|
| `ground_truth.nii` | The phantom as built. **This is the truth.** Exactly three values: `0` = air/void, `100` = resin matrix, `130` = glass fibre |
| `reconstruction_clean.nii` | Reconstruction from noise-free projections |
| `reconstruction_noisy.nii` | Reconstruction from Poisson-noised projections |
| `segmentation/labels.npy` | Segmentation of the clean reconstruction: `1` = matrix, `2` = fibre, `255` = outside specimen |
| `segmentation/labels.mhd` + `.zraw` | Same labels, for 3D Slicer / ITK |
| `run_parameters.hdf5` | Every parameter used for the run |

All volumes are voxel-aligned on the same grid. `.nii` files open directly in
3D Slicer.

> **Axis-order warning.** `nibabel` returns these as `(820, 600, 600)` with the
> fibre axis first; `SimpleITK` returns the axes *reversed*, as
> `(600, 600, 820)`. Both are used in this repo, deliberately and consistently
> per-script. If you mix them without transposing, your volumes will be
> silently misaligned and every metric will be quietly wrong.

---

## What the reference data actually shows

These are measured numbers from the shipped dataset, not estimates. Reproduce
them with the quick-start command above.

**Reconstruction fidelity** (clean, vs ground truth, inside the eroded specimen mask):

| Metric | Value |
|---|---|
| RMSE | 8.15 |
| PSNR | 24.06 dB |
| SSIM | 0.746 |

**Fibre volume fraction:**

| Method | Result | Error |
|---|---|---|
| **True** (from ground truth) | **12.18 %** | — |
| Otsu threshold | 16.01 % | +3.83 pp |
| Linear unmixing | 14.45 % | +2.27 pp |

**Segmentation agreement** (shipped labels vs ground-truth phases):

| Phase | Dice | IoU |
|---|---|---|
| matrix | 0.978 | 0.956 |
| fibre | 0.859 | 0.752 |

### Three caveats worth knowing before you trust anything

**1. The fibres are not properly resolved, so thresholding over-counts them.**

The fibres have a 1-voxel radius, which is at the edge of what the scan
geometry can resolve. In the reconstruction, true fibre voxels land at
111.8 ± 3.4 while true matrix voxels land at 102.5 ± 4.7 — the two populations
**overlap heavily** (discriminability *d′* = 2.27; roughly, *d′* > 3 is needed
before a threshold can cleanly separate two populations).

This means *no* intensity threshold can recover the true fibre fraction from
this data. The +3.83 pp Otsu error is not a bad choice of threshold, it is a
property of the reconstruction. Linear unmixing does better (+2.27 pp) because
it keeps partial-volume information instead of forcing each voxel to be wholly
one phase. `score_reconstruction.py` reports *d′* so you can check this for any
volume you generate.

If you need cleanly resolvable fibres, regenerate with a larger
`radius_range` or a finer voxel size.

**2. Voids are below the noise floor — the label volume is 2-phase.**

The phantom contains only 50 tiny voids (~314 voxels total, 0.0002 % of the
specimen). The FDK reconstruction's dark noise tail contains far more dark
voxels than there are real voids, and connected-component filtering cannot
separate them: loosen the threshold and the noise percolates into a single
2-million-voxel blob; tighten it and you find 18 voxels.

So the shipped labels contain **matrix and fibre only**. Do not read a porosity
figure off this dataset. If you segment it yourself, `segment_volume.py` warns
when the void track claims implausible porosity — heed that warning.

**3. The "noisy" reconstruction is barely noisier than the clean one.**

The config uses `i0 = 1e8` photons, which is a very high dose. The two
reconstructions score almost identically (RMSE 8.1479 vs 8.1485). If you want a
meaningful noise study, lower `i0` in the config — try `1e5` or `1e4` — and
regenerate.

---

## Generating new data

This needs ASTRA and a CUDA GPU:

```bash
conda install -c astra-toolbox astra-toolbox
```

Smoke test first — 128³, about a minute:

```bash
python main.py --config configs/smoke_test.json --out data/smoke_test
```

Then the full reference configuration (~20 min on the machine this was
developed on, and it needs a lot of GPU memory):

```bash
python main.py --config configs/reference_206k_fdk.json --out data/my_run
```

Copy a config and edit it to change the specimen. The parameters worth
reaching for first:

| Parameter | Meaning |
|---|---|
| `num_filaments` | Fibre count, and so the fibre volume fraction |
| `radius_range` | Fibre radius in voxels. `[1, 1]` is at the resolution limit — see caveat 1 |
| `orientation_jitter_deg` | Skin–core misalignment: fibres near the wall stay aligned, fibres near the axis tilt more |
| `generator_mode` | `straight`, `c_curve`, `kink_curve`, `full_wave_curve`, `half_wave_curve` |
| `defect_type` | `none`, `hole`, `square_notch`, `double_square_notch`, `v_notch`, `double_v_notch`, `reduced` |
| `i0` | Photon count. Lower means noisier — see caveat 3 |
| `algorithm` | `FDK_CUDA` (fast) or `SIRT3D_CUDA` (slower, iterative) |
| `num_angles`, `angular_range_degrees` | Projection count and rotation range |

Every run writes `run_parameters.json` next to its volumes, so any dataset can
be traced back to the settings that produced it.

---

## Repository layout

```
fiber_phantom/          the library
  generate_filaments.py   fibre placement, resin fill, voids
  next_point_generator.py fibre path shapes (straight, curves, kinks)
  defects.py              macro defects (holes, notches)
  perform_ASTRA.py        scan simulation and reconstruction
main.py                 CLI: config in, volumes out
configs/                scan/specimen parameter sets
analysis/
  score_reconstruction.py  ** scores a reconstruction against truth **
  diagnose_volume.py       intensity statistics and histogram; run before segmenting
  segment_volume.py        void/matrix/fibre segmentation
  export_for_voxelfem.py   bundle labels+intensity into voxelFEM's .npz layout
  compare_radial_profiles.py  compare radial artefacts against a real scan
data/                   volumes (git-ignored — transferred separately)
docs/                   reference figures and scores
```

`data/` is excluded from git: a single `.nii` here is 1.1 GB, well past what
git handles. Transfer it by copy, not by clone.

---

## Changes from upstream

Forked from [marychrisgo/Fiber-Phantoms](https://github.com/marychrisgo/Fiber-Phantoms)
(MIT). Substantive changes, in rough order of how much they affect results:

- **Fixed the ASTRA volume geometry axis order.** `create_vol_geom(Y, X, Z)`
  needs its data as `(Z, Y, X)`, which is not a simple two-axis swap.
  Verified against a dedicated marker-cube test.
- **Removed silent default clustering.** Fibre clustering defaulted *on*, so
  every run forced 50 % of fibres into one tight clump near `[180,180,180]`
  without the caller asking for it. Placement is now uniform unless clustering
  is explicitly requested.
- **Lowered fibre attenuation 255 → 130.** The extreme contrast against resin
  (100) was the main source of streak artefacts: in a controlled A/B test,
  background standard deviation fell from 11.877 to 3.264.
- **Made the attempt budget scale with fibre count** (was a fixed 10,000),
  which had made high fibre counts unreachable — collision rejection gets
  harder as density rises.
- **Vectorised the resin fill**, previously a 295-million-iteration Python
  loop and the dominant cost of a full-size run.
- **Added `angular_range_degrees`** so a full 360° rotation can be simulated to
  match the real scan; upstream was fixed at 180°.
- **Added `orientation_jitter_deg`**, a skin–core orientation gradient.
- **Fixed defect parameter selection**: `hole_params` was passed to the defect
  generator regardless of which `defect_type` was chosen, so selecting a notch
  silently configured it with hole settings.
- **Added `fiber_phantom/__init__.py`**, absent upstream, which made
  `find_packages()` return nothing and `pip install .` produce an empty package.
- **Fixed void-blob sizing** in segmentation (fragile `ndimage.sum` call, plus
  a `NameError` when no blobs were found).
- **Replaced hardcoded paths with CLI arguments** throughout; the output
  directory, dataset paths and volume index were previously edited in source.
- **Added `analysis/score_reconstruction.py`**, which did not exist upstream.

## License

MIT — see [LICENSE](LICENSE). Upstream: MC, Daan, Joost (Leiden University).
