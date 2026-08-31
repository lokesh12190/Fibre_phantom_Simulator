# Reconstructing from this dataset

You have a synthetic CT scan of a fibre-composite specimen where **the true
structure is known exactly**, because the specimen was built voxel-by-voxel
before it was scanned. That lets you put a number on how good your
reconstruction is — which a real scan can never do, since no ground truth
exists for it.

The workflow: reconstruct from the projections, then score your result against
the ground truth.

---

## What you have

| File | What it is |
|---|---|
| `projections_clean.npy` | Noise-free projections. Start here. |
| `projections_noisy.npy` | Same scan with Poisson photon noise. The realistic case. |
| `projections_clean_tiff/`, `projections_noisy_tiff/` | The same projections as one 32-bit TIFF per angle, if your software does not read `.npy`. Bit-identical to the `.npy` versions. |
| `angles.npy` | Rotation angle of each projection, in **radians** |
| `geometry.json` | The full scan geometry — everything you need to reconstruct |
| `ground_truth.nii` | **The truth.** Exactly three values: `0` air/void, `100` resin matrix, `130` glass fibre |
| `reconstruction_reference_clean.nii` | Plain ASTRA FDK reconstructed from the **clean** projections — the best case |
| `reconstruction_reference_noisy.nii` | Plain ASTRA FDK reconstructed from the **noisy** projections — the baseline to beat |
| `run_parameters.json` | Every parameter used, including the true phase fractions |

---

## Important: what the projection values are

The projections are **line integrals of attenuation**, not raw photon counts.
They are already log-corrected. **Do not take a log of them again.**

The noisy set was produced by drawing Poisson photon counts and converting
back:

```
counts  = Poisson( i0 * exp(-f * p) )
p_noisy = -log(counts / i0) / f
```

with `i0` and the absorption factor `f` both recorded in `geometry.json`. If
you would rather start from counts and do your own correction, you can invert
that relation, or ask and the raw counts can be exported too.

## Important: array layout

`projections_*.npy` has shape **`(detector_row, angle, detector_column)`** —
the angle is the *middle* axis, which is ASTRA's convention and is not what
most people expect. `geometry.json` states this in `projection_axis_order`.

`ground_truth.nii` loads via `nibabel` as `(820, 600, 600)`, with the fibre
axis first. Note that **SimpleITK reverses axis order** and would give you
`(600, 600, 820)`. If you mix the two loaders without transposing, your
volumes will be silently misaligned and every metric will come out wrong.

---

## Reconstructing

A complete worked example ships with the repo:

```bash
python analysis/reconstruct_from_projections.py \
    --dataset data/full_realistic_820 --which noisy
```

It reads nothing but `geometry.json` and the projection files. Run it on
`--which clean` and it will compare its output against
`reconstruction_reference_<clean|noisy>.nii` and report the difference — currently **exactly
zero**, which is the proof that `geometry.json` fully describes the scan.

To benchmark your own method, replace the ASTRA call in that script, or build
the geometry yourself from `geometry.json`:

```python
import json, numpy as np, astra

geom  = json.load(open("geometry.json"))
proj  = np.load("projections_noisy.npy")
angles = np.load("angles.npy")

dims = geom["volume_dimensions"]                       # [820, 600, 600]
vol_geom = astra.create_vol_geom(dims[1], dims[2], dims[0])   # note the order

proj_geom = astra.create_proj_geom(
    "cone",
    geom["det_width_u"], geom["det_width_v"],
    geom["det_count_x"], geom["det_count_y"],
    angles,
    geom["source_origin"], geom["origin_det"],
)
```

`astra.create_vol_geom(Y, X, Z)` expects its data shaped `(Z, Y, X)` — that is
not a plain two-axis swap, and getting it wrong is the single most common way
to produce a reconstruction that looks almost right but is subtly rotated.
`geometry.json` records the exact call as `astra_vol_geom_call`.

---

## Scoring your reconstruction

```bash
python analysis/score_reconstruction.py \
    --truth data/full_realistic_820/ground_truth.nii \
    --recon your_reconstruction.nii \
    --pipe-radius 290 --axial-margin 40
```

**Do not leave out `--axial-margin`.** In cone-beam geometry the slices at each
end of the rotation axis are not seen from every angle and reconstruct about
50 % low. They are not your method's fault, but if you include them they
dominate the statistics: on this dataset they cost 6 dB of PSNR and halve the
reported fibre/matrix separability. Use 40 for the realistic geometry, 60 for
the generic one, and ignore those slices in any measurement you make.

Your reconstruction can be on any intensity scale — the script fits it onto
the ground-truth scale by least squares before comparing, so it measures
structural fidelity rather than penalising a brightness offset.

It reports:

- **RMSE / PSNR / SSIM** against the truth
- **Fibre volume fraction**: the true value, and what you would measure from
  your reconstruction by Otsu thresholding and by linear unmixing
- **Phase separability**: the reconstructed intensity of each true phase, and
  a discriminability *d′* between matrix and fibre

Watch *d′* in particular. It says whether the fibres are resolved at all. Above
about 3 a threshold can cleanly separate matrix from fibre; below about 2 the
two populations overlap so heavily that **no** threshold can recover the true
fibre fraction, and any fibre-fraction number you compute is an artefact of the
reconstruction rather than a property of the material. This dataset was
deliberately built with radius-3 fibres to sit in the comfortable range.

---

## The two datasets are a matched pair

The two fibre datasets share a **bit-identical ground truth** — the same
specimen, scanned two ways:

| | geometry | cone half-angle | axial margin to skip |
|---|---|---|---|
| `full_realistic_820` | real scanner distances, magnification 11.92 | 4.5° | 40 slices |
| `full_generic_820` | round numbers, magnification 2 | 7.3° | 60 slices |

So comparing your results across the pair isolates the effect of scan geometry
alone, with everything else held fixed. The wider cone does not degrade the
interior; it makes the unusable margin at each end of the rotation axis
larger.

---

## Suggested order

1. Reconstruct `projections_clean.npy`. Score it. This is your ceiling — if
   this does not look right, the problem is in the geometry handling, not the
   algorithm.
2. Reconstruct `projections_noisy.npy`. Score it. The gap between the two is
   your method's noise robustness.
3. Compare against `reconstruction_reference_noisy.nii` (plain FDK on the same
   noisy input) to see where
   your method wins.

If you need a different specimen — other fibre content, curved or kinked
fibres, a defect such as a hole or notch, more or fewer projection angles, or
a different noise level — every one of those is a parameter in
`configs/full_realistic_820.json`, and a new dataset can be regenerated.
