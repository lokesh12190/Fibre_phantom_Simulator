#!/usr/bin/env python3
"""Score a reconstruction against the phantom's known ground truth.

This is the script that makes the synthetic dataset useful: it answers
"how good is this reconstruction, in numbers" for a volume where the true
answer is known exactly.

Reports three families of metric:

  1. Intensity fidelity  - RMSE / PSNR / SSIM against ground truth
  2. Fibre volume fraction - true vs. measured, by Otsu and by linear unmixing
  3. Segmentation agreement - per-class Dice and IoU, if labels are supplied

Ground truth uses the generator's attenuation constants (air/void 0,
resin 100, fibre 130). A reconstruction is on an arbitrary intensity scale,
so before comparing, it is affinely rescaled onto the ground-truth scale by
least squares. That measures structural fidelity rather than penalising a
global brightness or contrast offset, which is almost never what you care
about when judging a reconstruction algorithm.

For cone-beam data, pass --axial-margin. The slices at each end of the
rotation axis are not seen from every angle and reconstruct very badly, and
leaving them in dominates the statistics: on full_realistic_820 they cost
6 dB of PSNR and halve the reported matrix/fibre separability.

Usage:
    python analysis/score_reconstruction.py \
        --truth data/full_realistic_820/ground_truth.nii \
        --recon data/full_realistic_820/reconstruction_reference_noisy.nii \
        --pipe-radius 290 --axial-margin 40
"""

import argparse
import json

import numpy as np

# Attenuation constants the phantom was built with. Keep in sync with
# fiber_phantom/generate_filaments.py.
ATTENUATION_AIR = 0.0
ATTENUATION_RESIN = 100.0
ATTENUATION_FIBER = 130.0

# Label convention produced by analysis/segment_volume.py.
LABEL_VOID, LABEL_MATRIX, LABEL_FIBRE, LABEL_OUTSIDE = 0, 1, 2, 255
LABEL_NAMES = {LABEL_VOID: "void", LABEL_MATRIX: "matrix", LABEL_FIBRE: "fibre"}


def load_volume(path):
    """Load a .nii/.npy volume in the pipeline's native axis order.

    Both ground truth and reconstructions are written by nibabel, so loading
    both with nibabel keeps them voxel-aligned. Do NOT mix in SimpleITK here:
    it returns axes reversed, which would silently misalign the comparison.
    """
    if path.endswith(".npy"):
        return np.load(path)
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)


def pipe_mask(shape, pipe_radius, erode, axial_margin=0):
    """Cylindrical specimen mask, trimmed radially and optionally axially.

    Radial erosion matters: voxels at the pipe boundary are a blurred mix of
    resin and outside-air, and including them dominates the error statistics
    with an edge effect that says nothing about interior quality.

    Axial trimming matters for cone-beam data. The slices at each end of the
    rotation axis are not seen from every angle, so FDK reconstructs them very
    badly - measured on a uniform phantom, the outermost slice comes out about
    50% low, and the affected margin grows with the cone angle (about 35 slices
    at a 4.5 degree half-angle, 56 at 7.3). Leaving them in makes a
    reconstruction look far worse than it is everywhere it can actually be
    trusted. Default 0 keeps the whole volume; see --axial-margin.
    """
    radius = pipe_radius - erode
    cy, cz = shape[1] / 2, shape[2] / 2
    y, z = np.indices((shape[1], shape[2]))
    disc = ((y - cy) ** 2 + (z - cz) ** 2) <= radius ** 2
    mask = np.broadcast_to(disc[np.newaxis, :, :], shape).copy()
    if axial_margin > 0:
        if 2 * axial_margin >= shape[0]:
            raise ValueError(
                f"axial_margin {axial_margin} removes the whole volume "
                f"({shape[0]} slices).")
        mask[:axial_margin] = False
        mask[shape[0] - axial_margin:] = False
    return mask


def align_intensity(recon_vals, truth_vals):
    """Least-squares fit recon onto the truth scale: recon*a + b ~= truth."""
    a, b = np.polyfit(recon_vals, truth_vals, 1)
    return a, b


def intensity_metrics(truth, recon, mask):
    """RMSE / PSNR over the masked region, plus SSIM on central slices."""
    t = truth[mask].astype(np.float64)
    r = recon[mask].astype(np.float64)

    a, b = align_intensity(r, t)
    r_aligned = r * a + b

    rmse = float(np.sqrt(np.mean((t - r_aligned) ** 2)))
    data_range = ATTENUATION_FIBER - ATTENUATION_AIR
    psnr = float(20 * np.log10(data_range / rmse)) if rmse > 0 else float("inf")

    return {
        "intensity_scale_a": float(a),
        "intensity_offset_b": float(b),
        "rmse": rmse,
        "psnr_db": psnr,
        "ssim": slicewise_ssim(truth, recon, mask, a, b),
    }


def slicewise_ssim(truth, recon, mask, a, b):
    """Mean SSIM over evenly spaced slices along the fibre axis.

    Computed on slices rather than the full 295M-voxel volume: a volumetric
    SSIM would need several float64 copies of the whole array, and the
    slice mean is a faithful summary for a specimen this homogeneous.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None

    scores = []
    for i in np.linspace(0, truth.shape[0] - 1, 21).astype(int):
        m = mask[i]
        if not m.any():
            continue
        t_slice = truth[i].astype(np.float64)
        r_slice = recon[i].astype(np.float64) * a + b
        # Flatten everything outside the specimen to a constant so the
        # surrounding air cannot inflate the score.
        t_slice = np.where(m, t_slice, ATTENUATION_RESIN)
        r_slice = np.where(m, r_slice, ATTENUATION_RESIN)
        scores.append(structural_similarity(
            t_slice, r_slice,
            data_range=ATTENUATION_FIBER - ATTENUATION_AIR,
        ))
    return float(np.mean(scores)) if scores else None


# Intensity below which a voxel is treated as void rather than material,
# on the ground-truth scale. Sits well under the resin peak at 100.
VOID_MAX = 95.0


def fibre_fraction_metrics(truth, recon, mask):
    """True fibre fraction vs. what Otsu and linear unmixing recover.

    The two measurement routes disagree systematically, and the gap is itself
    the finding: Otsu forces every voxel to be wholly fibre or wholly matrix,
    so for fibres around a voxel wide it over-counts. Linear unmixing reads
    each voxel's intensity as a resin/fibre blend ratio and so keeps the
    partial-volume information.

    Otsu is run only after excluding voids. Voids are by far the strongest
    mode in the histogram, and with them included Otsu splits void-vs-material
    instead of matrix-vs-fibre, which reports nearly the whole specimen as
    fibre. Excluding them first reproduces the segmentation pipeline in
    analysis/segment_volume.py, so the numbers are comparable.
    """
    truth_in = truth[mask]
    true_fraction = float((truth_in >= ATTENUATION_FIBER).mean())

    # Work on the truth scale so VOID_MAX and the unmixing anchors both mean
    # something regardless of the reconstruction's native intensity units.
    recon_in = recon[mask].astype(np.float64)
    a, b = align_intensity(recon_in, truth_in.astype(np.float64))
    aligned = recon_in * a + b

    otsu_fraction = thr = None
    try:
        from skimage.filters import threshold_otsu
        material = aligned[aligned >= VOID_MAX].astype(np.float32)
        if material.size > 20_000_000:  # Otsu is a histogram method; sample
            rng = np.random.default_rng(0)
            material = material[rng.choice(material.size, 20_000_000, replace=False)]
        thr = float(threshold_otsu(material))
        otsu_fraction = float((aligned >= thr).mean())
    except ImportError:
        pass

    blend = (aligned - ATTENUATION_RESIN) / (ATTENUATION_FIBER - ATTENUATION_RESIN)
    unmix_fraction = float(np.clip(blend, 0.0, 1.0).mean())

    out = {
        "true_fibre_fraction": true_fraction,
        "otsu_fibre_fraction": otsu_fraction,
        "otsu_threshold": thr,
        "unmixing_fibre_fraction": unmix_fraction,
        "unmixing_absolute_error": unmix_fraction - true_fraction,
    }
    if otsu_fraction is not None:
        out["otsu_absolute_error"] = otsu_fraction - true_fraction
    return out


def phase_separability(truth, recon, mask):
    """How well the reconstruction separates the three known phases.

    For each true phase, report the reconstructed intensity it actually
    produced. This explains any fibre-fraction error above: if the fibre and
    matrix distributions overlap, no threshold can recover the true fraction,
    and the error is a property of the reconstruction rather than of the
    segmentation method.

    The reported discriminability d' is the standard
    |mean1 - mean2| / sqrt((var1 + var2) / 2). As a rule of thumb d' > 3 means
    a threshold can cleanly split the phases; below about 2 the populations
    overlap enough that thresholding will misclassify heavily.
    """
    truth_in = truth[mask]
    recon_in = recon[mask].astype(np.float64)
    a, b = align_intensity(recon_in, truth_in.astype(np.float64))
    aligned = recon_in * a + b

    stats = {}
    for value, name in ((ATTENUATION_AIR, "void"),
                        (ATTENUATION_RESIN, "matrix"),
                        (ATTENUATION_FIBER, "fibre")):
        sel = aligned[truth_in == value]
        if sel.size:
            stats[name] = {"mean": float(sel.mean()),
                           "std": float(sel.std()),
                           "voxels": int(sel.size)}

    if "matrix" in stats and "fibre" in stats:
        m, f = stats["matrix"], stats["fibre"]
        spread = np.sqrt((m["std"] ** 2 + f["std"] ** 2) / 2)
        stats["matrix_fibre_dprime"] = (
            float(abs(f["mean"] - m["mean"]) / spread) if spread > 0 else None
        )
    return stats


def segmentation_metrics(truth, labels, mask):
    """Per-class Dice and IoU of a label volume against ground-truth phases."""
    if labels.shape != truth.shape:
        # segment_volume.py writes labels in SimpleITK's reversed axis order.
        if labels.T.shape == truth.shape:
            labels = labels.T
        else:
            raise ValueError(
                f"Label shape {labels.shape} does not match truth "
                f"{truth.shape}, and transposing does not reconcile them."
            )

    truth_labels = np.full(truth.shape, LABEL_OUTSIDE, dtype=np.uint8)
    truth_labels[mask & (truth == ATTENUATION_AIR)] = LABEL_VOID
    truth_labels[mask & (truth == ATTENUATION_RESIN)] = LABEL_MATRIX
    truth_labels[mask & (truth == ATTENUATION_FIBER)] = LABEL_FIBRE

    results = {}
    for value, name in LABEL_NAMES.items():
        t = (truth_labels == value) & mask
        p = (labels == value) & mask
        inter = float(np.count_nonzero(t & p))
        union = float(np.count_nonzero(t | p))
        denom = float(np.count_nonzero(t) + np.count_nonzero(p))
        results[name] = {
            "dice": (2 * inter / denom) if denom else None,
            "iou": (inter / union) if union else None,
            "true_voxels": int(np.count_nonzero(t)),
            "predicted_voxels": int(np.count_nonzero(p)),
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--truth", required=True, help="ground_truth.nii")
    parser.add_argument("--recon", required=True, help="reconstruction to score")
    parser.add_argument("--labels", help="optional label volume (.npy/.mhd)")
    parser.add_argument("--pipe-radius", type=int, default=290,
                        help="specimen radius in voxels (default: 290)")
    parser.add_argument("--erode", type=int, default=20,
                        help="shrink mask inward to skip the blurred pipe "
                             "boundary (default: 20)")
    parser.add_argument("--axial-margin", type=int, default=0,
                        help="exclude this many slices at each end of the "
                             "rotation axis, where cone-beam reconstruction "
                             "is unreliable. Measure the right value with "
                             "analysis/measure_artefacts.py on a uniform "
                             "phantom; 40 suits the realistic 820 geometry "
                             "and 60 the generic one (default: 0, keep all)")
    parser.add_argument("--json", help="also write the report to this path")
    args = parser.parse_args()

    print("Loading ground truth...")
    truth = load_volume(args.truth)
    print(f"  {truth.shape} {truth.dtype}")

    print("Loading reconstruction...")
    recon = load_volume(args.recon)
    print(f"  {recon.shape} {recon.dtype}")

    if truth.shape != recon.shape:
        raise SystemExit(
            f"Shape mismatch: truth {truth.shape} vs recon {recon.shape}. "
            "Both must be on the same voxel grid."
        )

    mask = pipe_mask(truth.shape, args.pipe_radius, args.erode,
                     args.axial_margin)
    axial_note = (f", axial slices {args.axial_margin}-"
                  f"{truth.shape[0] - args.axial_margin}"
                  if args.axial_margin else ", full axial range")
    print(f"Specimen mask: radius {args.pipe_radius - args.erode}{axial_note} "
          f"({np.count_nonzero(mask):,} voxels)\n")

    report = {
        "truth": args.truth,
        "recon": args.recon,
        "mask_radius": args.pipe_radius - args.erode,
        "axial_margin": args.axial_margin,
    }

    print("--- Intensity fidelity ---")
    report["intensity"] = intensity_metrics(truth, recon, mask)
    print(f"  RMSE      : {report['intensity']['rmse']:.4f}")
    print(f"  PSNR      : {report['intensity']['psnr_db']:.2f} dB")
    ssim = report["intensity"]["ssim"]
    print(f"  SSIM      : {ssim:.4f}" if ssim is not None
          else "  SSIM      : (needs scikit-image)")
    print(f"  (recon aligned to truth scale as {report['intensity']['intensity_scale_a']:.4f}*x "
          f"+ {report['intensity']['intensity_offset_b']:.2f})\n")

    print("--- Fibre volume fraction ---")
    frac = fibre_fraction_metrics(truth, recon, mask)
    report["fibre_fraction"] = frac
    print(f"  true            : {100 * frac['true_fibre_fraction']:.3f} %")
    if frac["otsu_fibre_fraction"] is not None:
        print(f"  Otsu threshold  : {100 * frac['otsu_fibre_fraction']:.3f} % "
              f"(error {100 * frac['otsu_absolute_error']:+.3f} pp)")
    print(f"  linear unmixing : {100 * frac['unmixing_fibre_fraction']:.3f} % "
          f"(error {100 * frac['unmixing_absolute_error']:+.3f} pp)\n")

    print("--- Phase separability (reconstructed intensity per true phase) ---")
    sep = phase_separability(truth, recon, mask)
    report["phase_separability"] = sep
    for name in ("void", "matrix", "fibre"):
        if name in sep:
            s = sep[name]
            print(f"  {name:<7}: {s['mean']:7.2f} +/- {s['std']:5.2f}  "
                  f"({s['voxels']:,} voxels)")
    dprime = sep.get("matrix_fibre_dprime")
    if dprime is not None:
        verdict = ("cleanly separable" if dprime > 3
                   else "marginal" if dprime > 2
                   else "NOT separable - fibres are unresolved")
        print(f"  matrix vs fibre d' = {dprime:.2f}  ({verdict})")
        if dprime <= 2:
            print("  -> No intensity threshold can recover the true fibre "
                  "fraction from this\n     reconstruction. Prefer the "
                  "unmixing estimate, and treat per-fibre\n     segmentation "
                  "as unreliable at this voxel size.")
    print()

    if args.labels:
        print("--- Segmentation agreement ---")
        labels = load_volume(args.labels)
        seg = segmentation_metrics(truth, labels, mask)
        report["segmentation"] = seg
        for name, m in seg.items():
            dice = f"{m['dice']:.4f}" if m["dice"] is not None else "n/a"
            iou = f"{m['iou']:.4f}" if m["iou"] is not None else "n/a"
            print(f"  {name:<7}: Dice {dice}  IoU {iou}  "
                  f"(true {m['true_voxels']:,} / pred {m['predicted_voxels']:,})")
        print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.json}")


if __name__ == "__main__":
    main()
