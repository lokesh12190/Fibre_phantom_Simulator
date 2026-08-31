#!/usr/bin/env python3
"""Segment a reconstructed phantom into void / matrix / fibre.

    0 = void    1 = matrix (resin)    2 = fibre (glass)    255 = outside specimen

Two tracks, matching the pipeline used on the real scan data so results are
comparable:

  * Voids are found as connected dark blobs, not by threshold alone, so
    isolated noisy voxels are not mistaken for porosity.
  * Matrix and fibre are split by a single Otsu cut, computed after voids are
    removed. Voids are the strongest mode in the histogram; leaving them in
    makes Otsu split void-vs-material instead of matrix-vs-fibre.

The specimen mask is generated from the known pipe geometry rather than read
from a file, because for a synthetic phantom the geometry is exactly known.

Caveat: at 1-voxel fibre radius the reconstruction may not resolve individual
fibres, in which case no threshold recovers the true fibre fraction. Run
analysis/score_reconstruction.py first and check the matrix-vs-fibre d'.

Usage:
    python analysis/segment_volume.py \
        --input data/reference_206k_FDK/reconstruction_clean.nii \
        --outdir data/reference_206k_FDK/segmentation
"""

import argparse
import os

import numpy as np


def build_mask(shape, mask_radius):
    """Cylindrical specimen mask in SimpleITK's axis order.

    SimpleITK returns axes reversed relative to how the file was written, so
    the cross-section is shape[0]/shape[1] and the fibre axis is shape[2].
    """
    cy, cz = shape[0] / 2, shape[1] / 2
    y, z = np.indices((shape[0], shape[1]))
    disc = np.sqrt((y - cy) ** 2 + (z - cz) ** 2) <= mask_radius
    return np.broadcast_to(disc[:, :, np.newaxis], shape)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="reconstruction to segment")
    parser.add_argument("--outdir", required=True, help="directory for outputs")
    parser.add_argument("--tag", default="labels", help="output basename")
    parser.add_argument("--pipe-radius", type=int, default=290,
                        help="true specimen radius in voxels (default: 290)")
    parser.add_argument("--erode", type=int, default=20,
                        help="shrink the mask inward to exclude the blurred "
                             "pipe boundary (default: 20)")
    parser.add_argument("--void-max", type=float, default=95.0,
                        help="intensity below which a voxel may be void "
                             "(default: 95, on the generator's scale)")
    parser.add_argument("--min-void-voxels", type=int, default=4,
                        help="discard void blobs smaller than this (default: 4)")
    parser.add_argument("--no-voids", action="store_true",
                        help="skip void detection and emit matrix/fibre only. "
                             "Use this whenever the reconstruction's dark "
                             "noise tail overlaps the void intensity range - "
                             "see the caveat above")
    parser.add_argument("--no-preview", action="store_true",
                        help="skip writing the preview PNG")
    args = parser.parse_args()

    import SimpleITK as sitk
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    os.makedirs(args.outdir, exist_ok=True)
    out_mhd = os.path.join(args.outdir, f"{args.tag}.mhd")
    out_npy = os.path.join(args.outdir, f"{args.tag}.npy")

    img = sitk.ReadImage(args.input)
    vol = sitk.GetArrayFromImage(img)
    print(f"volume {vol.shape} {vol.dtype}")

    mask_radius = args.pipe_radius - args.erode
    mask = build_mask(vol.shape, mask_radius)
    n_in = int(mask.sum())
    print(f"inside mask: radius {mask_radius} "
          f"(true pipe_radius {args.pipe_radius})  ({n_in:,} voxels)")

    # ---- track 1: voids as connected dark blobs ----
    if args.no_voids:
        void_mask = np.zeros(vol.shape, dtype=bool)
        print("void detection: DISABLED (--no-voids), emitting matrix/fibre only")
    else:
        cand = (vol < args.void_max) & mask
        lab, n_blobs = ndimage.label(cand)
        if n_blobs:
            sizes = np.bincount(lab.ravel())
            keep = sizes >= args.min_void_voxels
            keep[0] = False  # index 0 is background
            void_mask = keep[lab]
            n_void = int(void_mask.sum())
            print(f"void blobs: kept {int(keep.sum())} of {n_blobs} "
                  f"({n_void:,} voxels)")
            # A real specimen's porosity is a small fraction of a percent. If
            # the void track claims percent-level porosity, the dark tail of
            # the reconstruction noise has percolated into large spurious
            # blobs and the void labels are meaningless - say so loudly rather
            # than writing a plausible-looking but wrong porosity figure.
            if n_void > 0.01 * n_in:
                print(f"  WARNING: that is {100 * n_void / n_in:.2f}% of the "
                      f"specimen. Void detection has almost certainly latched\n"
                      f"  onto reconstruction noise. Lower --void-max, raise "
                      f"--min-void-voxels, or re-run with --no-voids.")
        else:
            void_mask = np.zeros(vol.shape, dtype=bool)
            print("void blobs: none found")

    # ---- track 2: matrix vs fibre, one Otsu cut ----
    # The dark tail is excluded from the Otsu computation whether or not it is
    # being labelled as void. It is the strongest mode in the histogram, and
    # leaving it in makes Otsu split dark-vs-material instead of
    # matrix-vs-fibre, which reports almost the whole specimen as fibre.
    material_sel = mask & ~void_mask & (vol >= args.void_max)
    material_vals = vol[material_sel].astype(np.float32)
    if material_vals.size == 0:
        raise SystemExit(
            f"No voxels at or above --void-max={args.void_max} inside the "
            "mask; nothing to threshold. Check that --void-max matches this "
            "volume's intensity scale (run analysis/diagnose_volume.py)."
        )
    fib_thr = float(threshold_otsu(material_vals))
    print(f"matrix|fibre Otsu threshold: {fib_thr:.2f} "
          f"(computed over {material_vals.size:,} material voxels)")

    labels = np.full(vol.shape, 255, dtype=np.uint8)
    labels[mask] = np.where(vol[mask] >= fib_thr, 2, 1).astype(np.uint8)
    labels[void_mask] = 0

    names = {0: "void", 1: "matrix", 2: "fibre"}
    inside = labels[mask]
    print("\nphase fractions (inside specimen):")
    for k in (0, 1, 2):
        c = int((inside == k).sum())
        print(f"  {k} {names[k]:<7}: {c:>13,}  ({100 * c / n_in:6.3f} %)")

    lab_img = sitk.GetImageFromArray(labels)
    lab_img.CopyInformation(img)
    sitk.WriteImage(lab_img, out_mhd, useCompression=True)
    np.save(out_npy, labels)
    print(f"\nsaved -> {out_mhd}\nsaved -> {out_npy}")

    if not args.no_preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        preview = os.path.join(args.outdir, f"preview_{args.tag}.png")
        cmap = ListedColormap(["#1f77b4", "#888888", "#e41a1c"])
        zc, yc, xc = [s // 2 for s in vol.shape]
        panels = [("z", vol[zc], labels[zc]),
                  ("y", vol[:, yc, :], labels[:, yc, :]),
                  ("x", vol[:, :, xc], labels[:, :, xc])]
        fig, ax = plt.subplots(3, 2, figsize=(9, 12))
        for i, (axis, g, l) in enumerate(panels):
            show = np.where(l == 255, np.nan, l)
            ax[i, 0].imshow(g, cmap="gray")
            ax[i, 0].set_title(f"grayscale (mid-{axis})")
            ax[i, 1].imshow(show, cmap=cmap, vmin=0, vmax=2)
            ax[i, 1].set_title("labels (blue=void, gray=matrix, red=fibre)")
            for a in ax[i]:
                a.axis("off")
        plt.tight_layout()
        plt.savefig(preview, dpi=120)
        plt.close()
        print(f"saved -> {preview}")


if __name__ == "__main__":
    main()
