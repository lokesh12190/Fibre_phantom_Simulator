#!/usr/bin/env python3
"""Inspect a volume's intensity distribution. Read-only, writes nothing but a report.

Run this before segmenting anything: it reports the statistics, detects
whether the volume is already a label map or still grayscale, and saves a
histogram with the void / matrix / fibre landmarks marked, which is what you
need in order to choose sensible thresholds.

Usage:
    python analysis/diagnose_volume.py \
        --input data/reference_206k_FDK/reconstruction_clean.nii \
        --outdir data/reference_206k_FDK/segmentation
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tag", default="diagnostic")
    parser.add_argument("--pipe-radius", type=int, default=290)
    parser.add_argument("--erode", type=int, default=20,
                        help="shrink mask inward past the blurred pipe edge. "
                             "Without this the boundary blur shows up as a "
                             "large spurious low-intensity hump (default: 20)")
    args = parser.parse_args()

    import SimpleITK as sitk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.outdir, exist_ok=True)
    hist_png = os.path.join(args.outdir, f"histogram_{args.tag}.png")
    stats_txt = os.path.join(args.outdir, f"stats_{args.tag}.txt")

    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    vol = sitk.GetArrayFromImage(sitk.ReadImage(args.input))
    log(f"input        : {args.input}")
    log(f"shape (z,y,x): {vol.shape}")
    log(f"dtype        : {vol.dtype}")
    log(f"voxels       : {vol.size:,}")

    # SimpleITK reverses axis order, so the cross-section is shape[0]/shape[1].
    mask_radius = args.pipe_radius - args.erode
    cy, cz = vol.shape[0] / 2, vol.shape[1] / 2
    y, z = np.indices((vol.shape[0], vol.shape[1]))
    disc = np.sqrt((y - cy) ** 2 + (z - cz) ** 2) <= mask_radius
    mask = np.broadcast_to(disc[:, :, np.newaxis], vol.shape)
    log(f"mask         : radius {mask_radius} (true {args.pipe_radius})  "
        f"({mask.sum():,} voxels inside)")

    vals = vol[mask]
    vals = vals[np.isfinite(vals)].astype(np.float64)

    rng = np.random.default_rng(0)
    samp = vals if vals.size <= 1_000_000 else vals[rng.choice(vals.size, 1_000_000, replace=False)]
    n_unique = np.unique(samp).size
    is_label = np.issubdtype(vol.dtype, np.integer) and n_unique <= 64
    log(f"unique vals (1M sample): {n_unique}")
    log(f"-> interpreted as: "
        f"{'LABEL volume (already segmented)' if is_label else 'GRAYSCALE volume (needs segmentation)'}")
    log()

    if is_label:
        uniq, counts = np.unique(vals, return_counts=True)
        log("label / phase counts:")
        for u, c in zip(uniq, counts):
            log(f"  value {int(u):>4} : {c:>14,}  ({100 * c / vals.size:6.3f} %)")
    else:
        log(f"min / max  : {vals.min():.4f} / {vals.max():.4f}")
        log(f"mean / std : {vals.mean():.4f} / {vals.std():.4f}")
        pct = {}
        for p in (0.1, 1, 5, 50, 95, 99, 99.9):
            pct[p] = np.percentile(vals, p)
            log(f"  p{p:<5} : {pct[p]:.4f}")

        cnts, edges = np.histogram(vals, bins=512, range=(vals.min(), vals.max()))
        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.figure(figsize=(10, 5.5))
        plt.fill_between(centers, cnts, step="mid", color="#444")
        plt.yscale("log")
        plt.axvline(pct[50], color="#1f77b4", ls="--", lw=1,
                    label=f"median {pct[50]:.1f} (matrix-rich)")
        plt.axvline(pct[95], color="#d62728", ls="--", lw=1,
                    label=f"p95 {pct[95]:.1f} (glass shoulder)")
        plt.axvline(pct[1], color="#2ca02c", ls=":", lw=1,
                    label=f"p1 {pct[1]:.1f} (void/low tail)")
        plt.xlabel("voxel intensity (attenuation)")
        plt.ylabel("count (log)")
        plt.title(f"CT intensity histogram - {args.tag}")
        plt.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        plt.savefig(hist_png, dpi=130)
        plt.close()
        log(f"\nsaved -> {hist_png}")

    with open(stats_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"saved -> {stats_txt}")


if __name__ == "__main__":
    main()
