#!/usr/bin/env python3
"""Bundle labels + intensity into the .npz layout the voxelFEM pipeline expects.

Output keys:
    labels     uint8, 0=void 1=matrix 2=fibre 255=outside specimen
    intensity  float32 grey levels, same shape as labels

Both arrays are written with axis 0 as the height / tensile-loading axis.

Axis order is the thing to be careful about here. segment_volume.py loads via
SimpleITK, which returns axes reversed relative to how the file was written,
so its saved labels come out as (600, 600, 820) with height last. The
intensity volume is therefore loaded the same way, so the two are guaranteed
voxel-aligned, and only then is the height axis moved to the front.

Usage:
    python analysis/export_for_voxelfem.py \
        --intensity data/reference_206k_FDK/reconstruction_clean.nii \
        --labels data/reference_206k_FDK/segmentation/labels.npy \
        --output data/reference_206k_FDK/for_voxelfem.npz
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--intensity", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import SimpleITK as sitk

    print("Loading intensity (SimpleITK axis order)...")
    intensity_raw = sitk.GetArrayFromImage(
        sitk.ReadImage(args.intensity)).astype(np.float32)
    print(f"  {intensity_raw.shape}")

    print("Loading labels...")
    labels_raw = np.load(args.labels)
    print(f"  {labels_raw.shape}")

    if intensity_raw.shape != labels_raw.shape:
        raise SystemExit(
            f"Shape mismatch: intensity {intensity_raw.shape} vs "
            f"labels {labels_raw.shape}. Both must come from the same "
            "loading path to stay voxel-aligned."
        )

    # Height is the longest axis (820 vs 600x600). Move it to the front.
    height_axis = int(np.argmax(intensity_raw.shape))
    print(f"Height axis currently at position {height_axis} "
          f"(size {intensity_raw.shape[height_axis]})")

    intensity = np.moveaxis(intensity_raw, height_axis, 0)
    labels = np.moveaxis(labels_raw, height_axis, 0).astype(np.uint8)
    print(f"After move -> intensity {intensity.shape}, labels {labels.shape}")

    if not (intensity.shape[0] > intensity.shape[1]
            and intensity.shape[0] > intensity.shape[2]):
        raise SystemExit(
            "Axis 0 is not the longest dimension after the move. For a "
            "non-elongated volume the longest-axis heuristic cannot identify "
            "the height axis - move it manually."
        )

    uniq, counts = np.unique(labels, return_counts=True)
    print("\nLabel values present:")
    for u, c in zip(uniq, counts):
        print(f"  {u}: {c:,} voxels")

    np.savez(args.output, labels=labels, intensity=intensity)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
