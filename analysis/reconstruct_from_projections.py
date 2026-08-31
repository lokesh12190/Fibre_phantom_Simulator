#!/usr/bin/env python3
"""Reconstruct a volume from exported projections + geometry.json.

Two purposes:

  1. A worked example. It reads nothing but geometry.json and the projection
     files, so it shows exactly what a reconstruction needs from this dataset
     and that the export contains all of it.
  2. A check on the export. If this reproduces the reference reconstruction,
     the geometry description is complete and correct.

Replace the ASTRA call with your own reconstruction to benchmark it, then
score the result with analysis/score_reconstruction.py against ground_truth.nii.

Usage:
    python analysis/reconstruct_from_projections.py \
        --dataset data/pilot_generic --which noisy --out my_reconstruction.nii
"""

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="directory holding geometry.json and projections")
    parser.add_argument("--which", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--out", help="output .nii (default: inside --dataset)")
    parser.add_argument("--algorithm", help="override the reconstruction "
                                            "algorithm, e.g. SIRT3D_CUDA")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    import astra
    import nibabel as nib

    with open(os.path.join(args.dataset, "geometry.json")) as f:
        geom = json.load(f)

    projections = np.load(
        os.path.join(args.dataset, f"projections_{args.which}.npy"))
    angles = np.load(os.path.join(args.dataset, geom["angles_radians_file"]))
    print(f"projections {projections.shape}  {geom['projection_axis_order']}")
    print(f"angles      {angles.shape}, "
          f"{geom['angular_range_degrees']} degrees total")

    dims = geom["volume_dimensions"]
    vol_geom = astra.create_vol_geom(dims[1], dims[2], dims[0])

    if geom["geometry_type"] == "cone":
        proj_geom = astra.create_proj_geom(
            "cone",
            geom["det_width_u"], geom["det_width_v"],
            geom["det_count_x"], geom["det_count_y"],
            angles,
            geom["source_origin"], geom["origin_det"],
        )
    else:
        proj_geom = astra.create_proj_geom(
            geom["geometry_type"],
            geom["det_width_u"], geom["det_width_v"],
            geom["det_count_x"], geom["det_count_y"],
            angles,
        )

    algorithm = args.algorithm or "FDK_CUDA"
    print(f"Reconstructing with {algorithm}...")

    proj_id = astra.data3d.create("-sino", proj_geom, projections)
    rec_id = astra.data3d.create("-vol", vol_geom)
    cfg = astra.astra_dict(algorithm)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, args.iterations)
    rec = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)

    out = args.out or os.path.join(
        args.dataset, f"reconstruction_from_projections_{args.which}.nii")
    nib.save(nib.Nifti1Image(rec, affine=np.eye(4)), out)
    print(f"volume {rec.shape}, range {rec.min():.3f} to {rec.max():.3f}")
    print(f"Saved {out}")

    reference = os.path.join(args.dataset, "reconstruction_reference.nii")
    if args.which == "clean" and os.path.exists(reference):
        ref = np.asarray(nib.load(reference).dataobj)
        diff = float(np.abs(ref - rec).max())
        print(f"\nMax difference vs reconstruction_reference.nii: {diff:.6g}")
        print("  (near zero confirms geometry.json fully describes the scan)"
              if diff < 1e-3 else
              "  (large difference - the geometry description may be incomplete)")


if __name__ == "__main__":
    main()
