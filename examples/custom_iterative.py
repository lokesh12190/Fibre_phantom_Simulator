#!/usr/bin/env python3
"""Worked example: reconstruct with an iterative algorithm ASTRA does not provide.

The point of this file is the plumbing, not the algorithm. It shows that a
reconstruction method of your own can be scored against the ground truth
exactly like any other, because ASTRA supplies the forward and back projection
operators and you supply only the iteration.

Do not read the algorithm below as a good reconstruction method. Measured on
full_realistic_820 (noisy), it is nowhere near converged at low iteration
counts: 8 iterations gave matrix/fibre d' = 0.21 and 10 gave 0.72, both far
below the 3.94 that plain FDK achieves on the same data. The residual was
still halving every few iterations in both cases. Simple gradient methods need
hundreds of iterations at this problem size, which is exactly why real work
uses CGLS, SIRT, or a regularised method instead. Your own algorithm goes in
`reconstruct`; the surrounding workflow is what this file is for.

    W = astra.OpTomo(proj_id)
    W * x      forward projection   (A x)
    W.T * y    back projection      (A^T y)

Everything between those two calls is yours. The algorithm below is projected
gradient descent with a non-negativity constraint - deliberately simple, and
deliberately not one of ASTRA's built-ins. Replace `reconstruct` with your own
method and the rest of the workflow is unchanged.

Convergence is worth watching. Measured on full_realistic_820, noisy:

    iterations   d' (matrix vs fibre)   verdict
        8                0.21           fibres invisible
       10                0.72           fibres invisible
       60                3.80           cleanly separable

At 60 iterations this scores SSIM 0.836 against FDK's 0.776 on the same
data - smoother - but its Otsu fibre-fraction error is +7.2 pp against
FDK's +0.8 pp. Better by one measure, worse by another, which is the kind
of trade-off this dataset exists to expose. Roughly 11 minutes on an L40.

Usage:
    python examples/custom_iterative.py \
        --dataset data/full_realistic_820 --which noisy \
        --out my_iterative_result.nii

Then score it:
    python analysis/score_reconstruction.py \
        --truth data/full_realistic_820/ground_truth.nii \
        --recon my_iterative_result.nii \
        --pipe-radius 290 --axial-margin 40
"""

import argparse
import json
import os
import time

import numpy as np


def build_operator(geom, angles):
    """Wrap the scan geometry as a linear operator W with W*x and W.T*y."""
    import astra

    dims = geom["volume_dimensions"]
    vol_geom = astra.create_vol_geom(dims[1], dims[2], dims[0])
    if geom["geometry_type"] == "cone":
        proj_geom = astra.create_proj_geom(
            "cone", geom["det_width_u"], geom["det_width_v"],
            geom["det_count_x"], geom["det_count_y"], angles,
            geom["source_origin"], geom["origin_det"])
    else:
        proj_geom = astra.create_proj_geom(
            geom["geometry_type"], geom["det_width_u"], geom["det_width_v"],
            geom["det_count_x"], geom["det_count_y"], angles)

    proj_id = astra.create_projector("cuda3d", proj_geom, vol_geom)
    return astra.OpTomo(proj_id), tuple(dims)


def reconstruct(W, sinogram, shape, iterations, verbose=True):
    """Preconditioned gradient descent with a non-negativity constraint.

    REPLACE THIS with your own method. It receives the projection operator and
    the measured data, and must return a volume of `shape`.

        x <- max(0, x + C * W^T (R * (b - W x)))

    R and C are the reciprocal row and column sums of W. Without them, plain
    gradient descent needs a step size around 1/||W||^2 and converges far too
    slowly to be useful in a handful of iterations - measured here, eight
    iterations left the reconstruction so far from converged that fibre and
    matrix were indistinguishable (d' = 0.21). The preconditioner rescales
    each ray and each voxel by how much geometry passes through it, which is
    what makes this family of methods practical.
    """
    b = sinogram.ravel().astype(np.float32)
    n_vox = int(np.prod(shape))

    # Row sums: project a volume of ones. Column sums: back-project a
    # sinogram of ones. Both are geometry-only quantities, computed once.
    if verbose:
        print("  computing preconditioners...")
    row_sum = W * np.ones(n_vox, dtype=np.float32)
    col_sum = W.T * np.ones_like(b)
    R = np.zeros_like(row_sum)
    np.divide(1.0, row_sum, out=R, where=row_sum > 1e-6)
    C = np.zeros_like(col_sum)
    np.divide(1.0, col_sum, out=C, where=col_sum > 1e-6)

    x = np.zeros(n_vox, dtype=np.float32)
    for k in range(iterations):
        residual = b - W * x
        x = x + C * (W.T * (R * residual))
        np.maximum(x, 0, out=x)
        if verbose:
            print(f"  iteration {k + 1:3d}/{iterations}   "
                  f"residual norm {np.linalg.norm(residual):.6g}")
    return x.reshape(shape)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--which", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--iterations", type=int, default=100,
                        help="iteration count. Roughly 11 s per iteration on "
                             "the full 820 datasets, and low counts do not "
                             "converge - see the note at the top of this file "
                             "(default: 100)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import nibabel as nib

    with open(os.path.join(args.dataset, "geometry.json")) as f:
        geom = json.load(f)
    sino = np.load(os.path.join(args.dataset, f"projections_{args.which}.npy"))
    angles = np.load(os.path.join(args.dataset, geom["angles_radians_file"]))
    print(f"projections {sino.shape}  {geom['projection_axis_order']}")

    W, shape = build_operator(geom, angles)
    print(f"reconstructing {shape} with {args.iterations} iterations")
    start = time.time()
    volume = reconstruct(W, sino, shape, args.iterations)
    print(f"done in {time.time() - start:.1f} s")

    nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4)), args.out)
    print(f"Saved {args.out}")
    print("\nScore it with:")
    print(f"  python analysis/score_reconstruction.py \\\n"
          f"      --truth {args.dataset}/ground_truth.nii \\\n"
          f"      --recon {args.out} --pipe-radius 290 --axial-margin 40")


if __name__ == "__main__":
    main()
