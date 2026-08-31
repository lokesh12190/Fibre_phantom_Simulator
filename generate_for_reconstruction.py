#!/usr/bin/env python3
"""Generate a phantom and export projections + geometry for external reconstruction.

Use this when someone else will run the reconstruction. It writes the ground
truth, the projection images they reconstruct from, and the scan geometry they
need in order to do it.

    ground_truth.nii               the phantom as built (0=air, 100=resin, 130=fibre)
    projections_clean.npy          noise-free projections
    projections_noisy.npy          projections with Poisson photon noise
    angles.npy                     rotation angle per projection, radians
    geometry.json                  full scan geometry
    reconstruction_reference_clean.nii  ASTRA FDK from the clean projections
    reconstruction_reference_noisy.nii  ASTRA FDK from the noisy projections
    run_parameters.json            every parameter used

Requires ASTRA and a CUDA GPU.

Usage:
    python generate_for_reconstruction.py \
        --config configs/full_realistic_820.json --out data/my_run --tiff

    # skip the reference reconstructions (faster, smaller)
    python generate_for_reconstruction.py \
        --config configs/full_realistic_820.json --out data/my_run \
        --no-reference-reconstruction
"""

import argparse
import json
import os
import random
import time

import numpy as np

import fiber_phantom.generate_filaments as gf
from fiber_phantom.config import build_defect_generator, build_generator


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", type=int, default=0,
                        help="offsets the random seed (default: 0)")
    parser.add_argument("--no-reference-reconstruction", action="store_true",
                        help="skip ASTRA's own reconstruction")
    parser.add_argument("--tiff", action="store_true",
                        help="also write one 32-bit TIFF per projection angle, "
                             "for reconstruction software that does not read "
                             ".npy. Roughly doubles the dataset size")
    args = parser.parse_args()

    with open(args.config) as f:
        params = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    start = time.time()

    # Fail before building the phantom rather than after projecting it: a
    # detector too small for the specimen wastes the whole run.
    from fiber_phantom.scan_export import validate_geometry
    validate_geometry(params)

    random_seed = params["random_seed"] + args.index
    # Both generators must be seeded. Fibre starting positions come from the
    # standard library's `random` (next_point_generator.initialize_starting_point),
    # while orientation jitter and the photon noise come from numpy. Seeding
    # only numpy - as the pipeline previously did - left fibre placement
    # unseeded, so two runs of the same config produced different phantoms and
    # a dataset could not be regenerated from its recorded random_seed.
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Building phantom: {params['volume_dimensions']}, "
          f"{params['num_filaments']:,} filaments, seed {random_seed}")
    volume = np.zeros(params["volume_dimensions"], dtype=np.float32)
    placed, _ = gf.generate_and_count_filaments(
        volume,
        params["num_filaments"],
        build_generator(params),
        build_defect_generator(params),
        params["pipe_radius"],
        params["min_length"],
        params["max_length"],
        params["radius_range"],
        params["bias"],
        orientation_jitter_deg=params.get("orientation_jitter_deg", 0.0),
    )
    print(f"Placed {placed:,} filaments.")

    # Report the true composition now, while it is exactly knowable. This is
    # the number any downstream measurement should be checked against.
    cy, cz = volume.shape[1] / 2, volume.shape[2] / 2
    y, z = np.indices((volume.shape[1], volume.shape[2]))
    disc = ((y - cy) ** 2 + (z - cz) ** 2) <= params["pipe_radius"] ** 2
    mask = np.broadcast_to(disc[np.newaxis, :, :], volume.shape)
    n_in = int(mask.sum())
    inside = volume[mask]
    fractions = {
        "specimen_voxels": n_in,
        "void_fraction": float((inside == 0.0).mean()),
        "matrix_fraction": float((inside == 100.0).mean()),
        "fibre_fraction": float((inside == 130.0).mean()),
    }
    print(f"True composition inside the specimen ({n_in:,} voxels):")
    print(f"  void   {100 * fractions['void_fraction']:6.3f} %")
    print(f"  matrix {100 * fractions['matrix_fraction']:6.3f} %")
    print(f"  fibre  {100 * fractions['fibre_fraction']:6.3f} %")

    import nibabel as nib
    gt_path = os.path.join(args.out, "ground_truth.nii")
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), gt_path)
    print(f"Saved {gt_path}")

    # Imported here so --help works without ASTRA installed.
    from fiber_phantom.scan_export import simulate_and_export
    geometry = simulate_and_export(
        volume, params, args.out,
        save_reference_reconstruction=not args.no_reference_reconstruction,
        save_tiff=args.tiff,
    )

    resolved = dict(params,
                    random_seed=random_seed,
                    filaments_placed=placed,
                    true_composition=fractions,
                    absorption_factor=geometry["absorption_factor"])
    with open(os.path.join(args.out, "run_parameters.json"), "w") as f:
        json.dump(resolved, f, indent=4)

    print(f"\nDone in {time.time() - start:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
