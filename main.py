#!/usr/bin/env python3
"""Generate a synthetic fibre phantom and simulate a CT scan of it.

Produces the three volumes that make this useful as a reconstruction
benchmark, all voxel-aligned on the same grid:

    ground_truth.nii          the phantom as built (exact truth)
    reconstruction_clean.nii  noise-free reconstruction
    reconstruction_noisy.nii  reconstruction from Poisson-noised projections

Requires ASTRA Toolbox and a CUDA GPU (see requirements.txt).

Usage:
    python main.py --config configs/smoke_test.json --out data/smoke_test
    python main.py --config configs/reference_206k_fdk.json --out data/my_run
"""

import argparse
import json
import os
import time

import h5py
import numpy as np

import fiber_phantom.generate_filaments as gf
from fiber_phantom.defects import DefectGenerator
from fiber_phantom.next_point_generator import NextPointGenerator

# Maps defect_type -> the parameters.json key holding that defect's settings,
# so selecting a defect no longer means hand-editing which *_params block gets
# passed through (the previous version always passed hole_params regardless of
# which defect_type was actually selected).
DEFECT_PARAM_KEYS = {
    "hole": "hole_params",
    "square_notch": "square_notch_params",
    "double_square_notch": "square_notch_params",
    "v_notch": "v_notch_params",
    "double_v_notch": "v_notch_params",
    "reduced": "reduced_params",
    "none": None,
}


def build_generator(params):
    """Construct the fibre-path generator described by the config."""
    mode = params["generator_mode"]
    kwargs = {}
    if mode == "kink_curve":
        kwargs = {
            "bend_center": params["kink_bend_center"],
            "transition_range": params["kink_transition_range"],
            "return_center": params["kink_return_center"],
            "return_transition_range": params["kink_return_transition_range"],
        }
    elif mode == "straight":
        kwargs = {"volume_shape": params["volume_dimensions"]}
    return NextPointGenerator(mode=mode, **kwargs)


def build_defect_generator(params):
    """Pick the defect parameter block matching the selected defect_type."""
    defect_type = params["defect_type"]
    if defect_type not in DEFECT_PARAM_KEYS:
        raise ValueError(
            f"Unknown defect_type {defect_type!r}. "
            f"Expected one of: {', '.join(sorted(DEFECT_PARAM_KEYS))}"
        )
    key = DEFECT_PARAM_KEYS[defect_type]
    defect_params = params.get(key, []) if key else []
    return DefectGenerator(defect_type=defect_type, params=defect_params)


def generate_one(params, outdir, index):
    """Generate one phantom + its reconstructions into outdir."""
    volume = np.zeros(params["volume_dimensions"], dtype=np.float32)
    random_seed = params["random_seed"] + index
    np.random.seed(random_seed)

    print(f"Building phantom (seed {random_seed}, "
          f"{params['num_filaments']:,} filaments requested)...")
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

    gt_path = os.path.join(outdir, "ground_truth.nii")
    gf.save_as_nifti(volume, gt_path)
    print(f"Saved {gt_path}")

    if params["ASTRA_reconstruction"]:
        # Imported here, not at module scope, so that --help and config
        # validation work on a machine without ASTRA/CUDA installed.
        import fiber_phantom.perform_ASTRA as tomo

        print("Simulating scan and reconstructing (this is the slow part)...")
        clean_recon, noisy_recon = tomo.perform_tomography(
            volume,
            params["volume_dimensions"],
            params["num_angles"],
            params["geometry_type"],
            params["det_width_u"],
            params["det_width_v"],
            params["det_count_x"],
            params["det_count_y"],
            params["i0"],
            params["algorithm"],
            params["show_plots"],
            params["source_origin"],
            params["origin_det"],
            params["angular_range_degrees"],
        )
        for name, arr in (("reconstruction_clean.nii", clean_recon),
                          ("reconstruction_noisy.nii", noisy_recon)):
            path = os.path.join(outdir, name)
            gf.save_as_nifti(arr, path)
            print(f"Saved {path}")

    # Provenance, written two ways: JSON because it is readable and diffable,
    # HDF5 because the original pipeline consumed that format.
    resolved = dict(params, random_seed=random_seed, filaments_placed=placed)
    with open(os.path.join(outdir, "run_parameters.json"), "w") as f:
        json.dump(resolved, f, indent=4)
    with h5py.File(os.path.join(outdir, "run_parameters.hdf5"), "w") as h5f:
        for key, value in resolved.items():
            h5f.attrs[key] = str(value)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to a parameters JSON file (see configs/)")
    parser.add_argument("--out", required=True,
                        help="Output directory (created if absent)")
    parser.add_argument("--index", type=int, default=200,
                        help="Volume index; offsets the random seed so runs "
                             "are reproducible but distinct (default: 200)")
    args = parser.parse_args()

    with open(args.config) as f:
        params = json.load(f)

    num_volumes = params.get("num_volumes", 1)
    start = time.time()

    for i in range(args.index, args.index + num_volumes):
        # One volume goes straight into --out; several get numbered
        # subdirectories, so the single-volume case stays clean.
        outdir = args.out if num_volumes == 1 else os.path.join(args.out, f"volume_{i}")
        os.makedirs(outdir, exist_ok=True)
        generate_one(params, outdir, i)

    print(f"\nTotal processing time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
