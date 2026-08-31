#!/usr/bin/env python3
"""Generate a fibre-free reference phantom and export projections + geometry.

These are calibration objects, not specimens. The specimen datasets are full of
fibres, so every artefact in a reconstruction is tangled up with real structure.
Here the object is exactly known and deliberately simple, so anything in the
reconstruction that should not be there is measurable.

Two shapes:

  uniform     A solid resin cylinder and nothing else. Every voxel inside the
              specimen is the same value, so any variation in the
              reconstruction is artefact: cupping and beam hardening, ring
              artefacts, the noise floor, and edge blur at the wall.

  resolution  Rod pairs and isolated spheres at radii 1, 2, 3, 4, 6, 8 voxels,
              in fibre-contrast material. Measures what this scan geometry can
              actually resolve. The rods run along the fibre axis and each pair
              is separated by a gap equal to its own radius, so it tests both
              whether a feature of that size is visible at all and whether two
              adjacent ones can be told apart.

Both reuse the same geometry configs and the same export path as the specimen
datasets, so the output format is identical and every analysis script works on
them unchanged.

Usage:
    python make_reference_phantom.py --shape uniform \
        --config configs/full_generic_820.json --out data/uniform_generic_820 --tiff

    python make_reference_phantom.py --shape resolution \
        --config configs/full_realistic_820.json --out data/my_resolution_phantom --tiff
"""

import argparse
import json
import os
import time

import numpy as np

# Same attenuation values the fibre phantoms use, so contrast is comparable.
ATTENUATION_AIR = 0.0
ATTENUATION_RESIN = 100.0
ATTENUATION_FIBER = 130.0

# Feature radii in voxels. Absolute rather than scaled with the volume:
# resolution is a property of the voxel grid, so the same sizes are the
# interesting ones regardless of specimen size.
FEATURE_RADII = [1, 2, 3, 4, 6, 8]


def cylinder_mask_2d(shape_yz, radius, center=None):
    """Disc mask over the cross-section, broadcastable along the fibre axis."""
    ny, nz = shape_yz
    cy, cz = center if center else (ny / 2, nz / 2)
    y, z = np.indices((ny, nz))
    return ((y - cy) ** 2 + (z - cz) ** 2) <= radius ** 2


def build_uniform(dims, pipe_radius):
    """Solid resin cylinder. Nothing inside it."""
    volume = np.full(dims, ATTENUATION_AIR, dtype=np.float32)
    disc = cylinder_mask_2d((dims[1], dims[2]), pipe_radius)
    volume[np.broadcast_to(disc[np.newaxis, :, :], dims)] = ATTENUATION_RESIN
    return volume, {"shape": "uniform",
                    "description": "solid resin cylinder, no internal features",
                    "pipe_radius": pipe_radius,
                    "resin_value": ATTENUATION_RESIN,
                    "features": []}


def add_rod(volume, y0, z0, radius):
    """A cylinder running the full length of the fibre axis."""
    disc = cylinder_mask_2d((volume.shape[1], volume.shape[2]), radius, (y0, z0))
    volume[np.broadcast_to(disc[np.newaxis, :, :], volume.shape)] = ATTENUATION_FIBER


def add_sphere(volume, x0, y0, z0, radius):
    """A sphere, written into a local sub-box rather than over the whole grid."""
    r = int(np.ceil(radius))
    x0, y0, z0 = int(round(x0)), int(round(y0)), int(round(z0))
    xa, xb = max(x0 - r, 0), min(x0 + r + 1, volume.shape[0])
    ya, yb = max(y0 - r, 0), min(y0 + r + 1, volume.shape[1])
    za, zb = max(z0 - r, 0), min(z0 + r + 1, volume.shape[2])
    if xa >= xb or ya >= yb or za >= zb:
        return
    x, y, z = np.ogrid[xa:xb, ya:yb, za:zb]
    mask = ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) <= radius ** 2
    volume[xa:xb, ya:yb, za:zb][mask] = ATTENUATION_FIBER


def build_resolution(dims, pipe_radius):
    """Rod pairs and isolated spheres at a range of known radii."""
    volume, _ = build_uniform(dims, pipe_radius)
    cy, cz = dims[1] / 2, dims[2] / 2
    features = []

    # Rod pairs, one radius per angular station around a ring. The ring sits at
    # 62% of the pipe radius: far enough from the wall to avoid edge blur, far
    # enough from the axis to leave the centre free for the spheres.
    ring = 0.62 * pipe_radius
    for i, radius in enumerate(FEATURE_RADII):
        theta = 2 * np.pi * i / len(FEATURE_RADII)
        sy, sz = cy + ring * np.cos(theta), cz + ring * np.sin(theta)
        # Offset the pair tangentially. Centre separation of 3r leaves an
        # edge-to-edge gap of exactly one radius.
        ty, tz = -np.sin(theta), np.cos(theta)
        for sign in (-1, +1):
            y0 = sy + sign * 1.5 * radius * ty
            z0 = sz + sign * 1.5 * radius * tz
            add_rod(volume, y0, z0, radius)
            features.append({"type": "rod", "radius": radius,
                             "center_yz": [float(y0), float(z0)],
                             "axis": 0, "pair_station": i})

    # One isolated sphere per radius, spaced along the central axis.
    for i, radius in enumerate(FEATURE_RADII):
        x0 = dims[0] * (i + 0.5) / len(FEATURE_RADII)
        add_sphere(volume, x0, cy, cz, radius)
        features.append({"type": "sphere", "radius": radius,
                         "center_xyz": [float(x0), float(cy), float(cz)]})

    meta = {
        "shape": "resolution",
        "description": ("rod pairs (gap = one radius) and isolated spheres at "
                        "radii 1,2,3,4,6,8 voxels, in fibre-contrast material"),
        "pipe_radius": pipe_radius,
        "resin_value": ATTENUATION_RESIN,
        "feature_value": ATTENUATION_FIBER,
        "radii": FEATURE_RADII,
        "rod_ring_radius": float(ring),
        "features": features,
    }
    return volume, meta


BUILDERS = {"uniform": build_uniform, "resolution": build_resolution}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shape", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--config", required=True,
                        help="geometry config; only the scan parameters are "
                             "used, the specimen parameters are ignored")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-reference-reconstruction", action="store_true")
    parser.add_argument("--tiff", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        params = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    start = time.time()

    from fiber_phantom.scan_export import simulate_and_export, validate_geometry
    validate_geometry(params)

    dims = params["volume_dimensions"]
    pipe_radius = params["pipe_radius"]
    print(f"Building '{args.shape}' phantom: {dims}, pipe_radius {pipe_radius}")
    volume, meta = BUILDERS[args.shape](dims, pipe_radius)

    # The true composition is exact here by construction - there is no
    # randomness anywhere in these phantoms.
    disc = cylinder_mask_2d((dims[1], dims[2]), pipe_radius)
    mask = np.broadcast_to(disc[np.newaxis, :, :], dims)
    inside = volume[mask]
    n_in = int(mask.sum())
    composition = {
        "specimen_voxels": n_in,
        "resin_fraction": float((inside == ATTENUATION_RESIN).mean()),
        "feature_fraction": float((inside == ATTENUATION_FIBER).mean()),
    }
    print(f"True composition inside the specimen ({n_in:,} voxels):")
    print(f"  resin    {100 * composition['resin_fraction']:6.3f} %")
    print(f"  features {100 * composition['feature_fraction']:6.3f} %")

    import nibabel as nib
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)),
             os.path.join(args.out, "ground_truth.nii"))
    print(f"Saved {os.path.join(args.out, 'ground_truth.nii')}")

    # Exact locations and sizes of every feature, so measurement does not have
    # to rediscover where things are.
    meta["true_composition"] = composition
    with open(os.path.join(args.out, "phantom_features.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print("Saved phantom_features.json")

    geometry = simulate_and_export(
        volume, params, args.out,
        save_reference_reconstruction=not args.no_reference_reconstruction,
        save_tiff=args.tiff,
    )

    resolved = dict(params, phantom_shape=args.shape,
                    true_composition=composition,
                    absorption_factor=geometry["absorption_factor"])
    with open(os.path.join(args.out, "run_parameters.json"), "w") as f:
        json.dump(resolved, f, indent=4)

    print(f"\nDone in {time.time() - start:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
