"""Simulate a CT scan and export the projections plus the scan geometry.

This is the path to use when someone else will do the reconstruction: the
projections are the deliverable, and reconstructing here is optional and only
so there is a reference result to compare against.

What gets written:

    projections_clean.npy   noise-free projections
    projections_noisy.npy   projections with Poisson photon noise
    angles.npy              rotation angle of each projection, in radians
    geometry.json           everything needed to rebuild the scan geometry
    reconstruction_reference.nii   optional ASTRA reconstruction, for comparison

Projections are line integrals of attenuation, not raw photon counts: the
noisy set is produced by drawing Poisson counts and converting back with
-log(counts / i0) / absorption_factor. Feed them to a reconstructor directly;
do not take a log of them again.
"""

import json
import os

import numpy as np
import scipy.optimize as sco

# Fraction of the beam absorbed on average, used to scale attenuation into
# photon counts before adding noise.
AVERAGE_ABSORPTION_RATIO = 0.5


def estimate_absorption_factor(projections, average_absorption_ratio):
    """Fit the scale factor turning attenuation line integrals into photon counts.

    The phantom's attenuation values are arbitrary units, so before Poisson
    noise can be applied they need scaling to a physically sensible absorption
    level. This finds the factor `f` for which the mean of 1 - exp(-p*f) equals
    the requested average absorption ratio.
    """
    projs = projections[projections > 0]
    initial_guess = 1.0 / projs.mean()

    def error_function(fac):
        ratio = 1 - np.exp(-projs * fac[0]).mean()
        return (ratio - average_absorption_ratio) ** 2

    return sco.fmin(error_function, initial_guess, disp=False)[0]


def build_geometries(params):
    """Construct the ASTRA volume and projection geometries from a config.

    The volume geometry argument order is deliberate and easy to get wrong:
    astra.create_vol_geom(Y, X, Z) expects the data array to be shaped
    (Z, Y, X), which is not a plain two-axis swap. Passing
    (dims[1], dims[2], dims[0]) is what makes the geometry match a volume of
    shape (dims[0], dims[1], dims[2]).
    """
    import astra

    dims = params["volume_dimensions"]
    vol_geom = astra.create_vol_geom(dims[1], dims[2], dims[0])

    angular_range = params.get("angular_range_degrees", 180)
    angles = np.linspace(0, np.radians(angular_range), params["num_angles"], False)

    geometry_type = params["geometry_type"]
    if geometry_type == "parallel3d":
        # No point source, so there are no source/detector distances to give.
        proj_geom = astra.create_proj_geom(
            geometry_type,
            params["det_width_u"], params["det_width_v"],
            params["det_count_x"], params["det_count_y"],
            angles,
        )
    elif geometry_type == "cone":
        proj_geom = astra.create_proj_geom(
            geometry_type,
            params["det_width_u"], params["det_width_v"],
            params["det_count_x"], params["det_count_y"],
            angles,
            params["source_origin"], params["origin_det"],
        )
    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}")

    return vol_geom, proj_geom, angles


def check_truncation(projections, tolerance=1e-6):
    """Warn if the specimen runs off the edge of the detector.

    A truncated projection cannot be reconstructed correctly no matter how
    good the algorithm is, so this is worth catching at generation time
    rather than leaving someone to discover it as a mysterious artefact.
    Checks whether attenuation is non-zero along the detector border.
    """
    # Sinogram axes are (detector row, angle, detector column).
    border = np.concatenate([
        projections[0, :, :].ravel(), projections[-1, :, :].ravel(),
        projections[:, :, 0].ravel(), projections[:, :, -1].ravel(),
    ])
    max_border = float(np.abs(border).max())
    truncated = max_border > tolerance
    if truncated:
        print(f"  WARNING: non-zero attenuation ({max_border:.4g}) at the "
              f"detector border - the specimen is TRUNCATED.")
        print("  Increase det_count_x / det_count_y, or reduce the specimen "
              "size, before using this data.")
    else:
        print(f"  Truncation check passed (max border value {max_border:.4g}).")
    return truncated


def add_photon_noise(projections, i0, absorption_factor, chunk_angles=32):
    """Apply Poisson photon noise, one block of angles at a time.

    Chunked deliberately. At full scale the projection stack is around 6 GB in
    float32, and the obvious one-shot version transiently holds a float64
    exponential and an int64 Poisson draw of the whole stack at once - roughly
    an extra 25 GB for no benefit. Working in angle blocks keeps peak memory
    close to the size of one block.
    """
    noisy = np.empty_like(projections, dtype=np.float32)
    n_angles = projections.shape[1]  # (row, angle, col)
    for start in range(0, n_angles, chunk_angles):
        stop = min(start + chunk_angles, n_angles)
        block = projections[:, start:stop, :].astype(np.float64)
        counts = np.random.poisson(i0 * np.exp(-absorption_factor * block))
        np.maximum(counts, 1, out=counts)  # avoid log(0)
        noisy[:, start:stop, :] = (
            -np.log(counts / i0) / absorption_factor).astype(np.float32)
    return noisy


def save_tiff_stack(projections, outdir, name):
    """Write one 32-bit float TIFF per projection angle.

    The usual interchange format for reconstruction software that does not
    read numpy arrays. Each file is a single radiograph,
    (detector_row, detector_column), in the same line-integral units as the
    .npy stack.
    """
    import tifffile

    stack_dir = os.path.join(outdir, name)
    os.makedirs(stack_dir, exist_ok=True)
    n_angles = projections.shape[1]
    for i in range(n_angles):
        tifffile.imwrite(
            os.path.join(stack_dir, f"proj_{i:04d}.tif"),
            np.ascontiguousarray(projections[:, i, :], dtype=np.float32),
        )
    print(f"  saved {name}/ ({n_angles} TIFF files, "
          f"{projections.shape[0]}x{projections.shape[2]} each)")
    return stack_dir


def simulate_and_export(volume, params, outdir, save_reference_reconstruction=True,
                        save_tiff=False):
    """Scan `volume`, write projections + geometry + optional reconstruction."""
    import astra

    os.makedirs(outdir, exist_ok=True)
    vol_geom, proj_geom, angles = build_geometries(params)

    print("Forward-projecting...")
    proj_id, projections = astra.create_sino3d_gpu(volume, proj_geom, vol_geom)
    print(f"  projection stack shape {projections.shape} "
          f"(detector_row, angle, detector_column)")
    truncated = check_truncation(projections)

    # ---- photon noise ----
    absorption_factor = estimate_absorption_factor(
        projections, AVERAGE_ABSORPTION_RATIO)
    i0 = params["i0"]
    projections_noisy = add_photon_noise(projections, i0, absorption_factor)

    noise_std = float(np.std(projections_noisy - projections))
    print(f"  absorption_factor {absorption_factor:.6g}, i0 {i0:.4g}")
    print(f"  added noise: std {noise_std:.6g} in line-integral units")
    if noise_std < 0.01 * float(projections.std()):
        print("  NOTE: noise is under 1% of the signal spread. At this photon "
              "count the\n  'noisy' set is nearly identical to the clean one - "
              "lower i0 for a real noise study.")

    np.save(os.path.join(outdir, "projections_clean.npy"),
            projections.astype(np.float32))
    np.save(os.path.join(outdir, "projections_noisy.npy"), projections_noisy)
    np.save(os.path.join(outdir, "angles.npy"), angles)
    print(f"  saved projections_clean.npy / projections_noisy.npy / angles.npy")

    if save_tiff:
        save_tiff_stack(projections, outdir, "projections_clean_tiff")
        save_tiff_stack(projections_noisy, outdir, "projections_noisy_tiff")

    # ---- geometry description ----
    geometry = {
        "geometry_type": params["geometry_type"],
        "volume_dimensions": params["volume_dimensions"],
        "volume_array_shape": list(volume.shape),
        "num_angles": params["num_angles"],
        "angular_range_degrees": params.get("angular_range_degrees", 180),
        "angles_radians_file": "angles.npy",
        "det_width_u": params["det_width_u"],
        "det_width_v": params["det_width_v"],
        "det_count_x": params["det_count_x"],
        "det_count_y": params["det_count_y"],
        "projection_array_shape": list(projections.shape),
        "projection_axis_order": "(detector_row, angle, detector_column)",
        "tiff_stacks_written": bool(save_tiff),
        "tiff_layout": (
            "projections_{clean,noisy}_tiff/proj_NNNN.tif, one 32-bit float "
            "radiograph per angle, (detector_row, detector_column), same "
            "units as the .npy stack; angle NNNN corresponds to angles.npy[NNNN]"
        ) if save_tiff else None,
        "projection_units": (
            "line integral of attenuation; noisy set produced as "
            "-log(poisson(i0*exp(-f*p))/i0)/f"
        ),
        "i0": i0,
        "absorption_factor": float(absorption_factor),
        "truncated": bool(truncated),
        "attenuation_values": {"air_or_void": 0.0, "resin_matrix": 100.0,
                               "glass_fibre": 130.0},
        "astra_vol_geom_call": (
            f"astra.create_vol_geom({params['volume_dimensions'][1]}, "
            f"{params['volume_dimensions'][2]}, {params['volume_dimensions'][0]})"
        ),
    }
    if params["geometry_type"] == "cone":
        geometry["source_origin"] = params["source_origin"]
        geometry["origin_det"] = params["origin_det"]
        total = params["source_origin"] + params["origin_det"]
        geometry["magnification"] = total / params["source_origin"]
        geometry["voxel_size_at_isocenter"] = (
            params["det_width_u"] / geometry["magnification"])

    with open(os.path.join(outdir, "geometry.json"), "w") as f:
        json.dump(geometry, f, indent=4)
    print("  saved geometry.json")

    # ---- optional reference reconstruction ----
    if save_reference_reconstruction:
        print(f"Reconstructing reference volume ({params['algorithm']})...")
        import nibabel as nib

        rec_id = astra.data3d.create("-vol", vol_geom)
        cfg = astra.astra_dict(params["algorithm"])
        cfg["ReconstructionDataId"] = rec_id
        cfg["ProjectionDataId"] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 200)
        rec = astra.data3d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)

        path = os.path.join(outdir, "reconstruction_reference.nii")
        nib.save(nib.Nifti1Image(rec, affine=np.eye(4)), path)
        print(f"  saved reconstruction_reference.nii")

    astra.data3d.delete(proj_id)
    return geometry
