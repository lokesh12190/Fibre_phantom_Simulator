#!/usr/bin/env python3
"""Measure reconstruction artefacts using a uniform reference phantom.

Run against a `uniform` reference phantom. Inside the specimen every voxel is
the same value by construction, so anything that is not flat in the
reconstruction is an artefact, and can be quantified rather than eyeballed.

Reported:

  cupping        Radial trend from the specimen axis to its wall. Beam
                 hardening pulls the centre down relative to the edge, which
                 looks like a genuine density gradient and is a common way to
                 mis-measure a real specimen.

  noise floor    Local scatter with the radial trend removed - what a
                 reconstruction adds where the object contains nothing.

  ring artefacts Structure that is constant in angle at fixed radius, which is
                 what detector-element differences produce.

  axial drift    Variation along the rotation axis, where cone-beam methods
                 degrade away from the central plane.

Usage:
    python analysis/measure_artefacts.py \
        --dataset data/uniform_generic_820 --which clean
"""

import argparse
import json
import os

import numpy as np

RESIN = 100.0


def load(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--which", choices=["clean", "noisy"], default="clean")
    parser.add_argument("--recon", help="measure your own reconstruction "
                                        "instead of the shipped reference")
    parser.add_argument("--erode", type=int, default=20,
                        help="ignore this many voxels inside the wall, where "
                             "edge blur dominates (default: 20)")
    parser.add_argument("--json", help="also write the report here")
    args = parser.parse_args()

    features_path = os.path.join(args.dataset, "phantom_features.json")
    with open(features_path) as f:
        meta = json.load(f)
    if meta["shape"] != "uniform":
        raise SystemExit(f"This is a '{meta['shape']}' phantom; "
                         "measure_artefacts.py needs a 'uniform' one.")

    truth = load(os.path.join(args.dataset, "ground_truth.nii"))
    recon_path = args.recon or os.path.join(
        args.dataset, f"reconstruction_reference_{args.which}.nii")
    print(f"recon : {recon_path}")
    recon = load(recon_path).astype(np.float32)
    if recon.shape != truth.shape:
        raise SystemExit(f"Shape mismatch: {recon.shape} vs {truth.shape}")

    pipe_radius = meta["pipe_radius"]
    mask_radius = pipe_radius - args.erode
    cy, cz = recon.shape[1] / 2, recon.shape[2] / 2
    y, z = np.indices((recon.shape[1], recon.shape[2]))
    rr = np.sqrt((y - cy) ** 2 + (z - cz) ** 2)
    disc = rr <= mask_radius
    mask = np.broadcast_to(disc[np.newaxis, :, :], recon.shape)

    # Two-point calibration: air outside the specimen maps to 0, the mean of
    # the specimen interior maps to its true value.
    #
    # A least-squares fit against the ground truth CANNOT be used here. Inside
    # the mask the truth of a uniform phantom is a single constant, so the fit
    # returns slope 0 and intercept 100 and flattens the reconstruction into
    # that constant - reporting a flawless result and destroying every artefact
    # the script exists to measure. Anchoring on two distinct known materials
    # sets the scale while leaving all variation within the specimen intact.
    air_band = (rr > pipe_radius + 20) & (rr < pipe_radius + 80)
    if not air_band.any():
        raise SystemExit("No air region found outside the specimen to "
                         "calibrate against.")
    air_ref = float(recon[np.broadcast_to(air_band[np.newaxis, :, :],
                                          recon.shape)].mean())
    resin_ref = float(recon[mask].mean())
    if abs(resin_ref - air_ref) < 1e-9:
        raise SystemExit("Air and specimen reconstruct to the same value; "
                         "cannot calibrate.")
    recon = (recon - air_ref) / (resin_ref - air_ref) * RESIN
    print(f"calibrated: air {air_ref:.3f} -> 0, "
          f"specimen mean {resin_ref:.3f} -> {RESIN:.0f}")
    print(f"specimen mask: radius {mask_radius} "
          f"({int(mask.sum()):,} voxels), true value {RESIN:.0f}\n")

    vals = recon[mask]
    report = {"mean": float(vals.mean()), "std": float(vals.std()),
              "min": float(vals.min()), "max": float(vals.max())}
    print("OVERALL")
    print(f"  mean {report['mean']:.3f}   std {report['std']:.3f}   "
          f"range {report['min']:.2f} to {report['max']:.2f}")
    print(f"  -> a perfect reconstruction would read {RESIN:.0f} everywhere, "
          f"std 0\n")

    # ---- cupping: mean value as a function of radius ----
    n_bins = 12
    edges = np.linspace(0, mask_radius, n_bins + 1)
    mid_slab = recon[int(recon.shape[0] * 0.4):int(recon.shape[0] * 0.6)]
    print("CUPPING  (mean value vs radius, central slab)")
    radial = []
    for i in range(n_bins):
        band = (rr >= edges[i]) & (rr < edges[i + 1])
        m = np.broadcast_to(band[np.newaxis, :, :], mid_slab.shape)
        v = float(mid_slab[m].mean())
        radial.append(v)
        bar = "#" * max(int(round((v - RESIN) * 8 + 20)), 0)
        print(f"  r {edges[i]:6.0f}-{edges[i+1]:6.0f}: {v:8.3f}  {bar}")
    cupping = radial[-1] - radial[0]
    report["radial_profile"] = radial
    report["cupping_edge_minus_centre"] = cupping
    print(f"  edge - centre = {cupping:+.3f} "
          f"({'edge brighter, classic cupping' if cupping > 0 else 'centre brighter'})\n")

    # ---- noise floor, radial trend removed ----
    trend = np.interp(rr, (edges[:-1] + edges[1:]) / 2, radial)
    detrended = mid_slab - trend[np.newaxis, :, :]
    m = np.broadcast_to(disc[np.newaxis, :, :], mid_slab.shape)
    noise = float(detrended[m].std())
    report["noise_floor"] = noise
    print(f"NOISE FLOOR (radial trend removed): std {noise:.3f}")
    print(f"  -> for reference, true fibre-vs-resin contrast is 30.0, so this "
          f"noise is\n     {100 * noise / 30:.1f}% of the signal a fibre "
          f"would produce\n")

    # ---- ring artefacts: structure constant in angle at fixed radius ----
    # Averaging over angle at each radius leaves only radial structure; the
    # scatter of that curve beyond the smooth cupping trend indicates rings.
    fine = np.linspace(0, mask_radius, 200)
    prof = []
    slab = mid_slab.mean(axis=0)
    for i in range(len(fine) - 1):
        band = (rr >= fine[i]) & (rr < fine[i + 1])
        prof.append(float(slab[band].mean()) if band.any() else np.nan)
    prof = np.array(prof)
    good = np.isfinite(prof)
    smooth = np.convolve(prof[good], np.ones(9) / 9, mode="same")
    ring = float(np.nanstd((prof[good] - smooth)[4:-4]))
    report["ring_metric"] = ring
    print(f"RING ARTEFACTS: {ring:.4f}")
    print(f"  -> residual after removing the smooth radial trend; near zero "
          f"means no rings\n")

    # ---- axial drift along the rotation axis ----
    n_slabs = 10
    bounds = np.linspace(0, recon.shape[0], n_slabs + 1).astype(int)
    print("AXIAL DRIFT  (mean value along the rotation axis)")
    axial = []
    for i in range(n_slabs):
        v = float(recon[bounds[i]:bounds[i + 1]][
                      np.broadcast_to(disc[np.newaxis, :, :],
                                      (bounds[i + 1] - bounds[i],) + disc.shape)].mean())
        axial.append(v)
        print(f"  slice {bounds[i]:5d}-{bounds[i+1]:5d}: {v:8.3f}")
    drift = max(axial) - min(axial)
    report["axial_profile"] = axial
    report["axial_drift"] = drift
    print(f"  max - min = {drift:.3f}  "
          f"({'significant' if drift > 1.0 else 'small'} variation along the axis)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"dataset": args.dataset, "which": args.which,
                       **report}, f, indent=2)
        print(f"\nReport written to {args.json}")


if __name__ == "__main__":
    main()
