#!/usr/bin/env python3
"""Measure what feature size a reconstruction can actually resolve.

Run against a `resolution` reference phantom. It reads the exact feature
locations from phantom_features.json, so it measures where the features truly
are rather than trying to find them in a blurred image.

Two things are reported per radius:

  contrast recovery   How much of the true fibre-vs-resin contrast survives at
                      the centre of the feature. 100% means the feature
                      reconstructs at its true value; low values mean blur has
                      averaged it away with the surrounding resin, so the
                      feature is present but its intensity is wrong.

  pair separation     Whether two features of that size, separated by a gap of
                      one radius, can still be told apart. Measured as the dip
                      between the two peaks, relative to peak height. The
                      Rayleigh criterion puts the resolution limit near 26%;
                      below that the pair has merged into a single blob.

The second is the one that matters for fibre composites, where fibres sit close
together: a reconstruction can show that "something is there" long after it has
stopped being able to separate neighbouring fibres.

Usage:
    python analysis/measure_resolution.py \
        --dataset data/resolution_realistic_820 --which noisy
"""

import argparse
import json
import os

import numpy as np

RESIN = 100.0
FIBRE = 130.0
# Rayleigh's criterion: two point sources are conventionally called resolved
# when the dip between them reaches about 26% of the peak height.
RAYLEIGH_DIP = 0.26


def load(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)


def align_to_truth(recon, truth):
    """Put the reconstruction on the ground-truth intensity scale."""
    a, b = np.polyfit(recon.ravel()[::97].astype(np.float64),
                      truth.ravel()[::97].astype(np.float64), 1)
    return recon.astype(np.float32) * a + b, float(a), float(b)


def sample_disc(plane, cy, cz, radius):
    """Mean value inside a small disc of `plane` centred on (cy, cz)."""
    r = max(int(np.ceil(radius)), 1)
    y0, y1 = int(round(cy)) - r, int(round(cy)) + r + 1
    z0, z1 = int(round(cz)) - r, int(round(cz)) + r + 1
    sub = plane[max(y0, 0):y1, max(z0, 0):z1]
    if sub.size == 0:
        return np.nan
    y, z = np.ogrid[max(y0, 0) - cy:y1 - cy, max(z0, 0) - cz:z1 - cz]
    return float(sub[(y ** 2 + z ** 2) <= radius ** 2].mean())


def profile_between(plane, p0, p1, n=64, pad=1.5):
    """Sample `plane` along the line through p0 and p1, extended past both."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    mid, half = (p0 + p1) / 2, (p1 - p0) / 2
    ts = np.linspace(-pad, pad, n)
    pts = mid[None, :] + ts[:, None] * half[None, :]
    yi = np.clip(np.round(pts[:, 0]).astype(int), 0, plane.shape[0] - 1)
    zi = np.clip(np.round(pts[:, 1]).astype(int), 0, plane.shape[1] - 1)
    return ts, plane[yi, zi]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--which", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--recon", help="score your own reconstruction instead "
                                        "of the shipped reference")
    parser.add_argument("--json", help="also write the report here")
    args = parser.parse_args()

    with open(os.path.join(args.dataset, "phantom_features.json")) as f:
        meta = json.load(f)
    if meta["shape"] != "resolution":
        raise SystemExit(f"This is a '{meta['shape']}' phantom; "
                         "measure_resolution.py needs a 'resolution' one.")

    truth = load(os.path.join(args.dataset, "ground_truth.nii"))
    recon_path = args.recon or os.path.join(
        args.dataset, f"reconstruction_reference_{args.which}.nii")
    print(f"truth : {os.path.join(args.dataset, 'ground_truth.nii')}")
    print(f"recon : {recon_path}")
    recon = load(recon_path)
    if recon.shape != truth.shape:
        raise SystemExit(f"Shape mismatch: {recon.shape} vs {truth.shape}")

    recon, a, b = align_to_truth(recon, truth)
    print(f"aligned to truth scale as {a:.4f}*x + {b:.2f}\n")

    # Rods run the full length of axis 0. Averaging a central slab along that
    # axis suppresses noise without blurring the cross-section being measured.
    x0, x1 = int(truth.shape[0] * 0.3), int(truth.shape[0] * 0.7)
    plane = recon[x0:x1].mean(axis=0)

    # Background resin, sampled from an annulus that holds no features: inside
    # the rod ring but outside the central spheres.
    cy, cz = truth.shape[1] / 2, truth.shape[2] / 2
    yy, zz = np.indices(plane.shape)
    rr = np.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
    ring = meta["rod_ring_radius"]
    background = float(plane[(rr > 0.25 * ring) & (rr < 0.70 * ring)].mean())
    print(f"background resin measures {background:.2f} "
          f"(true {RESIN:.0f})\n")

    rods = {}
    for f in meta["features"]:
        if f["type"] == "rod":
            rods.setdefault(f["radius"], []).append(f)

    print("RODS  (pairs separated by a gap of one radius)")
    print(f"  {'radius':>6}  {'contrast recovery':>18}  {'pair dip':>9}  verdict")
    results = {"rods": {}, "spheres": {}}
    for radius in sorted(rods):
        pair = rods[radius]
        peaks = [sample_disc(plane, *f["center_yz"], max(radius * 0.6, 0.8))
                 for f in pair]
        peak = float(np.mean(peaks))
        recovery = (peak - background) / (FIBRE - RESIN)

        _, prof = profile_between(plane, pair[0]["center_yz"], pair[1]["center_yz"])
        centre = len(prof) // 2
        dip = float(prof[centre - 1:centre + 2].min())
        height = peak - background
        dip_frac = (peak - dip) / height if height > 1e-6 else 0.0

        if recovery < 0.15:
            verdict = "not detected"
        elif dip_frac < RAYLEIGH_DIP:
            verdict = "detected, pair MERGED"
        elif recovery < 0.80:
            # Separable, but the feature does not reconstruct at its true
            # value. Counting such voxels by thresholding will undercount.
            verdict = f"resolved, contrast down to {100 * recovery:.0f}%"
        else:
            verdict = "resolved"
        print(f"  {radius:>6}  {100 * recovery:>16.1f} %  "
              f"{100 * dip_frac:>7.1f} %  {verdict}")
        results["rods"][radius] = {"contrast_recovery": recovery,
                                   "pair_dip_fraction": dip_frac,
                                   "peak_value": peak,
                                   "verdict": verdict}

    print("\nSPHERES  (isolated, no neighbour)")
    print(f"  {'radius':>6}  {'contrast recovery':>18}  verdict")
    for f in sorted((f for f in meta["features"] if f["type"] == "sphere"),
                    key=lambda f: f["radius"]):
        x, y, z = f["center_xyz"]
        radius = f["radius"]
        r = max(int(round(radius * 0.6)), 1)
        xi = int(round(x))
        box = recon[max(xi - r, 0):xi + r + 1,
                    int(y) - r:int(y) + r + 1,
                    int(z) - r:int(z) + r + 1]
        peak = float(box.mean()) if box.size else np.nan
        recovery = (peak - background) / (FIBRE - RESIN)
        verdict = "detected" if recovery >= 0.15 else "not detected"
        print(f"  {radius:>6}  {100 * recovery:>16.1f} %  {verdict}")
        results["spheres"][radius] = {"contrast_recovery": recovery,
                                      "peak_value": peak, "verdict": verdict}

    separable = [r for r, v in results["rods"].items()
                 if v["pair_dip_fraction"] >= RAYLEIGH_DIP
                 and v["contrast_recovery"] >= 0.15]
    full_contrast = [r for r, v in results["rods"].items()
                     if r in separable and v["contrast_recovery"] >= 0.80]
    print("\n" + "-" * 68)
    if separable:
        print(f"Smallest radius separable as a pair : {min(separable)} voxels")
    else:
        print("No rod pair was separable at any tested radius.")
    if full_contrast:
        print(f"Smallest radius at full contrast    : {min(full_contrast)} voxels")

    print("""
IMPORTANT - this is an ISOLATED-feature limit, not a limit for a real
specimen. These rods sit in clean resin with nothing nearby. In a real
fibre composite at 15 vol% every fibre has neighbours, their blur overlaps
and fills the gaps between them, and separability collapses well before
this figure suggests. Treat the number above as the best case the scan
geometry can offer, and check the matrix/fibre d' reported by
score_reconstruction.py on an actual fibre dataset for the realistic one.""")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"dataset": args.dataset, "which": args.which,
                       "background": background, "results": results}, f, indent=2)
        print(f"\nReport written to {args.json}")


if __name__ == "__main__":
    main()
