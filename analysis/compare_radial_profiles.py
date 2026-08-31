#!/usr/bin/env python3
"""Compare the radial intensity profile of a real scan against a synthetic one.

Takes one cross-sectional slice from each volume and plots mean intensity as a
function of distance from the centre. Use it to check whether cupping and
other radial artefacts in the synthetic data match what the real scanner
produces - if both curves have the same shape, the artefact is a genuine
reconstruction effect rather than something the simulation invented.

Two normalisations make the curves comparable:
  * Intensity is scaled to 0-1 per volume, since real scanner units and the
    simulation's attenuation units are unrelated.
  * Radius is scaled so 1.0 is each specimen's OWN detected edge. A real
    specimen may fill half its array while the synthetic fills nearly all of
    it, and without this they would not line up.

Usage:
    python analysis/compare_radial_profiles.py \
        --real  /path/to/real_scan.npy \
        --synthetic data/reference_206k_FDK/reconstruction_clean.nii \
        --output docs/radial_profile_comparison.png
"""

import argparse

import numpy as np


def load_volume(path):
    if path.lower().endswith(".npy"):
        return np.load(path)
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def radial_profile(volume, slice_frac=0.5, n_bins=50, exclude_zero=False):
    """Mean intensity vs. normalised radius, for one slice through the volume."""
    sl = volume[int(volume.shape[0] * slice_frac), :, :].astype(float)

    cy, cz = sl.shape[0] / 2, sl.shape[1] / 2
    y, z = np.indices(sl.shape)
    r = np.sqrt((y - cy) ** 2 + (z - cz) ** 2)

    if exclude_zero:
        specimen = sl != 0
    else:
        # Treat the bottom 5% of the intensity range as background, purely to
        # locate the specimen edge.
        specimen = sl > sl.min() + 0.05 * (sl.max() - sl.min())

    specimen_radius = r[specimen].max() if specimen.any() else r.max()
    print(f"  detected specimen radius: {specimen_radius:.1f} px "
          f"(array half-width {r.max():.1f}, ratio {specimen_radius / r.max():.2f})")

    r_norm = r / specimen_radius
    bins = np.linspace(0, 1, n_bins + 1)
    profile = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (r_norm >= bins[i]) & (r_norm < bins[i + 1])
        if exclude_zero:
            m &= sl != 0
        if m.any():
            profile[i] = sl[m].mean()

    valid = ~np.isnan(profile)
    if valid.any():
        lo, hi = profile[valid].min(), profile[valid].max()
        if hi > lo:
            profile[valid] = (profile[valid] - lo) / (hi - lo)
    return (bins[:-1] + bins[1:]) / 2, profile


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--real", required=True, help="real scan volume")
    parser.add_argument("--synthetic", required=True, help="synthetic reconstruction")
    parser.add_argument("--output", default="radial_profile_comparison.png")
    parser.add_argument("--slice-frac", type=float, default=0.5,
                        help="position along axis 0 to slice (default: 0.5)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Loading real volume...")
    real = load_volume(args.real)
    print(f"  shape {real.shape}")
    print("Loading synthetic volume...")
    synth = load_volume(args.synthetic)
    print(f"  shape {synth.shape}")

    print("Real profile:")
    r_real, p_real = radial_profile(real, args.slice_frac, exclude_zero=True)
    print("Synthetic profile:")
    r_syn, p_syn = radial_profile(synth, args.slice_frac)

    plt.figure(figsize=(8, 6))
    plt.plot(r_real, p_real, label="Real scan", linewidth=2)
    plt.plot(r_syn, p_syn, label="Synthetic reconstruction", linewidth=2)
    plt.xlabel("Normalised radius (0 = centre, 1 = specimen edge)")
    plt.ylabel("Normalised intensity (shape only)")
    plt.title("Radial intensity profile: real vs synthetic")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
