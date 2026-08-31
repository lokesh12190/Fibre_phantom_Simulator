from setuptools import find_packages, setup

setup(
    name="fiber_phantom",
    version="1.1.0",
    description=(
        "Synthetic fibre-composite phantoms with simulated CT scanning, "
        "for validating reconstruction and segmentation against known ground truth"
    ),
    # Upstream copyright and authorship are recorded in LICENSE.
    author="Fiber-Phantoms contributors",
    packages=find_packages(include=["fiber_phantom", "fiber_phantom.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.24.2",
        "scipy==1.13.1",
        "nibabel==5.2.1",
        "matplotlib==3.7.1",
        "tifffile==2025.5.10",
    ],
    extras_require={
        # Needed only by the scripts in analysis/
        "analysis": ["SimpleITK==2.3.1", "scikit-image==0.22.0"],
    },
    # NOTE: astra-toolbox is deliberately absent - it is conda-only and needs
    # a CUDA GPU. See requirements.txt.
)
