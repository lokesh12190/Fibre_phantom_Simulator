"""Synthetic fibre-composite phantom generation and CT scan simulation.

Generates a unidirectional fibre phantom (glass fibres in a resin matrix,
inside a cylindrical specimen) and simulates a cone-beam X-ray CT scan of it
with the ASTRA Toolbox. Because the phantom is built voxel-by-voxel before it
is scanned, the resulting reconstruction has exact known ground truth.
"""

__version__ = "1.1.0"
