"""Turn a parameters JSON file into the objects the generator needs."""

from fiber_phantom.defects import DefectGenerator
from fiber_phantom.next_point_generator import NextPointGenerator

# Maps defect_type -> the config key holding that defect's settings. Without
# this, selecting a defect means hand-editing which *_params block gets passed
# through, and an earlier version of the pipeline always passed hole_params
# regardless of defect_type, so choosing a notch silently configured it with
# hole settings.
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
