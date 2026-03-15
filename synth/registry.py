"""
Synthesis registry. Maps synth_type strings to callables with metadata.
"""

from .tame_synth import tame_synthesize
from .tame_synth_orders import tame_orders_synthesize
from .tame_synth_critic import tame_critic_synthesize
from .tame_synth_fusion import tame_fusion_synthesize
from .ctgan_tvae_synth import ctgan_synthesize, tvae_synthesize
from .reference_synth import (
    full_synthesize,
    random_ipc_synthesize,
    vq_synthesize,
    voronoi_synthesize,
    gonzalez_synthesize,
)

SYNTH_REGISTRY = {
    # TAME variants
    "tame": {
        "fn": tame_synthesize,
        "type": "condensation",
        "teacher_required": False,
    },
    "tame_orders": {
        "fn": tame_orders_synthesize,
        "type": "condensation",
        "teacher_required": False,
    },
    "tame_critic": {
        "fn": tame_critic_synthesize,
        "type": "condensation",
        "teacher_required": False,
    },
    "tame_fusion": {
        "fn": tame_fusion_synthesize,
        "type": "condensation",
        "teacher_required": False,
    },

    # deep generative baselines
    "ctgan": {
        "fn": ctgan_synthesize,
        "type": "generative",
        "teacher_required": False,
    },
    "tvae": {
        "fn": tvae_synthesize,
        "type": "generative",
        "teacher_required": False,
    },

    # reference methods
    "full": {
        "fn": full_synthesize,
        "type": "reference",
        "teacher_required": False,
    },
    "random": {
        "fn": random_ipc_synthesize,
        "type": "reference",
        "teacher_required": False,
    },
    "vq": {
        "fn": vq_synthesize,
        "type": "reference",
        "teacher_required": False,
    },
    "voronoi": {
        "fn": voronoi_synthesize,
        "type": "reference",
        "teacher_required": False,
    },
    "gonzalez": {
        "fn": gonzalez_synthesize,
        "type": "reference",
        "teacher_required": False,
    },
}


def synthesize(synth_type, data, config):
    if synth_type not in SYNTH_REGISTRY:
        raise ValueError(
            f"Unknown synth_type '{synth_type}'. "
            f"Available: {list(SYNTH_REGISTRY.keys())}"
        )
    entry = SYNTH_REGISTRY[synth_type]
    if entry["teacher_required"]:
        assert "teacher" in data, f"'{synth_type}' requires data['teacher']"
    return entry["fn"](data, config)
