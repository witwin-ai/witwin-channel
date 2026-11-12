# Copyright Xingyu Chen.
# Tests that every solver's result metadata keys match its declared TypedDict.

from __future__ import annotations

from typing import Any, get_args

import pytest
import torch

from witwin.channel.capabilities import Combine, Estimator, Quantity, ResultSemantics
from witwin.channel.components import DiscoveryMetadata
from witwin.channel.runtime import KernelMetadata

from tests.support.solvers import solver as _solver

_SOLVERS = ("deterministic", "montecarlo.basic", "montecarlo.bdpt", "path")
_ALL_COMPONENTS = frozenset({"los", "reflection", "diffraction", "transmission", "scattering"})


def _declared(typed_dict: Any) -> tuple[frozenset[str], frozenset[str]]:
    required = frozenset(typed_dict.__required_keys__)
    return required, required | frozenset(typed_dict.__optional_keys__)


def _assert_matches(metadata: dict[str, object], typed_dict: Any) -> None:
    required, declared = _declared(typed_dict)
    keys = frozenset(metadata)
    assert not required - keys, f"missing required keys {sorted(required - keys)}"
    assert not keys - declared, f"undeclared keys {sorted(keys - declared)}"


@pytest.mark.parametrize("solver", sorted(_SOLVERS))
def test_solver_metadata_types_carry_the_shared_blocks(solver):
    metadata_type = _solver(solver).Metadata
    _, declared = _declared(metadata_type)
    assert frozenset(ResultSemantics.__annotations__) <= declared
    assert "kernel" in declared
    if solver != "path":
        # The path result also accepts caller-supplied metadata, so its keys are all optional.
        required, _ = _declared(metadata_type)
        assert frozenset(ResultSemantics.__required_keys__) <= required


def _scene():
    from witwin.core import Scene

    from tests.support.core_world import make_receiver, make_transmitter
    from tests.support.scenes import rough_wall_structure

    wall = rough_wall_structure(2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=1.0)
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.0])),
            make_receiver(position=torch.tensor([0.0, 1.0, 0.0])),
        ],
    )


_CONFIGS = {
    "path": {"max_depth": 1, "scattering_samples_per_m2": 32.0},
    "deterministic": {"max_depth": 1, "scattering_samples_per_m2": 32.0},
    "montecarlo.basic": {"max_depth": 1, "samples": 256},
    "montecarlo.bdpt": {"max_depth": 1, "samples": 64},
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="solves need CUDA")
@pytest.mark.parametrize("solver", sorted(_SOLVERS))
@pytest.mark.parametrize("components", [frozenset({"los"}), _ALL_COMPONENTS], ids=["los", "all"])
def test_solver_metadata_keys_are_declared(solver, components):
    module = _solver(solver)
    result = module.solve(
        _scene(),
        module.Config(components=components, **_CONFIGS[solver]),
        reference_frequency_hz=3.0e9,
    )
    metadata = result.metadata
    assert type(metadata) is dict
    _assert_matches(metadata, module.Metadata)
    _assert_matches(metadata["kernel"], KernelMetadata)
    assert metadata["estimator"] in get_args(Estimator)
    assert metadata["combine"] in get_args(Combine)
    assert set(metadata["quantity"].values()) <= set(get_args(Quantity))
    if "discovery" in metadata:
        _assert_matches(metadata["discovery"], DiscoveryMetadata)
    if solver == "path":
        _assert_matches(result.filter_by_type(0).metadata, module.Metadata)
