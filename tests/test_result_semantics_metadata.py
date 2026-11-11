# Copyright Xingyu Chen.
# Tests that every solver publishes estimator, quantity, combine and complete in its metadata.

from __future__ import annotations

import pytest
import torch

from witwin.channel.capabilities import estimator_metadata

_KEYS = ("estimator", "quantity", "combine", "complete")
_LOS = {"los": "point_sample"}
_POINTS = {"los": "point_sample", "reflection": "point_sample"}
_MC = {"los": "point_sample", "reflection": "cell_average"}


def test_estimator_metadata_validates_its_vocabulary():
    assert estimator_metadata(
        estimator="sampled", quantity={"reflection": "cell_average", "los": "point_sample"},
        combine="incoherent", complete=False,
    ) == {"estimator": "sampled", "quantity": {"los": "point_sample", "reflection": "cell_average"},
          "combine": "incoherent", "complete": False}
    with pytest.raises(ValueError, match="estimator"):
        estimator_metadata(estimator="mc", quantity=_LOS, combine="none", complete=None)
    with pytest.raises(ValueError, match="quantity"):
        estimator_metadata(estimator="enumerated", quantity={"los": "cell"}, combine="none",
                           complete=None)
    with pytest.raises(TypeError, match="quantity"):
        estimator_metadata(estimator="enumerated", quantity="point_sample", combine="none",
                           complete=None)
    with pytest.raises(ValueError, match="combine"):
        estimator_metadata(estimator="enumerated", quantity=_LOS, combine="sum", complete=None)
    with pytest.raises(TypeError, match="complete"):
        estimator_metadata(estimator="enumerated", quantity=_LOS, combine="none", complete=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="solves need CUDA")
@pytest.mark.parametrize("solver,config,expected", [
    ("path", {}, ("enumerated", _POINTS, "none", True)),
    ("deterministic", {}, ("enumerated", _POINTS, "coherent", True)),
    ("deterministic", {"coherent": False}, ("enumerated", _POINTS, "incoherent", True)),
    ("path", {"discovery": "rays"}, ("enumerated", _POINTS, "none", False)),
    ("deterministic", {"discovery": "rays"}, ("enumerated", _POINTS, "coherent", False)),
    ("montecarlo.basic", {"samples": 256}, ("sampled", _MC, "incoherent", False)),
    ("montecarlo.bdpt", {"samples": 64}, ("sampled", _POINTS, "incoherent", False)),
    ("montecarlo.bdpt", {"samples": 64, "coherent": True},
     ("sampled", _POINTS, "coherent", False)),
])
def test_every_solver_publishes_result_semantics(solver, config, expected):
    import importlib

    from tests.support.scenes import same_side_wall_reflection_scene

    module = importlib.import_module(f"witwin.channel.{solver}")
    result = module.solve(
        same_side_wall_reflection_scene(),
        module.Config(components={"los", "reflection"}, **config),
        reference_frequency_hz=3.0e9,
    )
    assert tuple(result.metadata[key] for key in _KEYS) == expected
    if expected[0] == "enumerated":
        assert result.metadata["complete"] is result.metadata["discovery"]["complete"]
