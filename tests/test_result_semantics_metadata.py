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
        estimator="stochastic", quantity={"reflection": "cell_average", "los": "point_sample"},
        combine="power", complete=False,
    ) == {"estimator": "stochastic",
          "quantity": {"los": "point_sample", "reflection": "cell_average"},
          "combine": "power", "complete": False}
    with pytest.raises(ValueError, match="estimator"):
        estimator_metadata(estimator="mc", quantity=_LOS, combine="none", complete=None)
    for retired in ("enumerated", "sampled"):
        with pytest.raises(ValueError, match="estimator"):
            estimator_metadata(estimator=retired, quantity=_LOS, combine="none", complete=None)
    with pytest.raises(ValueError, match="quantity"):
        estimator_metadata(estimator="coherent", quantity={"los": "cell"}, combine="none",
                           complete=None)
    with pytest.raises(TypeError, match="quantity"):
        estimator_metadata(estimator="coherent", quantity="point_sample", combine="none",
                           complete=None)
    for bad in ("sum", "coherent", "incoherent"):
        with pytest.raises(ValueError, match="combine"):
            estimator_metadata(estimator="coherent", quantity=_LOS, combine=bad, complete=None)
    with pytest.raises(TypeError, match="complete"):
        estimator_metadata(estimator="coherent", quantity=_LOS, combine="none", complete=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="solves need CUDA")
@pytest.mark.parametrize("solver,config,expected", [
    ("path", {}, ("coherent", _POINTS, "none", True)),
    ("Coherent", {}, ("coherent", _POINTS, "field", True)),
    ("Coherent", {"combine": "power"}, ("coherent", _POINTS, "power", True)),
    ("path", {"discovery": "rays"}, ("coherent", _POINTS, "none", False)),
    ("Coherent", {"discovery": "rays"}, ("coherent", _POINTS, "field", False)),
    ("Stochastic", {"samples": 256}, ("stochastic", _MC, "power", False)),
    ("bdpt", {"samples": 64}, ("stochastic", _POINTS, "power", False)),
    ("bdpt", {"samples": 64, "coherent": True}, ("stochastic", _POINTS, "field", False)),
])
def test_every_solver_publishes_result_semantics(solver, config, expected):
    from witwin.channel import path, radiomap
    from witwin.channel.experimental import bdpt

    from tests.support.scenes import same_side_wall_reflection_scene

    config_type, solve = {
        "path": (path.Config, path.solve),
        "Coherent": (radiomap.Coherent, radiomap.solve),
        "Stochastic": (radiomap.Stochastic, radiomap.solve),
        "bdpt": (bdpt.Config, bdpt.solve),
    }[solver]
    result = solve(
        same_side_wall_reflection_scene(),
        config_type(components={"los", "reflection"}, **config),
        reference_frequency_hz=3.0e9,
    )
    assert tuple(result.metadata[key] for key in _KEYS) == expected
    if solver in {"path", "Coherent"}:
        assert result.metadata["complete"] is result.metadata["discovery"]["complete"]
