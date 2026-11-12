# Copyright Xingyu Chen.
# Tests that the consumer evaluates the diffraction component it advertises.

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize(
    "components",
    [frozenset({"diffraction"}), frozenset({"los", "reflection", "diffraction"})],
)
def test_consumer_diffraction_matches_the_deterministic_path_table(components) -> None:
    from witwin.core import AntennaState, Scene
    from witwin.core.identity import new_antenna_id

    from witwin.channel.radiomap import Coherent, solve
    from witwin.channel.propagation import consumer
    from witwin.channel.scene import compile
    from tests.propagation.consumer.test_consumer_input_guards import _batch, _scene

    assert components <= consumer.capabilities().components_for("scalar_transport")
    scene = _scene()
    request = consumer.PropagationRequest(
        sources=_batch([[0.0, 0.0, 1.5]], [1], [1.0]),
        sinks=_batch([[1.0, 1.0, 1.5], [4.0, 1.0, 1.0]], [201, 202]),
        reference_frequency_hz=3.5e9,
        components=components,
        max_depth=1,
        response="scalar_transport",
        topology_mode="discover",
        ad_mode="none",
    )
    paths = consumer.evaluate(compile(scene, reference_frequency_hz=3.5e9), request).paths
    assert int((paths.topology.component_id == 2).sum()) > 0

    endpoints = [
        AntennaState(new_antenna_id(), "tx", torch.tensor([0.0, 0.0, 1.5])),
        AntennaState(new_antenna_id(), "rx", torch.tensor([1.0, 1.0, 1.5])),
        AntennaState(new_antenna_id(), "rx", torch.tensor([4.0, 1.0, 1.0])),
    ]
    table = solve(
        Scene(structures=scene.structures, endpoints=endpoints),
        Coherent(components=set(components), max_depth=1, export_paths=True),
        reference_frequency_hz=3.5e9,
    ).paths
    assert table is not None
    assert sorted(paths.topology.component_id.tolist()) == sorted(table.component_id.tolist())
    torch.testing.assert_close(
        paths.transport.coefficient.abs().sum(), table.coefficient.abs().sum(), rtol=1e-6, atol=0.0
    )
