# Copyright Xingyu Chen.
# Tests the consumer discovery mode and the completeness flag of an evaluate result.

from __future__ import annotations

import pytest
import torch

from tests.propagation.consumer._reflection_world import endpoints, two_wall_scene
from witwin.channel.components import DISCOVERY_MODES as ENGINE_DISCOVERY_MODES
from witwin.channel.propagation.consumer import (
    DISCOVERY_MODES,
    PropagationRequest,
    capabilities,
    evaluate,
)


def _request(discovery: str | None = None) -> PropagationRequest:
    sources, sinks = endpoints()
    options = {} if discovery is None else {"discovery": discovery}
    return PropagationRequest(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=1.0e9,
        components=frozenset({"los", "reflection"}),
        max_depth=2,
        response="scalar_transport",
        topology_mode="discover",
        ad_mode="none",
        **options,
    )


def test_discovery_vocabulary_matches_the_engine_and_the_capability_record() -> None:
    assert DISCOVERY_MODES == frozenset(ENGINE_DISCOVERY_MODES)
    assert capabilities().discovery_modes == DISCOVERY_MODES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unknown_discovery_mode_is_refused_before_evaluation() -> None:
    with pytest.raises(NotImplementedError, match="unsupported discovery 'guess'"):
        _request("guess")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "discovery,complete", [(None, True), ("auto", True), ("exhaustive", True), ("rays", False)]
)
def test_evaluate_reports_completeness_of_the_requested_discovery(discovery, complete) -> None:
    compiled = two_wall_scene()

    result = evaluate(compiled, _request(discovery))

    assert result.complete is complete
    assert result.paths.path_count > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exhaustive_and_auto_discovery_publish_the_same_rows() -> None:
    compiled = two_wall_scene()

    auto = evaluate(compiled, _request("auto"))
    exhaustive = evaluate(compiled, _request("exhaustive"))

    assert auto.paths.path_count == exhaustive.paths.path_count
    torch.testing.assert_close(auto.paths.geometry.path_length_m, exhaustive.paths.geometry.path_length_m)
