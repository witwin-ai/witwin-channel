# Copyright Xingyu Chen.
# Tests that the consumer refuses stale, cross-device and out-of-range inputs before native work.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.support.isolated import run_isolated

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_FREQUENCY = 3.5e9
_WALL = [[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [2.5, -2.0, 3.0], [2.5, 2.0, 3.0]]


def _scene(vertices: torch.Tensor | None = None):
    from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

    floor = torch.tensor([[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [-10.0, 10.0, 0.0], [10.0, 10.0, 0.0]])
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])

    def structure(v):
        return Structure(
            Mesh(v, faces, recenter=False, fill_mode="surface", topology_diagnostics=False),
            PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        )

    return Scene(structures=[structure(floor), structure(torch.tensor(_WALL) if vertices is None else vertices)])


def _batch(positions, ids, power=None):
    from witwin.channel.propagation.consumer import EndpointBatch

    values = torch.tensor(positions, dtype=torch.float32, device="cuda").reshape(-1, 3)
    count = values.shape[0]
    return EndpointBatch(
        stable_ids=torch.tensor(ids, dtype=torch.int64, device="cuda"),
        positions_m=values,
        polarizations=torch.tensor([[0.0, 0.0, 1.0]], device="cuda").repeat(count, 1).contiguous(),
        powers_w=None if power is None else torch.tensor(power, device="cuda"),
    )


def _request(**kwargs):
    from witwin.channel.propagation.consumer import PropagationRequest

    kwargs.setdefault("max_depth", 1)
    return PropagationRequest(
        sources=_batch([[0.0, 0.0, 1.5]], [1], [1.0]),
        sinks=_batch([[1.0, 1.0, 1.5]], [201]),
        reference_frequency_hz=_FREQUENCY,
        components=frozenset({"los", "reflection"}),
        response="scalar_transport",
        topology_mode="discover",
        ad_mode="none",
        **kwargs,
    )


def _fixed(topology, **kwargs):
    from witwin.channel.propagation.consumer import FixedTopologyRequest

    return FixedTopologyRequest(
        sources=_batch([[0.0, 0.0, 1.5]], [1], [1.0]),
        sinks=_batch([[1.0, 1.0, 1.5]], [201]),
        reference_frequency_hz=_FREQUENCY,
        topology=topology,
        response="scalar_transport",
        ad_mode="none",
        **kwargs,
    )


def test_in_place_source_edit_after_compile_is_refused_by_evaluate_and_reevaluate() -> None:
    from witwin.channel.propagation import consumer
    from witwin.channel.scene import compile

    vertices = torch.tensor(_WALL)
    scene = _scene(vertices)
    compiled = compile(scene, reference_frequency_hz=_FREQUENCY)
    discovered = consumer.evaluate(compiled, _request())
    prepared = consumer.prepare_fixed_topology(discovered.paths.topology)
    vertices[:, 0] += 1.0
    with pytest.raises(ValueError, match="the compiled scene is stale: its source scene's geometry_version changed"):
        consumer.evaluate(compiled, _request())
    with pytest.raises(ValueError, match="the compiled scene is stale"):
        consumer.reevaluate(compiled, _fixed(prepared))
    fresh = compile(scene, reference_frequency_hz=_FREQUENCY)
    assert consumer.evaluate(fresh, _request()).paths.path_count > 0


def test_endpoint_device_must_match_the_compiled_scene() -> None:
    from witwin.channel.propagation.consumer import _require_scene_device

    compiled = SimpleNamespace(geometry=SimpleNamespace(vertices=SimpleNamespace(device=torch.device("cuda", 7))))
    sources = SimpleNamespace(device=torch.device("cuda", 0))
    with pytest.raises(ValueError, match=r"endpoint batches are on cuda:0 but the compiled scene lives on cuda:7"):
        _require_scene_device(compiled, sources, sources)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sources are on cuda:0 but sinks are on cuda:7"):
        _require_scene_device(
            compiled, sources, SimpleNamespace(device=torch.device("cuda", 7))  # type: ignore[arg-type]
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two visible GPUs")
def test_endpoints_on_another_gpu_are_refused() -> None:
    from witwin.channel.propagation import consumer
    from witwin.channel.scene import compile

    compiled = compile(_scene(), reference_frequency_hz=_FREQUENCY)
    with torch.cuda.device(1):
        request = _request()
    with pytest.raises(ValueError, match="but the compiled scene lives on"):
        consumer.evaluate(compiled, request)


def test_max_paths_overflow_fails_instead_of_truncating() -> None:
    from witwin.channel.propagation import consumer
    from witwin.channel.scene import compile

    compiled = compile(_scene(), reference_frequency_hz=_FREQUENCY)
    count = consumer.evaluate(compiled, _request()).paths.path_count
    assert count >= 2
    assert consumer.evaluate(compiled, _request(max_paths=count)).paths.path_count == count
    with pytest.raises(RuntimeError, match=f"more than max_paths={count - 1} paths"):
        consumer.evaluate(compiled, _request(max_paths=count - 1))


def test_tampered_or_foreign_frozen_topology_is_refused_and_the_context_survives() -> None:
    outcomes = run_isolated(
        """
        import dataclasses
        from witwin.channel.propagation import consumer as c
        from witwin.channel.scene import compile
        def batch(positions, ids, power=None):
            p = torch.tensor(positions, dtype=torch.float32, device="cuda").reshape(-1, 3)
            return c.EndpointBatch(stable_ids=torch.tensor(ids, device="cuda"), positions_m=p,
                polarizations=torch.tensor([[0.0, 0.0, 1.0]], device="cuda").repeat(p.shape[0], 1).contiguous(),
                powers_w=None if power is None else torch.tensor(power, device="cuda"))
        src, snk = batch([[0.0, 0.0, 1.5]], [1], [1.0]), batch([[1.0, 1.0, 1.5]], [201])
        compiled = compile(scene(), reference_frequency_hz=3.5e9)
        found = c.evaluate(compiled, c.PropagationRequest(sources=src, sinks=snk, reference_frequency_hz=3.5e9,
            components=frozenset({"los", "reflection"}), max_depth=1, response="scalar_transport",
            topology_mode="discover", ad_mode="none")).paths.topology
        def replay(target, topology):
            return c.reevaluate(target, c.FixedTopologyRequest(sources=src, sinks=snk, reference_frequency_hz=3.5e9,
                topology=c.prepare_fixed_topology(topology), response="scalar_transport", ad_mode="none")).row_valid.tolist()
        fields = {f.name: getattr(found, f.name) for f in dataclasses.fields(found)}
        big = dict(fields, primitive_sequence=torch.where(found.primitive_sequence >= 0, torch.full_like(found.primitive_sequence, 10**6), found.primitive_sequence))
        case("primitive_oob", lambda: replay(compiled, type(found)(**big)))
        small = compile(Scene(structures=[mesh(torch.tensor([[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [2.5, 0.0, 3.0]]), torch.tensor([[0, 1, 2]]))]), reference_frequency_hz=3.5e9)
        case("foreign_scene", lambda: replay(small, dataclasses.replace(found, provenance=None)))
        case("valid", lambda: replay(compiled, found))
        """
    )
    assert outcomes.pop("__health__")["value"] is True
    assert outcomes["valid"]["status"] == "returned"
    for name in ("primitive_oob", "foreign_scene"):
        outcome = outcomes[name]
        assert outcome["status"] == "raised" and outcome["type"] == "ValueError", (name, outcome)
        assert "primitive_sequence face index outside the compiled scene's faces" in str(outcome["message"])
