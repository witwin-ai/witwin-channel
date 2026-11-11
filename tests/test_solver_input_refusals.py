# Copyright Xingyu Chen.
# Tests that solvers refuse unsupported receiver, AD and endpoint inputs before native work.

from __future__ import annotations

import importlib
import math

import pytest
import torch
from witwin.core import AntennaState, Mesh, PhysicalMaterial, ReceiverGrid, Scene, Structure
from witwin.core.identity import new_antenna_id

from tests.support.isolated import run_isolated

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

MODULES = {
    "det": "witwin.channel.deterministic",
    "path": "witwin.channel.path",
    "basic": "witwin.channel.montecarlo.basic",
    "bdpt": "witwin.channel.montecarlo.bdpt",
}
WALL = [[2.5, -3.0, -1.0], [2.5, 3.0, -1.0], [2.5, -3.0, 2.4], [2.5, 3.0, 2.4]]


def _wall(vertices=None) -> Structure:
    vertices = torch.tensor(WALL) if vertices is None else vertices
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], device=vertices.device)
    return Structure(
        Mesh(vertices, faces, recenter=False, fill_mode="surface", topology_diagnostics=False),
        PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
    )


def _tx(position=(0.0, -1.0, 0.5), **kwargs) -> AntennaState:
    value = position if isinstance(position, torch.Tensor) else torch.tensor(position)
    return AntennaState(new_antenna_id(), "tx", value, **kwargs)


def _rx(position=(0.0, 1.0, 0.5)) -> AntennaState:
    value = position if isinstance(position, torch.Tensor) else torch.tensor(position)
    return AntennaState(new_antenna_id(), "rx", value)


def _grid(origin=(0.5, -1.0, 0.5), shape=(4, 4), spacing=(0.5, 0.5)) -> ReceiverGrid:
    return ReceiverGrid(
        new_antenna_id(), torch.tensor(origin), torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]), shape, spacing,
    )


def _solve(solver: str, scene: Scene, **config):
    module = importlib.import_module(MODULES[solver])
    if solver in ("basic", "bdpt"):
        config.setdefault("samples", 256)
    return module.solve(scene, module.Config(**config), reference_frequency_hz=3.5e9)


def _forbid_compile(monkeypatch, solver: str) -> None:
    module = importlib.import_module(MODULES[solver])

    def fail(*_args, **_kwargs):
        raise AssertionError("the refusal must come before scene compilation")

    monkeypatch.setattr(module, "compile_scene", fail)


def test_mc_basic_solves_diffraction_on_point_receivers() -> None:
    result = _solve(
        "basic", Scene(structures=[_wall()], endpoints=[_tx(), _rx()]),
        components={"los", "diffraction"},
    )
    assert result.path_gain.shape == (1, 1)
    assert result.component_maps["diffraction"].shape == (1, 1)


@pytest.mark.parametrize("components", [{"reflection"}, {"los", "transmission"}, {"los"}])
def test_mc_basic_solves_point_receivers(components) -> None:
    result = _solve(
        "basic", Scene(structures=[_wall()], endpoints=[_tx(), _rx()]), components=components,
        samples=1 << 18,
    )
    assert result.path_gain.shape == (1, 1)
    for name in components:
        assert result.component_maps[name].shape == (1, 1)
    if "los" in components or "reflection" in components:
        assert float(result.path_gain.sum()) > 0.0


@pytest.mark.parametrize("layout", ["two_grids", "grid_and_point"])
def test_bdpt_solves_multiple_or_mixed_receivers_as_one_flat_list(layout: str) -> None:
    receivers = [_grid(), _grid(shape=(2, 2))] if layout == "two_grids" else [_grid(), _rx()]
    result = _solve("bdpt", Scene(structures=[_wall()], endpoints=[_tx(), *receivers]), components={"los"})
    assert result.path_gain.shape == (1, 16 + (4 if layout == "two_grids" else 1))
    assert result.component_maps is None
    assert float(result.path_gain.sum()) > 0.0


@pytest.mark.parametrize("layout", ["two_grids", "grid_and_point"])
def test_mc_basic_solves_multiple_or_mixed_receivers(layout: str) -> None:
    receivers = [_grid(), _grid(shape=(2, 2))] if layout == "two_grids" else [_grid(), _rx()]
    result = _solve(
        "basic", Scene(structures=[_wall()], endpoints=[_tx(), *receivers]),
        components={"los", "reflection"},
    )
    expected = 20 if layout == "two_grids" else 17
    assert result.path_gain.shape == (1, expected)


@pytest.mark.parametrize("leaf", ["tx", "rx", "vertices"])
def test_bdpt_refuses_geometry_leaves_under_ad(leaf: str) -> None:
    value = torch.tensor(WALL if leaf == "vertices" else [0.0, 1.0, 0.5], device="cuda").requires_grad_(True)
    scene = Scene(
        structures=[_wall(value if leaf == "vertices" else None)],
        endpoints=[_tx(value if leaf == "tx" else (0.0, -1.0, 0.5)), _rx(value if leaf == "rx" else (0.0, 1.0, 0.5))],
    )
    with pytest.raises(NotImplementedError, match="differentiates material, frequency and transmitter-power leaves only"):
        _solve("bdpt", scene, components={"reflection"}, ad_mode="vjp")


def test_mc_basic_refuses_vertex_leaves_through_reflection() -> None:
    vertices = torch.tensor(WALL, device="cuda").requires_grad_(True)
    scene = Scene(structures=[_wall(vertices)], endpoints=[_tx(), _grid()])
    with pytest.raises(NotImplementedError, match=r"does not differentiate mesh vertices through \['reflection'\]"):
        _solve("basic", scene, components={"los", "reflection"}, ad_mode="vjp")


@pytest.mark.parametrize("leaf", ["tx", "rx", "vertices"])
def test_path_none_mode_returns_no_graph(leaf: str) -> None:
    value = torch.tensor(WALL if leaf == "vertices" else [0.0, 1.0, 0.5], device="cuda").requires_grad_(True)
    scene = Scene(
        structures=[_wall(value if leaf == "vertices" else None)],
        endpoints=[_tx(value if leaf == "tx" else (0.0, -1.0, 0.5)), _rx(value if leaf == "rx" else (0.0, 1.0, 0.5))],
    )
    result = _solve("path", scene, components={"los", "reflection"}, ad_mode="none")
    live = [
        name
        for name in ("a", "tau", "theta_t", "phi_t", "theta_r", "phi_r", "field_xyz")
        if isinstance(getattr(result, name, None), torch.Tensor) and getattr(result, name).requires_grad
    ]
    assert live == []


@pytest.mark.parametrize("solver", ["det", "path"])
def test_reverse_mode_with_no_valid_path_returns_a_connected_zero_gradient(solver: str) -> None:
    receiver = torch.tensor([5.0, 1.0, 0.5], device="cuda").requires_grad_(True)
    scene = Scene(structures=[_wall()], endpoints=[_tx(), _rx(receiver)])
    result = _solve(solver, scene, components={"los", "reflection"}, ad_mode="vjp")
    output = result.a if solver == "path" else result.field
    assert output.requires_grad
    loss = output.abs().sum() if output.numel() else output.real.sum()
    loss.backward()
    assert receiver.grad is not None
    assert torch.count_nonzero(receiver.grad) == 0


def test_bdpt_without_structures_returns_los_like_the_other_solvers() -> None:
    scene = Scene(structures=[], endpoints=[_tx(), _rx()])
    values = {
        solver: float(_solve(solver, scene, components={"los", "reflection"}).path_gain.sum())
        if solver != "path"
        else float(_solve(solver, scene, components={"los", "reflection"}).a.abs().sum())
        for solver in MODULES
        if solver != "basic"
    }
    assert all(value > 0.0 and math.isfinite(value) for value in values.values()), values


def test_non_finite_endpoints_are_refused_before_native_work_in_every_solver() -> None:
    outcomes = run_isolated(
        """
        nan, inf = float("nan"), float("inf")
        def bad(name, endpoints):
            return lambda: solve(name, scene([floor(), wall()], endpoints), components={"los", "transmission"})
        for name in ("det", "path", "basic", "bdpt"):
            receiver = grid() if name == "basic" else rx()
            case(f"nan_tx/{name}", bad(name, [tx((nan, 0.0, 1.5)), receiver]))
            case(f"inf_tx_power/{name}", bad(name, [tx(power_w=inf), receiver]))
            case(f"negative_tx_power/{name}", bad(name, [tx(power_w=-1.0), receiver]))
            case(f"nan_orientation/{name}", bad(name, [tx(orientation=torch.tensor([nan, 0.0, 0.0])), receiver]))
            if name != "basic":
                case(f"inf_rx/{name}", bad(name, [tx(), rx((inf, 0.0, 1.5))]))
            case(f"nan_grid_origin/{name}", bad(name, [tx(), grid(origin=(nan, -2.0, 1.0))]))
            case(f"nan_grid_spacing/{name}", bad(name, [tx(), grid(spacing=torch.tensor([nan, 0.5]))]))
        """
    )
    assert outcomes.pop("__health__")["value"] is True
    for name, outcome in outcomes.items():
        assert outcome["status"] == "raised" and outcome["type"] == "ValueError", (name, outcome)
        message = str(outcome["message"])
        assert "must be finite" in message, (name, message)


@pytest.mark.parametrize("solver", list(MODULES))
def test_receiver_at_the_transmitter_position_is_refused_in_every_solver(solver: str) -> None:
    scene = Scene(structures=[_wall()], endpoints=[_tx((0.0, -1.0, 0.5)), _rx((0.0, -1.0, 0.5))])
    with pytest.raises(ValueError, match="is at the same position as transmitter"):
        _solve(solver, scene, components={"los"})


@pytest.mark.parametrize("solver", list(MODULES))
def test_scene_built_under_inference_mode_gets_a_channel_message(solver: str) -> None:
    with torch.inference_mode():
        scene = Scene(structures=[_wall()], endpoints=[_tx(), _rx()])
    with pytest.raises(ValueError, match=r"created under torch\.inference_mode\(\)"):
        _solve(solver, scene, components={"los"})


@pytest.mark.parametrize("solver", ["det", "path"])
@pytest.mark.parametrize("transform", ["grad", "vmap"])
def test_torch_func_transforms_are_refused_with_a_named_alternative(
    solver: str, transform: str,
) -> None:
    def loss(position: torch.Tensor) -> torch.Tensor:
        result = _solve(
            solver,
            Scene(structures=[_wall()], endpoints=[_tx(), _rx(position)]),
            components={"los"},
            ad_mode="vjp" if transform == "grad" else "none",
        )
        output = result.a if solver == "path" else result.field
        return output.abs().sum()

    position = torch.tensor([0.0, 1.0, 0.5], device="cuda")
    with pytest.raises(NotImplementedError, match="does not run under torch.func transforms"):
        if transform == "grad":
            torch.func.grad(loss)(position)
        else:
            torch.vmap(loss)(position.reshape(1, 3))


def test_compiled_scene_refuses_pickle_and_deepcopy_with_guidance() -> None:
    import copy
    import pickle

    from witwin.channel.scene import compile

    compiled = compile(Scene(structures=[_wall()], endpoints=[_tx(), _rx()]), reference_frequency_hz=3.5e9)
    for operation in (copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="cannot be pickled or copied"):
            operation(compiled)
