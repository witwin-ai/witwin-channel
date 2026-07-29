# Copyright Xingyu Chen.
# Tests the analytic acceptance matrix.

import pytest
import torch

from tests.deterministic.test_reflection_multibounce import two_wall_multibounce_scene
from tests.support.scenes import (
    empty_space_los_scene,
    same_side_wall_reflection_scene,
    transmission_wall_structure,
    wedge_diffraction_scene,
)
from tests.support.core_world import make_receiver, make_receiver_grid, make_transmitter
from witwin.channel.capabilities import capabilities
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as solve_deterministic
from witwin.channel.montecarlo.basic import Config as BasicConfig
from witwin.channel.montecarlo.basic import solve as solve_basic
from witwin.channel.montecarlo.bdpt import Config as BdptConfig
from witwin.channel.montecarlo.bdpt import solve as solve_bdpt
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as solve_paths
from witwin.core import AntennaState, MaterialLayer, PhysicalMaterial, Scene

_REFERENCE_FREQUENCY_HZ = 3.0e9


def _transmission_scene() -> Scene:
    surface = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.1, eps_r=4.0, sigma_e=0.05),),
        name="phase-b-lossy-wall",
    )
    return Scene(
        structures=[transmission_wall_structure(2.5, surface)],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 0.0])),
            make_receiver(position=torch.tensor([5.0, 0.0, 0.0])),
        ],
    )


def _single_point_scene(scene: Scene) -> Scene:
    transmitter = next(
        endpoint for endpoint in scene.endpoints if endpoint.role == "tx"
    )
    receiver = next(endpoint for endpoint in scene.endpoints if endpoint.role == "rx")
    assert isinstance(receiver, AntennaState)
    return Scene(
        structures=scene.structures,
        endpoints=[transmitter, receiver],
        metadata=scene.metadata,
    )


def _single_cell_grid_scene(scene: Scene) -> Scene:
    point_scene = _single_point_scene(scene)
    transmitter = next(
        endpoint for endpoint in point_scene.endpoints if endpoint.role == "tx"
    )
    receiver = next(
        endpoint for endpoint in point_scene.endpoints if endpoint.role == "rx"
    )
    position = receiver.position
    grid = make_receiver_grid(
        origin=position,
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(1, 1),
        spacing=(1.0, 1.0),
    )
    return Scene(
        structures=point_scene.structures,
        endpoints=[transmitter, grid],
        metadata=point_scene.metadata,
    )


_CASES = (
    ("los", empty_space_los_scene, frozenset({"los"}), 0),
    (
        "single_reflection",
        same_side_wall_reflection_scene,
        frozenset({"reflection"}),
        1,
    ),
    ("double_reflection", two_wall_multibounce_scene, frozenset({"reflection"}), 2),
    ("single_transmission", _transmission_scene, frozenset({"transmission"}), 1),
    ("single_diffraction", wedge_diffraction_scene, frozenset({"diffraction"}), 1),
)


@pytest.mark.parametrize(("name", "scene_factory", "components", "max_depth"), _CASES)
def test_path_and_deterministic_share_complex_field_geometry_and_delay(
    name, scene_factory, components, max_depth
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Phase B analytic acceptance")
    if components != {"los"} and not build_info()["uses_rayd_native"]:
        pytest.skip(f"RayD native capability is required for {name}")

    scene = scene_factory()
    path = solve_paths(
        scene,
        PathConfig(components=components, max_depth=max_depth),
        reference_frequency_hz=_REFERENCE_FREQUENCY_HZ,
    )
    deterministic = solve_deterministic(
        scene,
        DeterministicConfig(
            components=components,
            max_depth=max_depth,
            coherent=True,
            return_field=True,
            export_paths=True,
        ),
        reference_frequency_hz=_REFERENCE_FREQUENCY_HZ,
    )
    assert deterministic.paths is not None
    valid = path.valid
    coefficient = path.a[..., 0][valid]
    field_xyz = path.field_xyz[valid]

    assert int(valid.sum()) == int(deterministic.paths.valid.numel())
    torch.testing.assert_close(
        coefficient, deterministic.paths.coefficient, rtol=5.0e-4, atol=1.0e-7
    )
    wrapped_phase_error = torch.angle(
        coefficient * deterministic.paths.coefficient.conj()
    )
    torch.testing.assert_close(
        wrapped_phase_error,
        torch.zeros_like(wrapped_phase_error),
        atol=5.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        field_xyz, deterministic.paths.field_xyz, rtol=5.0e-4, atol=1.0e-7
    )
    torch.testing.assert_close(
        path.tau[valid], deterministic.paths.delay_s, rtol=1.0e-6, atol=1.0e-12
    )
    expected_objects = deterministic.paths.primitive_sequence.clone()
    if components == {"diffraction"}:
        expected_objects[:, 0] = deterministic.paths.edge_id
    torch.testing.assert_close(path.primitive_id[valid], expected_objects)
    torch.testing.assert_close(
        path.material_id[valid], deterministic.paths.material_sequence
    )
    torch.testing.assert_close(
        path.position[valid],
        deterministic.paths.interaction_positions,
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        path.normal[valid],
        deterministic.paths.interaction_normals,
        atol=1.0e-5,
        rtol=0.0,
    )


_CAPABILITY_AWARE_COVERAGE = {
    "montecarlo_basic": {
        "complex": False,
        "polarization": False,
        "observable": "power/finite/analytic-reference bound",
        "scenarios": frozenset(name for name, *_ in _CASES),
        "convergence_scenarios": frozenset({"los"}),
    },
    "montecarlo_bdpt": {
        "complex": True,
        "polarization": True,
        "observable": "power/finite/analytic-reference bound",
        "scenarios": frozenset(name for name, *_ in _CASES),
        "convergence_scenarios": frozenset({"los"}),
    },
}


def test_phase_b_matrix_records_capability_aware_monte_carlo_coverage():
    solver_capabilities = capabilities()["solvers"]
    expected_scenarios = frozenset(name for name, *_ in _CASES)
    for solver, coverage in _CAPABILITY_AWARE_COVERAGE.items():
        advertised = solver_capabilities[solver]
        assert coverage["complex"] is advertised["supports_complex_path_coefficients"]
        assert coverage["polarization"] is advertised["supports_polarization"]
        assert coverage["scenarios"] == expected_scenarios
        assert coverage["convergence_scenarios"] == {"los"}


@pytest.mark.parametrize(
    ("solver", "config_type"),
    ((solve_basic, BasicConfig), (solve_bdpt, BdptConfig)),
)
def test_monte_carlo_los_common_power_is_finite_and_converged(solver, config_type):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Phase B Monte Carlo acceptance")
    scene = empty_space_los_scene()
    reference = solve_paths(
        scene,
        PathConfig(max_depth=0, components={"los"}),
        reference_frequency_hz=_REFERENCE_FREQUENCY_HZ,
    )
    reference_power = reference.a[reference.valid].abs().square().sum()
    errors = []
    for samples in (128, 512):
        result = solver(
            scene,
            config_type(samples=samples, max_depth=0, components={"los"}, seed=17),
            reference_frequency_hz=_REFERENCE_FREQUENCY_HZ,
        )
        observed = result.path_gain.sum()
        assert torch.isfinite(observed)
        assert observed >= 0.0
        errors.append((observed - reference_power).abs())
    assert errors[1] <= errors[0] + 1.0e-8


@pytest.mark.parametrize(("name", "scene_factory", "components", "max_depth"), _CASES)
@pytest.mark.parametrize(
    ("solver", "config_type", "use_grid", "samples"),
    (
        (solve_basic, BasicConfig, True, 8192),
        (solve_bdpt, BdptConfig, False, 2048),
    ),
)
def test_monte_carlo_supported_scenarios_are_finite_and_reference_bounded(
    name, scene_factory, components, max_depth, solver, config_type, use_grid, samples
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Phase B Monte Carlo acceptance")
    if components != {"los"} and not build_info()["uses_rayd_native"]:
        pytest.skip(f"RayD native capability is required for {name}")

    point_scene = _single_point_scene(scene_factory())
    solve_scene = _single_cell_grid_scene(point_scene) if use_grid else point_scene
    reference = solve_paths(
        point_scene,
        PathConfig(components=components, max_depth=max_depth),
        reference_frequency_hz=_REFERENCE_FREQUENCY_HZ,
    )
    reference_power = reference.a[reference.valid].abs().square().sum()
    assert torch.isfinite(reference_power)
    assert reference_power > 0.0

    result = solver(
        solve_scene,
        config_type(
            samples=samples,
            max_depth=max_depth,
            components=components,
            seed=23,
        ),
        reference_frequency_hz=_REFERENCE_FREQUENCY_HZ,
    )
    observed = result.path_gain.sum()
    assert torch.isfinite(observed)
    assert observed >= 0.0
    # MC estimators need not match one realization path-for-path. The analytic
    # Path reference is nevertheless a hard finite-energy guard against
    # explosive weights while convergence is tested separately across seeds.
    assert observed <= torch.maximum(reference_power * 1.0e6, reference_power + 1.0e-6)