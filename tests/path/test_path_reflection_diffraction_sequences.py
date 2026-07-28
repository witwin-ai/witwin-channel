import pytest
import torch

from tests.support.scenes import coupled_wall_wedge_scene, wedge_diffraction_scene
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as solve_deterministic
from witwin.channel.path import Config, solve
from witwin.channel.kernels import geometry as geometry_kernels


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native diffraction topology")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")


def test_single_wedge_sequence_length_and_delay_match_deterministic():
    _require_native()
    scene = wedge_diffraction_scene()
    config = Config(components={"diffraction"}, max_depth=1)

    paths = solve(scene, config, reference_frequency_hz=3.0e9)
    deterministic = solve_deterministic(
        scene,
        DeterministicConfig(components={"diffraction"}, max_depth=1, export_paths=True),
        reference_frequency_hz=3.0e9,
    )

    assert deterministic.paths is not None
    torch.testing.assert_close(
        paths.primitive_id[paths.valid, 0], deterministic.paths.edge_id
    )
    torch.testing.assert_close(
        paths.path_length_m[paths.valid],
        deterministic.paths.path_length_m,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        paths.tau[paths.valid], deterministic.paths.delay_s, rtol=0.0, atol=0.0
    )
    assert torch.all(paths.interaction_type[paths.valid] == 2)
    torch.testing.assert_close(
        paths.primitive_id[paths.valid, 0], deterministic.paths.edge_id
    )


def test_solve_exports_bounded_reflection_diffraction_sequences():
    _require_native()
    result = solve(
        coupled_wall_wedge_scene(),
        Config(
            components={"reflection", "diffraction"},
            max_depth=2,
            coupled_paths=True,
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.metadata["coupled_paths"]["geometry"] == "native_1r1d_reciprocal"
    assert result.metadata["coupled_paths"]["coefficient"] == "unified_complex3_jones"
    active = result.interaction_type[result.valid]
    assert bool((active == torch.tensor([1, 2], device=active.device)).all(dim=1).any())
    assert bool((active == torch.tensor([2, 1], device=active.device)).all(dim=1).any())
    coupled = (active[:, 0] != 0) & (active[:, 1] != 0) & (active[:, 0] != active[:, 1])
    coupled_types = active[coupled]
    coupled_objects = result.primitive_id[result.valid][coupled]
    coupled_materials = result.material_id[result.valid][coupled]
    coupled_positions = result.position[result.valid][coupled]
    coupled_normals = result.normal[result.valid][coupled]
    canonical = torch.cat((coupled_types, coupled_objects), dim=1)
    assert torch.unique(canonical, dim=0).shape[0] == canonical.shape[0]
    assert torch.isfinite(result.a[result.valid][coupled]).all()
    assert torch.count_nonzero(result.a[result.valid][coupled].abs() > 0.0) > 0
    assert torch.isfinite(coupled_positions).all()
    assert torch.isfinite(coupled_normals).all()
    assert bool((coupled_objects >= 0).all())
    reflection = coupled_types == 1
    diffraction = coupled_types == 2
    assert bool((coupled_materials[reflection] >= 0).all())
    assert bool((coupled_materials[diffraction] == -1).all())
    assert bool(
        (torch.linalg.vector_norm(coupled_normals[reflection], dim=-1) > 0.0).all()
    )
    assert result.metadata["coupled_paths"]["coefficient"] == "unified_complex3_jones"


def test_flat_solve_exports_finite_coupled_power():
    _require_native()
    config = Config(
        components={"reflection", "diffraction"},
        max_depth=2,
        coupled_paths=True,
    )
    result = solve(coupled_wall_wedge_scene(), config, reference_frequency_hz=3.0e9)
    active = result.interaction_type[result.valid]
    coupled = (active[:, 0] != 0) & (active[:, 1] != 0) & (active[:, 0] != active[:, 1])
    assert bool(coupled.any())
    assert torch.isfinite(result.a[result.valid][coupled]).all()


def test_coupled_topology_rejects_candidate_space_before_kernel_launch(monkeypatch):
    _require_native()
    monkeypatch.setattr(
        geometry_kernels,
        "coupled_rd_geometry_forward",
        lambda *_args, **_kwargs: pytest.fail("coupled kernel launched before guard"),
    )
    config = Config(
        components={"reflection", "diffraction"},
        max_depth=2,
        coupled_paths=True,
        coupled_candidate_limit=1,
    )
    with pytest.raises(RuntimeError, match="exceeding coupled_candidate_limit=1"):
        solve(coupled_wall_wedge_scene(), config, reference_frequency_hz=3.0e9)
