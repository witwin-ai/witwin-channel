import pytest
import torch

from tests.support.scenes import coupled_wall_wedge_scene, wedge_diffraction_scene
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.deterministic import Config as DeterministicConfig
from witwin.channel_native.deterministic import solve as solve_deterministic
from witwin.channel_native.path import Config, solve, solve_v2


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native diffraction topology")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")


def test_single_wedge_sequence_length_and_delay_match_deterministic():
    _require_native()
    scene = wedge_diffraction_scene()
    config = Config(components={"diffraction"}, max_depth=1)

    flat = solve(scene, config)
    padded = solve_v2(scene, config)
    deterministic = solve_deterministic(
        scene,
        DeterministicConfig(components={"diffraction"}, max_depth=1, export_paths=True),
    )

    assert deterministic.paths is not None
    torch.testing.assert_close(flat.edge_id, deterministic.paths.edge_id)
    torch.testing.assert_close(
        flat.path_length_m, deterministic.paths.path_length_m, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        flat.delay_s, deterministic.paths.delay_s, rtol=0.0, atol=0.0
    )
    assert torch.all(padded.interaction_type[padded.valid] == 2)
    torch.testing.assert_close(
        padded.primitive_id[padded.valid, 0], deterministic.paths.edge_id
    )


def test_solve_v2_exports_bounded_reflection_diffraction_sequences():
    _require_native()
    result = solve_v2(
        coupled_wall_wedge_scene(),
        Config(
            components={"reflection", "diffraction"},
            max_depth=2,
            coupled_paths=True,
        ),
    )

    assert (
        result.metadata["semantic_capabilities"][
            "supports_reflection_diffraction_coupling"
        ]
        is True
    )
    assert (
        result.metadata["semantic_capabilities"][
            "supports_reflection_diffraction_coupling_geometry"
        ]
        is True
    )
    active = result.interaction_type[result.valid]
    assert bool((active == torch.tensor([1, 2], device=active.device)).all(dim=1).any())
    assert bool((active == torch.tensor([2, 1], device=active.device)).all(dim=1).any())
    coupled = (active[:, 0] != 0) & (active[:, 1] != 0) & (active[:, 0] != active[:, 1])
    canonical = torch.cat(
        (active[coupled], result.primitive_id[result.valid][coupled]), dim=1
    )
    assert torch.unique(canonical, dim=0).shape[0] == canonical.shape[0]
    assert torch.isfinite(result.a[result.valid][coupled]).all()
    assert torch.count_nonzero(result.a[result.valid][coupled].abs() > 0.0) > 0
    assert result.metadata["coupled_paths"]["coefficient"] == "unified_complex3_jones"


def test_flat_solve_exports_finite_coupled_power():
    _require_native()
    config = Config(
        components={"reflection", "diffraction"},
        max_depth=2,
        coupled_paths=True,
    )
    result = solve(coupled_wall_wedge_scene(), config)
    coupled = (result.component_id == 3) | (result.component_id == 4)
    assert bool(coupled.any())
    assert torch.isfinite(result.path_gain[coupled]).all()


def test_coupled_topology_rejects_candidate_space_before_kernel_launch(monkeypatch):
    _require_native()
    monkeypatch.setattr(
        ops,
        "raydn_coupled_rd_geometry_forward",
        lambda *_args, **_kwargs: pytest.fail("coupled kernel launched before guard"),
    )
    config = Config(
        components={"reflection", "diffraction"},
        max_depth=2,
        coupled_paths=True,
        coupled_candidate_limit=1,
    )
    with pytest.raises(RuntimeError, match="exceeding coupled_candidate_limit=1"):
        solve_v2(coupled_wall_wedge_scene(), config)
