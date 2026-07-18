"""Deterministic coupled reflection-diffraction paths (ADR-011).

Kernel-behaviour tests: they need the rebuilt native extension whose flat
accumulator materialises six slots (the coupled field slot 5). Against a
five-slot extension the accumulator facade's shape check fails before any
assertion, so these run post-build.
"""

from __future__ import annotations

import pytest
import torch

from tests.support.scenes import coupled_wall_wedge_scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.deterministic import Config, solve
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native coupled topology")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")


_BASE_COMPONENTS = frozenset({"los", "reflection", "diffraction"})


def test_coupled_off_is_deterministic_and_omits_coupled():
    """Coupled-off solves keep slots 0-4 and never surface a coupled row.

    A default config (coupled_paths defaults False) and an explicit
    coupled_paths=False config produce byte-identical field/component tensors,
    the coupled component is absent, and no cid 3/4 rows are enumerated. Adding
    the sixth accumulator slot must not perturb the coupled-off result.
    """

    _require_native()
    scene = coupled_wall_wedge_scene()
    default_result = solve(
        scene,
        Config(components=_BASE_COMPONENTS, max_depth=2, export_paths=True),
    )
    explicit_off = solve(
        scene,
        Config(
            components=_BASE_COMPONENTS,
            max_depth=2,
            coupled_paths=False,
            export_paths=True,
        ),
    )

    torch.testing.assert_close(
        default_result.field, explicit_off.field, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        default_result.path_gain, explicit_off.path_gain, rtol=0.0, atol=0.0
    )
    assert set(default_result.component_fields) == set(explicit_off.component_fields)
    for name, tensor in default_result.component_fields.items():
        torch.testing.assert_close(
            tensor, explicit_off.component_fields[name], rtol=0.0, atol=0.0
        )
    for name, tensor in default_result.component_power.items():
        torch.testing.assert_close(
            tensor, explicit_off.component_power[name], rtol=0.0, atol=0.0
        )

    assert "coupled" not in default_result.component_fields
    assert "coupled" not in default_result.component_power
    assert default_result.metadata["coupled_paths"]["requested"] is False
    assert default_result.paths is not None
    cid = default_result.paths.component_id
    assert not bool(((cid == 3) | (cid == 4)).any())


def test_coupled_on_exports_finite_nonzero_coupled_component():
    """Coupled-on grid solve materialises the compensator and keeps the total.

    Coupled rows exist, the coupled component map is finite and non-zero, the
    coherent field total equals the sum of the coherent field components
    (including coupled), and nothing is NaN.
    """

    _require_native()
    scene = coupled_wall_wedge_scene()
    result = solve(
        scene,
        Config(
            components=_BASE_COMPONENTS,
            max_depth=2,
            coupled_paths=True,
            export_paths=True,
        ),
    )

    assert result.metadata["coupled_paths"]["geometry"] == "native_1r1d_reciprocal"
    assert result.metadata["coupled_paths"]["coefficient"] == "unified_complex3_jones"
    assert result.metadata["counts"]["components"].get("coupled", 0) > 0

    assert result.paths is not None
    cid = result.paths.component_id
    assert bool(((cid == 3) | (cid == 4)).any())

    assert "coupled" in result.component_fields
    assert "coupled" in result.component_power
    coupled_field = result.component_fields["coupled"]
    assert torch.isfinite(coupled_field.real).all()
    assert torch.isfinite(coupled_field.imag).all()
    assert float(coupled_field.abs().sum()) > 0.0

    assert torch.isfinite(result.field.real).all()
    assert torch.isfinite(result.field.imag).all()
    assert torch.isfinite(result.path_gain).all()

    # Coherent field total is the sum of the coherent field components; the
    # coupled slot joins it like los / reflection / diffraction.
    component_sum = torch.zeros_like(result.field)
    for tensor in result.component_fields.values():
        component_sum = component_sum + tensor
    torch.testing.assert_close(result.field, component_sum, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize("ad_mode", ("jvp", "vjp"))
def test_coupled_on_runs_under_ad_modes(ad_mode):
    """AD dispatch through the coupled rows and the six-slot accumulator runs.

    No scene input requires grad, so the coupled mesh-vertex refusal does not
    trigger; the solve must complete and route the native companions.
    """

    _require_native()
    scene = coupled_wall_wedge_scene()
    result = solve(
        scene,
        Config(
            components=_BASE_COMPONENTS,
            max_depth=2,
            coupled_paths=True,
            ad_mode=ad_mode,
        ),
    )
    assert "coupled" in result.component_fields
    assert torch.isfinite(result.field.real).all()
    assert torch.isfinite(result.field.imag).all()


def test_coupled_candidate_budget_fails_loudly_before_launch(monkeypatch):
    """The per-block candidate guard fires without a Torch/CPU fallback.

    A one-candidate budget cannot fit even a single receiver block, so the
    shared plan raises before any coupled geometry kernel launches; nothing
    silently falls back to a reduced result.
    """

    _require_native()
    monkeypatch.setattr(
        geometry_bridge,
        "raydn_coupled_rd_geometry_forward",
        lambda *_args, **_kwargs: pytest.fail(
            "coupled kernel launched before the candidate guard"
        ),
    )
    config = Config(
        components=_BASE_COMPONENTS,
        max_depth=2,
        coupled_paths=True,
        coupled_candidate_limit=1,
    )
    with pytest.raises(RuntimeError, match="exceeding coupled_candidate_limit=1"):
        solve(coupled_wall_wedge_scene(), config)
