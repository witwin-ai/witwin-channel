# Copyright Xingyu Chen.
# Tests coupled paths.

"""Tests coupled paths."""

from __future__ import annotations

import pytest
import torch

from tests.support.scenes import coupled_wall_wedge_scene
from witwin.channel.deployment import build_info
from witwin.channel.radiomap import Coherent, solve


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native coupled topology")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")


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
        Coherent(components=_BASE_COMPONENTS, max_depth=2, export_paths=True),
        reference_frequency_hz=3.0e9,
    )
    explicit_off = solve(
        scene,
        Coherent(
            components=_BASE_COMPONENTS,
            max_depth=2,
            coupled_paths=False,
            export_paths=True,
        ),
        reference_frequency_hz=3.0e9,
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
        Coherent(
            components=_BASE_COMPONENTS,
            max_depth=2,
            coupled_paths=True,
            export_paths=True,
        ),
        reference_frequency_hz=3.0e9,
    )

    assert (
        result.metadata["coupled_paths"]["geometry"] == "native_interaction_chains_plus_dd"
    )
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
        Coherent(
            components=_BASE_COMPONENTS,
            max_depth=2,
            coupled_paths=True,
            ad_mode=ad_mode,
        ),
        reference_frequency_hz=3.0e9,
    )
    assert "coupled" in result.component_fields
    assert torch.isfinite(result.field.real).all()
    assert torch.isfinite(result.field.imag).all()


def test_coupled_candidate_limit_chunks_the_validator_without_capping():
    """coupled_candidate_limit sizes the validator launches; it never drops or rejects rows.

    A one-candidate limit validates every candidate in its own launch and must reproduce the
    default solve: the same coupled rows with the same coefficients.
    """

    _require_native()
    scene = coupled_wall_wedge_scene()

    def coupled_rows(limit: int) -> torch.Tensor:
        config = Coherent(
            components=_BASE_COMPONENTS, max_depth=2, coupled_paths=True,
            coupled_candidate_limit=limit, export_paths=True,
        )
        paths = solve(scene, config, reference_frequency_hz=3.0e9).paths
        assert paths is not None
        coupled = (paths.component_id == 3) | (paths.component_id == 4) | (paths.component_id == 7)
        return paths.coefficient[coupled]

    chunked = coupled_rows(1)
    assert int(chunked.numel()) > 0
    torch.testing.assert_close(chunked, coupled_rows(1_000_000), rtol=0.0, atol=0.0)


def test_coupled_on_double_diffraction_keeps_coupled_component_finite_nonzero():
    """With D->D in the coupled union the coupled component stays finite/nonzero.

 coupled double diffraction: cid 7 double-diffraction rows aggregate into the same coupled
 slot as cid 3/4, and the path table keeps cid 7 distinct for audits. The
 coupled component map must remain finite and non-zero, the coherent total
 must still equal the sum of the coherent components, and any cid 7 rows the
 scene produces must carry finite fields (never NaN, never a silent zero).
 """

    _require_native()
    scene = coupled_wall_wedge_scene()
    result = solve(
        scene,
        Coherent(
            components=_BASE_COMPONENTS,
            max_depth=2,
            coupled_paths=True,
            export_paths=True,
        ),
        reference_frequency_hz=3.0e9,
    )

    assert "coupled" in result.component_fields
    coupled_field = result.component_fields["coupled"]
    assert torch.isfinite(coupled_field.real).all()
    assert torch.isfinite(coupled_field.imag).all()
    assert float(coupled_field.abs().sum()) > 0.0

    assert result.paths is not None
    cid = result.paths.component_id
    # cid 7 is a distinct exported component id (kept separate from the
    # aggregated coupled cids 3/4); when the scene produces such rows the
    # exported field must be finite rather than a silent zero.
    dd_rows = cid == 7
    if bool(dd_rows.any()):
        assert torch.isfinite(result.paths.field_real[dd_rows]).all()
        assert torch.isfinite(result.paths.field_imag[dd_rows]).all()

    component_sum = torch.zeros_like(result.field)
    for tensor in result.component_fields.values():
        component_sum = component_sum + tensor
    torch.testing.assert_close(result.field, component_sum, rtol=1.0e-5, atol=1.0e-6)