"""Path solver export of Kirchhoff scattering paths (plan 05 wave 3).

Scattering paths are INCOHERENT power paths: one path per contributing patch
sample, ``a`` magnitude ``sqrt(power)`` with zero (non-physical) phase, tau
from the tx->patch->rx geometric length, and the metadata flag
``scattering_paths_incoherent`` documenting the phase semantics.
"""

import pytest
import torch

from tests.support.scenes import rough_wall_structure
from witwin.channel_native import ReceiverPoint, Scene, Transmitter
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.path import Config, InteractionType, solve

_FREQUENCY_HZ = 3.0e9
_LIGHT_SPEED = 299_792_458.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)


def _require_raydn() -> None:
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native scene capability is not built")


def _scene(rms_height_m: float = 0.015) -> Scene:
    wall = rough_wall_structure(
        2.5, rms_height_m=rms_height_m, corr_length_m=0.15, half_size=2.0
    )
    return Scene(
        structures=[wall],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 0.0]))],
        frequency=_FREQUENCY_HZ,
    )


def _config(**overrides) -> Config:
    settings = {
        "max_depth": 1,
        "components": {"los", "reflection", "scattering"},
        "scattering_samples_per_m2": 16.0,
    }
    settings.update(overrides)
    return Config(**settings)


def _scattering_mask(result):
    return result.valid & (
        result.interaction_type == int(InteractionType.SCATTERING)
    ).any(dim=-1)


def test_scattering_paths_export_result():
    _require_raydn()
    result = solve(_scene(), _config())
    scattering = _scattering_mask(result)
    count = int(scattering.sum())
    assert count > 0
    assert result.metadata["components"]["scattering"] == "enabled"
    assert result.metadata["scattering"]["scattering_paths_incoherent"] is True
    assert result.metadata["scattering"]["path_count"] == count
    # Incoherent power paths carry positive gains and depth 1.
    assert bool((result.a[scattering].abs() > 0.0).all())
    # tau is the tx -> patch -> rx geometric delay, longer than the LoS.
    los_delay = 2.0 / _LIGHT_SPEED
    assert bool((result.tau[scattering] > los_delay).all())


def test_scattering_paths_export_contract():
    _require_raydn()
    scene = _scene()
    result = solve(scene, _config())
    types = result.interaction_type
    scattering_paths = result.valid & (types == int(InteractionType.SCATTERING)).any(
        dim=-1
    )
    assert int(scattering_paths.sum()) > 0
    assert result.metadata["scattering"]["scattering_paths_incoherent"] is True

    # Per scattering path: single SCATTERING event at slot 0, patch position
    # on the wall plane, tau consistent with |tx-p| + |p-rx|, and a zero-phase
    # (incoherent) coefficient.
    tx = torch.tensor([0.0, -1.0, 0.0], device=result.a.device)
    rx = torch.tensor([0.0, 1.0, 0.0], device=result.a.device)
    positions = result.position[scattering_paths][:, 0, :]
    assert torch.allclose(
        positions[:, 0], torch.full_like(positions[:, 0], 2.5), atol=1.0e-4
    )
    r1 = torch.linalg.vector_norm(positions - tx, dim=-1)
    r2 = torch.linalg.vector_norm(rx - positions, dim=-1)
    expected_tau = (r1 + r2) / _LIGHT_SPEED
    tau = result.tau[scattering_paths]
    torch.testing.assert_close(tau, expected_tau, rtol=1.0e-5, atol=1.0e-12)

    a = result.a[scattering_paths]
    assert bool((a.imag == 0.0).all())
    assert bool((a.real > 0.0).all())

    depth_types = types[scattering_paths]
    assert bool((depth_types[:, 0] == int(InteractionType.SCATTERING)).all())
    assert bool((depth_types[:, 1:] == int(InteractionType.NONE)).all())


def test_scattering_path_cap_keeps_strongest():
    _require_raydn()
    full = solve(_scene(), _config())
    capped = solve(_scene(), _config(scattering_max_paths_per_pair=8))
    full_rows = _scattering_mask(full)
    full_count = int(full_rows.sum())
    assert full_count > 8
    capped_rows = _scattering_mask(capped)
    assert int(capped_rows.sum()) == 8
    assert capped.metadata["scattering"]["capped_path_count"] == full_count - 8
    # The kept rows are the strongest ones of the full set.
    strongest = torch.topk(full.a[full_rows, 0].abs().square(), 8).values.sort().values
    kept = capped.a[capped_rows, 0].abs().square().sort().values
    torch.testing.assert_close(kept, strongest, rtol=1.0e-6, atol=0.0)


def test_scattering_power_threshold_filters_rows():
    _require_raydn()
    full = solve(_scene(), _config())
    full_rows = _scattering_mask(full)
    gains = full.a[full_rows, 0].abs().square()
    cutoff = float(gains.median())
    filtered = solve(_scene(), _config(scattering_power_threshold=cutoff))
    filtered_rows = _scattering_mask(filtered)
    kept = filtered.a[filtered_rows, 0].abs().square()
    assert int(kept.numel()) < int(gains.numel())
    assert bool((kept > cutoff).all())


def test_smooth_scene_reports_no_scattering_paths():
    _require_raydn()
    result = solve(_scene(0.0), _config())
    assert int(_scattering_mask(result).sum()) == 0
    assert result.metadata["components"]["scattering"] == "enabled_no_paths"


def test_out_of_domain_roughness_raises():
    _require_raydn()
    wall = rough_wall_structure(
        2.5, rms_height_m=0.008, corr_length_m=0.05, half_size=2.0
    )
    scene = Scene(
        structures=[wall],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 0.0]))],
        frequency=_FREQUENCY_HZ,
    )
    with pytest.raises((RuntimeError, ValueError), match="kirchhoff_domain_exceeded"):
        solve(scene, _config())
