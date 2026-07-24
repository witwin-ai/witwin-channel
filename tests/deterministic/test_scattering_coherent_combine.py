"""Deterministic opt-in coherent scattering combine (ADR-021 D3).

The combine sums the complex ``path_field`` of scattering rows per (tx, rx)
and finalizes ``|sum|^2`` (the ADR-019 per-component phasor precedent), instead
of the default incoherent ``SUM |field|^2`` power sum. It is opt-in via
``Config.scattering_coherent`` and defaults OFF, keeping the scattering slot
bit-identical to today. It is physical only for realization-coherent
phase-screen rows (true complex field); ensemble rows are zero-phase power rows
and an ensemble-only solve is refused loudly.
"""

import pytest
import torch

from tests.support.scenes import rough_wall_structure
from witwin.core import Scene
from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.core.kernels.extension import build_info
from witwin.core import PhaseScreen, SurfaceRoughness
from witwin.channel.deterministic import Config, solve
from witwin.channel.scattering import (
    generate_gaussian_realization,
    realization_seed,
)

_FREQUENCY_HZ = 3.0e9

# Only the solve-based tests need CUDA; the config-validation tests are pure
# Python and run on CPU-only validation.
_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)


def _require_rayd() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scene capability is not built")


def _screen(realization_id: int) -> PhaseScreen:
    rough = SurfaceRoughness(
        rms_height_m=0.008, correlation_length_x_m=0.15, correlation_length_y_m=0.15
    )
    height = generate_gaussian_realization(
        rough,
        extent_m=2.0,
        resolution=256,
        seed=realization_seed(0, 1, realization_id),
        device="cpu",
    )
    return PhaseScreen(
        height=height,
        height_scale_m=1.0,
        realization_id=realization_id,
        mode="realization_coherent",
    )


def _wall(x_m: float, *, realization_id: int, surface_id: int, name: str):
    return rough_wall_structure(
        x_m,
        rms_height_m=0.008,
        corr_length_m=0.15,
        half_size=1.0,
        phase_screen=_screen(realization_id),
        with_uv=True,
        name=name,
        surface_id=surface_id,
    )


def _scene(structures) -> Scene:
    return Scene(
        structures=structures,
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.0])),
            make_receiver(position=torch.tensor([0.0, 1.0, 0.0])),
        ],
    )


def _config(**overrides) -> Config:
    settings = {
        "max_depth": 1,
        "components": {"reflection", "scattering"},
        "scattering_samples_per_m2": 64.0,
        "scattering_max_paths_per_pair": 65536,
    }
    settings.update(overrides)
    return Config(**settings)


# --- config-level validation (no CUDA solve needed) -----------------------


def test_config_refuses_coherent_without_scattering_component():
    with pytest.raises(RuntimeError, match="requires the 'scattering' component"):
        Config(scattering_coherent=True, components={"reflection"})


def test_config_default_components_refuse_coherent():
    # The default component set has no scattering; opting into the combine
    # without requesting scattering is a config-level error.
    with pytest.raises(RuntimeError, match="requires the 'scattering' component"):
        Config(scattering_coherent=True)


def test_config_default_flag_is_false():
    assert Config().scattering_coherent is False
    assert Config(scattering_coherent=False) == Config()


# --- default bitwise identity ---------------------------------------------


@_needs_cuda
def test_default_flag_absent_matches_explicit_false():
    """Flag absent and scattering_coherent=False produce identical results."""

    _require_rayd()
    scene = _scene([_wall(2.5, realization_id=1, surface_id=1, name="w1")])
    absent = solve(scene, _config(), reference_frequency_hz=_FREQUENCY_HZ)
    explicit = solve(
        scene, _config(scattering_coherent=False), reference_frequency_hz=_FREQUENCY_HZ
    )
    assert torch.equal(
        absent.component_power["scattering"],
        explicit.component_power["scattering"],
    )
    assert torch.equal(absent.path_gain, explicit.path_gain)
    assert torch.equal(absent.field, explicit.field)
    assert absent.metadata["scattering"]["combine_domain"] == "incoherent_power"


# --- single-row parity -----------------------------------------------------


@_needs_cuda
def test_single_row_coherent_equals_incoherent():
    """One scattering row per (tx, rx): |sum|^2 == sum |.|^2 exactly."""

    _require_rayd()
    scene = _scene([_wall(2.5, realization_id=1, surface_id=1, name="w1")])
    incoherent = solve(scene, _config(), reference_frequency_hz=_FREQUENCY_HZ)
    coherent = solve(
        scene, _config(scattering_coherent=True), reference_frequency_hz=_FREQUENCY_HZ
    )
    # A single realization structure emits exactly one scattering row per
    # (tx, rx), so the coherent |sum|^2 collapses to the single-row power.
    assert incoherent.metadata["scattering"]["path_count"] == 1
    torch.testing.assert_close(
        coherent.component_power["scattering"],
        incoherent.component_power["scattering"],
        rtol=1.0e-6,
        atol=1.0e-30,
    )
    assert coherent.metadata["scattering"]["combine_domain"] == "coherent"


# --- multi-row interference ------------------------------------------------


@_needs_cuda
def test_multi_row_coherent_differs_and_is_reproducible():
    """Two scattering rows per (tx, rx) interfere; the combine is bit-exact
    run-to-run."""

    _require_rayd()
    structures = [
        _wall(2.5, realization_id=1, surface_id=1, name="w1"),
        _wall(-2.5, realization_id=2, surface_id=2, name="w2"),
    ]
    incoherent = solve(
        _scene(structures), _config(), reference_frequency_hz=_FREQUENCY_HZ
    )
    coherent_a = solve(
        _scene(structures),
        _config(scattering_coherent=True),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    coherent_b = solve(
        _scene(structures),
        _config(scattering_coherent=True),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    assert incoherent.metadata["scattering"]["path_count"] == 2
    # Interference: the coherent |sum|^2 is not the incoherent power sum.
    assert not torch.equal(
        coherent_a.component_power["scattering"],
        incoherent.component_power["scattering"],
    )
    # Deterministic speckle: the same scene/realization is bit-for-bit
    # reproducible.
    assert torch.equal(
        coherent_a.component_power["scattering"],
        coherent_b.component_power["scattering"],
    )
    assert torch.equal(coherent_a.path_gain, coherent_b.path_gain)


# --- solve-time refusal of ensemble-only / empty-realization scenes --------


@_needs_cuda
def test_pipeline_refuses_ensemble_only_scene():
    """A rough material without a realization phase screen is ensemble-only;
    the coherent combine is refused loudly."""

    _require_rayd()
    ensemble_wall = rough_wall_structure(
        2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=1.0
    )
    scene = _scene([ensemble_wall])
    with pytest.raises(RuntimeError, match="ensemble scattering surfaces"):
        solve(
            scene,
            _config(scattering_coherent=True),
            reference_frequency_hz=_FREQUENCY_HZ,
        )


@_needs_cuda
def test_pipeline_refuses_scene_without_scattering_surfaces():
    """Scattering requested but no rough/realization surface: nothing to
    combine coherently."""

    _require_rayd()
    smooth_wall = rough_wall_structure(
        2.5, rms_height_m=0.0, corr_length_m=0.15, half_size=1.0
    )
    scene = _scene([smooth_wall])
    with pytest.raises(RuntimeError, match="realization_coherent phase-screen"):
        solve(
            scene,
            _config(scattering_coherent=True),
            reference_frequency_hz=_FREQUENCY_HZ,
        )
