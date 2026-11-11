# Copyright Xingyu Chen.
# Tests coherent combine.

"""Tests coherent combine."""

import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.core import ReceiverGrid, Scene
from tests.support.core_world import make_receiver_grid
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as deterministic_solve
from witwin.channel.montecarlo.bdpt import Config as BDPTConfig
from witwin.channel.montecarlo.bdpt import solve as bdpt_solve


_GATE_LOW = 0.5
_GATE_HIGH = 2.0


def _grid() -> ReceiverGrid:
    return make_receiver_grid(
        origin=torch.tensor([3.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _with_grid(base: Scene, grid: ReceiverGrid) -> Scene:
    return base.with_endpoints(
        (*tuple(endpoint for endpoint in base.endpoints if endpoint.role == "tx"), grid)
    )


def _diffraction(result) -> torch.Tensor:
    return result.component_power["diffraction"]


def _skip_unless_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT coherent combine")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")


# --- config-level (no CUDA) -------------------------------------------------


def test_default_is_off():
    assert BDPTConfig().coherent is False


def test_coherent_refuses_transmission():
    with pytest.raises(ValueError, match=r"coherent=True supports only the components"):
        BDPTConfig(components={"transmission"}, coherent=True)


def test_coherent_refuses_scattering():
    with pytest.raises(ValueError, match=r"coherent=True supports only the components"):
        BDPTConfig(components={"scattering"}, coherent=True)


def test_coherent_allows_los_reflection_diffraction():
    config = BDPTConfig(
        components={"los", "reflection", "diffraction"}, coherent=True, max_depth=1
    )
    assert config.coherent is True


def test_coherent_refuses_ad_mode():
    # BDPT has no AD in this release, so ad_mode != 'none' is already refused;
    # the coherent path documents the same stance (the boundary taper precedent).
    with pytest.raises((RuntimeError, ValueError)):
        BDPTConfig(components={"diffraction"}, coherent=True, ad_mode="reverse")


# --- native acceptance gates ------------------------------------------------


def test_default_off_is_bit_identical_to_incoherent():
    """Gate (a): OFF path untouched, bitwise-equal to the incoherent reference.

    The OFF path never enters the coherent kernel; it must be reproducible
    bit-for-bit across runs. It consumes the same enumerated first-order UTD
    rows as the deterministic solver and reduces each cell as a fixed-order
    float64 sum cast to float32 once, so it equals that reduction of the
    deterministic solver's exported per-row powers bit for bit. The
    deterministic solver itself sums the same rows with float32 atomics, so its
    cell differs from the float64 reduction by float32 rounding of the partial
    sums only: at most (rows - 1) half-ulps of the sum, bounded below by
    rows * eps32 * sum.
    """

    _skip_unless_native()
    scene = wedge_diffraction_scene()

    deterministic = deterministic_solve(
        scene,
        DeterministicConfig(
            components={"diffraction"}, max_depth=1, coherent=False, export_paths=True
        ),
        reference_frequency_hz=3.0e9,
    )
    reference = _diffraction(deterministic)
    rows = deterministic.paths.path_gain[deterministic.paths.valid]
    assert rows.numel() >= 1
    # The scene's edge set gives the receiver several diffraction rows, so the
    # reduction order matters below float32 resolution.
    float64_reduction = rows.to(torch.float64).sum().to(torch.float32)

    off_a = _diffraction(
        bdpt_solve(
            scene,
            BDPTConfig(
                components={"diffraction"},
                samples=512,
                seed=7,
                receiver_strategy="point_sphere",
            ),
            reference_frequency_hz=3.0e9,
        )
    )
    off_b = _diffraction(
        bdpt_solve(
            scene,
            BDPTConfig(
                components={"diffraction"},
                samples=512,
                seed=7,
                receiver_strategy="point_sphere",
            ),
            reference_frequency_hz=3.0e9,
        )
    )

    assert torch.equal(off_a, off_b)
    assert torch.equal(off_a.reshape(()), float64_reduction)
    bound = rows.numel() * torch.finfo(torch.float32).eps * float(float64_reduction)
    assert abs(float(reference.reshape(())) - float(float64_reduction)) <= bound


def test_coherent_converges_to_deterministic_coherent():
    """Gate (b): coherent ON tracks the deterministic coherent component power."""

    _skip_unless_native()
    scene = wedge_diffraction_scene()

    reference = float(
        _diffraction(
            deterministic_solve(
                scene,
                DeterministicConfig(
                    components={"diffraction"}, max_depth=1, coherent=True
                ),
                reference_frequency_hz=3.0e9,
            )
        ).sum()
    )
    observed = float(
        _diffraction(
            bdpt_solve(
                scene,
                BDPTConfig(
                    components={"diffraction"},
                    samples=512,
                    seed=7,
                    coherent=True,
                    receiver_strategy="point_sphere",
                ),
                reference_frequency_hz=3.0e9,
            )
        ).sum()
    )

    assert reference > 0.0
    assert observed > 0.0
    ratio = observed / reference
    assert _GATE_LOW <= ratio <= _GATE_HIGH, f"coherent ratio {ratio} outside [0.5, 2]"


def test_coherent_differs_from_incoherent_like_the_deterministic_phasor_sum():
    """The wedge fixture's diffraction rows interfere, so the coherent combine differs from the
 incoherent estimate on the same side as the deterministic phasor sum does (proves the phasor
 sum is active, not a relabelled incoherent accumulation)."""

    _skip_unless_native()
    scene = wedge_diffraction_scene()
    coherent = float(
        _diffraction(
            bdpt_solve(
                scene,
                BDPTConfig(
                    components={"diffraction"},
                    samples=512,
                    seed=7,
                    coherent=True,
                    receiver_strategy="point_sphere",
                ),
                reference_frequency_hz=3.0e9,
            )
        ).sum()
    )
    incoherent = float(
        _diffraction(
            bdpt_solve(
                scene,
                BDPTConfig(
                    components={"diffraction"},
                    samples=512,
                    seed=7,
                    receiver_strategy="point_sphere",
                ),
                reference_frequency_hz=3.0e9,
            )
        ).sum()
    )
    deterministic = {
        mode: float(
            _diffraction(
                deterministic_solve(
                    scene,
                    DeterministicConfig(components={"diffraction"}, max_depth=1, coherent=mode),
                    reference_frequency_hz=3.0e9,
                )
            ).sum()
        )
        for mode in (False, True)
    }
    expected = deterministic[True] - deterministic[False]
    assert abs(expected) > 1.0e-2 * deterministic[False]
    assert (coherent - incoherent) * expected > 0.0
    assert abs(coherent - incoherent) > 1.0e-2 * incoherent


def test_three_mis_modes_consistent_under_coherent():
    """Gate (c): coherent combine is MIS-invariant across the three modes."""

    _skip_unless_native()
    scene = wedge_diffraction_scene()

    def coherent_diffraction(mis: str) -> torch.Tensor:
        return _diffraction(
            bdpt_solve(
                scene,
                BDPTConfig(
                    components={"diffraction"},
                    samples=512,
                    seed=7,
                    coherent=True,
                    mis=mis,
                    receiver_strategy="point_sphere",
                ),
                reference_frequency_hz=3.0e9,
            )
        )

    balance = coherent_diffraction("balance")
    assert torch.equal(coherent_diffraction("none"), balance)
    assert torch.equal(coherent_diffraction("power_heuristic"), balance)


def test_metadata_records_combine_domain():
    _skip_unless_native()
    scene = wedge_diffraction_scene()

    off = bdpt_solve(
        scene,
        BDPTConfig(components={"diffraction"}, samples=256, seed=1),
        reference_frequency_hz=3.0e9,
    )
    on = bdpt_solve(
        scene,
        BDPTConfig(components={"diffraction"}, samples=256, seed=1, coherent=True),
        reference_frequency_hz=3.0e9,
    )
    assert off.metadata["combine_domain"] == "power"
    assert on.metadata["combine_domain"] == "coherent"
    assert on.metadata["coherent"] is True