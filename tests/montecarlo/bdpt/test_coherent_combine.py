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
    with pytest.raises(RuntimeError, match="coherent combine supports only"):
        BDPTConfig(components={"transmission"}, coherent=True)


def test_coherent_refuses_scattering():
    with pytest.raises(RuntimeError, match="coherent combine supports only"):
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
 bit-for-bit across runs and must reproduce the deterministic incoherent
 component power exactly (both consume the same enumerated first-order UTD
 evaluation and the same power accumulator). The single point receiver makes
 the cross-solver reduction bit-exact.
 """

    _skip_unless_native()
    scene = wedge_diffraction_scene()

    reference = _diffraction(
        deterministic_solve(
            scene,
            DeterministicConfig(
                components={"diffraction"}, max_depth=1, coherent=False
            ),
            reference_frequency_hz=3.0e9,
        )
    )
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
    assert torch.equal(off_a.reshape(()), reference.reshape(()))


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


def test_coherent_is_below_incoherent_here():
    """The wedge fixture has 4 diffraction rows per bin that partially cancel,
 so coherent combine is strictly below the incoherent estimate (proves the
 phasor sum is active, not a relabelled incoherent accumulation)."""

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
    assert coherent < incoherent


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