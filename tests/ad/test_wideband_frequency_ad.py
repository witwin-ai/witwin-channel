# Copyright Xingyu Chen.
# Tests wideband frequency ad.

"""Tests wideband frequency ad."""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from witwin.channel.constants import C0
from witwin.channel.propagation.consumer import (
    EndpointBatch,
    FixedTopologyRequest,
    PropagationRequest,
    evaluate,
    prepare_fixed_topology,
    reevaluate,
)
from witwin.channel.scene import compile as compile_scene
from witwin.core import Scene

from tests.ad._tolerances import FD_REL_STEP_FREQUENCY
from tests.support.scenes import rough_wall_structure
from witwin.channel.propagation.consumer import native_frequency_resolution_hz


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for frequency AD"
)


FREQUENCY_HZ = 3.0e9
WALL_X_M = 2.0

# The native launch grid is float32, so an FD step or an offset that is not a
# multiple of one ULP at the launch frequency is silently rounded and biases
# the difference by up to half a ULP. Both are quantized onto that grid using
# the published resolution function, which is what the function is for.
_RESOLUTION_HZ = native_frequency_resolution_hz(FREQUENCY_HZ)


def _on_launch_grid(value: float) -> float:
    return _RESOLUTION_HZ * round(value / _RESOLUTION_HZ)


OFFSETS = tuple(_on_launch_grid(value) for value in (-4.0e8, 0.0, 2.5e8, 6.0e8))
FD_STEP_HZ = _on_launch_grid(FREQUENCY_HZ * FD_REL_STEP_FREQUENCY)

# The FD oracle runs the SAME float32 native forward the AD path runs, so its
# noise floor is the float32 forward precision divided by the phase change over
# one step: measured at 5e-4 (line of sight) and 2.2e-3 (reflection) at this
# step, and worse at every other step tried. This bound sits above that floor
# and is still five times tighter than the suite-wide REL_TOL_GENERAL the
# existing single-frequency frequency-AD test uses for the same quantity.
#
# The tight per-column claim is carried by
# ``test_forward_and_reverse_agree_column_by_column`` instead: forward and
# reverse are two independent native companions over the same launch, so their
# agreement has no finite-difference noise in it at all.
REL_TOL = 1.0e-2


def _endpoints() -> tuple[EndpointBatch, EndpointBatch]:
    device = torch.device("cuda")

    def batch(position, stable_id, power):
        return EndpointBatch(
            stable_ids=torch.tensor([stable_id], dtype=torch.int64, device=device),
            positions_m=torch.tensor(
                [position], dtype=torch.float32, device=device
            ),
            polarizations=torch.tensor(
                [[0.0, 0.0, 1.0]], dtype=torch.float32, device=device
            ),
            powers_w=(
                None
                if power is None
                else torch.full((1,), power, dtype=torch.float32, device=device)
            ),
        )

    return (
        batch((0.0, -0.5, 0.0), 101, 1.0),
        batch((0.0, 0.5, 0.0), 707, None),
    )


def _scene(component: str) -> Scene:
    if component == "los":
        return Scene(structures=())
    return Scene(
        structures=(
            rough_wall_structure(
                WALL_X_M, rms_height_m=0.0, corr_length_m=0.1
            ),
        )
    )


def _frozen(component: str, frequency: float | torch.Tensor):
    """Compile, discover once, and freeze the rows of one component."""

    compiled = compile_scene(_scene(component), reference_frequency_hz=frequency)
    sources, sinks = _endpoints()
    discovered = evaluate(
        compiled,
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=frequency,
            components=frozenset({component}),
            max_depth=0 if component == "los" else 1,
            response="scalar_transport",
            topology_mode="discover",
            ad_mode="none",
        ),
    )
    assert discovered.paths.path_count == 1
    return compiled, sources, sinks, prepare_fixed_topology(
        discovered.paths.topology
    )


def _sweep(compiled, sources, sinks, prepared, frequency, ad_mode: str):
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=frequency,
            topology=prepared,
            response="scalar_transport",
            ad_mode=ad_mode,
            frequency_offsets_hz=OFFSETS,
        ),
    )


def _column_loss(payload: torch.Tensor, column: int) -> torch.Tensor:
    values = payload[:, column]
    return values.real.sum() + 0.5 * values.imag.sum()


def _fd_column_gradients(component: str) -> list[float]:
    """Central difference of every column with respect to the reference.

 The scene is recompiled at ``f +- h`` and replayed against the SAME frozen
 topology. That is legal precisely because frequency is not one of the four
 world version domains, so a frequency-only recompile leaves the provenance
 the frozen rows were discovered against unchanged.
 """

    step = FD_STEP_HZ
    values = []
    for frequency in (FREQUENCY_HZ + step, FREQUENCY_HZ - step):
        compiled, sources, sinks, prepared = _frozen(component, frequency)
        result = _sweep(compiled, sources, sinks, prepared, frequency, "none")
        payload = result.paths.transport.coefficient_offsets.detach()
        values.append(
            [
                float(_column_loss(payload.to(torch.complex128), column))
                for column in range(len(OFFSETS))
            ]
        )
    return [
        (plus - minus) / (2.0 * step)
        for plus, minus in zip(values[0], values[1], strict=True)
    ]


def _reverse_column_gradients(component: str) -> list[float]:
    frequency = torch.tensor(
        FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    compiled, sources, sinks, prepared = _frozen(component, frequency)
    result = _sweep(compiled, sources, sinks, prepared, frequency, "vjp")
    payload = result.paths.transport.coefficient_offsets
    assert payload.requires_grad
    gradients = []
    for column in range(len(OFFSETS)):
        frequency.grad = None
        _column_loss(payload, column).backward(retain_graph=True)
        assert frequency.grad is not None
        gradients.append(float(frequency.grad.detach()))
    return gradients


def _forward_column_gradients(component: str) -> tuple[list[float], object]:
    with forward_ad.dual_level():
        primal = torch.tensor(FREQUENCY_HZ, dtype=torch.float64, device="cuda")
        frequency = forward_ad.make_dual(primal, torch.ones_like(primal))
        compiled, sources, sinks, prepared = _frozen(component, frequency)
        result = _sweep(compiled, sources, sinks, prepared, frequency, "jvp")
        payload = result.paths.transport.coefficient_offsets
        tangent = forward_ad.unpack_dual(payload).tangent
        assert tangent is not None, "every column must carry a frequency tangent"
        gradients = [
            float(_column_loss(tangent.to(torch.complex128), column))
            for column in range(len(OFFSETS))
        ]
        delay_tangent = forward_ad.unpack_dual(result.paths.geometry.delay_s).tangent
    return gradients, delay_tangent


def test_every_launch_frequency_shares_one_float32_resolution() -> None:
    """The whole sweep lives in one float32 binade, so one quantum fits it."""

    for offset in OFFSETS:
        assert native_frequency_resolution_hz(FREQUENCY_HZ + offset) == (
            _RESOLUTION_HZ
        )
    assert FD_STEP_HZ % _RESOLUTION_HZ == 0.0
    assert FREQUENCY_HZ % _RESOLUTION_HZ == 0.0


@pytest.mark.parametrize("component", ("los", "reflection"))
def test_reverse_frequency_gradient_matches_fd_at_every_column(component) -> None:
    reverse = _reverse_column_gradients(component)
    reference = _fd_column_gradients(component)

    for column, offset in enumerate(OFFSETS):
        scale = max(abs(reference[column]), abs(reverse[column]))
        assert scale > 0.0, (component, offset)
        error = abs(reverse[column] - reference[column]) / scale
        assert error <= REL_TOL, (component, offset, error)


@pytest.mark.parametrize("component", ("los", "reflection"))
def test_forward_frequency_tangent_matches_fd_at_every_column(component) -> None:
    forward, _delay_tangent = _forward_column_gradients(component)
    reference = _fd_column_gradients(component)

    for column, offset in enumerate(OFFSETS):
        scale = max(abs(reference[column]), abs(forward[column]))
        assert scale > 0.0, (component, offset)
        error = abs(forward[column] - reference[column]) / scale
        assert error <= REL_TOL, (component, offset, error)


@pytest.mark.parametrize("component", ("los", "reflection"))
def test_forward_and_reverse_agree_column_by_column(component) -> None:
    """Different companions, same launch: they must not merely be close."""

    forward, _delay_tangent = _forward_column_gradients(component)
    reverse = _reverse_column_gradients(component)

    for column, offset in enumerate(OFFSETS):
        scale = max(abs(forward[column]), abs(reverse[column]))
        error = abs(forward[column] - reverse[column]) / scale
        assert error <= 1.0e-6, (component, offset, error)


@pytest.mark.parametrize("component", ("los", "reflection"))
def test_the_columns_are_not_the_same_derivative(component) -> None:
    """A per-column claim is only meaningful if the columns differ.

 Without this the three tests above would pass on an implementation that
 evaluated every column at the reference frequency.
 """

    reverse = _reverse_column_gradients(component)
    baseline = reverse[OFFSETS.index(0.0)]
    for column, offset in enumerate(OFFSETS):
        if offset == 0.0:
            continue
        assert abs(reverse[column] - baseline) > 1.0e-3 * abs(baseline), (
            component,
            offset,
        )


@pytest.mark.parametrize("component", ("los", "reflection"))
def test_geometry_liveness_is_one_decision_for_the_whole_sweep(component) -> None:
    """forward-mode liveness through a wideband loop: endpoint duals, every column live.

 A forward-only dual on the sink positions - no ``requires_grad`` anywhere -
 must produce the geometry tangents AND a complete payload. Deciding
 liveness inside the column loop, or letting the first column decide for the
 rest, is the defect forward-mode liveness and is what this asserts against.
 """

    compiled, sources, sinks, prepared = _frozen(component, FREQUENCY_HZ)
    velocity = torch.tensor(
        [[0.0, 12.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    with forward_ad.dual_level():
        moving = EndpointBatch(
            stable_ids=sinks.stable_ids,
            positions_m=forward_ad.make_dual(sinks.positions_m, velocity),
            polarizations=sinks.polarizations,
        )
        result = reevaluate(
            compiled,
            FixedTopologyRequest(
                sources=sources,
                sinks=moving,
                reference_frequency_hz=FREQUENCY_HZ,
                topology=prepared,
                response="scalar_transport",
                ad_mode="jvp",
                frequency_offsets_hz=OFFSETS,
            ),
        )
        payload = result.paths.transport.coefficient_offsets
        payload_tangent = forward_ad.unpack_dual(payload).tangent
        delay_tangent = forward_ad.unpack_dual(result.paths.geometry.delay_s).tangent
        length_tangent = forward_ad.unpack_dual(
            result.paths.geometry.path_length_m
        ).tangent
        assert payload_tangent is not None
        assert payload_tangent.shape == (result.paths.path_count, len(OFFSETS))
        assert delay_tangent is not None, "ADR-038: delay_s must stay live"
        assert length_tangent is not None
        delay_rate = float(delay_tangent[0])
        length_rate = float(length_tangent[0])
    # The published geometry tangent is the reference column's and is a fact
    # about motion, not frequency: delay rate is the length rate over c.
    assert abs(delay_rate - length_rate / C0) <= 1.0e-6 * abs(delay_rate)
    assert abs(delay_rate) > 0.0
    # Every column carries a nonzero Doppler tangent, not just the first.
    magnitudes = payload_tangent.abs().reshape(-1)
    assert float(magnitudes.min()) > 0.0