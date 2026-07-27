"""Wideband frequency offsets on the fixed-topology consumer route (ADR-042).

Version 4 published one coefficient at the compiled reference frequency and a
narrowband law a caller could apply to shift it. These tests pin the capability
that replaces that law with an evaluation: the same frozen rows, evaluated
natively at ``F`` declared absolute frequencies, published as a ``[K, F]``
payload paired with the grid it was evaluated on.

Four things are worth stating about what is asserted here, because each one is
a claim the design rests on.

The reference identity is BITWISE, not a tolerance. A ``0.0`` entry re-launches
at the same float32 frequency with the same inputs, so it must reproduce the
reference column exactly; anything less would mean the wideband route is a
different evaluation wearing the same name.

The material comparisons are against closed forms, not oracles in the loose
sense. ``fresnel_interface`` and ``layer_stack_rt`` are the analytic
half-space and transfer-matrix expressions, and the free-space factor
``(c/f)/(4*pi*L)*exp(-j*2*pi*f*L/c)`` is exact. The fixture geometry is
deliberately arranged so the source and sink polarizations are both the
out-of-plane axis, which makes the row a pure TE response with a unit
projection and removes every basis convention from the comparison.

The multilayer sweep is the fixture that FALSIFIES a narrowband
implementation. A 0.1 m eps_r=4 slab fringes every 755 MHz at this incidence,
so a grid spanning 2.4 GHz crosses three nulls that a narrowband law cannot
express at all.

The three scene-dependent refusals are each reached by a case that trips only
that one. A refusal that is only reachable through another refusal is not a
discoverable limit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from witwin.channel.constants import C0
from witwin.channel.propagation.consumer import (
    Complex3Transport,
    FixedTopologyRequest,
    PropagationConvention,
    PropagationRequest,
    PropagationTopology,
    ScalarTransport,
    capabilities,
    evaluate,
    native_frequency_resolution_hz,
    prepare_fixed_topology,
    reevaluate,
    replicate_over_slots,
)
from witwin.channel.scene import compile as compile_scene
from witwin.core import MaterialLayer, PhysicalMaterial, Scene

from tests.propagation.consumer._multi_endpoint_world import (
    FROZEN_ROW_COUNT,
    SINK_POSITIONS,
    SOURCE_POSITIONS,
    compiled_world,
    cuda_positions,
    frozen_topology,
    replay,
    sink_batch,
    slot_offsets,
    source_batch,
    stack_over_slots,
)
from tests.propagation.consumer._reflection_world import endpoints
from tests.reference.em_oracle import (
    fresnel_interface,
    layer_stack_rt,
    medium_params,
    vacuum_medium,
)
from tests.support.scenes import rough_wall_structure


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


WALL_X_M = 2.0
# Thick and lossy enough that |exp(-j*k_z*d)|^2 < 1e-10 at the reference
# frequency, so the layer stack degenerates to the bare vacuum/dielectric
# interface and the half-space closed form is exact.
HALF_SPACE = {"eps_r": 4.0, "sigma_e": 0.5, "thickness_m": 0.3}
# The slab that fringes: 0.1 m of eps_r 4 fringes every 755 MHz at this
# incidence, so a 2.4 GHz sweep crosses three nulls.
SLAB = {"eps_r": 4.0, "sigma_e": 0.01, "thickness_m": 0.1}
# The survey's default wall, used for the narrowband error bound at 77 GHz.
DEFAULT_WALL = {"eps_r": 4.0, "sigma_e": 0.02, "thickness_m": 0.1}


def _wall_scene(frequency_hz: float, **material) -> object:
    scene = Scene(
        structures=(
            rough_wall_structure(
                WALL_X_M, rms_height_m=0.0, corr_length_m=0.1, **material
            ),
        )
    )
    return compile_scene(scene, reference_frequency_hz=frequency_hz)


def _empty_scene(frequency_hz: float) -> object:
    return compile_scene(Scene(structures=()), reference_frequency_hz=frequency_hz)


def _discover(
    compiled,
    sources,
    sinks,
    frequency_hz: float,
    *,
    components: frozenset[str],
    max_depth: int = 1,
    response: str = "scalar_transport",
):
    return evaluate(
        compiled,
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=frequency_hz,
            components=components,
            max_depth=max_depth,
            response=response,
            topology_mode="discover",
            ad_mode="none",
        ),
    )


def _reflection_world(
    frequency_hz: float,
    material: dict,
    *,
    response: str = "scalar_transport",
):
    """One frozen reflection row off a single wall, ready for a sweep."""

    compiled = _wall_scene(frequency_hz, **material)
    sources, sinks = endpoints(with_basis=False)
    discovered = _discover(
        compiled,
        sources,
        sinks,
        frequency_hz,
        components=frozenset({"reflection"}),
        response=response,
    )
    assert discovered.paths.path_count == 1
    prepared = prepare_fixed_topology(discovered.paths.topology)
    return compiled, sources, sinks, prepared


def _sweep(
    compiled,
    sources,
    sinks,
    prepared,
    frequency_hz: float,
    offsets: tuple[float, ...],
    *,
    response: str = "scalar_transport",
    ad_mode: str = "none",
):
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=frequency_hz,
            topology=prepared,
            response=response,
            ad_mode=ad_mode,
            frequency_offsets_hz=offsets,
        ),
    )


def _incidence_cosine(geometry, source_position: torch.Tensor, row: int = 0) -> float:
    """Read the incidence cosine off the geometry the replay published.

    Taken from the interaction table the native transport itself consumed
    rather than rederived from the wall equation, so the closed form is
    compared against the geometry the coefficient was actually evaluated on.
    """

    hit = geometry.interaction_positions_m[row, 0].detach().cpu().numpy()
    normal = geometry.interaction_normals[row, 0].detach().cpu().numpy()
    incident = hit - source_position.detach().cpu().numpy()
    incident = incident / np.linalg.norm(incident)
    return abs(float(np.dot(incident, normal)))


def _free_space_factor(frequency_hz: float, path_length_m: float) -> complex:
    """Exact ``lambda/(4*pi*L)*exp(-j*2*pi*f*L/c)`` at unit source power."""

    return (
        (C0 / frequency_hz)
        / (4.0 * math.pi * path_length_m)
        * np.exp(-2j * math.pi * frequency_hz * path_length_m / C0)
    )


def _relative_error(measured: complex, reference: complex) -> float:
    return abs(measured - reference) / abs(reference)


# --------------------------------------------------------------------------
# 1. Structural contract: identity, pairing, geometry, validity, budget
# --------------------------------------------------------------------------


def test_a_zero_offset_column_is_bitwise_the_reference_coefficient() -> None:
    frequency_hz = 3.0e9
    compiled, sources, sinks, prepared = _reflection_world(frequency_hz, SLAB)
    offsets = (-2.0e7, 0.0, 5.0e7)

    result = _sweep(compiled, sources, sinks, prepared, frequency_hz, offsets)
    transport = result.paths.transport

    assert isinstance(transport, ScalarTransport)
    assert transport.frequency_offsets_hz == offsets
    assert transport.coefficient_offsets.shape == (result.paths.path_count, 3)
    assert transport.coefficient_offsets.dtype == torch.complex64
    # Bitwise, not a tolerance: a zero offset is the same launch.
    assert torch.equal(transport.coefficient_offsets[:, 1], transport.coefficient)


def test_a_zero_offset_column_is_bitwise_the_reference_field() -> None:
    frequency_hz = 3.0e9
    compiled, sources, sinks, prepared = _reflection_world(
        frequency_hz, SLAB, response="complex3_transport"
    )
    offsets = (0.0, 5.0e7)

    result = _sweep(
        compiled,
        sources,
        sinks,
        prepared,
        frequency_hz,
        offsets,
        response="complex3_transport",
    )
    transport = result.paths.transport

    assert isinstance(transport, Complex3Transport)
    assert transport.field_offsets.shape == (result.paths.path_count, 2, 3)
    assert transport.direction.shape == (result.paths.path_count, 3)
    assert torch.equal(transport.field_offsets[:, 0], transport.field)


def test_offset_columns_publish_the_same_geometry_as_the_reference() -> None:
    """Geometry is a fact about where the path goes, not about frequency."""

    frequency_hz = 3.0e9
    compiled, sources, sinks, prepared = _reflection_world(frequency_hz, SLAB)
    offsets = (-4.0e8, 0.0, 4.0e8)

    narrow = _sweep(compiled, sources, sinks, prepared, frequency_hz, None)
    wide = _sweep(compiled, sources, sinks, prepared, frequency_hz, offsets)

    for name in (
        "path_length_m",
        "delay_s",
        "field_direction",
        "interaction_positions_m",
        "interaction_normals",
    ):
        assert torch.equal(
            getattr(wide.paths.geometry, name), getattr(narrow.paths.geometry, name)
        ), name
    assert torch.equal(wide.paths.transport.coefficient, narrow.paths.transport.coefficient)


def test_row_validity_stays_one_mask_over_rows_and_zeroes_every_column() -> None:
    """``row_valid`` is ``[K]``, never ``[K, F]``: validity is geometric."""

    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    offsets = (0.0, 1.0e8, -1.0e8)
    # Move the sinks far off the wall so at least one frozen row stops existing.
    sources = source_batch(cuda_positions(SOURCE_POSITIONS))
    sinks = sink_batch(cuda_positions(((2.0, 40.0, 0.0), (2.0, 60.0, 0.0))))

    result = replay(
        compiled, prepared, sources, sinks, frequency_offsets_hz=offsets
    )

    assert result.row_valid is not None
    assert tuple(result.row_valid.shape) == (result.paths.path_count,)
    payload = result.paths.transport.coefficient_offsets
    assert payload.shape == (result.paths.path_count, len(offsets))
    dead = ~result.row_valid
    assert bool(dead.any()), "the fixture must actually kill a row"
    assert torch.equal(payload[dead], torch.zeros_like(payload[dead]))


def test_the_budget_is_flat_in_the_frequency_column_count() -> None:
    """One validation copy and one synchronization, whatever ``F`` is.

    The row gather owns both and runs once, above the column loop. The launch
    count is the thing that grows, and the diagnostics field reports it rather
    than hiding it.
    """

    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    sources = source_batch(cuda_positions(SOURCE_POSITIONS))
    sinks = sink_batch(cuda_positions(SINK_POSITIONS))

    narrow = replay(compiled, prepared, sources, sinks)
    assert narrow.diagnostics.frequency_column_count == 1

    for count in (1, 8, 64):
        offsets = tuple(float(1.0e6 * (index + 1)) for index in range(count))
        result = replay(
            compiled, prepared, sources, sinks, frequency_offsets_hz=offsets
        )
        diagnostics = result.diagnostics
        assert diagnostics.frequency_column_count == count
        assert diagnostics.validation_d2h_copies == 1
        assert diagnostics.validation_d2h_bytes == 4
        assert diagnostics.validation_sync_count == 1
        assert diagnostics.compact_count_d2h_copies == 0
        assert diagnostics.compact_sync_count == 0
        assert result.paths.transport.coefficient_offsets.shape == (
            FROZEN_ROW_COUNT,
            count,
        )
        # The row axis and the pair segmentation are untouched by frequency.
        assert torch.equal(result.paths.pair_offsets, narrow.paths.pair_offsets)
        assert result.paths.pair_count == narrow.paths.pair_count


def test_the_launch_count_follows_the_published_law() -> None:
    """``launches = (1 + F) * buckets``, asserted rather than asserted-about."""

    from witwin.channel.propagation.fields.kernels import functional

    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    buckets = len(prepared.buckets)
    assert buckets >= 2, "the fixture must exercise more than one bucket"
    sources = source_batch(cuda_positions(SOURCE_POSITIONS))
    sinks = sink_batch(cuda_positions(SINK_POSITIONS))

    counted: list[int] = [0]
    originals = {
        name: getattr(functional, name)
        for name in ("field_free_space", "field_reflection_sequence")
    }

    def counting(original):
        def wrapper(*args, **kwargs):
            counted[0] += 1
            return original(*args, **kwargs)

        return wrapper

    try:
        for name, original in originals.items():
            setattr(functional, name, counting(original))
        measured = {}
        for count in (None, 1, 8, 64):
            offsets = (
                None
                if count is None
                else tuple(float(1.0e6 * (index + 1)) for index in range(count))
            )
            counted[0] = 0
            replay(compiled, prepared, sources, sinks, frequency_offsets_hz=offsets)
            measured[count] = counted[0]
    finally:
        for name, original in originals.items():
            setattr(functional, name, original)

    assert measured[None] == buckets
    for count in (1, 8, 64):
        assert measured[count] == (1 + count) * buckets, (count, measured)


def test_slot_batching_and_frequency_offsets_compose() -> None:
    """``[slot_count*K, F]``: the two axes are orthogonal and neither tiles."""

    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 4
    offsets = tuple(float(2.0e6 * (index + 1)) for index in range(8))
    displacements = slot_offsets(slot_count)
    sources = source_batch(
        stack_over_slots(cuda_positions(SOURCE_POSITIONS), displacements)
    )
    sinks = sink_batch(
        stack_over_slots(cuda_positions(SINK_POSITIONS), displacements)
    )

    batched = replay(
        compiled,
        replicate_over_slots(
            prepared,
            slot_count,
            source_count=len(SOURCE_POSITIONS),
            sink_count=len(SINK_POSITIONS),
        ),
        sources,
        sinks,
        slot_count=slot_count,
        frequency_offsets_hz=offsets,
    )
    payload = batched.paths.transport.coefficient_offsets
    assert payload.shape == (slot_count * FROZEN_ROW_COUNT, len(offsets))
    assert batched.diagnostics.validation_d2h_copies == 1
    assert batched.diagnostics.validation_sync_count == 1

    for slot in range(slot_count):
        one = replay(
            compiled,
            prepared,
            source_batch(cuda_positions(SOURCE_POSITIONS) + displacements[slot]),
            sink_batch(cuda_positions(SINK_POSITIONS) + displacements[slot]),
            frequency_offsets_hz=offsets,
        )
        rows = slice(slot * FROZEN_ROW_COUNT, (slot + 1) * FROZEN_ROW_COUNT)
        single = one.paths.transport.coefficient_offsets
        assert torch.equal(payload[rows].real, single.real), slot
        assert torch.equal(payload[rows].imag, single.imag), slot


# --------------------------------------------------------------------------
# 2. Closed-form physics
# --------------------------------------------------------------------------


def test_line_of_sight_offsets_match_the_free_space_closed_form() -> None:
    """Free space has an exact answer: a ``f_ref/f`` tilt and a delay phase."""

    frequency_hz = 1.0e9
    compiled = _empty_scene(frequency_hz)
    sources, sinks = endpoints(with_basis=False)
    discovered = _discover(
        compiled,
        sources,
        sinks,
        frequency_hz,
        components=frozenset({"los"}),
        max_depth=0,
    )
    assert discovered.paths.path_count == 1
    # Zero-width interaction sequences: this is the raw fixed-LoS route, which
    # owns its own column loop.
    topology = discovered.paths.topology
    assert topology.primitive_sequence.shape[1] == 0
    assert isinstance(topology, PropagationTopology)

    offsets = (-5.0e7, 0.0, 2.5e7, 5.0e7)
    result = _sweep(compiled, sources, sinks, topology, frequency_hz, offsets)
    payload = result.paths.transport.coefficient_offsets[0].detach().cpu().numpy()
    delay_s = float(result.paths.geometry.delay_s[0])
    reference = complex(payload[offsets.index(0.0)])

    for index, offset in enumerate(offsets):
        measured = complex(payload[index])
        magnitude_ratio = abs(measured) / abs(reference)
        expected_ratio = frequency_hz / (frequency_hz + offset)
        assert abs(magnitude_ratio - expected_ratio) <= 1.0e-6 * expected_ratio, offset
        residual = float(
            np.angle(measured) - np.angle(reference) + 2.0 * math.pi * offset * delay_s
        )
        residual = (residual + math.pi) % (2.0 * math.pi) - math.pi
        assert abs(residual) <= 1.0e-4, (offset, residual)


def test_half_space_sweep_matches_the_fresnel_closed_form() -> None:
    """A layer thick enough to hide its backing IS the single interface.

    ``sigma_e`` alone makes ``eps_c(f)`` frequency dependent, with no
    ``DispersionSpec`` anywhere, which is what separates genuine frequency
    dependence from the dispersion term the contract refuses.
    """

    frequency_hz = 1.0e9
    compiled, sources, sinks, prepared = _reflection_world(frequency_hz, HALF_SPACE)
    offsets = tuple(float(-4.0e8 + 5.0e7 * index) for index in range(17))

    result = _sweep(compiled, sources, sinks, prepared, frequency_hz, offsets)
    geometry = result.paths.geometry
    cos_theta_i = _incidence_cosine(geometry, sources.positions_m[0])
    path_length_m = float(geometry.path_length_m[0])
    payload = result.paths.transport.coefficient_offsets[0].detach().cpu().numpy()

    worst = 0.0
    for index, offset in enumerate(offsets):
        frequency = frequency_hz + offset
        interface = fresnel_interface(
            cos_theta_i,
            vacuum_medium(frequency),
            medium_params(
                HALF_SPACE["eps_r"], HALF_SPACE["sigma_e"], 1.0, frequency
            ),
        )
        stack = layer_stack_rt(
            [
                (
                    HALF_SPACE["thickness_m"],
                    HALF_SPACE["eps_r"],
                    HALF_SPACE["sigma_e"],
                    1.0,
                )
            ],
            cos_theta_i,
            frequency,
        )
        # The backing really is invisible: the stack and the bare interface
        # agree, so the half-space closed form is the right reference.
        assert abs(stack.r_te - interface.r_te) <= 1.0e-8 * abs(interface.r_te)
        reference = _free_space_factor(frequency, path_length_m) * interface.r_te
        worst = max(worst, _relative_error(complex(payload[index]), reference))
    assert worst <= 4.0e-5, worst


def test_multilayer_sweep_crosses_fringes_and_falsifies_the_narrowband_law() -> None:
    """The slab fringe is the thing a narrowband coefficient cannot express."""

    frequency_hz = 3.0e9
    compiled, sources, sinks, prepared = _reflection_world(frequency_hz, SLAB)
    step_hz = 5.0e7
    offsets = tuple(float(-1.2e9 + step_hz * index) for index in range(49))

    result = _sweep(compiled, sources, sinks, prepared, frequency_hz, offsets)
    geometry = result.paths.geometry
    cos_theta_i = _incidence_cosine(geometry, sources.positions_m[0])
    path_length_m = float(geometry.path_length_m[0])
    delay_s = float(geometry.delay_s[0])
    payload = result.paths.transport.coefficient_offsets[0].detach().cpu().numpy()

    layers = [
        (SLAB["thickness_m"], SLAB["eps_r"], SLAB["sigma_e"], 1.0)
    ]
    worst = 0.0
    measured_envelope = []
    oracle_envelope = []
    for index, offset in enumerate(offsets):
        frequency = frequency_hz + offset
        r_te = layer_stack_rt(layers, cos_theta_i, frequency).r_te
        reference = _free_space_factor(frequency, path_length_m) * r_te
        worst = max(worst, _relative_error(complex(payload[index]), reference))
        # Divide the monotone spreading tilt out so a magnitude minimum is a
        # material null rather than a slope.
        measured_envelope.append(abs(payload[index]) * frequency)
        oracle_envelope.append(abs(r_te))
    assert worst <= 1.0e-4, worst

    # The analytic fringe period at this incidence, from Snell.
    sin_theta_t = math.sqrt(1.0 - cos_theta_i**2) / math.sqrt(SLAB["eps_r"])
    cos_theta_t = math.sqrt(1.0 - sin_theta_t**2)
    fringe_hz = C0 / (
        2.0 * math.sqrt(SLAB["eps_r"]) * SLAB["thickness_m"] * cos_theta_t
    )
    span_hz = offsets[-1] - offsets[0]
    assert span_hz / fringe_hz >= 2.0, (span_hz, fringe_hz)

    def local_minima(values: list[float]) -> list[int]:
        return [
            index
            for index in range(1, len(values) - 1)
            if values[index] < values[index - 1] and values[index] < values[index + 1]
        ]

    measured_nulls = local_minima(measured_envelope)
    oracle_nulls = local_minima(oracle_envelope)
    assert len(oracle_nulls) >= 2, oracle_nulls
    # Each measured null lands in the same grid bin as the analytic one, which
    # is well inside one grid step of the analytic fringe position.
    assert measured_nulls == oracle_nulls

    # The narrowband law is not merely imprecise here, it is wrong by an order
    # of magnitude: it holds the wall reflectivity fixed across three nulls.
    reference = complex(payload[offsets.index(0.0)])
    worst_law = 0.0
    for index, offset in enumerate(offsets):
        law = reference * np.exp(-2j * math.pi * offset * delay_s)
        worst_law = max(worst_law, abs(abs(payload[index]) / abs(law) - 1.0))
    assert worst_law >= 2.0, worst_law


def test_the_narrowband_law_is_measurably_wrong_at_one_megahertz() -> None:
    """Why the wideband route exists, as a number rather than an adjective.

    A 1 MHz offset at 77 GHz is a fractional shift of 1.3e-5. The spreading
    term is negligible there; the whole error is the slab's frequency
    selectivity, which is exactly the term the wideband route removes.
    """

    frequency_hz = 77.0e9
    compiled, sources, sinks, prepared = _reflection_world(
        frequency_hz, DEFAULT_WALL
    )
    offsets = (0.0, 1.0e6)

    result = _sweep(compiled, sources, sinks, prepared, frequency_hz, offsets)
    payload = result.paths.transport.coefficient_offsets[0].detach().cpu().numpy()
    delay_s = float(result.paths.geometry.delay_s[0])

    reference = complex(payload[0])
    true_value = complex(payload[1])
    law = reference * np.exp(-2j * math.pi * offsets[1] * delay_s)

    magnitude_error = abs(abs(true_value) - abs(law)) / abs(law)
    phase_error = abs(float(np.angle(true_value / law)))
    assert magnitude_error > 5.0e-3, magnitude_error
    assert phase_error > 5.0e-3, phase_error


# --------------------------------------------------------------------------
# 3. Structural refusals
# --------------------------------------------------------------------------


def _request(offsets, **overrides) -> None:
    sources, sinks = endpoints(with_basis=True)
    compiled = _wall_scene(1.0e9, **SLAB)
    discovered = _discover(
        compiled,
        sources,
        sinks,
        1.0e9,
        components=frozenset({"reflection"}),
    )
    keywords = {
        "sources": sources,
        "sinks": sinks,
        "reference_frequency_hz": 1.0e9,
        "topology": prepare_fixed_topology(discovered.paths.topology),
        "response": "scalar_transport",
        "ad_mode": "none",
        "frequency_offsets_hz": offsets,
    }
    keywords.update(overrides)
    FixedTopologyRequest(**keywords)


def test_a_tensor_offset_grid_is_refused_by_name() -> None:
    with pytest.raises(TypeError, match="host declaration"):
        _request(torch.tensor([0.0, 1.0e6]))


def test_an_empty_offset_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _request(())


def test_a_non_finite_offset_is_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        _request((0.0, float("inf")))


def test_a_duplicate_offset_is_refused() -> None:
    with pytest.raises(ValueError, match="repeat an offset"):
        _request((0.0, 1.0e6, 1.0e6))


def test_polarimetric_transport_refuses_an_offset_grid_by_capability() -> None:
    with pytest.raises(NotImplementedError, match="wideband_responses"):
        _request((0.0, 1.0e6), response="polarimetric_transport")


# --------------------------------------------------------------------------
# 4. Scene-dependent refusals, each reached on its own
# --------------------------------------------------------------------------


def _dispersive_wall_scene(frequency_hz: float):
    class _PowerLaw:
        """Minimal DispersionSpec: eps_r falls as a power of the frequency."""

        def complex_eps(self, frequency_hz):
            return 4.0 * (frequency_hz / 1.0e9) ** -0.05 - 0.0j

    from tests.support.scenes import transmission_wall_structure

    material = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.1, dispersion=_PowerLaw()),),
        name="wideband-dispersive-wall",
    )
    scene = Scene(structures=(transmission_wall_structure(WALL_X_M, material),))
    return compile_scene(scene, reference_frequency_hz=frequency_hz)


def _replay_offsets(compiled, frequency_hz: float, offsets, *, max_depth: int = 1):
    sources, sinks = endpoints(with_basis=False)
    discovered = _discover(
        compiled,
        sources,
        sinks,
        frequency_hz,
        components=frozenset({"reflection"}),
        max_depth=max_depth,
    )
    prepared = prepare_fixed_topology(discovered.paths.topology)
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=frequency_hz,
            topology=prepared,
            response="scalar_transport",
            ad_mode="none",
            frequency_offsets_hz=offsets,
        ),
    )


def test_w1_a_dispersive_scene_is_refused_at_every_ad_mode() -> None:
    """The primal half of the frozen-record defect, not only the AD half."""

    frequency_hz = 1.0e9
    compiled = _dispersive_wall_scene(frequency_hz)
    assert compiled.materials.frequency_dependent
    # Non-dispersive-independent conditions all satisfied: smooth scene, grid
    # resolvable at 1 GHz (64 Hz resolution), ad_mode='none'.
    assert native_frequency_resolution_hz(frequency_hz) < 1.0e6
    with pytest.raises(NotImplementedError, match="frequency-dependent materials"):
        _replay_offsets(compiled, frequency_hz, (0.0, 1.0e6))


def test_w2_an_unresolvable_offset_grid_is_refused_with_the_resolution() -> None:
    """Below one float32 ULP the offset does not exist at all."""

    frequency_hz = 77.0e9
    resolution_hz = native_frequency_resolution_hz(frequency_hz)
    assert resolution_hz == 8192.0
    compiled, sources, sinks, prepared = _reflection_world(frequency_hz, SLAB)
    assert not compiled.materials.frequency_dependent

    with pytest.raises(ValueError, match="below the native frequency resolution"):
        _sweep(compiled, sources, sinks, prepared, frequency_hz, (0.0, 100.0))
    with pytest.raises(ValueError, match="closer than the native frequency"):
        _sweep(
            compiled,
            sources,
            sinks,
            prepared,
            frequency_hz,
            (1.0e6, 1.0e6 + 100.0),
        )
    # One ULP apart is resolvable and is accepted.
    _sweep(
        compiled,
        sources,
        sinks,
        prepared,
        frequency_hz,
        (0.0, resolution_hz),
    )


def test_w4_a_rough_scene_is_refused_by_resident_table_lifetime() -> None:
    frequency_hz = 1.0e9
    scene = Scene(
        structures=(
            rough_wall_structure(
                WALL_X_M, rms_height_m=1.0e-3, corr_length_m=0.05
            ),
        )
    )
    compiled = compile_scene(scene, reference_frequency_hz=frequency_hz)
    assert not compiled.materials.frequency_dependent

    with pytest.raises(NotImplementedError, match="rough materials"):
        _replay_offsets(compiled, frequency_hz, (0.0, 1.0e6))


def test_the_capability_record_publishes_the_wideband_contract() -> None:
    record = capabilities()
    convention = PropagationConvention()

    assert record.contract_version == 6
    assert record.supports_wideband_offsets is True
    assert record.wideband_responses == frozenset(
        {"scalar_transport", "complex3_transport"}
    )
    assert record.wideband_components == frozenset({"los", "reflection"})
    assert record.wideband_dispersive_materials is False
    assert record.wideband_rough_materials is False
    assert record.max_frequency_offset_count is None
    assert record.native_frequency_resolution_law == (
        "resolution_hz = ulp_float32(reference_frequency_hz)"
    )
    assert "frequency_minor" in convention.wideband_offset_layout
    assert "row_valid stays [K]" in convention.wideband_offset_layout
    assert "ulp_float32" in convention.wideband_frequency_quantization_law
    # A caller computes the same number the refusal quotes.
    assert native_frequency_resolution_hz(77.0e9) == 8192.0
    assert native_frequency_resolution_hz(1.0e9) == 64.0
    assert native_frequency_resolution_hz(3.0e9) == 256.0


def test_the_grid_is_a_propagation_frequency_grid_and_nothing_else() -> None:
    """No waveform parameter enters the Channel contract with the grid.

    ``frequency_offsets_hz`` names absolute frequencies at which a field is
    evaluated. It is not a subcarrier count, an FFT size, a bandwidth, or a
    sample count, and no field with such a meaning may ride in with it: that
    is the boundary criterion this capability is most likely to erode.
    """

    import dataclasses

    forbidden = ("subcarrier", "fft", "bandwidth", "sample", "symbol", "waveform")
    surfaces = (
        FixedTopologyRequest,
        PropagationRequest,
        ScalarTransport,
        Complex3Transport,
        PropagationConvention,
        type(capabilities()),
    )
    for surface in surfaces:
        for field in dataclasses.fields(surface):
            lowered = field.name.lower()
            assert not any(token in lowered for token in forbidden), (
                surface.__name__,
                field.name,
            )
    # Absent by default, so the grid is opt-in and every existing caller keeps
    # exactly the single-frequency answer it had.
    assert (
        FixedTopologyRequest.__dataclass_fields__["frequency_offsets_hz"].default
        is None
    )
    assert (
        ScalarTransport.__dataclass_fields__["coefficient_offsets"].default is None
    )
