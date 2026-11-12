# Copyright Xingyu Chen.
# Gate: path, deterministic export and the propagation consumer publish the same enumerated rows.

"""Gate: path, deterministic export and the propagation consumer publish the same enumerated rows."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

FREQUENCY_HZ = 3.0e9
POLARIZATION = (0.0, 0.0, 1.0)


def _scenes():
    from tests.support import scenes

    return {
        "two_tx_two_rx_free_space": (scenes.empty_space_los_scene, {"los"}, 1),
        "same_side_wall": (scenes.same_side_wall_reflection_scene, {"los", "reflection"}, 1),
        "same_side_wall_depth2": (scenes.same_side_wall_reflection_scene, {"los", "reflection"}, 2),
        "wedge": (scenes.wedge_diffraction_scene, {"los", "reflection", "diffraction"}, 1),
        "wall_and_edge": (scenes.coupled_wall_wedge_scene, {"los", "reflection", "diffraction"}, 1),
    }


SCENES = tuple(_scenes())


def _endpoints(scene, role):
    return [endpoint for endpoint in scene.endpoints if endpoint.role == role]


def _batch(endpoints, first_id, *, with_power):
    from witwin.channel.propagation.consumer import EndpointBatch

    positions = torch.stack(
        [torch.as_tensor(e.position, dtype=torch.float32) for e in endpoints]
    ).cuda()
    count = len(endpoints)
    return EndpointBatch(
        stable_ids=torch.arange(first_id, first_id + count, dtype=torch.int64, device="cuda"),
        positions_m=positions,
        polarizations=torch.tensor([POLARIZATION] * count, device="cuda"),
        powers_w=(
            torch.tensor([float(e.power_w) for e in endpoints], device="cuda")
            if with_power
            else None
        ),
    )


def _solve_all(name, *, max_paths=None):
    from witwin.channel import radiomap, path
    from witwin.channel.propagation import consumer
    from witwin.channel.scene import compile as compile_scene

    build, components, depth = _scenes()[name]
    scene = build()
    paths = path.solve(
        scene, path.Config(components=components, max_depth=depth, max_paths=max_paths),
        reference_frequency_hz=FREQUENCY_HZ,
    )
    table = radiomap.solve(
        scene,
        radiomap.Coherent(components=components, max_depth=depth, export_paths=True),
        reference_frequency_hz=FREQUENCY_HZ,
    ).paths
    sources = _endpoints(scene, "tx")
    request = consumer.PropagationRequest(
        sources=_batch(sources, 1, with_power=True),
        sinks=_batch(_endpoints(scene, "rx"), 1000, with_power=False),
        reference_frequency_hz=FREQUENCY_HZ,
        components=frozenset(components),
        max_depth=depth,
        response="scalar_transport",
        topology_mode="discover",
        ad_mode="none",
    )
    rows = consumer.evaluate(compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ), request).paths
    power = torch.tensor([float(e.power_w) for e in sources], device="cuda")
    return paths, table, rows, power


def _path_rows(paths):
    """Valid rows of a padded ``PathResult`` as flat (rx, tx, coefficient, delay)."""

    valid = paths.valid[:, 0, :, 0, :]
    rx, tx, _ = torch.nonzero(valid, as_tuple=True)
    return rx, tx, paths.a[:, 0, :, 0, :, 0][valid], paths.tau[:, 0, :, 0, :][valid]


@pytest.mark.parametrize("name", SCENES)
def test_path_and_deterministic_export_publish_identical_rows(name):
    paths, table, _rows, _power = _solve_all(name)
    rx, tx, coefficient, delay = _path_rows(paths)
    assert table is not None and int(table.valid.sum()) == int(table.valid.numel()) > 0
    # Both entries order rows receiver-major, then transmitter, then the engine's row order.
    assert torch.equal(rx, table.rx_id.long())
    assert torch.equal(tx, table.tx_id.long())
    assert torch.equal(coefficient, table.coefficient)
    assert torch.equal(delay, table.delay_s)


@pytest.mark.parametrize("name", SCENES)
def test_consumer_publishes_the_same_rows_with_source_amplitude(name):
    """The consumer's scalar coefficient is the unit-excitation row times sqrt(source power)."""

    _paths, table, rows, power = _solve_all(name)
    assert rows.path_count == int(table.valid.numel())
    assert torch.equal(rows.topology.source_index, table.tx_id.long())
    assert torch.equal(rows.topology.sink_index, table.rx_id.long())
    assert torch.equal(rows.topology.component_id, table.component_id.long())
    assert torch.equal(rows.geometry.delay_s, table.delay_s)
    expected = table.coefficient * power.sqrt()[table.tx_id.long()]
    torch.testing.assert_close(rows.transport.coefficient, expected, rtol=1e-6, atol=0.0)


def test_max_paths_scope_is_an_intended_selection_difference():
    """``path`` caps rows per pair; ``deterministic`` defaults to one global cap.

    With ``max_paths_scope="per_pair"`` the deterministic export selects exactly the path rows.
    """

    from witwin.channel import radiomap, path

    build, components, depth = _scenes()["two_tx_two_rx_free_space"]
    scene = build()
    per_pair = path.solve(
        scene, path.Config(components=components, max_depth=depth, max_paths=1),
        reference_frequency_hz=FREQUENCY_HZ,
    )
    assert int(per_pair.valid.sum()) == 4
    with pytest.raises(ValueError, match="path max_paths_scope"):
        path.Config(max_paths=1, max_paths_scope="global")

    def export(scope):
        return radiomap.solve(
            scene,
            radiomap.Coherent(
                components=components, max_depth=depth, export_paths=True, max_paths=1,
                max_paths_scope=scope,
            ),
            reference_frequency_hz=FREQUENCY_HZ,
        ).paths

    assert int(export("global").valid.numel()) == 1
    matched = export("per_pair")
    _rx, _tx, coefficient, delay = _path_rows(per_pair)
    assert torch.equal(coefficient, matched.coefficient)
    assert torch.equal(delay, matched.delay_s)


@pytest.mark.parametrize("coherent", [True, False])
def test_deterministic_accumulation_combines_the_exported_rows(coherent):
    """Intended difference: ``deterministic`` combines rows per receiver, the other entries do not.

    Coherent mode squares the summed complex field; incoherent mode sums row powers.
    """

    from witwin.channel import radiomap

    build, components, depth = _scenes()["wall_and_edge"]
    result = radiomap.solve(
        build(),
        radiomap.Coherent(
            components={"los", "reflection"}, max_depth=depth, export_paths=True,
            combine=("field" if coherent else "power"),
        ),
        reference_frequency_hz=FREQUENCY_HZ,
    )
    table = result.paths
    assert table is not None and int(table.valid.numel()) >= 2
    field = torch.complex(table.field_real, table.field_imag)
    if coherent:
        expected = field.to(torch.complex128).sum().abs().square()
    else:
        expected = table.path_gain.double().sum()
    torch.testing.assert_close(
        result.path_gain.double().reshape(()), expected, rtol=1e-5, atol=0.0
    )


def _layout_pair(coherent):
    """Diffraction component power on a 3 x 2 grid and at the same six points as receivers."""

    from witwin.core import Scene

    from witwin.channel import radiomap
    from tests.support import scenes
    from tests.support.core_world import make_receiver, make_receiver_grid

    base = scenes.wedge_diffraction_scene()
    tx = _endpoints(base, "tx")
    origin, spacing, shape = (2.6, 0.6, 0.5), (0.3, 0.4), (3, 2)
    receiver_grid = make_receiver_grid(
        origin=torch.tensor(origin), x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]), shape=shape, spacing=spacing,
    )
    points = [
        make_receiver(torch.tensor(
            [origin[0] + row * spacing[0], origin[1] + col * spacing[1], origin[2]]
        ))
        for row in range(shape[0]) for col in range(shape[1])
    ]
    config = radiomap.Coherent(components={"los", "reflection", "diffraction"}, combine=("field" if coherent else "power"))

    def solve(receivers):
        return radiomap.solve(
            Scene(structures=base.structures, endpoints=[*tx, *receivers]), config,
            reference_frequency_hz=FREQUENCY_HZ,
        )

    return solve([receiver_grid]), solve(points)


@pytest.mark.parametrize("coherent", [True, False])
def test_diffraction_accumulation_is_independent_of_receiver_layout(coherent):
    """A grid cell and a point receiver at the same position get the same diffraction power."""

    on_grid, at_points = _layout_pair(coherent)
    # Points are receiver row-major; compare as sorted values so the check is layout-neutral.
    grid_values = on_grid.component_power["diffraction"].reshape(-1).sort().values
    point_values = at_points.component_power["diffraction"].reshape(-1).sort().values
    assert bool((point_values > 0).all())
    torch.testing.assert_close(grid_values, point_values, rtol=1e-5, atol=0.0)
    torch.testing.assert_close(
        on_grid.path_gain.reshape(-1).sort().values, at_points.path_gain.reshape(-1).sort().values,
        rtol=1e-5, atol=0.0,
    )
