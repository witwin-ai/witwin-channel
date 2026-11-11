# Copyright Xingyu Chen.
# The fused canonical-row field launch is bitwise the per-operator field kernels on every row kind.

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="field kernels need CUDA")

_FREQUENCY_HZ = 3.5e9
_WIDTH = 3
_COLUMNS = ("field_xyz", "coefficient", "path_field", "path_gain", "path_length", "delay", "direction")
# (component, depth) of every row: LoS, reflection depth 1..3, order-1 diffraction, and rows the
# launch must leave untouched (transmission, coupled R-D, D-R, D-D, scattering, reflection depth 0).
_ROWS = (
    (0, 0), (1, 1), (2, 1), (1, 2), (5, 2), (1, 3), (0, 0), (3, 2), (1, 1), (2, 1), (4, 2),
    (7, 2), (6, 1), (1, 0), (1, 2), (0, 0), (1, 3), (2, 1),
)


def _bits(value: torch.Tensor) -> torch.Tensor:
    value = torch.view_as_real(value) if value.is_complex() else value
    return value.contiguous().view(torch.int32)


def _assert_bitwise(observed: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    assert observed.shape == expected.shape, name
    assert torch.equal(_bits(observed), _bits(expected)), name


def _canonical_inputs(*, seed: int = 3) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)

    def uniform(*shape: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
        return (low + (high - low) * torch.rand(shape, generator=generator)).cuda()

    count = len(_ROWS)
    tx_count, rx_count, face_count = 3, 4, 6
    component = torch.tensor([row[0] for row in _ROWS], dtype=torch.int32).cuda()
    depth = torch.tensor([row[1] for row in _ROWS], dtype=torch.int32).cuda()
    normals = uniform(count, _WIDTH, 3)
    fields = {
        "field_xyz": torch.complex(uniform(count, 3), uniform(count, 3)),
        "coefficient": torch.complex(uniform(count), uniform(count)),
        "path_field": torch.complex(uniform(count), uniform(count)),
        "path_gain": uniform(count, low=0.0),
        "path_length": uniform(count, low=1.0, high=50.0),
        "delay": uniform(count, low=0.0, high=1.0e-7),
        "direction": uniform(count, 3),
    }
    return {
        "component_id": component,
        "depth": depth,
        "tx_id": torch.randint(0, tx_count, (count,), generator=generator, dtype=torch.int32).cuda(),
        "rx_id": torch.randint(0, rx_count, (count,), generator=generator, dtype=torch.int32).cuda(),
        "primitive_sequence": torch.randint(
            0, face_count, (count, _WIDTH), generator=generator, dtype=torch.int32
        ).cuda(),
        "tx_positions": uniform(tx_count, 3, low=-5.0, high=5.0),
        "tx_power": uniform(tx_count, low=0.5, high=2.0),
        "tx_polarization": uniform(tx_count, 3),
        "rx_positions": uniform(rx_count, 3, low=10.0, high=20.0),
        "rx_polarization": uniform(rx_count, 3),
        "interaction_positions": uniform(count, _WIDTH, 3, low=2.0, high=8.0),
        "interaction_normals": normals / normals.norm(dim=-1, keepdim=True),
        "fields": fields,
        "face": {
            "eps_r": uniform(face_count, low=2.0, high=8.0),
            "sigma_e": uniform(face_count, low=0.0, high=0.1),
            "mu_r": uniform(face_count, low=1.0, high=1.2),
            "gain": uniform(face_count, low=0.5, high=1.0),
            "thickness": uniform(face_count, low=0.05, high=0.3),
            "material_id": torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.int32).cuda(),
            "layer_offset": torch.tensor([0, 1, 3], dtype=torch.int32).cuda(),
            "layer_count": torch.tensor([1, 2, 3], dtype=torch.int32).cuda(),
            "layer_thickness_m": uniform(6, low=0.01, high=0.2),
            "layer_eps_r": uniform(6, low=2.0, high=7.0),
            "layer_sigma_e": uniform(6, low=0.0, high=0.05),
            "layer_mu_r": torch.ones(6).cuda(),
            "rough_sigma_h_m": uniform(face_count, low=0.002, high=0.02),
            "rough": torch.tensor([True, False, True, True, False, False]).cuda(),
            "replaced": torch.tensor([False, True, False, False, False, True]).cuda(),
        },
    }


def _face_tables(face: dict[str, torch.Tensor], *, multilayer: bool, rough: bool):
    from witwin.channel.kernels.fields import CanonicalFaceTables

    return CanonicalFaceTables(
        eps_r=face["eps_r"], sigma_e=face["sigma_e"], mu_r=face["mu_r"], gain=face["gain"],
        thickness=face["thickness"], material=face["material_id"],
        layers=tuple(
            face[name]
            for name in (
                "layer_offset", "layer_count", "layer_thickness_m", "layer_eps_r",
                "layer_sigma_e", "layer_mu_r",
            )
        ),
        multilayer=multilayer,
        rough_sigma=face["rough_sigma_h_m"] if rough else None,
        rough=face["rough"] if rough else None,
        replaced=face["replaced"] if rough else None,
    )


def _launch(inputs: dict[str, object], *, order, multilayer: bool, rough: bool, evaluate_los=True):
    from witwin.channel.kernels.fields import field_canonical_rows

    arguments = {key: value for key, value in inputs.items() if key != "face"}
    return field_canonical_rows(
        order=order,
        faces=_face_tables(inputs["face"], multilayer=multilayer, rough=rough),
        evaluate_los=evaluate_los,
        frequency_hz=_FREQUENCY_HZ,
        **arguments,
    )


def _per_operator_reference(inputs: dict[str, object], *, multilayer: bool, rough: bool):
    """The field stage as one operator launch per row kind, the shape it had before fusion."""

    from witwin.channel.kernels import fields as field_kernels
    from witwin.channel.kernels import geometry as geometry_kernels

    fields = inputs["fields"]
    face = inputs["face"]
    out = {name: value.clone() for name, value in fields.items()}
    tx = inputs["tx_id"].long()
    rx = inputs["rx_id"].long()
    source = inputs["tx_positions"][tx]
    target = inputs["rx_positions"][rx]
    power = inputs["tx_power"][tx]
    tx_pol = inputs["tx_polarization"][tx]
    rx_pol = inputs["rx_polarization"][rx]
    component = inputs["component_id"]
    depth = inputs["depth"]

    def publish(rows: torch.Tensor, values: dict[str, torch.Tensor]) -> None:
        for name, key in (
            ("field_xyz", "field_vector"), ("coefficient", "coefficient"),
            ("path_field", "path_field"), ("path_gain", "path_gain"),
            ("path_length", "path_length_m"), ("delay", "delay_s"), ("direction", "direction"),
        ):
            out[name].index_copy_(0, rows, values[key])

    los = torch.nonzero(component == 0).reshape(-1)
    publish(los, field_kernels.field_free_space(
        source[los].contiguous(), target[los].contiguous(), power[los].contiguous(),
        tx_pol[los].contiguous(), rx_pol[los].contiguous(), frequency_hz=_FREQUENCY_HZ))
    for bounces in range(1, _WIDTH + 1):
        rows = torch.nonzero((component == 1) & (depth == bounces)).reshape(-1)
        face_id = inputs["primitive_sequence"][rows][:, :bounces].long()
        positions = inputs["interaction_positions"][rows][:, :bounces].contiguous()
        normals = inputs["interaction_normals"][rows][:, :bounces].contiguous()
        layers = (
            field_kernels.reflection_layers(face, face_id, multilayer=True) if multilayer else None
        )
        values = field_kernels.field_reflection_sequence(
            source[rows].contiguous(), target[rows].contiguous(), positions, normals,
            power[rows].contiguous(), tx_pol[rows].contiguous(), rx_pol[rows].contiguous(),
            *(face[name][face_id].contiguous() for name in ("eps_r", "sigma_e", "mu_r", "gain", "thickness")),
            layers=layers, frequency_hz=_FREQUENCY_HZ,
        )
        if rough:
            replaced = (
                face["replaced"][face_id[:, 0]].contiguous()
                if bounces == 1
                else torch.zeros(rows.shape, dtype=torch.bool, device="cuda")
            )
            scaled = field_kernels.field_rough_reflection_scale(
                values["field_vector"], values["coefficient"], values["path_field"],
                values["path_gain"], positions, normals, source[rows].contiguous(),
                face["rough_sigma_h_m"][face_id].contiguous(), face["rough"][face_id].contiguous(),
                replaced, frequency_hz=_FREQUENCY_HZ,
            )
            values = {**values, **{key: scaled[key] for key in (
                "field_vector", "coefficient", "path_field", "path_gain")}}
        publish(rows, values)
    rows = torch.nonzero(component == 2).reshape(-1)
    arrival = geometry_kernels.deterministic_normalize_vec3(
        (target[rows] - inputs["interaction_positions"][rows][:, 0]).contiguous(), eps=1.0e-6)
    powered = fields["field_xyz"][rows].contiguous()
    projected = field_kernels.field_project_complex3(powered, arrival, rx_pol[rows].contiguous())
    amplitude = power[rows].clamp_min(1.0e-30).sqrt()
    out["field_xyz"].index_copy_(0, rows, powered / amplitude[:, None])
    out["path_field"].index_copy_(0, rows, projected["coefficient"])
    out["coefficient"].index_copy_(0, rows, projected["coefficient"] / amplitude)
    out["path_gain"].index_copy_(0, rows, projected["path_gain"])
    out["direction"].index_copy_(0, rows, arrival)
    return out


@pytest.mark.parametrize(
    ("multilayer", "rough"), [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("grouped", [False, True])
def test_canonical_launch_is_bitwise_the_per_operator_field_stage(multilayer, rough, grouped):
    inputs = _canonical_inputs()
    order = (
        torch.argsort(inputs["component_id"].long() * 16 + inputs["depth"].long(), stable=True)
        if grouped
        else None
    )
    observed = _launch(inputs, order=order, multilayer=multilayer, rough=rough)
    expected = _per_operator_reference(inputs, multilayer=multilayer, rough=rough)
    for name in _COLUMNS:
        _assert_bitwise(observed[name], expected[name], name)
    untouched = torch.tensor(
        [index for index, (component, depth) in enumerate(_ROWS)
         if component not in (0, 1, 2) or (component == 1 and depth == 0)],
        device="cuda",
    )
    for name in _COLUMNS:
        _assert_bitwise(observed[name][untouched], inputs["fields"][name][untouched], name)
        assert observed[name].data_ptr() != inputs["fields"][name].data_ptr()


def test_line_of_sight_rows_keep_their_inputs_when_evaluated_elsewhere():
    inputs = _canonical_inputs()
    observed = _launch(inputs, order=None, multilayer=False, rough=False, evaluate_los=False)
    los = torch.nonzero(inputs["component_id"] == 0).reshape(-1)
    for name in _COLUMNS:
        _assert_bitwise(observed[name][los], inputs["fields"][name][los], name)


def test_rows_with_an_index_outside_their_table_are_inert_and_the_context_survives():
    inputs = _canonical_inputs()
    inputs["primitive_sequence"][1, 0] = 99
    inputs["rx_id"][0] = 7
    observed = _launch(inputs, order=None, multilayer=False, rough=False)
    for row in (0, 1):
        for name in _COLUMNS:
            assert not bool(observed[name][row].abs().any()), (row, name)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(observed["path_gain"][2:]).all())


def test_negative_indices_count_from_the_end_like_a_torch_gather():
    inputs = _canonical_inputs()
    wrapped = dict(inputs)
    wrapped["tx_id"] = torch.where(inputs["tx_id"] == 2, -1, inputs["tx_id"]).to(torch.int32)
    wrapped["primitive_sequence"] = torch.where(
        inputs["primitive_sequence"] == 5, -1, inputs["primitive_sequence"]
    ).to(torch.int32)
    observed = _launch(wrapped, order=None, multilayer=True, rough=True)
    expected = _launch(inputs, order=None, multilayer=True, rough=True)
    for name in _COLUMNS:
        _assert_bitwise(observed[name], expected[name], name)
