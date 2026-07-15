import pytest
import torch
import math

from witwin.channel_native.core.kernels import ops
from witwin.channel_native.montecarlo.bdpt.kernels import paths
from witwin.channel_native.runtime import symbols


def _endpoint_subpath_state(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    launch_tx_id: torch.Tensor,
    light_seed: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    tx_polarization = torch.zeros_like(tx_positions)
    tx_polarization[:, 2] = 1.0
    rx_polarization = torch.zeros_like(rx_positions)
    rx_polarization[:, 2] = 1.0
    return ops.bdpt_endpoint_subpath_state(
        tx_positions,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        launch_tx_id,
        light_seed,
    )


def _complete_subpath_field_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    count = int(state["origin"].shape[0])
    field_real = torch.zeros((count, 3), device="cuda", dtype=torch.float32)
    field_real[:, 2] = 1.0
    state["field_real"] = field_real
    state["field_imag"] = torch.zeros_like(field_real)
    state["source_power"] = state["throughput_real"].square()
    state["event_type"] = torch.zeros((count,), device="cuda", dtype=torch.int32)
    return state


def test_bdpt_launch_state_validates_cuda_reference():
    reference = torch.zeros((1, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="reference must be a CUDA tensor"):
        ops.bdpt_launch_state(
            reference,
            tx_count=1,
            samples=2,
            sample_streams=1,
            seed=3,
        )


def test_bdpt_launch_state_returns_stable_cuda_tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT launch state")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)

    first = ops.bdpt_launch_state(reference, tx_count=2, samples=3, sample_streams=2, seed=11)
    second = ops.bdpt_launch_state(reference, tx_count=2, samples=3, sample_streams=2, seed=11)
    changed = ops.bdpt_launch_state(reference, tx_count=2, samples=3, sample_streams=2, seed=12)

    assert set(first) == {"tx_id", "sample_id", "stream_id", "light_seed"}
    assert first["tx_id"].shape == (12,)
    assert first["sample_id"].shape == (12,)
    assert first["stream_id"].shape == (12,)
    assert first["light_seed"].dtype == torch.int64
    torch.testing.assert_close(first["tx_id"], second["tx_id"])
    torch.testing.assert_close(first["sample_id"], second["sample_id"])
    torch.testing.assert_close(first["stream_id"], second["stream_id"])
    torch.testing.assert_close(first["light_seed"], second["light_seed"])
    assert not torch.equal(first["light_seed"], changed["light_seed"])


def test_bdpt_launch_state_rejects_bad_counts_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT launch state")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="samples"):
        ops.bdpt_launch_state(reference, tx_count=1, samples=0, sample_streams=1, seed=0)


def test_bdpt_launch_state_has_no_python_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT launch state")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(paths, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="bdpt_launch_state CUDA kernel is required"):
        ops.bdpt_launch_state(reference, tx_count=1, samples=1, sample_streams=1, seed=0)


def test_bdpt_empty_subpath_state_returns_native_schema():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT subpath state")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)

    state = ops.bdpt_empty_subpath_state(reference)

    expected = {
        "origin": ((0, 3), torch.float32),
        "direction": ((0, 3), torch.float32),
        "throughput_real": ((0,), torch.float32),
        "throughput_imag": ((0,), torch.float32),
        "pdf_forward": ((0,), torch.float32),
        "pdf_reverse": ((0,), torch.float32),
        "depth": ((0,), torch.int32),
        "component_mask": ((0,), torch.int32),
        "primitive_id": ((0,), torch.int32),
        "edge_id": ((0,), torch.int32),
        "tx_id": ((0,), torch.int32),
        "rx_id": ((0,), torch.int32),
        "grid_linear_id": ((0,), torch.int32),
        "valid": ((0,), torch.bool),
        "path_length": ((0,), torch.float32),
        "field_real": ((0, 3), torch.float32),
        "field_imag": ((0, 3), torch.float32),
        "source_power": ((0,), torch.float32),
        "event_type": ((0,), torch.int32),
    }
    assert set(state) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert state[name].shape == shape
        assert state[name].dtype == dtype
        assert state[name].is_cuda
        assert state[name].is_contiguous()


def test_bdpt_endpoint_subpath_state_generates_native_light_and_sensor_endpoints():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT endpoint subpaths")

    tx_positions = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        device="cuda",
        dtype=torch.float32,
    )
    tx_power = torch.tensor([10.0, 20.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        device="cuda",
        dtype=torch.float32,
    )
    launch_tx_id = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    light_seed = torch.tensor([11, 22, 33], device="cuda", dtype=torch.int64)

    endpoints = _endpoint_subpath_state(tx_positions, tx_power, rx_positions, launch_tx_id, light_seed)

    light = endpoints["light"]
    sensor = endpoints["sensor"]
    torch.testing.assert_close(light["origin"], tx_positions[launch_tx_id.to(torch.long)])
    torch.testing.assert_close(
        light["throughput_real"],
        torch.tensor([10.0, 20.0, 10.0], device="cuda").sqrt(),
    )
    torch.testing.assert_close(
        light["pdf_forward"],
        torch.full((3,), 0.07957747154594767, device="cuda"),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    torch.testing.assert_close(light["pdf_reverse"], light["pdf_forward"])
    torch.testing.assert_close(light["event_type"], torch.zeros(3, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(light["tx_id"], launch_tx_id)
    torch.testing.assert_close(light["rx_id"], torch.full((3,), -1, device="cuda", dtype=torch.int32))
    assert light["valid"].all()

    torch.testing.assert_close(sensor["origin"], rx_positions)
    torch.testing.assert_close(sensor["throughput_real"], torch.ones(2, device="cuda"))
    torch.testing.assert_close(sensor["pdf_forward"], torch.ones(2, device="cuda"))
    torch.testing.assert_close(sensor["pdf_reverse"], torch.ones(2, device="cuda"))
    torch.testing.assert_close(sensor["tx_id"], torch.full((2,), -1, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(sensor["rx_id"], torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(sensor["grid_linear_id"], torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(sensor["event_type"], torch.zeros(2, device="cuda", dtype=torch.int32))
    assert sensor["valid"].all()


def test_bdpt_endpoint_subpath_state_uses_launch_seeded_light_directions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT endpoint subpaths")

    tx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    first_launch = ops.bdpt_launch_state(tx_positions, tx_count=1, samples=4, sample_streams=1, seed=101)
    second_launch = ops.bdpt_launch_state(tx_positions, tx_count=1, samples=4, sample_streams=1, seed=101)
    changed_launch = ops.bdpt_launch_state(tx_positions, tx_count=1, samples=4, sample_streams=1, seed=102)

    first = _endpoint_subpath_state(
        tx_positions,
        tx_power,
        rx_positions,
        first_launch["tx_id"],
        first_launch["light_seed"],
    )
    second = _endpoint_subpath_state(
        tx_positions,
        tx_power,
        rx_positions,
        second_launch["tx_id"],
        second_launch["light_seed"],
    )
    changed = _endpoint_subpath_state(
        tx_positions,
        tx_power,
        rx_positions,
        changed_launch["tx_id"],
        changed_launch["light_seed"],
    )

    torch.testing.assert_close(first["light"]["direction"], second["light"]["direction"])
    assert not torch.equal(first["light"]["direction"], changed["light"]["direction"])
    for row in first["light"]["direction"].detach().cpu().tolist():
        norm = sum(float(component) * float(component) for component in row) ** 0.5
        assert norm == pytest.approx(1.0, rel=1.0e-6, abs=1.0e-6)


def test_bdpt_reflected_light_subpath_state_uses_native_hit_geometry():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflected light subpaths")

    light = {
        "origin": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "direction": torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        "throughput_real": torch.tensor([2.0], device="cuda", dtype=torch.float32),
        "throughput_imag": torch.tensor([0.0], device="cuda", dtype=torch.float32),
        "pdf_forward": torch.tensor([0.25], device="cuda", dtype=torch.float32),
        "pdf_reverse": torch.tensor([0.0], device="cuda", dtype=torch.float32),
        "depth": torch.tensor([0], device="cuda", dtype=torch.int32),
        "component_mask": torch.tensor([1], device="cuda", dtype=torch.int32),
        "primitive_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "edge_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "tx_id": torch.tensor([3], device="cuda", dtype=torch.int32),
        "rx_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "valid": torch.tensor([True], device="cuda", dtype=torch.bool),
        "path_length": torch.tensor([0.0], device="cuda", dtype=torch.float32),
    }
    light = _complete_subpath_field_state(light)
    intersection = {
        "t": torch.tensor([1.0], device="cuda", dtype=torch.float32),
        "p": torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        "n": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "geo_n": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "uv": torch.empty((1, 2), device="cuda", dtype=torch.float32),
        "barycentric": torch.empty((1, 3), device="cuda", dtype=torch.float32),
        "shape_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "prim_id": torch.tensor([7], device="cuda", dtype=torch.int32),
        "local_prim_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "global_prim_id": torch.tensor([7], device="cuda", dtype=torch.int32),
    }

    reflected = ops.bdpt_reflected_light_subpath_state(
        light,
        intersection,
        material_gain=torch.ones((8,), device="cuda", dtype=torch.float32),
        material_valid=torch.ones((8,), device="cuda", dtype=torch.bool),
        material_eps_r=torch.ones((8,), device="cuda", dtype=torch.float32),
        material_sigma_e=torch.full((8,), 1.0e14, device="cuda", dtype=torch.float32),
        material_mu_r=torch.ones((8,), device="cuda", dtype=torch.float32),
        material_thickness=torch.full((8,), 0.1, device="cuda", dtype=torch.float32),
        frequency_hz=3.0e9,
    )

    torch.testing.assert_close(
        reflected["origin"].cpu(),
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        reflected["direction"].cpu(),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(reflected["throughput_real"].cpu(), torch.tensor([2.0], dtype=torch.float32))
    torch.testing.assert_close(reflected["pdf_forward"].cpu(), torch.tensor([0.25], dtype=torch.float32))
    torch.testing.assert_close(reflected["pdf_reverse"].cpu(), torch.tensor([0.25], dtype=torch.float32))
    torch.testing.assert_close(reflected["depth"].cpu(), torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(reflected["component_mask"].cpu(), torch.tensor([3], dtype=torch.int32))
    torch.testing.assert_close(reflected["event_type"].cpu(), torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(reflected["primitive_id"].cpu(), torch.tensor([7], dtype=torch.int32))
    torch.testing.assert_close(reflected["tx_id"].cpu(), torch.tensor([3], dtype=torch.int32))
    torch.testing.assert_close(reflected["valid"].cpu(), torch.tensor([True], dtype=torch.bool))


def test_bdpt_reflected_light_subpath_state_applies_native_material_gain_and_validity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflected light subpaths")

    light = {
        "origin": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "direction": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        "throughput_real": torch.tensor([2.0, 4.0], device="cuda", dtype=torch.float32),
        "throughput_imag": torch.tensor([3.0, 5.0], device="cuda", dtype=torch.float32),
        "pdf_forward": torch.tensor([0.25, 0.5], device="cuda", dtype=torch.float32),
        "pdf_reverse": torch.tensor([0.0, 0.0], device="cuda", dtype=torch.float32),
        "depth": torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        "component_mask": torch.tensor([1, 1], device="cuda", dtype=torch.int32),
        "primitive_id": torch.tensor([-1, -1], device="cuda", dtype=torch.int32),
        "edge_id": torch.tensor([-1, -1], device="cuda", dtype=torch.int32),
        "tx_id": torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        "rx_id": torch.tensor([-1, -1], device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.tensor([-1, -1], device="cuda", dtype=torch.int32),
        "valid": torch.tensor([True, True], device="cuda", dtype=torch.bool),
        "path_length": torch.tensor([0.0, 0.0], device="cuda", dtype=torch.float32),
    }
    light = _complete_subpath_field_state(light)
    intersection = {
        "t": torch.tensor([1.0, 1.0], device="cuda", dtype=torch.float32),
        "p": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        "n": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "geo_n": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "uv": torch.empty((2, 2), device="cuda", dtype=torch.float32),
        "barycentric": torch.empty((2, 3), device="cuda", dtype=torch.float32),
        "shape_id": torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        "prim_id": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        "local_prim_id": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        "global_prim_id": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
    }
    material_gain = torch.tensor([0.5, 0.25], device="cuda", dtype=torch.float32)
    material_valid = torch.tensor([True, False], device="cuda", dtype=torch.bool)

    reflected = ops.bdpt_reflected_light_subpath_state(
        light,
        intersection,
        material_gain=material_gain,
        material_valid=material_valid,
        material_eps_r=torch.ones((2,), device="cuda", dtype=torch.float32),
        material_sigma_e=torch.full((2,), 1.0e14, device="cuda", dtype=torch.float32),
        material_mu_r=torch.ones((2,), device="cuda", dtype=torch.float32),
        material_thickness=torch.full((2,), 0.1, device="cuda", dtype=torch.float32),
        frequency_hz=3.0e9,
    )

    # Throughput is a real amplitude proxy: specular reflection scales it by
    # sqrt(material_gain * R_eff); the near-PEC wall has R_eff ~= 1.
    amplitude = math.sqrt(0.5)
    torch.testing.assert_close(
        reflected["throughput_real"].cpu(),
        torch.tensor([2.0 * amplitude, 0.0], dtype=torch.float32),
        rtol=1.0e-4,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        reflected["throughput_imag"].cpu(),
        torch.tensor([3.0 * amplitude, 0.0], dtype=torch.float32),
        rtol=1.0e-4,
        atol=1.0e-6,
    )
    torch.testing.assert_close(reflected["valid"].cpu(), torch.tensor([True, False], dtype=torch.bool))


def test_bdpt_subpath_intersection_inputs_expose_native_raydn_ray_schema():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT subpath intersection inputs")

    tx_positions = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    launch_tx_id = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    light_seed = torch.tensor([11, 22], device="cuda", dtype=torch.int64)
    light = _endpoint_subpath_state(tx_positions, tx_power, rx_positions, launch_tx_id, light_seed)["light"]

    inputs = ops.bdpt_subpath_intersection_inputs(light)

    assert set(inputs) == {"ray_o", "ray_d", "ray_tmax", "active"}
    torch.testing.assert_close(inputs["ray_o"], light["origin"])
    torch.testing.assert_close(inputs["ray_d"], light["direction"])
    torch.testing.assert_close(inputs["active"], light["valid"])
    assert inputs["ray_tmax"].shape == (0,)
    assert inputs["ray_tmax"].dtype == torch.float32
    assert inputs["ray_tmax"].is_cuda


def test_bdpt_endpoint_connection_samples_classifies_reflected_light_as_reflection():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT endpoint connections")

    light = {
        "origin": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "direction": torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        "throughput_real": torch.tensor([1.0], device="cuda", dtype=torch.float32),
        "throughput_imag": torch.tensor([0.0], device="cuda", dtype=torch.float32),
        "pdf_forward": torch.tensor([0.25], device="cuda", dtype=torch.float32),
        "pdf_reverse": torch.tensor([0.0], device="cuda", dtype=torch.float32),
        "depth": torch.tensor([0], device="cuda", dtype=torch.int32),
        "component_mask": torch.tensor([1], device="cuda", dtype=torch.int32),
        "primitive_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "edge_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "tx_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "rx_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "valid": torch.tensor([True], device="cuda", dtype=torch.bool),
        "path_length": torch.tensor([0.0], device="cuda", dtype=torch.float32),
    }
    light = _complete_subpath_field_state(light)
    intersection = {
        "t": torch.tensor([1.0], device="cuda", dtype=torch.float32),
        "p": torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        "n": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "geo_n": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "uv": torch.empty((1, 2), device="cuda", dtype=torch.float32),
        "barycentric": torch.empty((1, 3), device="cuda", dtype=torch.float32),
        "shape_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "prim_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "local_prim_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "global_prim_id": torch.tensor([0], device="cuda", dtype=torch.int32),
    }
    reflected = ops.bdpt_reflected_light_subpath_state(
        light,
        intersection,
        material_gain=torch.ones((1,), device="cuda", dtype=torch.float32),
        material_valid=torch.ones((1,), device="cuda", dtype=torch.bool),
        material_eps_r=torch.ones((1,), device="cuda", dtype=torch.float32),
        material_sigma_e=torch.full((1,), 1.0e14, device="cuda", dtype=torch.float32),
        material_mu_r=torch.ones((1,), device="cuda", dtype=torch.float32),
        material_thickness=torch.full((1,), 0.1, device="cuda", dtype=torch.float32),
        frequency_hz=3.0e9,
    )
    sensor = ops.bdpt_empty_subpath_state(light["origin"])
    sensor["origin"] = torch.tensor([[0.0, 0.0, 2.0]], device="cuda", dtype=torch.float32)
    sensor["direction"] = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
    sensor["throughput_real"] = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    sensor["throughput_imag"] = torch.tensor([0.0], device="cuda", dtype=torch.float32)
    sensor["pdf_forward"] = torch.tensor([0.0], device="cuda", dtype=torch.float32)
    sensor["pdf_reverse"] = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    sensor["depth"] = torch.tensor([0], device="cuda", dtype=torch.int32)
    sensor["component_mask"] = torch.tensor([1], device="cuda", dtype=torch.int32)
    sensor["primitive_id"] = torch.tensor([-1], device="cuda", dtype=torch.int32)
    sensor["edge_id"] = torch.tensor([-1], device="cuda", dtype=torch.int32)
    sensor["tx_id"] = torch.tensor([-1], device="cuda", dtype=torch.int32)
    sensor["rx_id"] = torch.tensor([2], device="cuda", dtype=torch.int32)
    sensor["grid_linear_id"] = torch.tensor([5], device="cuda", dtype=torch.int32)
    sensor["valid"] = torch.tensor([True], device="cuda", dtype=torch.bool)
    sensor["path_length"] = torch.tensor([0.0], device="cuda", dtype=torch.float32)
    sensor["field_real"] = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    sensor["field_imag"] = torch.zeros((1, 3), device="cuda")
    sensor["source_power"] = torch.zeros((1,), device="cuda")
    sensor["event_type"] = torch.zeros((1,), device="cuda", dtype=torch.int32)

    samples = ops.bdpt_endpoint_connection_samples(
        reflected,
        sensor,
        frequency_hz=3.0e9,
        samples_per_tx=1,
        max_paths=None,
    )

    torch.testing.assert_close(samples["component_id"].cpu(), torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(samples["light_depth"].cpu(), torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(samples["topology"].cpu(), torch.tensor([[0, 2, 1, 1]], dtype=torch.int32))


def test_bdpt_endpoint_connection_samples_emit_native_connection_schema():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT endpoint connections")

    tx_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    tx_power = torch.tensor([2.0, 0.5], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor(
        [[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    launch_tx_id = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    light_seed = torch.tensor([11, 22], device="cuda", dtype=torch.int64)

    endpoints = _endpoint_subpath_state(tx_positions, tx_power, rx_positions, launch_tx_id, light_seed)
    samples = ops.bdpt_endpoint_connection_samples(
        endpoints["light"],
        endpoints["sensor"],
        frequency_hz=3.0e9,
        samples_per_tx=1,
        max_paths=None,
    )

    expected_fields = {
        "topology",
        "contribution",
        "pdf",
        "mis_weight",
        "component_id",
        "valid",
        "tx_id",
        "rx_id",
        "grid_linear_id",
        "light_depth",
        "sensor_depth",
        "path_length_m",
    }
    assert set(samples) == expected_fields
    assert samples["topology"].shape == (4, 4)
    torch.testing.assert_close(samples["tx_id"], torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["rx_id"], torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["grid_linear_id"], samples["rx_id"])
    torch.testing.assert_close(samples["component_id"], torch.zeros(4, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["light_depth"], torch.zeros(4, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["sensor_depth"], torch.zeros(4, device="cuda", dtype=torch.int32))
    assert samples["valid"].all()

    distance = torch.tensor([5.0, 10.0, 13.0**0.5, 72.0**0.5], device="cuda", dtype=torch.float32)
    wavelength = 299_792_458.0 / 3.0e9
    expected = torch.tensor([2.0, 2.0, 0.5, 0.5], device="cuda", dtype=torch.float32) / (
        (4.0 * torch.pi * distance / wavelength) ** 2
    )
    torch.testing.assert_close(samples["path_length_m"], distance, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(samples["contribution"], expected, rtol=1.0e-6, atol=1.0e-12)
    assert torch.all(samples["pdf"] > 0.0)
    assert not torch.allclose(samples["pdf"], samples["valid"].to(torch.float32))
    torch.testing.assert_close(samples["mis_weight"], torch.ones(4, device="cuda"))


def test_bdpt_filter_connection_samples_applies_native_visibility_mask():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT connection filtering")

    tx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    endpoints = _endpoint_subpath_state(
        tx_positions,
        tx_power,
        rx_positions,
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.tensor([11], device="cuda", dtype=torch.int64),
    )
    samples = ops.bdpt_endpoint_connection_samples(
        endpoints["light"],
        endpoints["sensor"],
        frequency_hz=3.0e9,
        samples_per_tx=1,
        max_paths=None,
    )
    visible = torch.tensor([True, False], device="cuda", dtype=torch.bool)

    filtered = ops.bdpt_filter_connection_samples(samples, visible)

    torch.testing.assert_close(filtered["valid"].cpu(), torch.tensor([True, False], dtype=torch.bool))
    assert filtered["contribution"][0].item() > 0.0
    torch.testing.assert_close(filtered["contribution"][1].cpu(), torch.tensor(0.0, dtype=torch.float32))
    torch.testing.assert_close(filtered["pdf"][1].cpu(), torch.tensor(0.0, dtype=torch.float32))
    torch.testing.assert_close(filtered["mis_weight"][1].cpu(), torch.tensor(0.0, dtype=torch.float32))


def test_bdpt_concat_connection_samples_concatenates_native_blocks():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT connection concatenation")

    tx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    first_rx = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    second_rx = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    launch_tx_id = torch.tensor([0], device="cuda", dtype=torch.int32)
    light_seed = torch.tensor([11], device="cuda", dtype=torch.int64)

    first_endpoints = _endpoint_subpath_state(tx_positions, tx_power, first_rx, launch_tx_id, light_seed)
    second_endpoints = _endpoint_subpath_state(tx_positions, tx_power, second_rx, launch_tx_id, light_seed)
    first = ops.bdpt_endpoint_connection_samples(
        first_endpoints["light"],
        first_endpoints["sensor"],
        frequency_hz=3.0e9,
        samples_per_tx=1,
    )
    second = ops.bdpt_endpoint_connection_samples(
        second_endpoints["light"],
        second_endpoints["sensor"],
        frequency_hz=3.0e9,
        samples_per_tx=1,
    )

    merged = ops.bdpt_concat_connection_samples((first, second))

    assert merged["valid"].shape == (3,)
    torch.testing.assert_close(merged["rx_id"].cpu(), torch.tensor([0, 0, 1], dtype=torch.int32))
    torch.testing.assert_close(merged["topology"][:1].cpu(), first["topology"].cpu())
    torch.testing.assert_close(merged["topology"][1:].cpu(), second["topology"].cpu())
    torch.testing.assert_close(merged["contribution"][:1].cpu(), first["contribution"].cpu())
    torch.testing.assert_close(merged["contribution"][1:].cpu(), second["contribution"].cpu())


def test_bdpt_endpoint_connection_visibility_inputs_match_connection_pair_order():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT endpoint connection visibility inputs")

    tx_positions = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0, 1.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    endpoints = _endpoint_subpath_state(
        tx_positions,
        tx_power,
        rx_positions,
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        torch.tensor([11, 22], device="cuda", dtype=torch.int64),
    )

    visibility = ops.bdpt_endpoint_connection_visibility_inputs(
        endpoints["light"],
        endpoints["sensor"],
        sample_count=3,
    )

    assert set(visibility) == {"start", "end", "active"}
    torch.testing.assert_close(
        visibility["start"].cpu(),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        visibility["end"].cpu(),
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(visibility["active"].cpu(), torch.tensor([True, True, True], dtype=torch.bool))


def test_bdpt_endpoint_connection_samples_has_no_python_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT endpoint connections")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    endpoints = _endpoint_subpath_state(
        reference,
        torch.ones((1,), device="cuda", dtype=torch.float32),
        reference,
        torch.zeros((1,), device="cuda", dtype=torch.int32),
        torch.ones((1,), device="cuda", dtype=torch.int64),
    )
    monkeypatch.setattr(symbols, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="bdpt_endpoint_connection_samples CUDA kernel is required"):
        ops.bdpt_endpoint_connection_samples(
            endpoints["light"],
            endpoints["sensor"],
            frequency_hz=3.0e9,
            samples_per_tx=1,
            max_paths=None,
        )


def test_legacy_bdpt_matrix_export_facades_are_not_public():
    for name in (
        "bdpt_export_paths",
        "bdpt_export_component_paths",
        "bdpt_component_connection_samples",
        "bdpt_variance_estimate",
        "bdpt_connection_samples_from_path_block",
        "bdpt_sample_path_block",
    ):
        assert not hasattr(ops, name)

    try:
        native = ops.native_extension()
    except ModuleNotFoundError:
        native = None
    if native is not None:
        for name in (
            "bdpt_export_paths",
            "bdpt_export_component_paths",
            "bdpt_component_connection_samples",
            "bdpt_variance_estimate",
            "bdpt_connection_samples_from_path_block",
            "bdpt_sample_path_block",
        ):
            assert not hasattr(native, name)


def test_bdpt_diffraction_connection_samples_from_tape_emits_native_schema():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction tape connections")

    tape = {
        "active": torch.tensor([True, False], device="cuda", dtype=torch.bool),
        "state_idx": torch.tensor([0, -1], device="cuda", dtype=torch.int32),
        "cell": torch.tensor([0, -1], device="cuda", dtype=torch.int32),
        "material_idx": torch.tensor([0, -1], device="cuda", dtype=torch.int32),
        "edge_u": torch.tensor([0.5, 0.0], device="cuda", dtype=torch.float32),
    }
    states = (
        torch.tensor([7], device="cuda", dtype=torch.int32),
        torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        torch.tensor([0.0], device="cuda", dtype=torch.float32),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.tensor([1], device="cuda", dtype=torch.int32),
        torch.tensor([math.pi], device="cuda", dtype=torch.float32),
        torch.tensor([[-1.0, 0.0, 0.5]], device="cuda", dtype=torch.float32),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
    )

    samples = ops.bdpt_diffraction_connection_samples_from_tape(
        tape,
        states,
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([True], device="cuda", dtype=torch.bool),
        tx_index=3,
        state_count=1,
        grid_axis=0,
        grid_position=1.0,
        grid_coord0_min=-0.5,
        grid_coord0_max=0.5,
        grid_coord1_min=0.0,
        grid_coord1_max=1.0,
        grid_resolution0=1,
        grid_resolution1=1,
        grid_cell_area=1.0,
        wavelength=0.125,
        direct_samples=2,
        keller_samples=0,
        mis="none",
        beta=2.0,
        strategy_count=1,
    )

    assert set(samples) == {
        "topology",
        "contribution",
        "pdf",
        "mis_weight",
        "component_id",
        "valid",
        "tx_id",
        "rx_id",
        "grid_linear_id",
        "light_depth",
        "sensor_depth",
        "path_length_m",
    }
    torch.testing.assert_close(samples["tx_id"], torch.tensor([3, 3], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["rx_id"], torch.tensor([0, -1], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["grid_linear_id"], samples["rx_id"])
    torch.testing.assert_close(samples["component_id"], torch.full((2,), 2, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(
        samples["topology"],
        torch.tensor([[3, 0, 2, 1], [3, -1, 2, 1]], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(samples["light_depth"], torch.ones(2, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["sensor_depth"], torch.zeros(2, device="cuda", dtype=torch.int32))
    assert samples["valid"][0].item() is True
    assert samples["valid"][1].item() is False
    assert samples["contribution"][0].item() > 0.0
    assert samples["pdf"][0].item() > 0.0
    assert samples["path_length_m"][0].item() > 0.0


def test_bdpt_diffraction_tape_mis_uses_direct_keller_pdf_sums():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction tape connections")

    tape = {
        "active": torch.tensor([True, True, True], device="cuda", dtype=torch.bool),
        "state_idx": torch.tensor([0, 0, 0], device="cuda", dtype=torch.int32),
        "cell": torch.tensor([0, 0, 0], device="cuda", dtype=torch.int32),
        "material_idx": torch.tensor([0, 0, 0], device="cuda", dtype=torch.int32),
        "edge_u": torch.tensor([0.25, 0.5, 0.75], device="cuda", dtype=torch.float32),
    }
    states = (
        torch.tensor([7], device="cuda", dtype=torch.int32),
        torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        torch.tensor([0.0], device="cuda", dtype=torch.float32),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.tensor([1], device="cuda", dtype=torch.int32),
        torch.tensor([math.pi], device="cuda", dtype=torch.float32),
        torch.tensor([[-1.0, 0.0, 0.5]], device="cuda", dtype=torch.float32),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
    )

    def export(mis: str) -> dict[str, torch.Tensor]:
        return ops.bdpt_diffraction_connection_samples_from_tape(
            tape,
            states,
            torch.tensor([1.0], device="cuda", dtype=torch.float32),
            torch.tensor([True], device="cuda", dtype=torch.bool),
            tx_index=3,
            state_count=1,
            grid_axis=0,
            grid_position=1.0,
            grid_coord0_min=-0.5,
            grid_coord0_max=0.5,
            grid_coord1_min=0.0,
            grid_coord1_max=1.0,
            grid_resolution0=1,
            grid_resolution1=1,
            grid_cell_area=1.0,
            wavelength=0.125,
            direct_samples=1,
            keller_samples=2,
            mis=mis,
            beta=2.0,
            strategy_count=2,
        )

    balance = export("balance")
    power = export("power_heuristic")

    torch.testing.assert_close(
        balance["mis_weight"],
        torch.tensor([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], device="cuda", dtype=torch.float32),
    )
    torch.testing.assert_close(
        power["mis_weight"],
        torch.tensor([1.0 / 5.0, 4.0 / 5.0, 4.0 / 5.0], device="cuda", dtype=torch.float32),
    )
    assert not torch.allclose(balance["mis_weight"], torch.full((3,), 0.5, device="cuda"))


def test_bdpt_diffraction_point_connection_samples_emits_native_schema_and_visibility_segments():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT point diffraction connections")

    states = (
        torch.tensor([7], device="cuda", dtype=torch.int32),
        torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        torch.tensor([0.0], device="cuda", dtype=torch.float32),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.tensor([1], device="cuda", dtype=torch.int32),
        torch.tensor([math.pi], device="cuda", dtype=torch.float32),
        torch.tensor([[-1.0, 0.0, 0.5]], device="cuda", dtype=torch.float32),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
    )

    exported = ops.bdpt_diffraction_point_connection_samples(
        torch.tensor([[1.0, 0.0, 0.5]], device="cuda", dtype=torch.float32),
        states,
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([True], device="cuda", dtype=torch.bool),
        tx_index=3,
        state_count=1,
        direct_samples=2,
        keller_samples=0,
        seed=123,
        wavelength=0.125,
        mis="none",
        beta=2.0,
        strategy_count=1,
    )
    assert set(exported) == {"samples", "source_start", "source_end", "target_start", "target_end", "visibility_active"}
    samples = exported["samples"]
    assert isinstance(samples, dict)
    torch.testing.assert_close(samples["tx_id"], torch.tensor([3, 3], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["rx_id"], torch.tensor([0, 0], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(samples["component_id"], torch.full((2,), 2, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(
        samples["topology"],
        torch.tensor([[3, 0, 2, 1], [3, 0, 2, 1]], device="cuda", dtype=torch.int32),
    )
    assert samples["valid"].all()
    assert samples["contribution"][0].item() > 0.0
    assert samples["pdf"][0].item() > 0.0
    for name in ("source_start", "source_end", "target_start", "target_end"):
        assert exported[name].shape == (2, 3)
    torch.testing.assert_close(exported["source_end"], exported["target_start"])
    assert exported["visibility_active"].all()


@pytest.mark.parametrize("accumulation_strategy", ["atomic", "staged", "compact"])
def test_bdpt_accumulate_connection_samples_applies_mis_weight(accumulation_strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT connection accumulation")

    samples = {
        "topology": torch.zeros((2, 4), device="cuda", dtype=torch.int32),
        "contribution": torch.tensor([4.0, 8.0], device="cuda", dtype=torch.float32),
        "pdf": torch.ones((2,), device="cuda", dtype=torch.float32),
        "mis_weight": torch.tensor([0.25, 0.5], device="cuda", dtype=torch.float32),
        "component_id": torch.tensor([1, 1], device="cuda", dtype=torch.int32),
        "valid": torch.ones((2,), device="cuda", dtype=torch.bool),
        "tx_id": torch.zeros((2,), device="cuda", dtype=torch.int32),
        "rx_id": torch.zeros((2,), device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.zeros((2,), device="cuda", dtype=torch.int32),
        "light_depth": torch.ones((2,), device="cuda", dtype=torch.int32),
        "sensor_depth": torch.zeros((2,), device="cuda", dtype=torch.int32),
        "path_length_m": torch.ones((2,), device="cuda", dtype=torch.float32),
    }

    accumulated = ops.bdpt_accumulate_connection_samples(
        samples,
        tx_count=1,
        rx_count=1,
        accumulation_strategy=accumulation_strategy,
    )

    expected = torch.tensor([[5.0]], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(accumulated["path_gain"], expected)
    torch.testing.assert_close(accumulated["reflection"], expected)
    torch.testing.assert_close(accumulated["los"], torch.zeros_like(expected))
    torch.testing.assert_close(accumulated["diffraction"], torch.zeros_like(expected))
    torch.testing.assert_close(accumulated["transmission"], torch.zeros_like(expected))


def test_bdpt_count_valid_connection_samples_uses_native_valid_mask():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT connection sample counting")

    samples = {
        "topology": torch.zeros((4, 4), device="cuda", dtype=torch.int32),
        "contribution": torch.ones((4,), device="cuda", dtype=torch.float32),
        "pdf": torch.ones((4,), device="cuda", dtype=torch.float32),
        "mis_weight": torch.ones((4,), device="cuda", dtype=torch.float32),
        "component_id": torch.zeros((4,), device="cuda", dtype=torch.int32),
        "valid": torch.tensor([True, False, True, False], device="cuda", dtype=torch.bool),
        "tx_id": torch.zeros((4,), device="cuda", dtype=torch.int32),
        "rx_id": torch.zeros((4,), device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.zeros((4,), device="cuda", dtype=torch.int32),
        "light_depth": torch.zeros((4,), device="cuda", dtype=torch.int32),
        "sensor_depth": torch.zeros((4,), device="cuda", dtype=torch.int32),
        "path_length_m": torch.ones((4,), device="cuda", dtype=torch.float32),
    }

    assert ops.bdpt_count_valid_connection_samples(samples) == 2


def test_bdpt_accumulate_connection_samples_has_no_python_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT connection accumulation")

    samples = {
        "topology": torch.zeros((1, 4), device="cuda", dtype=torch.int32),
        "contribution": torch.ones((1,), device="cuda", dtype=torch.float32),
        "pdf": torch.ones((1,), device="cuda", dtype=torch.float32),
        "mis_weight": torch.ones((1,), device="cuda", dtype=torch.float32),
        "component_id": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "valid": torch.ones((1,), device="cuda", dtype=torch.bool),
        "tx_id": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "rx_id": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "light_depth": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "sensor_depth": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "path_length_m": torch.ones((1,), device="cuda", dtype=torch.float32),
    }
    monkeypatch.setattr(symbols, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="bdpt_accumulate_connection_samples CUDA kernel is required"):
        ops.bdpt_accumulate_connection_samples(samples, tx_count=1, rx_count=1)


def test_bdpt_accumulate_connection_samples_passes_strategy_id_to_native(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT connection accumulation")

    calls: list[int] = []

    class FakeNative:
        def bdpt_accumulate_connection_samples(self, samples, tx_count, rx_count, accumulation_strategy):
            calls.append(accumulation_strategy)
            return {
                "path_gain": torch.zeros((tx_count, rx_count), device="cuda", dtype=torch.float32),
                "los": torch.zeros((tx_count, rx_count), device="cuda", dtype=torch.float32),
                "reflection": torch.zeros((tx_count, rx_count), device="cuda", dtype=torch.float32),
                "diffraction": torch.zeros((tx_count, rx_count), device="cuda", dtype=torch.float32),
                "transmission": torch.zeros((tx_count, rx_count), device="cuda", dtype=torch.float32),
                "scattering": torch.zeros((tx_count, rx_count), device="cuda", dtype=torch.float32),
            }

    samples = {
        "topology": torch.zeros((1, 4), device="cuda", dtype=torch.int32),
        "contribution": torch.ones((1,), device="cuda", dtype=torch.float32),
        "pdf": torch.ones((1,), device="cuda", dtype=torch.float32),
        "mis_weight": torch.ones((1,), device="cuda", dtype=torch.float32),
        "component_id": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "valid": torch.ones((1,), device="cuda", dtype=torch.bool),
        "tx_id": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "rx_id": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "light_depth": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "sensor_depth": torch.zeros((1,), device="cuda", dtype=torch.int32),
        "path_length_m": torch.ones((1,), device="cuda", dtype=torch.float32),
    }
    monkeypatch.setattr(symbols, "native_extension", lambda: FakeNative())

    for strategy in ("atomic", "staged", "compact"):
        ops.bdpt_accumulate_connection_samples(
            samples,
            tx_count=1,
            rx_count=1,
            accumulation_strategy=strategy,
        )

    assert calls == [0, 1, 2]


def test_bdpt_accumulation_strategies_agree_with_invalid_and_duplicate_samples():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT connection accumulation")

    samples = {
        "topology": torch.zeros((9, 4), device="cuda", dtype=torch.int32),
        "contribution": torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0], device="cuda"),
        "pdf": torch.ones((9,), device="cuda", dtype=torch.float32),
        "mis_weight": torch.tensor([1.0, 0.5, 0.25, 0.125, 1.0, 1.0, 1.0, 1.0, 1.0], device="cuda"),
        "component_id": torch.tensor([0, 5, 1, 2, 1, 2, 1, 0, 7], device="cuda", dtype=torch.int32),
        "valid": torch.tensor([True, True, True, True, False, True, True, True, True], device="cuda", dtype=torch.bool),
        "tx_id": torch.tensor([0, 0, 0, 0, 1, 1, 3, 1, 1], device="cuda", dtype=torch.int32),
        "rx_id": torch.tensor([0, 0, 1, 1, 0, 2, 0, 5, 1], device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.zeros((9,), device="cuda", dtype=torch.int32),
        "light_depth": torch.zeros((9,), device="cuda", dtype=torch.int32),
        "sensor_depth": torch.zeros((9,), device="cuda", dtype=torch.int32),
        "path_length_m": torch.ones((9,), device="cuda", dtype=torch.float32),
    }

    atomic = ops.bdpt_accumulate_connection_samples(
        samples,
        tx_count=2,
        rx_count=3,
        accumulation_strategy="atomic",
    )
    for strategy in ("staged", "compact"):
        accumulated = ops.bdpt_accumulate_connection_samples(
            samples,
            tx_count=2,
            rx_count=3,
            accumulation_strategy=strategy,
        )
        for name in ("path_gain", "los", "reflection", "diffraction", "transmission"):
            torch.testing.assert_close(accumulated[name], atomic[name], rtol=2.0e-6, atol=1.0e-12)


def test_bdpt_los_component_maps_from_matrix_uses_native_grid_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT component-map layout")

    los = torch.arange(6, device="cuda", dtype=torch.float32).reshape(1, 6).contiguous()

    maps = ops.bdpt_los_component_maps_from_matrix(los, rows=2, cols=3)

    assert maps.shape == (1, 3, 2)
    expected = torch.tensor([[[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]], device="cuda")
    torch.testing.assert_close(maps, expected)


def test_bdpt_face_material_tensors_from_host_expands_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT material tensors")

    result = ops.bdpt_face_material_tensors_from_host(
        (2.0, 5.0),
        (0.1, 0.2),
        (1.0, 1.5),
        (1, 0, 1),
    )

    assert result["eps_r"].is_cuda
    torch.testing.assert_close(result["eps_r"].cpu(), torch.tensor([5.0, 2.0, 5.0]))
    torch.testing.assert_close(result["sigma_e"].cpu(), torch.tensor([0.2, 0.1, 0.2]))
    torch.testing.assert_close(result["mu_r"].cpu(), torch.tensor([1.5, 1.0, 1.5]))
    torch.testing.assert_close(result["gain"].cpu(), torch.ones(3))
    torch.testing.assert_close(result["valid"].cpu(), torch.ones(3, dtype=torch.bool))


def test_bdpt_face_material_tensors_from_host_has_no_python_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT material tensors")

    monkeypatch.setattr(symbols, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="bdpt_face_material_tensors_from_host CUDA kernel is required"):
        ops.bdpt_face_material_tensors_from_host((2.0,), (0.1,), (1.0,), (0,))
