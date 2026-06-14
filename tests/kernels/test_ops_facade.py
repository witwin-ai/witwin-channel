import pytest
import torch

from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.kernels import raydn_backend


def test_validate_cuda_tensor_accepts_matching_cuda_tensor_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for positive CUDA tensor validation")

    tensor = torch.zeros((2, 3), device="cuda", dtype=torch.float32)

    assert ops.validate_cuda_tensor(
        "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    ) is tensor


def test_validate_cuda_tensor_rejects_cpu_tensor():
    tensor = torch.zeros((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="points must be a CUDA tensor"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_validate_cuda_tensor_rejects_wrong_dtype_before_shape():
    tensor = torch.zeros((2, 3), dtype=torch.float64)

    with pytest.raises(TypeError, match="points must have dtype torch.float32"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_validate_cuda_tensor_rejects_trailing_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to isolate shape validation after device validation")

    tensor = torch.zeros((2, 2), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="points must end with shape"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_noop_metadata_reports_valid_schema():
    metadata = ops.noop_metadata(accumulation_strategy="atomic_add")

    assert metadata["primitive"] == "noop_metadata"
    assert metadata["accumulation_strategy"] == "atomic_add"
    ops.validate_metadata(metadata)


def test_path_los_export_returns_cuda_path_tensors_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path LoS export")

    tx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[3.0, 4.0, 0.0]], device="cuda", dtype=torch.float32)

    result = ops.path_los_export(
        tx_positions,
        tx_power,
        rx_positions,
        frequency_hz=3.0e9,
    )

    assert result["tx_id"].is_cuda
    assert result["rx_id"].is_cuda
    assert result["path_length_m"].is_cuda
    assert result["path_gain"].is_cuda
    assert result["path_length_m"].item() == pytest.approx(5.0)


def test_path_los_export_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path LoS export")

    tx_positions = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="path_los_export CUDA kernel is required"):
        ops.path_los_export(tx_positions, tx_power, rx_positions, frequency_hz=3.0e9)


def test_deterministic_los_field_matches_python_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic LoS field")

    from witwin.channel_native.deterministic.field import free_space_complex_field

    path_gain = torch.tensor([1.0e-4, 2.5e-5, 0.0], device="cuda", dtype=torch.float32)
    path_length = torch.tensor([1.0, 3.25, 9.5], device="cuda", dtype=torch.float32)

    result = ops.deterministic_los_field(
        path_gain=path_gain,
        path_length_m=path_length,
        frequency_hz=3.0e9,
    )

    field = torch.complex(result["field_real"], result["field_imag"])
    expected = free_space_complex_field(path_gain, path_length, 3.0e9)
    torch.testing.assert_close(field, expected, rtol=2.0e-5, atol=1.0e-7)
    torch.testing.assert_close(result["path_gain"], path_gain, rtol=0.0, atol=0.0)


def test_deterministic_diffraction_vector_field_matches_python_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diffraction field")

    from witwin.channel_native.deterministic.field import equivalent_field_from_vector_components

    x_re = torch.tensor([1.0, -0.5, 0.0], device="cuda", dtype=torch.float32)
    x_im = torch.tensor([0.5, 0.25, 0.0], device="cuda", dtype=torch.float32)
    y_re = torch.tensor([0.25, 0.75, 0.0], device="cuda", dtype=torch.float32)
    y_im = torch.tensor([-0.125, 0.5, 0.0], device="cuda", dtype=torch.float32)
    z_re = torch.tensor([0.125, -0.25, 0.0], device="cuda", dtype=torch.float32)
    z_im = torch.tensor([0.0625, -0.5, 0.0], device="cuda", dtype=torch.float32)

    result = ops.deterministic_diffraction_vector_field(
        x_re=x_re,
        x_im=x_im,
        y_re=y_re,
        y_im=y_im,
        z_re=z_re,
        z_im=z_im,
    )

    field = torch.complex(result["field_real"], result["field_imag"])
    expected_gain, expected_field = equivalent_field_from_vector_components(x_re, x_im, y_re, y_im, z_re, z_im)
    torch.testing.assert_close(result["path_gain"], expected_gain, rtol=2.0e-6, atol=1.0e-7)
    torch.testing.assert_close(field, expected_field, rtol=2.0e-6, atol=1.0e-7)


def test_deterministic_reflection_field_returns_native_complex_field():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection field")
    native = ops.native_extension()
    if native is None or not hasattr(native, "deterministic_reflection_field"):
        pytest.skip("native deterministic reflection field kernel is not built")

    tx_position = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    hit_position = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    rx_position = torch.tensor([[1.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    normal = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    one = torch.ones((1,), device="cuda", dtype=torch.float32)
    eps_r = torch.full((1,), 4.0, device="cuda", dtype=torch.float32)
    sigma_e = torch.zeros((1,), device="cuda", dtype=torch.float32)

    result = ops.deterministic_reflection_field(
        tx_position=tx_position,
        rx_position=rx_position,
        hit_position=hit_position,
        normal=normal,
        tx_power=one,
        eps_r=eps_r,
        sigma_e=sigma_e,
        mu_r=one,
        gain=one,
        frequency_hz=3.0e9,
    )

    assert result["path_gain"].is_cuda
    assert result["field_real"].is_cuda
    assert result["field_imag"].is_cuda
    assert result["path_gain"].item() > 0.0
    field = torch.complex(result["field_real"], result["field_imag"])
    torch.testing.assert_close(result["path_gain"], field.abs().square(), rtol=2.0e-4, atol=1.0e-10)


def test_deterministic_reflection_field_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection field")

    tensor = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    one = torch.ones((1,), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="deterministic_reflection_field CUDA kernel is required"):
        ops.deterministic_reflection_field(
            tx_position=tensor,
            rx_position=tensor,
            hit_position=tensor,
            normal=tensor,
            tx_power=one,
            eps_r=one,
            sigma_e=one,
            mu_r=one,
            gain=one,
            frequency_hz=3.0e9,
        )


def test_deterministic_reflection_sequence_field_matches_python_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection sequence field")

    from witwin.channel_native.deterministic.field import reflection_sequence_complex_field

    tx_position = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    hit_positions = torch.tensor(
        [
            [[2.0, 0.0, 1.0], [2.0, 2.0, 1.0]],
            [[2.0, 0.1, 1.0], [2.0, 2.0, 1.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    rx_position = torch.tensor([[0.0, 2.0, 1.0], [0.1, 2.0, 1.0]], device="cuda", dtype=torch.float32)
    normals = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    tx_power = torch.tensor([1.0, 0.5], device="cuda", dtype=torch.float32)
    eps_r = torch.tensor([[3.0, 4.0], [3.5, 4.5]], device="cuda", dtype=torch.float32)
    sigma_e = torch.tensor([[0.005, 0.01], [0.004, 0.009]], device="cuda", dtype=torch.float32)
    mu_r = torch.ones((2, 2), device="cuda", dtype=torch.float32)
    gain = torch.ones((2, 2), device="cuda", dtype=torch.float32)

    result = ops.deterministic_reflection_sequence_field(
        tx_position=tx_position,
        rx_position=rx_position,
        hit_positions=hit_positions,
        normals=normals,
        tx_power=tx_power,
        eps_r=eps_r,
        sigma_e=sigma_e,
        mu_r=mu_r,
        gain=gain,
        frequency_hz=3.0e9,
    )
    expected_gain, expected_field, expected_length = reflection_sequence_complex_field(
        tx_position=tx_position,
        rx_position=rx_position,
        hit_positions=hit_positions,
        normals=normals,
        tx_power_w=tx_power,
        eps_r=eps_r,
        sigma_e=sigma_e,
        mu_r=mu_r,
        gain=gain,
        frequency_hz=3.0e9,
    )

    field = torch.complex(result["field_real"], result["field_imag"])
    torch.testing.assert_close(result["path_length_m"], expected_length, rtol=2.0e-5, atol=1.0e-6)
    torch.testing.assert_close(result["path_gain"], expected_gain, rtol=5.0e-4, atol=1.0e-10)
    torch.testing.assert_close(field, expected_field, rtol=5.0e-4, atol=1.0e-7)


def test_mc_los_path_gain_backward_and_jvp_match_free_space_formula():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS path-gain AD kernels")

    tx_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    tx_power = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    rx_positions = torch.tensor(
        [[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    grad_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float32).t()
    rx_tangent = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    power_tangent = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.float32)
    tx_tangent = torch.zeros_like(tx_positions)

    grad_tx, grad_power, grad_rx = ops.mc_los_path_gain_backward(
        tx_positions,
        tx_power,
        rx_positions,
        grad_output,
        frequency_hz=3.0e9,
    )
    jvp = ops.mc_los_path_gain_jvp(
        tx_positions,
        tx_power,
        rx_positions,
        tx_tangent,
        power_tangent,
        rx_tangent,
        False,
        True,
        True,
        frequency_hz=3.0e9,
    )

    scale = (299_792_458.0 / 3.0e9 / (4.0 * torch.pi)) ** 2
    diff = tx_positions[:, None, :] - rx_positions[None, :, :]
    distance_sq = (diff * diff).sum(dim=-1)
    inv_d2 = 1.0 / distance_sq
    expected_grad_power = (grad_output * scale * inv_d2).sum(dim=1)
    coeff = grad_output * 2.0 * tx_power[:, None] * scale * inv_d2 * inv_d2
    expected_grad_rx = (coeff[:, :, None] * diff).sum(dim=0)
    expected_grad_tx = -(coeff[:, :, None] * diff).sum(dim=1)
    expected_jvp = power_tangent[:, None] * scale * inv_d2
    expected_jvp = expected_jvp + 2.0 * tx_power[:, None] * scale * inv_d2 * inv_d2 * (
        diff * rx_tangent[None, :, :]
    ).sum(dim=-1)

    torch.testing.assert_close(grad_tx, expected_grad_tx, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(grad_power, expected_grad_power, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(grad_rx, expected_grad_rx, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(jvp, expected_jvp, rtol=1e-6, atol=1e-9)


def test_mc_los_path_gain_ad_kernels_require_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS path-gain AD kernels")

    tx_positions = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    tx_power = torch.ones((1,), device="cuda", dtype=torch.float32)
    rx_positions = torch.ones((1, 3), device="cuda", dtype=torch.float32)
    grad_output = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_los_path_gain_backward CUDA kernel is required"):
        ops.mc_los_path_gain_backward(
            tx_positions,
            tx_power,
            rx_positions,
            grad_output,
            frequency_hz=3.0e9,
        )
    with pytest.raises(RuntimeError, match="mc_los_path_gain_jvp CUDA kernel is required"):
        ops.mc_los_path_gain_jvp(
            tx_positions,
            tx_power,
            rx_positions,
            tx_positions,
            tx_power,
            rx_positions,
            False,
            False,
            False,
            frequency_hz=3.0e9,
        )


def test_mc_finalize_component_maps_fuses_total_and_power_reductions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC finalize")

    los = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device="cuda", dtype=torch.float32)
    reflection = torch.tensor([[[0.5, 0.0], [1.5, 2.0]]], device="cuda", dtype=torch.float32)
    diffraction = torch.tensor([[[0.25, 0.75], [0.0, 1.0]]], device="cuda", dtype=torch.float32)

    result = ops.mc_finalize_component_maps(los, reflection, diffraction)

    expected_total = (los + reflection + diffraction).reshape(1, -1)
    torch.testing.assert_close(result["path_gain"], expected_total)
    torch.testing.assert_close(result["los_power"], los.sum())
    torch.testing.assert_close(result["reflection_power"], reflection.sum())
    torch.testing.assert_close(result["diffraction_power"], diffraction.sum())


def test_mc_finalize_component_maps_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC finalize")

    los = torch.zeros((1, 2, 2), device="cuda", dtype=torch.float32)
    reflection = torch.zeros_like(los)
    diffraction = torch.zeros_like(los)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_finalize_component_maps CUDA kernel is required"):
        ops.mc_finalize_component_maps(los, reflection, diffraction)


def test_mc_component_map_buffer_and_store_kernels_write_tx_slots():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC component map stores")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    maps = ops.mc_component_map_buffer(reference, tx_count=2, dim0=2, dim1=2)
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float32)
    scale = torch.tensor([10.0, 0.5], device="cuda", dtype=torch.float32)

    ops.mc_store_component_map(maps, source, tx_index=0)
    ops.mc_store_scaled_component_map(maps, source, scale, tx_index=1, scale_index=1)

    expected = torch.stack((source, source * 0.5), dim=0)
    torch.testing.assert_close(maps, expected)


def test_mc_component_map_store_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC component map stores")

    maps = torch.zeros((1, 2, 2), device="cuda", dtype=torch.float32)
    source = torch.ones((2, 2), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_store_component_map CUDA kernel is required"):
        ops.mc_store_component_map(maps, source, tx_index=0)


def test_mc_sample_directions_matches_golden_ratio_sequence():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC sample directions")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    directions = ops.mc_sample_directions(5, reference)
    indices = torch.arange(5, device="cuda", dtype=torch.float64)
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    azimuth_u = torch.frac(indices / golden_ratio)
    elevation_v = indices / 4.0
    phi = 2.0 * torch.pi * azimuth_u
    z = 1.0 - 2.0 * elevation_v
    radial = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    expected = torch.stack((radial * torch.cos(phi), radial * torch.sin(phi), z), dim=1).to(torch.float32)

    assert directions.is_cuda
    assert directions.shape == (5, 3)
    torch.testing.assert_close(directions, expected, rtol=1e-6, atol=1e-6)


def test_mc_sample_directions_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC sample directions")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_sample_directions CUDA kernel is required"):
        ops.mc_sample_directions(1, reference)


def test_mc_transmitter_tensors_creates_cuda_positions_and_power():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC transmitter tensors")

    result = ops.mc_transmitter_tensors(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        (0.5, 2.0),
    )

    expected_positions = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        device="cuda",
        dtype=torch.float32,
    )
    expected_power = torch.tensor([0.5, 2.0], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result["positions"], expected_positions)
    torch.testing.assert_close(result["power"], expected_power)


def test_mc_transmitter_tensors_requires_native_cuda_helper(monkeypatch):
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_transmitter_tensors CUDA helper is required"):
        ops.mc_transmitter_tensors((0.0, 0.0, 0.0), (1.0,))


def test_mc_pack_vec3_interleaves_cuda_vectors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC vec3 packing")

    x = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], device="cuda", dtype=torch.float32)
    z = torch.tensor([5.0, 6.0], device="cuda", dtype=torch.float32)

    packed = ops.mc_pack_vec3(x, y, z)

    expected = torch.tensor(
        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
        device="cuda",
        dtype=torch.float32,
    )
    torch.testing.assert_close(packed, expected)


def test_mc_pack_vec3_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC vec3 packing")

    x = torch.zeros((1,), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_pack_vec3 CUDA kernel is required"):
        ops.mc_pack_vec3(x, x, x)


def test_mc_los_component_maps_converts_to_public_grid_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS component maps")

    los = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    maps = ops.mc_los_component_maps(los)

    torch.testing.assert_close(maps, los.transpose(1, 2).contiguous())


def test_mc_apply_los_visibility_masks_one_transmitter_in_public_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS visibility")

    los = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device="cuda", dtype=torch.float32)
    maps = ops.mc_los_component_maps(los)
    visible = torch.tensor([True, False, False, True], device="cuda", dtype=torch.bool)

    out = ops.mc_apply_los_visibility(maps, los, visible, tx_index=0)

    expected = torch.tensor([[[1.0, 0.0], [0.0, 4.0]]], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_mc_apply_los_visibility_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS visibility")

    los = torch.zeros((1, 2, 2), device="cuda", dtype=torch.float32)
    maps = torch.zeros((1, 2, 2), device="cuda", dtype=torch.float32)
    visible = torch.ones((4,), device="cuda", dtype=torch.bool)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_apply_los_visibility CUDA kernel is required"):
        ops.mc_apply_los_visibility(maps, los, visible, tx_index=0)


def test_mc_los_visibility_inputs_fill_start_and_active():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS visibility inputs")

    tx_positions = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        device="cuda",
        dtype=torch.float32,
    )

    result = ops.mc_los_visibility_inputs(tx_positions, tx_index=1, rx_count=4)

    expected_start = tx_positions[1].expand(4, 3).contiguous()
    torch.testing.assert_close(result["start"], expected_start)
    assert result["active"].dtype == torch.bool
    assert result["active"].all()


def test_mc_los_visibility_inputs_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC LoS visibility inputs")

    tx_positions = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_los_visibility_inputs CUDA kernel is required"):
        ops.mc_los_visibility_inputs(tx_positions, tx_index=0, rx_count=1)


def test_mc_receiver_grid_points_matches_receiver_grid_order():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC receiver grid points")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)

    points = ops.mc_receiver_grid_points(
        reference,
        origin=(1.0, 2.0, 3.0),
        x_axis=(1.0, 0.0, 0.0),
        y_axis=(0.0, 0.0, 1.0),
        shape=(2, 3),
        spacing=(0.5, 2.0),
    )

    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 5.0],
            [1.0, 2.0, 7.0],
            [1.5, 2.0, 3.0],
            [1.5, 2.0, 5.0],
            [1.5, 2.0, 7.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    torch.testing.assert_close(points, expected)


def test_mc_receiver_grid_points_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC receiver grid points")

    reference = torch.empty((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_receiver_grid_points CUDA kernel is required"):
        ops.mc_receiver_grid_points(
            reference,
            origin=(0.0, 0.0, 0.0),
            x_axis=(1.0, 0.0, 0.0),
            y_axis=(0.0, 1.0, 0.0),
            shape=(1, 1),
            spacing=(1.0, 1.0),
        )


def test_mc_reflection_launch_inputs_fill_ray_origin_active_and_polarization():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC reflection launch inputs")

    tx_positions = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        device="cuda",
        dtype=torch.float32,
    )

    result = ops.mc_reflection_launch_inputs(tx_positions, tx_index=1, sample_count=4)

    torch.testing.assert_close(result["ray_o"], tx_positions[1].expand(4, 3).contiguous())
    assert result["ray_tmax"].shape == (0,)
    assert result["active"].dtype == torch.bool
    assert result["active"].all()
    expected_pol = torch.zeros((4, 3), device="cuda", dtype=torch.float32)
    expected_pol[:, 0] = 1.0
    torch.testing.assert_close(result["tx_pol"], expected_pol)


def test_mc_reflection_launch_inputs_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC reflection launch inputs")

    tx_positions = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_reflection_launch_inputs CUDA kernel is required"):
        ops.mc_reflection_launch_inputs(tx_positions, tx_index=0, sample_count=1)


def test_mc_diffraction_state_wi_matches_normalized_edge_to_source_direction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC diffraction state directions")

    edge_pos = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    src = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )

    state_wi = ops.mc_diffraction_state_wi(edge_pos, src)

    expected = torch.nn.functional.normalize(edge_pos - src, dim=1, eps=1.0e-6)
    torch.testing.assert_close(state_wi, expected, rtol=1e-6, atol=1e-6)


def test_mc_diffraction_state_wi_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC diffraction state directions")

    edge_pos = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    src = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_diffraction_state_wi CUDA kernel is required"):
        ops.mc_diffraction_state_wi(edge_pos, src)


def test_mc_selected_edge_indices_compacts_bool_mask_in_edge_order():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC selected edge compaction")

    selected = torch.tensor([False, True, True, False, True], device="cuda", dtype=torch.bool)

    indices = ops.mc_selected_edge_indices(selected)

    expected = torch.tensor([1, 2, 4], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(indices, expected)


def test_mc_selected_edge_indices_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC selected edge compaction")

    selected = torch.ones((1,), device="cuda", dtype=torch.bool)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_selected_edge_indices CUDA kernel is required"):
        ops.mc_selected_edge_indices(selected)


def test_mc_diffraction_state_pack_gathers_edge_state_tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC diffraction state packing")

    edge_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    edge_pos = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        device="cuda",
        dtype=torch.float32,
    )
    edge_dir = edge_pos + 10.0
    line_min = torch.tensor([-1.0, -2.0, -3.0], device="cuda", dtype=torch.float32)
    line_max = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    n0 = edge_pos + 20.0
    n1 = edge_pos + 30.0
    face0 = torch.tensor([10, 11, 12], device="cuda", dtype=torch.int32)
    face1 = torch.tensor([20, 21, 22], device="cuda", dtype=torch.int32)
    exterior_angle = torch.tensor([0.5, 1.5, 2.5], device="cuda", dtype=torch.float32)
    tx = torch.tensor([9.0, 8.0, 7.0], device="cuda", dtype=torch.float32)
    tx_power = torch.tensor(4.0, device="cuda", dtype=torch.float32)

    states = ops.mc_diffraction_state_pack(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
    )

    idx = edge_indices.to(dtype=torch.long)
    torch.testing.assert_close(states[0], edge_indices)
    torch.testing.assert_close(states[1], edge_pos[idx])
    torch.testing.assert_close(states[2], edge_dir[idx])
    torch.testing.assert_close(states[3], line_min[idx])
    torch.testing.assert_close(states[4], line_max[idx])
    torch.testing.assert_close(states[5], n0[idx])
    torch.testing.assert_close(states[6], n1[idx])
    torch.testing.assert_close(states[7], face0[idx])
    torch.testing.assert_close(states[8], face1[idx])
    torch.testing.assert_close(states[9], exterior_angle[idx])
    torch.testing.assert_close(states[10], tx.expand(2, 3).contiguous())
    torch.testing.assert_close(states[11], tx_power.expand(2).contiguous())


def test_mc_diffraction_state_pack_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC diffraction state packing")

    edge_indices = torch.zeros((1,), device="cuda", dtype=torch.int32)
    edge_pos = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    line = torch.zeros((1,), device="cuda", dtype=torch.float32)
    face = torch.zeros((1,), device="cuda", dtype=torch.int32)
    tx = torch.zeros((3,), device="cuda", dtype=torch.float32)
    tx_power = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_diffraction_state_pack CUDA kernel is required"):
        ops.mc_diffraction_state_pack(
            edge_indices,
            edge_pos,
            edge_pos,
            line,
            line,
            edge_pos,
            edge_pos,
            face,
            face,
            line,
            tx,
            tx_power,
        )


def test_raydn_diffraction_discover_edges_counted_uses_gpu_hit_count():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RayDN counted diffraction edge discovery")

    raydn_backend.require_native_extension()
    tx_pos = torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32)
    ray_dir = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    prim_index = torch.tensor([0, 0, -1, -1], device="cuda", dtype=torch.int32)
    hit_p = torch.tensor(
        [
            [1.0, 0.2, 0.0],
            [1.0, 0.3, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    hit_n = torch.tensor([[0.0, 1.0, 0.0]] * 4, device="cuda", dtype=torch.float32)
    hit_geo_n = hit_n.contiguous()
    hit_count = torch.tensor([2], device="cuda", dtype=torch.int32)
    triangle_edge_count = torch.tensor([1], device="cuda", dtype=torch.int32)
    triangle_edge_indices = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    edge_pos = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    edge_dir = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    edge_n0 = torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32)
    edge_n1 = torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32)
    line_min = torch.tensor([-1.0], device="cuda", dtype=torch.float32)
    line_max = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    face1 = torch.tensor([-1], device="cuda", dtype=torch.int32)

    sliced = torch.ops.raydn.diffraction_discover_edges(
        tx_pos,
        ray_dir[:2].contiguous(),
        prim_index[:2].contiguous(),
        hit_p[:2].contiguous(),
        hit_n[:2].contiguous(),
        hit_geo_n[:2].contiguous(),
        triangle_edge_count,
        triangle_edge_indices,
        edge_pos,
        edge_dir,
        edge_n0,
        edge_n1,
        line_min,
        line_max,
        face1,
    )
    counted = torch.ops.raydn.diffraction_discover_edges_counted(
        tx_pos,
        ray_dir,
        prim_index,
        hit_p,
        hit_n,
        hit_geo_n,
        hit_count,
        triangle_edge_count,
        triangle_edge_indices,
        edge_pos,
        edge_dir,
        edge_n0,
        edge_n1,
        line_min,
        line_max,
        face1,
    )

    torch.testing.assert_close(counted, sliced)


def test_mc_face_material_tensors_expands_material_ids():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC material expansion")

    eps_r = torch.tensor([1.0, 4.0], device="cuda", dtype=torch.float32)
    sigma_e = torch.tensor([0.0, 0.2], device="cuda", dtype=torch.float32)
    mu_r = torch.tensor([1.0, 1.5], device="cuda", dtype=torch.float32)
    face_material_id = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)

    result = ops.mc_face_material_tensors(eps_r, sigma_e, mu_r, face_material_id)

    torch.testing.assert_close(result["eps_r"], torch.tensor([4.0, 1.0, 4.0], device="cuda"))
    torch.testing.assert_close(result["sigma_e"], torch.tensor([0.2, 0.0, 0.2], device="cuda"))
    torch.testing.assert_close(result["mu_r"], torch.tensor([1.5, 1.0, 1.5], device="cuda"))
    torch.testing.assert_close(result["gain"], torch.ones((3,), device="cuda"))
    assert result["valid"].dtype == torch.bool
    assert result["valid"].all()


def test_mc_face_material_tensors_requires_native_cuda_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC material expansion")

    eps_r = torch.ones((1,), device="cuda", dtype=torch.float32)
    sigma_e = torch.zeros((1,), device="cuda", dtype=torch.float32)
    mu_r = torch.ones((1,), device="cuda", dtype=torch.float32)
    face_material_id = torch.zeros((1,), device="cuda", dtype=torch.int32)
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="mc_face_material_tensors CUDA kernel is required"):
        ops.mc_face_material_tensors(eps_r, sigma_e, mu_r, face_material_id)


def test_deterministic_accumulate_flat_matches_torch_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic accumulation")

    tx_id = torch.tensor([0, 0, 0, 1], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int32)
    component_id = torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32)
    path_gain = torch.tensor([1.0, 4.0, 9.0, 16.0], device="cuda", dtype=torch.float32)
    field = torch.tensor([1.0 + 0.0j, 0.0 + 2.0j, 3.0 + 0.0j, 0.0 + 4.0j], device="cuda", dtype=torch.complex64)

    result = ops.deterministic_accumulate_flat(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field.real.contiguous(),
        field.imag.contiguous(),
        num_tx=2,
        num_rx=2,
        coherent=True,
    )

    expected_component_field = torch.zeros((3, 2, 2), device="cuda", dtype=torch.complex64)
    expected_component_power = torch.zeros((3, 2, 2), device="cuda", dtype=torch.float32)
    for index in range(int(tx_id.numel())):
        cid = int(component_id[index])
        tx = int(tx_id[index])
        rx = int(rx_id[index])
        expected_component_field[cid, tx, rx] += field[index]
    expected_component_power = expected_component_field.abs().square()
    expected_field_total = expected_component_field.sum(dim=0)
    expected_power_total = expected_field_total.abs().square()

    torch.testing.assert_close(result["component_power"], expected_component_power)
    torch.testing.assert_close(torch.complex(result["component_field_real"], result["component_field_imag"]), expected_component_field)
    torch.testing.assert_close(result["power_total"], expected_power_total)
    torch.testing.assert_close(torch.complex(result["field_total_real"], result["field_total_imag"]), expected_field_total)


def test_deterministic_accumulate_flat_incoherent_sums_power():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic accumulation")

    tx_id = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    component_id = torch.tensor([1, 1], device="cuda", dtype=torch.int32)
    path_gain = torch.tensor([4.0, 9.0], device="cuda", dtype=torch.float32)
    field_real = torch.tensor([2.0, 3.0], device="cuda", dtype=torch.float32)
    field_imag = torch.zeros((2,), device="cuda", dtype=torch.float32)

    result = ops.deterministic_accumulate_flat(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field_real,
        field_imag,
        num_tx=1,
        num_rx=1,
        coherent=False,
    )

    torch.testing.assert_close(result["component_power"][1], torch.tensor([[13.0]], device="cuda"))
    torch.testing.assert_close(result["power_total"], torch.tensor([[13.0]], device="cuda"))
    torch.testing.assert_close(result["field_total_real"], torch.sqrt(torch.tensor([[13.0]], device="cuda")))
    torch.testing.assert_close(result["field_total_imag"], torch.zeros((1, 1), device="cuda"))
