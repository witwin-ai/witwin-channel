#include <torch/extension.h>

#include <tuple>

namespace {

void check_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype,
    int64_t dimensions) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_path_los_export_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz);

pybind11::dict cn_path_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz) {
    check_cuda_tensor(tx_positions, "tx_positions", torch::kFloat32, 2);
    check_cuda_tensor(tx_power, "tx_power", torch::kFloat32, 1);
    check_cuda_tensor(rx_positions, "rx_positions", torch::kFloat32, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(rx_positions.size(1) == 3, "rx_positions must have shape (M, 3)");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto [tx_id, rx_id, path_length, delay, path_gain, path_gain_matrix] =
        cn_path_los_export_cuda(tx_positions, tx_power, rx_positions, frequency_hz);

    pybind11::dict out;
    out["tx_id"] = tx_id;
    out["rx_id"] = rx_id;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["path_gain"] = path_gain;
    out["path_gain_matrix"] = path_gain_matrix;
    return out;
}
