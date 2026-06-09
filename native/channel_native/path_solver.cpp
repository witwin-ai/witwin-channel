#include <torch/extension.h>

namespace {

constexpr double kLightSpeedMetersPerSecond = 299792458.0;

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

    const auto tx_count = tx_positions.size(0);
    const auto rx_count = rx_positions.size(0);
    auto int_options = tx_positions.options().dtype(torch::kInt32);
    auto rx_id = torch::arange(rx_count, int_options).repeat_interleave(tx_count).contiguous();
    auto tx_id = torch::arange(tx_count, int_options).repeat({rx_count}).contiguous();
    auto tx_index = tx_id.to(torch::kInt64);
    auto rx_index = rx_id.to(torch::kInt64);
    auto tx_for_path = tx_positions.index_select(0, tx_index);
    auto rx_for_path = rx_positions.index_select(0, rx_index);
    auto path_length = torch::linalg_norm(tx_for_path - rx_for_path, c10::nullopt, {-1}, false, c10::nullopt)
                           .clamp_min(1.0e-6)
                           .to(torch::kFloat32)
                           .contiguous();
    auto delay = (path_length / kLightSpeedMetersPerSecond).to(torch::kFloat32).contiguous();
    const auto wavelength = kLightSpeedMetersPerSecond / frequency_hz;
    auto free_space_gain = torch::square(wavelength / (4.0 * M_PI * path_length));
    auto path_gain = (tx_power.index_select(0, tx_index) * free_space_gain).to(torch::kFloat32).contiguous();

    pybind11::dict out;
    out["tx_id"] = tx_id;
    out["rx_id"] = rx_id;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["path_gain"] = path_gain;
    return out;
}
