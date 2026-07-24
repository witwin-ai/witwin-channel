#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>

namespace {

constexpr int kBlockSize = 256;
constexpr float kPi = 3.14159265358979323846f;

using channel::check_tensor;
using Complex = c10::complex<float>;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

void check_same_device(const at::Tensor& tensor, const at::Tensor& reference, const char* name) {
    TORCH_CHECK(tensor.get_device() == reference.get_device(), name, " must share valid device");
}

void check_row_vector(
    const at::Tensor& tensor,
    const at::Tensor& valid,
    const char* name,
    at::ScalarType dtype) {
    check_tensor(tensor, name, dtype, 1);
    check_same_device(tensor, valid, name);
    TORCH_CHECK(tensor.size(0) == valid.size(0), name, " must match row capacity");
}

void check_row_matrix(
    const at::Tensor& tensor,
    const at::Tensor& valid,
    const char* name,
    at::ScalarType dtype,
    int64_t width) {
    check_tensor(tensor, name, dtype, 2);
    check_same_device(tensor, valid, name);
    TORCH_CHECK(
        tensor.size(0) == valid.size(0) && tensor.size(1) == width,
        name,
        " has invalid shape");
}

void check_row_sequence(
    const at::Tensor& tensor,
    const at::Tensor& valid,
    const char* name,
    at::ScalarType dtype,
    int64_t sequence_width) {
    check_tensor(tensor, name, dtype, 3);
    check_same_device(tensor, valid, name);
    TORCH_CHECK(
        tensor.size(0) == valid.size(0) && tensor.size(1) == sequence_width && tensor.size(2) == 3,
        name,
        " has invalid shape");
}

struct ForwardInput {
    const bool* valid;
    const int32_t* tx_id;
    const int32_t* rx_id;
    const int32_t* depth;
    const int32_t* component_id;
    const int32_t* primitive_id;
    const int32_t* edge_id;
    const int32_t* material_id;
    const int32_t* primitive_sequence;
    const int32_t* material_sequence;
    const float* path_length_m;
    const float* delay_s;
    const float* field_direction;
    const float* interaction_position;
    const float* interaction_normal;
    const float* interaction_positions;
    const float* interaction_normals;
    const float* path_gain;
    const Complex* path_field;
    const Complex* field_xyz;
    const Complex* coefficient;
};

struct ForwardOutput {
    bool* valid;
    int32_t* tx_id;
    int32_t* rx_id;
    int32_t* depth;
    int32_t* component_id;
    int32_t* primitive_id;
    int32_t* edge_id;
    int32_t* material_id;
    int32_t* primitive_sequence;
    int32_t* material_sequence;
    int32_t* interaction_count;
    float* phase_rad;
    float* path_length_m;
    float* delay_s;
    float* path_gain;
    float* interaction_position;
    float* interaction_normal;
    float* interaction_positions;
    float* interaction_normals;
    float* field_real;
    float* field_imag;
    Complex* coefficient;
    Complex* field_xyz;
    float* field_direction;
};

// ADR-029 bitwise lockstep duplicate of deterministic_phase_from_field_kernel.
// Keep the expression and compile flags unchanged; direct tests compare every
// valid-row phase bit. Numerical deduplication requires a separate change.
__device__ float path_table_phase_from_field(Complex value) {
    float phase = -atan2f(value.imag(), value.real());
    phase = fmodf(phase, 2.0f * kPi);
    if (phase < 0.0f) {
        phase += 2.0f * kPi;
    }
    return phase;
}

__global__ void path_table_pack_kernel(
    ForwardInput input,
    const int32_t* failure_state,
    const bool* overflow,
    bool include_fields,
    int64_t rows,
    int64_t sequence_width,
    ForwardOutput output) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        // Overflow and validity precede every discrete id and numerical read.
        const bool row_valid = failure_state[0] == 0 && !overflow[0] && input.valid[row];
        output.valid[row] = row_valid;
        if (!row_valid) {
            output.tx_id[row] = -1;
            output.rx_id[row] = -1;
            output.depth[row] = 0;
            output.component_id[row] = -1;
            output.primitive_id[row] = -1;
            output.edge_id[row] = -1;
            output.material_id[row] = -1;
            output.interaction_count[row] = 0;
            output.phase_rad[row] = 0.0f;
            output.path_length_m[row] = -1.0f;
            output.delay_s[row] = -1.0f;
            output.path_gain[row] = 0.0f;
            output.field_real[row] = 0.0f;
            output.field_imag[row] = 0.0f;
            output.coefficient[row] = Complex(0.0f, 0.0f);
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t index = row * 3 + axis;
                output.interaction_position[index] = 0.0f;
                output.interaction_normal[index] = 0.0f;
                output.field_xyz[index] = Complex(0.0f, 0.0f);
                output.field_direction[index] = 0.0f;
            }
            for (int64_t bounce = 0; bounce < sequence_width; ++bounce) {
                const int64_t scalar = row * sequence_width + bounce;
                output.primitive_sequence[scalar] = -1;
                output.material_sequence[scalar] = -1;
                for (int axis = 0; axis < 3; ++axis) {
                    const int64_t index = scalar * 3 + axis;
                    output.interaction_positions[index] = 0.0f;
                    output.interaction_normals[index] = 0.0f;
                }
            }
            continue;
        }

        output.tx_id[row] = input.tx_id[row];
        output.rx_id[row] = input.rx_id[row];
        output.depth[row] = input.depth[row];
        output.component_id[row] = input.component_id[row];
        output.primitive_id[row] = input.primitive_id[row];
        output.edge_id[row] = input.edge_id[row];
        output.material_id[row] = input.material_id[row];
        output.interaction_count[row] = input.depth[row];
        output.path_length_m[row] = input.path_length_m[row];
        output.delay_s[row] = input.delay_s[row];
        output.path_gain[row] = input.path_gain[row];
        output.coefficient[row] = input.coefficient[row];
        if (include_fields) {
            const Complex path_field = input.path_field[row];
            output.field_real[row] = path_field.real();
            output.field_imag[row] = path_field.imag();
            output.phase_rad[row] = path_table_phase_from_field(path_field);
        } else {
            output.field_real[row] = 0.0f;
            output.field_imag[row] = 0.0f;
            output.phase_rad[row] = 0.0f;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t index = row * 3 + axis;
            output.interaction_position[index] = input.interaction_position[index];
            output.interaction_normal[index] = input.interaction_normal[index];
            output.field_xyz[index] = input.field_xyz[index];
            output.field_direction[index] = input.field_direction[index];
        }
        for (int64_t bounce = 0; bounce < sequence_width; ++bounce) {
            const int64_t scalar = row * sequence_width + bounce;
            output.primitive_sequence[scalar] = input.primitive_sequence[scalar];
            output.material_sequence[scalar] = input.material_sequence[scalar];
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t index = scalar * 3 + axis;
                output.interaction_positions[index] = input.interaction_positions[index];
                output.interaction_normals[index] = input.interaction_normals[index];
            }
        }
    }
}

__global__ void path_table_count_kernel(
    const int32_t* input,
    const int32_t* failure_state,
    const bool* overflow,
    int64_t pair_count,
    int32_t* output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < pair_count) {
        output[index] = failure_state[0] != 0 || overflow[0] ? 0 : input[index];
    }
}

struct ContinuousPointers {
    const float* path_length_m;
    const float* delay_s;
    const float* field_direction;
    const float* interaction_position;
    const float* interaction_normal;
    const float* interaction_positions;
    const float* interaction_normals;
    const float* path_gain;
    const Complex* path_field;
    const Complex* field_xyz;
    const Complex* coefficient;
};

struct TablePointers {
    const float* path_length_m;
    const float* delay_s;
    const float* path_gain;
    const float* interaction_position;
    const float* interaction_normal;
    const float* interaction_positions;
    const float* interaction_normals;
    const float* field_real;
    const float* field_imag;
    const Complex* coefficient;
    const Complex* field_xyz;
    const float* field_direction;
};

struct MutableContinuousPointers {
    float* path_length_m;
    float* delay_s;
    float* field_direction;
    float* interaction_position;
    float* interaction_normal;
    float* interaction_positions;
    float* interaction_normals;
    float* path_gain;
    Complex* path_field;
    Complex* field_xyz;
    Complex* coefficient;
};

struct MutableTablePointers {
    float* path_length_m;
    float* delay_s;
    float* path_gain;
    float* interaction_position;
    float* interaction_normal;
    float* interaction_positions;
    float* interaction_normals;
    float* field_real;
    float* field_imag;
    Complex* coefficient;
    Complex* field_xyz;
    float* field_direction;
};

__device__ float load_or_zero(const float* pointer, int64_t index) {
    return pointer == nullptr ? 0.0f : pointer[index];
}

__device__ Complex load_or_zero(const Complex* pointer, int64_t index) {
    return pointer == nullptr ? Complex(0.0f, 0.0f) : pointer[index];
}

__global__ void path_table_backward_kernel(
    const bool* valid,
    bool include_fields,
    TablePointers gradients,
    int64_t rows,
    int64_t sequence_width,
    MutableContinuousPointers output) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const bool active = valid[row];
    output.path_length_m[row] = active ? load_or_zero(gradients.path_length_m, row) : 0.0f;
    output.delay_s[row] = active ? load_or_zero(gradients.delay_s, row) : 0.0f;
    output.path_gain[row] = active ? load_or_zero(gradients.path_gain, row) : 0.0f;
    output.path_field[row] = active && include_fields
        ? Complex(load_or_zero(gradients.field_real, row), load_or_zero(gradients.field_imag, row))
        : Complex(0.0f, 0.0f);
    output.coefficient[row] = active ? load_or_zero(gradients.coefficient, row) : Complex(0.0f, 0.0f);
    for (int axis = 0; axis < 3; ++axis) {
        const int64_t index = row * 3 + axis;
        output.field_direction[index] = active ? load_or_zero(gradients.field_direction, index) : 0.0f;
        output.interaction_position[index] = active ? load_or_zero(gradients.interaction_position, index) : 0.0f;
        output.interaction_normal[index] = active ? load_or_zero(gradients.interaction_normal, index) : 0.0f;
        output.field_xyz[index] = active ? load_or_zero(gradients.field_xyz, index) : Complex(0.0f, 0.0f);
    }
    for (int64_t bounce = 0; bounce < sequence_width; ++bounce) {
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t index = (row * sequence_width + bounce) * 3 + axis;
            output.interaction_positions[index] = active ? load_or_zero(gradients.interaction_positions, index) : 0.0f;
            output.interaction_normals[index] = active ? load_or_zero(gradients.interaction_normals, index) : 0.0f;
        }
    }
}

__global__ void path_table_jvp_kernel(
    const bool* valid,
    bool include_fields,
    ContinuousPointers tangents,
    int64_t rows,
    int64_t sequence_width,
    MutableTablePointers output) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const bool active = valid[row];
    output.path_length_m[row] = active ? load_or_zero(tangents.path_length_m, row) : 0.0f;
    output.delay_s[row] = active ? load_or_zero(tangents.delay_s, row) : 0.0f;
    output.path_gain[row] = active ? load_or_zero(tangents.path_gain, row) : 0.0f;
    const Complex field = active && include_fields ? load_or_zero(tangents.path_field, row) : Complex(0.0f, 0.0f);
    output.field_real[row] = field.real();
    output.field_imag[row] = field.imag();
    output.coefficient[row] = active ? load_or_zero(tangents.coefficient, row) : Complex(0.0f, 0.0f);
    for (int axis = 0; axis < 3; ++axis) {
        const int64_t index = row * 3 + axis;
        output.field_direction[index] = active ? load_or_zero(tangents.field_direction, index) : 0.0f;
        output.interaction_position[index] = active ? load_or_zero(tangents.interaction_position, index) : 0.0f;
        output.interaction_normal[index] = active ? load_or_zero(tangents.interaction_normal, index) : 0.0f;
        output.field_xyz[index] = active ? load_or_zero(tangents.field_xyz, index) : Complex(0.0f, 0.0f);
    }
    for (int64_t bounce = 0; bounce < sequence_width; ++bounce) {
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t index = (row * sequence_width + bounce) * 3 + axis;
            output.interaction_positions[index] = active ? load_or_zero(tangents.interaction_positions, index) : 0.0f;
            output.interaction_normals[index] = active ? load_or_zero(tangents.interaction_normals, index) : 0.0f;
        }
    }
}

at::Tensor optional_tensor(
    const std::optional<at::Tensor>& value,
    const at::Tensor& valid,
    const char* name,
    at::ScalarType dtype,
    at::IntArrayRef shape) {
    if (!value.has_value()) return at::Tensor();
    check_tensor(*value, name, dtype, static_cast<int64_t>(shape.size()));
    check_same_device(*value, valid, name);
    TORCH_CHECK(value->sizes() == shape, name, " has invalid shape");
    return *value;
}

const float* float_pointer(const at::Tensor& tensor) {
    return tensor.defined() ? tensor.data_ptr<float>() : nullptr;
}

const Complex* complex_pointer(const at::Tensor& tensor) {
    return tensor.defined() ? tensor.data_ptr<Complex>() : nullptr;
}

struct ContinuousOutputs {
    at::Tensor path_length_m, delay_s, field_direction, interaction_position, interaction_normal;
    at::Tensor interaction_positions, interaction_normals, path_gain, path_field, field_xyz, coefficient;
};

ContinuousOutputs allocate_continuous(const at::Tensor& valid, int64_t rows, int64_t width) {
    auto f = valid.options().dtype(at::kFloat);
    auto c = valid.options().dtype(at::kComplexFloat);
    return {at::empty({rows}, f), at::empty({rows}, f), at::empty({rows, 3}, f),
            at::empty({rows, 3}, f), at::empty({rows, 3}, f), at::empty({rows, width, 3}, f),
            at::empty({rows, width, 3}, f), at::empty({rows}, f), at::empty({rows}, c),
            at::empty({rows, 3}, c), at::empty({rows}, c)};
}

pybind11::dict continuous_dict(const ContinuousOutputs& value) {
    pybind11::dict out;
    out["path_length_m"] = value.path_length_m;
    out["delay_s"] = value.delay_s;
    out["field_direction"] = value.field_direction;
    out["interaction_position"] = value.interaction_position;
    out["interaction_normal"] = value.interaction_normal;
    out["interaction_positions"] = value.interaction_positions;
    out["interaction_normals"] = value.interaction_normals;
    out["path_gain"] = value.path_gain;
    out["path_field"] = value.path_field;
    out["field_xyz"] = value.field_xyz;
    out["coefficient"] = value.coefficient;
    return out;
}

}  // namespace

pybind11::dict channel_deterministic_path_table_capacity_pack(
    at::Tensor failure_state,
    at::Tensor valid, at::Tensor tx_id, at::Tensor rx_id, at::Tensor depth,
    at::Tensor component_id, at::Tensor primitive_id, at::Tensor edge_id,
    at::Tensor material_id, at::Tensor primitive_sequence, at::Tensor material_sequence,
    at::Tensor path_length_m, at::Tensor delay_s, at::Tensor field_direction,
    at::Tensor interaction_position, at::Tensor interaction_normal,
    at::Tensor interaction_positions, at::Tensor interaction_normals,
    at::Tensor path_gain, at::Tensor path_field, at::Tensor field_xyz, at::Tensor coefficient,
    at::Tensor num_paths, at::Tensor overflow, bool include_fields,
    int64_t pair_count, int64_t path_capacity_per_pair) {
    check_tensor(failure_state, "failure_state", at::kInt, 1);
    TORCH_CHECK(failure_state.size(0) == 1, "failure_state must have shape (1,)");
    check_tensor(valid, "valid", at::kBool, 1);
    check_same_device(failure_state, valid, "failure_state");
    TORCH_CHECK(pair_count >= 0 && path_capacity_per_pair >= 0, "capacity metadata must be non-negative");
    TORCH_CHECK(
        pair_count == 0 ||
            path_capacity_per_pair <= std::numeric_limits<int64_t>::max() / pair_count,
        "pair-major row capacity overflows int64");
    const int64_t expected_rows = pair_count * path_capacity_per_pair;
    TORCH_CHECK(valid.size(0) == expected_rows, "valid must match pair-major row capacity");
    for (auto item : std::initializer_list<std::pair<at::Tensor*, const char*>>{
             {&tx_id, "tx_id"}, {&rx_id, "rx_id"}, {&depth, "depth"},
             {&component_id, "component_id"}, {&primitive_id, "primitive_id"},
             {&edge_id, "edge_id"}, {&material_id, "material_id"}}) {
        check_row_vector(*item.first, valid, item.second, at::kInt);
    }
    check_tensor(primitive_sequence, "primitive_sequence", at::kInt, 2);
    check_same_device(primitive_sequence, valid, "primitive_sequence");
    TORCH_CHECK(primitive_sequence.size(0) == valid.size(0), "primitive_sequence must match rows");
    const int64_t width = primitive_sequence.size(1);
    check_row_matrix(material_sequence, valid, "material_sequence", at::kInt, width);
    check_row_vector(path_length_m, valid, "path_length_m", at::kFloat);
    check_row_vector(delay_s, valid, "delay_s", at::kFloat);
    check_row_matrix(field_direction, valid, "field_direction", at::kFloat, 3);
    check_row_matrix(interaction_position, valid, "interaction_position", at::kFloat, 3);
    check_row_matrix(interaction_normal, valid, "interaction_normal", at::kFloat, 3);
    check_row_sequence(interaction_positions, valid, "interaction_positions", at::kFloat, width);
    check_row_sequence(interaction_normals, valid, "interaction_normals", at::kFloat, width);
    check_row_vector(path_gain, valid, "path_gain", at::kFloat);
    check_row_vector(path_field, valid, "path_field", at::kComplexFloat);
    check_row_matrix(field_xyz, valid, "field_xyz", at::kComplexFloat, 3);
    check_row_vector(coefficient, valid, "coefficient", at::kComplexFloat);
    check_tensor(num_paths, "num_paths", at::kInt, 1);
    check_same_device(num_paths, valid, "num_paths");
    TORCH_CHECK(num_paths.size(0) == pair_count, "num_paths must match pair_count");
    check_tensor(overflow, "overflow", at::kBool, 1);
    check_same_device(overflow, valid, "overflow");
    TORCH_CHECK(overflow.size(0) == 1, "overflow must have shape (1,)");

    const int64_t rows = valid.size(0);
    auto b = valid.options().dtype(at::kBool);
    auto i = valid.options().dtype(at::kInt);
    auto f = valid.options().dtype(at::kFloat);
    auto c = valid.options().dtype(at::kComplexFloat);
    auto out_valid = at::empty({rows}, b);
    auto out_num_paths = at::empty({pair_count}, i);
    auto out_tx = at::empty({rows}, i); auto out_rx = at::empty({rows}, i);
    auto out_depth = at::empty({rows}, i); auto out_component = at::empty({rows}, i);
    auto out_primitive = at::empty({rows}, i); auto out_edge = at::empty({rows}, i);
    auto out_material = at::empty({rows}, i); auto out_primitive_seq = at::empty({rows, width}, i);
    auto out_material_seq = at::empty({rows, width}, i); auto out_count = at::empty({rows}, i);
    auto out_phase = at::empty({rows}, f); auto out_length = at::empty({rows}, f);
    auto out_delay = at::empty({rows}, f); auto out_gain = at::empty({rows}, f);
    auto out_position = at::empty({rows, 3}, f); auto out_normal = at::empty({rows, 3}, f);
    auto out_positions = at::empty({rows, width, 3}, f); auto out_normals = at::empty({rows, width, 3}, f);
    auto out_re = at::empty({rows}, f); auto out_im = at::empty({rows}, f);
    auto out_coefficient = at::empty({rows}, c); auto out_xyz = at::empty({rows, 3}, c);
    auto out_direction = at::empty({rows, 3}, f);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    if (rows > 0) {
        path_table_pack_kernel<<<launch_blocks(rows), kBlockSize, 0, stream>>>(
            {valid.data_ptr<bool>(), tx_id.data_ptr<int32_t>(), rx_id.data_ptr<int32_t>(),
             depth.data_ptr<int32_t>(), component_id.data_ptr<int32_t>(), primitive_id.data_ptr<int32_t>(),
             edge_id.data_ptr<int32_t>(), material_id.data_ptr<int32_t>(), primitive_sequence.data_ptr<int32_t>(),
             material_sequence.data_ptr<int32_t>(), path_length_m.data_ptr<float>(), delay_s.data_ptr<float>(),
             field_direction.data_ptr<float>(), interaction_position.data_ptr<float>(), interaction_normal.data_ptr<float>(),
             interaction_positions.data_ptr<float>(), interaction_normals.data_ptr<float>(), path_gain.data_ptr<float>(),
             path_field.data_ptr<Complex>(), field_xyz.data_ptr<Complex>(), coefficient.data_ptr<Complex>()},
            failure_state.data_ptr<int32_t>(), overflow.data_ptr<bool>(), include_fields, rows, width,
            {out_valid.data_ptr<bool>(), out_tx.data_ptr<int32_t>(), out_rx.data_ptr<int32_t>(),
             out_depth.data_ptr<int32_t>(), out_component.data_ptr<int32_t>(), out_primitive.data_ptr<int32_t>(),
             out_edge.data_ptr<int32_t>(), out_material.data_ptr<int32_t>(), out_primitive_seq.data_ptr<int32_t>(),
             out_material_seq.data_ptr<int32_t>(), out_count.data_ptr<int32_t>(), out_phase.data_ptr<float>(),
             out_length.data_ptr<float>(), out_delay.data_ptr<float>(), out_gain.data_ptr<float>(),
             out_position.data_ptr<float>(), out_normal.data_ptr<float>(), out_positions.data_ptr<float>(),
             out_normals.data_ptr<float>(), out_re.data_ptr<float>(), out_im.data_ptr<float>(),
             out_coefficient.data_ptr<Complex>(), out_xyz.data_ptr<Complex>(), out_direction.data_ptr<float>()});
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (pair_count > 0) {
        path_table_count_kernel<<<launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            num_paths.data_ptr<int32_t>(), failure_state.data_ptr<int32_t>(), overflow.data_ptr<bool>(),
            pair_count, out_num_paths.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["valid"] = out_valid; out["num_paths"] = out_num_paths;
    out["tx_id"] = out_tx; out["rx_id"] = out_rx; out["depth"] = out_depth;
    out["component_id"] = out_component; out["primitive_id"] = out_primitive; out["edge_id"] = out_edge;
    out["material_id"] = out_material; out["primitive_sequence"] = out_primitive_seq;
    out["material_sequence"] = out_material_seq; out["interaction_count"] = out_count;
    out["phase_rad"] = out_phase; out["path_length_m"] = out_length; out["delay_s"] = out_delay;
    out["path_gain"] = out_gain; out["interaction_position"] = out_position;
    out["interaction_normal"] = out_normal; out["interaction_positions"] = out_positions;
    out["interaction_normals"] = out_normals; out["field_real"] = out_re; out["field_imag"] = out_im;
    out["coefficient"] = out_coefficient; out["field_xyz"] = out_xyz; out["field_direction"] = out_direction;
    return out;
}

pybind11::dict channel_deterministic_path_table_capacity_pack_backward(
    at::Tensor valid, bool include_fields,
    std::optional<at::Tensor> grad_path_length_m, std::optional<at::Tensor> grad_delay_s,
    std::optional<at::Tensor> grad_path_gain, std::optional<at::Tensor> grad_interaction_position,
    std::optional<at::Tensor> grad_interaction_normal, std::optional<at::Tensor> grad_interaction_positions,
    std::optional<at::Tensor> grad_interaction_normals, std::optional<at::Tensor> grad_field_real,
    std::optional<at::Tensor> grad_field_imag, std::optional<at::Tensor> grad_coefficient,
    std::optional<at::Tensor> grad_field_xyz, std::optional<at::Tensor> grad_field_direction,
    int64_t sequence_width) {
    check_tensor(valid, "valid", at::kBool, 1);
    const int64_t rows = valid.size(0); TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    const auto r = at::IntArrayRef(&rows, 1); const int64_t shape3_data[] = {rows, 3};
    const int64_t shape_seq_data[] = {rows, sequence_width, 3};
    auto length = optional_tensor(grad_path_length_m, valid, "grad_path_length_m", at::kFloat, r);
    auto delay = optional_tensor(grad_delay_s, valid, "grad_delay_s", at::kFloat, r);
    auto gain = optional_tensor(grad_path_gain, valid, "grad_path_gain", at::kFloat, r);
    auto position = optional_tensor(grad_interaction_position, valid, "grad_interaction_position", at::kFloat, shape3_data);
    auto normal = optional_tensor(grad_interaction_normal, valid, "grad_interaction_normal", at::kFloat, shape3_data);
    auto positions = optional_tensor(grad_interaction_positions, valid, "grad_interaction_positions", at::kFloat, shape_seq_data);
    auto normals = optional_tensor(grad_interaction_normals, valid, "grad_interaction_normals", at::kFloat, shape_seq_data);
    auto re = optional_tensor(grad_field_real, valid, "grad_field_real", at::kFloat, r);
    auto im = optional_tensor(grad_field_imag, valid, "grad_field_imag", at::kFloat, r);
    auto coefficient = optional_tensor(grad_coefficient, valid, "grad_coefficient", at::kComplexFloat, r);
    auto xyz = optional_tensor(grad_field_xyz, valid, "grad_field_xyz", at::kComplexFloat, shape3_data);
    auto direction = optional_tensor(grad_field_direction, valid, "grad_field_direction", at::kFloat, shape3_data);
    auto out = allocate_continuous(valid, rows, sequence_width);
    if (rows > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
        path_table_backward_kernel<<<launch_blocks(rows), kBlockSize, 0, stream>>>(
            valid.data_ptr<bool>(), include_fields,
            {float_pointer(length), float_pointer(delay), float_pointer(gain), float_pointer(position),
             float_pointer(normal), float_pointer(positions), float_pointer(normals), float_pointer(re),
             float_pointer(im), complex_pointer(coefficient), complex_pointer(xyz), float_pointer(direction)},
            rows, sequence_width,
            {out.path_length_m.data_ptr<float>(), out.delay_s.data_ptr<float>(), out.field_direction.data_ptr<float>(),
             out.interaction_position.data_ptr<float>(), out.interaction_normal.data_ptr<float>(),
             out.interaction_positions.data_ptr<float>(), out.interaction_normals.data_ptr<float>(),
             out.path_gain.data_ptr<float>(), out.path_field.data_ptr<Complex>(), out.field_xyz.data_ptr<Complex>(),
             out.coefficient.data_ptr<Complex>()});
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return continuous_dict(out);
}

pybind11::dict channel_deterministic_path_table_capacity_pack_jvp(
    at::Tensor valid, bool include_fields,
    std::optional<at::Tensor> tangent_path_length_m, std::optional<at::Tensor> tangent_delay_s,
    std::optional<at::Tensor> tangent_field_direction, std::optional<at::Tensor> tangent_interaction_position,
    std::optional<at::Tensor> tangent_interaction_normal, std::optional<at::Tensor> tangent_interaction_positions,
    std::optional<at::Tensor> tangent_interaction_normals, std::optional<at::Tensor> tangent_path_gain,
    std::optional<at::Tensor> tangent_path_field, std::optional<at::Tensor> tangent_field_xyz,
    std::optional<at::Tensor> tangent_coefficient, int64_t sequence_width) {
    check_tensor(valid, "valid", at::kBool, 1);
    const int64_t rows = valid.size(0); TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    const auto r = at::IntArrayRef(&rows, 1); const int64_t shape3_data[] = {rows, 3};
    const int64_t shape_seq_data[] = {rows, sequence_width, 3};
    auto length = optional_tensor(tangent_path_length_m, valid, "tangent_path_length_m", at::kFloat, r);
    auto delay = optional_tensor(tangent_delay_s, valid, "tangent_delay_s", at::kFloat, r);
    auto direction = optional_tensor(tangent_field_direction, valid, "tangent_field_direction", at::kFloat, shape3_data);
    auto position = optional_tensor(tangent_interaction_position, valid, "tangent_interaction_position", at::kFloat, shape3_data);
    auto normal = optional_tensor(tangent_interaction_normal, valid, "tangent_interaction_normal", at::kFloat, shape3_data);
    auto positions = optional_tensor(tangent_interaction_positions, valid, "tangent_interaction_positions", at::kFloat, shape_seq_data);
    auto normals = optional_tensor(tangent_interaction_normals, valid, "tangent_interaction_normals", at::kFloat, shape_seq_data);
    auto gain = optional_tensor(tangent_path_gain, valid, "tangent_path_gain", at::kFloat, r);
    auto field = optional_tensor(tangent_path_field, valid, "tangent_path_field", at::kComplexFloat, r);
    auto xyz = optional_tensor(tangent_field_xyz, valid, "tangent_field_xyz", at::kComplexFloat, shape3_data);
    auto coefficient = optional_tensor(tangent_coefficient, valid, "tangent_coefficient", at::kComplexFloat, r);
    auto f = valid.options().dtype(at::kFloat); auto c = valid.options().dtype(at::kComplexFloat);
    auto out_length = at::empty({rows}, f); auto out_delay = at::empty({rows}, f); auto out_gain = at::empty({rows}, f);
    auto out_position = at::empty({rows, 3}, f); auto out_normal = at::empty({rows, 3}, f);
    auto out_positions = at::empty({rows, sequence_width, 3}, f); auto out_normals = at::empty({rows, sequence_width, 3}, f);
    auto out_re = at::empty({rows}, f); auto out_im = at::empty({rows}, f); auto out_coefficient = at::empty({rows}, c);
    auto out_xyz = at::empty({rows, 3}, c); auto out_direction = at::empty({rows, 3}, f);
    if (rows > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
        path_table_jvp_kernel<<<launch_blocks(rows), kBlockSize, 0, stream>>>(
            valid.data_ptr<bool>(), include_fields,
            {float_pointer(length), float_pointer(delay), float_pointer(direction), float_pointer(position),
             float_pointer(normal), float_pointer(positions), float_pointer(normals), float_pointer(gain),
             complex_pointer(field), complex_pointer(xyz), complex_pointer(coefficient)},
            rows, sequence_width,
            {out_length.data_ptr<float>(), out_delay.data_ptr<float>(), out_gain.data_ptr<float>(),
             out_position.data_ptr<float>(), out_normal.data_ptr<float>(), out_positions.data_ptr<float>(),
             out_normals.data_ptr<float>(), out_re.data_ptr<float>(), out_im.data_ptr<float>(),
             out_coefficient.data_ptr<Complex>(), out_xyz.data_ptr<Complex>(), out_direction.data_ptr<float>()});
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["path_length_m"] = out_length; out["delay_s"] = out_delay; out["path_gain"] = out_gain;
    out["interaction_position"] = out_position; out["interaction_normal"] = out_normal;
    out["interaction_positions"] = out_positions; out["interaction_normals"] = out_normals;
    out["field_real"] = out_re; out["field_imag"] = out_im; out["coefficient"] = out_coefficient;
    out["field_xyz"] = out_xyz; out["field_direction"] = out_direction;
    return out;
}
