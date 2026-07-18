#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <array>
#include <utility>

namespace {

constexpr int kBlockSize = 256;
// Named component counts exported by cn_deterministic_component_counts
// (los / reflection / diffraction only).
constexpr int kComponentCount = 3;
// Slots materialized by the flat accumulator: los, reflection, diffraction,
// transmission, scattering and coupled. Scattering is an incoherent POWER slot
// (plan 05 sections 6.7.3 / 7.3): its rows fold into the totals in the power
// domain and never enter the coherent field sum; its complex cell field is
// kept as a diagnostic only. Coupled is an ordinary coherent field slot
// (ADR-011): reflection-diffraction and its reciprocal both land there and sum
// coherently in-cell, joining field_total / power_total like the first three
// slots.
constexpr int kAccumSlotCount = 6;
constexpr int kScatteringSlot = 4;
constexpr int kCoupledSlot = 5;

// Path component ids: 0=los, 1=reflection, 2=diffraction, 3/4=coupled
// reflection-diffraction and its reciprocal (ADR-011), 7=coupled double
// diffraction (ADR-013). Ids 3/4/7 all map to the single coherent coupled slot
// 5 and sum in-cell. 5=transmission, 6=scattering. Ids without a slot return -1
// and are dropped by the scatter/gather gates.
__device__ __forceinline__ int accum_slot(int component_id) {
    if (component_id >= 0 && component_id < kComponentCount) {
        return component_id;
    }
    if (component_id == 3 || component_id == 4 || component_id == 7) {
        return kCoupledSlot;
    }
    if (component_id == 5) {
        return 3;
    }
    if (component_id == 6) {
        return kScatteringSlot;
    }
    return -1;
}

void check_flat_tensor(const at::Tensor &tensor, const char *name, c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == 1, name, " must have shape (path_count,)");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Exact per-dtype intrinsics so the float32 instantiation keeps the primal
// kernel's code byte-identical while the float64 gradcheck companion shares
// the same source.
__device__ __forceinline__ float accum_sqrt(float value) { return sqrtf(value); }
__device__ __forceinline__ double accum_sqrt(double value) { return sqrt(value); }
__device__ __forceinline__ float accum_max_zero(float value) { return fmaxf(value, 0.0f); }
__device__ __forceinline__ double accum_max_zero(double value) { return fmax(value, 0.0); }

template <typename T>
__global__ void deterministic_accumulate_paths_kernel(
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int *__restrict__ component_id,
    const T *__restrict__ path_gain,
    const T *__restrict__ field_real,
    const T *__restrict__ field_imag,
    T *__restrict__ component_power,
    T *__restrict__ component_field_real,
    T *__restrict__ component_field_imag,
    int64_t path_count,
    int64_t num_tx,
    int64_t num_rx) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t cell_count = num_tx * num_rx;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        const int slot = accum_slot(component_id[idx]);
        const int tx = tx_id[idx];
        const int rx = rx_id[idx];
        if (slot < 0 || tx < 0 || rx < 0 || tx >= num_tx || rx >= num_rx) {
            continue;
        }
        const int64_t cell = static_cast<int64_t>(tx) * num_rx + rx;
        const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
        atomicAdd(component_power + out, path_gain[idx]);
        atomicAdd(component_field_real + out, field_real[idx]);
        atomicAdd(component_field_imag + out, field_imag[idx]);
    }
}

template <typename T>
__global__ void deterministic_finalize_accumulation_kernel(
    T *__restrict__ component_power,
    const T *__restrict__ component_field_real,
    const T *__restrict__ component_field_imag,
    T *__restrict__ power_total,
    T *__restrict__ field_total_real,
    T *__restrict__ field_total_imag,
    int64_t cell_count,
    int coherent,
    int scattering_coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < cell_count;
         cell += stride) {
        T real_sum = T(0);
        T imag_sum = T(0);
        T power_sum = T(0);
        T scattering_power = T(0);
        for (int slot = 0; slot < kAccumSlotCount; ++slot) {
            const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
            const T real = component_field_real[out];
            const T imag = component_field_imag[out];
            if (coherent) {
                if (slot == kScatteringSlot) {
                    if (scattering_coherent) {
                        // ADR-021 D3 (opt-in): scattering rows combine
                        // coherently. The slot already holds the summed
                        // complex path field (scattered by the paths
                        // kernel); its |sum|^2 replaces the incoherent gain
                        // sum as the scattering component power and still
                        // folds into power_total as a power term (components
                        // stay mutually incoherent, exactly the ADR-019
                        // per-component phasor precedent).
                        const T coherent_power = real * real + imag * imag;
                        component_power[out] = coherent_power;
                        scattering_power += coherent_power;
                    } else {
                        // Power-domain slot: keep the scattered gains as the
                        // component power and fold them after the field
                        // square.
                        scattering_power += component_power[out];
                    }
                    continue;
                }
                const T coherent_power = real * real + imag * imag;
                component_power[out] = coherent_power;
            } else {
                if (slot == kScatteringSlot && scattering_coherent) {
                    // ADR-021 D3 in an incoherent solve: scattering rows
                    // still interfere with each other, but the combined
                    // power adds incoherently to the other components.
                    const T coherent_power = real * real + imag * imag;
                    component_power[out] = coherent_power;
                    power_sum += coherent_power;
                } else {
                    power_sum += component_power[out];
                }
            }
            real_sum += real;
            imag_sum += imag;
        }
        if (coherent) {
            field_total_real[cell] = real_sum;
            field_total_imag[cell] = imag_sum;
            power_total[cell] =
                real_sum * real_sum + imag_sum * imag_sum + scattering_power;
        } else {
            power_total[cell] = power_sum;
            field_total_real[cell] = accum_sqrt(accum_max_zero(power_sum));
            field_total_imag[cell] = T(0);
        }
    }
}

// VJP of the flat accumulation (plan 07). Every output is either a linear
// scatter of the per-path field/power (adjoint: gather through the same
// frozen slot/tx/rx gates) or a per-cell |.|^2 / sqrt nonlinearity
// linearized at the saved forward cell values. One gather per path, no
// atomics: dropped rows write exact zeros.
template <typename T>
__global__ void deterministic_accumulate_backward_kernel(
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int *__restrict__ component_id,
    const T *__restrict__ component_field_real,
    const T *__restrict__ component_field_imag,
    const T *__restrict__ field_total_real,
    const T *__restrict__ field_total_imag,
    const T *__restrict__ power_total,
    const T *__restrict__ grad_power_total,
    const T *__restrict__ grad_field_total_real,
    const T *__restrict__ grad_field_total_imag,
    const T *__restrict__ grad_component_power,
    const T *__restrict__ grad_component_field_real,
    const T *__restrict__ grad_component_field_imag,
    T *__restrict__ grad_path_gain,
    T *__restrict__ grad_field_real,
    T *__restrict__ grad_field_imag,
    int64_t path_count,
    int64_t num_tx,
    int64_t num_rx,
    int coherent,
    int scattering_coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t cell_count = num_tx * num_rx;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        const int slot = accum_slot(component_id[idx]);
        const int tx = tx_id[idx];
        const int rx = rx_id[idx];
        if (slot < 0 || tx < 0 || rx < 0 || tx >= num_tx || rx >= num_rx) {
            grad_path_gain[idx] = T(0);
            grad_field_real[idx] = T(0);
            grad_field_imag[idx] = T(0);
            continue;
        }
        const int64_t cell = static_cast<int64_t>(tx) * num_rx + rx;
        const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
        T g_real = T(0);
        T g_imag = T(0);
        T g_gain = T(0);
        // component_field = scatter(field): plain gather in both modes.
        if (grad_component_field_real != nullptr) {
            g_real += grad_component_field_real[out];
        }
        if (grad_component_field_imag != nullptr) {
            g_imag += grad_component_field_imag[out];
        }
        if (scattering_coherent && slot == kScatteringSlot) {
            // ADR-021 D3 / ADR-022 accumulate spec: the scattering slot's
            // component_power and its contribution to power_total (and, in an
            // incoherent solve, to field_total via sqrt) are all |S|^2 with
            // S the summed complex field. d|S|^2 = 2 Re(S) dRe + 2 Im(S) dIm,
            // so every cotangent routes through the field with the pair
            // convention (grad_c = 2 grad_P S); the gain reaches no total.
            const T sr = component_field_real[out];
            const T si = component_field_imag[out];
            if (grad_component_power != nullptr) {
                const T g = grad_component_power[out];
                g_real += T(2) * sr * g;
                g_imag += T(2) * si * g;
            }
            if (grad_power_total != nullptr) {
                const T g = grad_power_total[cell];
                g_real += T(2) * sr * g;
                g_imag += T(2) * si * g;
            }
            if (!coherent && grad_field_total_real != nullptr) {
                const T total = power_total[cell];
                if (total > T(0)) {
                    const T factor =
                        grad_field_total_real[cell] / (T(2) * accum_sqrt(total));
                    g_real += T(2) * sr * factor;
                    g_imag += T(2) * si * factor;
                }
            }
        } else if (coherent && slot == kScatteringSlot) {
            // Power-domain slot inside the coherent totals:
            // component_power = scatter(gain) and power_total adds it
            // linearly after the field square; the cell field is a
            // diagnostic scatter that reaches no total.
            if (grad_component_power != nullptr) {
                g_gain += grad_component_power[out];
            }
            if (grad_power_total != nullptr) {
                g_gain += grad_power_total[cell];
            }
        } else if (coherent) {
            // component_power = |F|^2 and power_total = |sum_s F|^2 +
            // P_scatter with field_total = sum_s F over the coherent slots:
            // d|z|^2 = 2 Re(z) dRe + 2 Im(z) dIm.
            if (grad_component_power != nullptr) {
                const T g = grad_component_power[out];
                g_real += T(2) * component_field_real[out] * g;
                g_imag += T(2) * component_field_imag[out] * g;
            }
            if (grad_field_total_real != nullptr) {
                g_real += grad_field_total_real[cell];
            }
            if (grad_field_total_imag != nullptr) {
                g_imag += grad_field_total_imag[cell];
            }
            if (grad_power_total != nullptr) {
                const T g = grad_power_total[cell];
                g_real += T(2) * field_total_real[cell] * g;
                g_imag += T(2) * field_total_imag[cell] * g;
            }
        } else {
            // component_power = scatter(gain), power_total = sum_s of it and
            // field_total = sqrt(max(power_total, 0)) + 0j: the pseudo-field
            // chain is 1 / (2 sqrt(P)) with a zero subgradient at P <= 0
            // (the primal clamp gates negative sums to a constant zero).
            if (grad_component_power != nullptr) {
                g_gain += grad_component_power[out];
            }
            if (grad_power_total != nullptr) {
                g_gain += grad_power_total[cell];
            }
            if (grad_field_total_real != nullptr) {
                const T total = power_total[cell];
                if (total > T(0)) {
                    g_gain += grad_field_total_real[cell] / (T(2) * accum_sqrt(total));
                }
            }
            // field_total_imag is identically zero in incoherent mode; its
            // cotangent reaches no input.
        }
        grad_path_gain[idx] = g_gain;
        grad_field_real[idx] = g_real;
        grad_field_imag[idx] = g_imag;
    }
}

// JVP scatter: the same frozen-gate atomic scatter as the primal, with each
// tangent stream optional so absent tangents stay exact zeros. Coherent
// cells overwrite the tangent power of every field slot in the finalize, so
// their gain tangents never scatter; the power-domain scattering slot keeps
// its gain tangents in both modes.
template <typename T>
__global__ void deterministic_accumulate_tangent_scatter_kernel(
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int *__restrict__ component_id,
    const T *__restrict__ tangent_path_gain,
    const T *__restrict__ tangent_field_real,
    const T *__restrict__ tangent_field_imag,
    T *__restrict__ t_component_power,
    T *__restrict__ t_component_field_real,
    T *__restrict__ t_component_field_imag,
    int64_t path_count,
    int64_t num_tx,
    int64_t num_rx,
    int coherent,
    int scattering_coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t cell_count = num_tx * num_rx;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        const int slot = accum_slot(component_id[idx]);
        const int tx = tx_id[idx];
        const int rx = rx_id[idx];
        if (slot < 0 || tx < 0 || rx < 0 || tx >= num_tx || rx >= num_rx) {
            continue;
        }
        const int64_t cell = static_cast<int64_t>(tx) * num_rx + rx;
        const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
        // Gain tangents scatter into the power buffer only where the finalize
        // derives that slot's power from the gain sum. The scattering slot
        // does so unless the ADR-021 D3 coherent combine is active, in which
        // case its power comes from the summed field tangents instead.
        const bool scatter_gain = (slot == kScatteringSlot)
                                      ? !scattering_coherent
                                      : !coherent;
        if (tangent_path_gain != nullptr && scatter_gain) {
            atomicAdd(t_component_power + out, tangent_path_gain[idx]);
        }
        if (tangent_field_real != nullptr) {
            atomicAdd(t_component_field_real + out, tangent_field_real[idx]);
        }
        if (tangent_field_imag != nullptr) {
            atomicAdd(t_component_field_imag + out, tangent_field_imag[idx]);
        }
    }
}

// JVP finalize: push the scattered tangents through the cell nonlinearities
// linearized at the saved forward cell values. The coherent field total is
// re-summed from the component fields in the same cid order as the primal
// finalize, so the linearization point matches the forward bit for bit.
template <typename T>
__global__ void deterministic_accumulate_jvp_finalize_kernel(
    const T *__restrict__ component_field_real,
    const T *__restrict__ component_field_imag,
    const T *__restrict__ power_total,
    T *__restrict__ t_component_power,
    const T *__restrict__ t_component_field_real,
    const T *__restrict__ t_component_field_imag,
    T *__restrict__ t_power_total,
    T *__restrict__ t_field_total_real,
    T *__restrict__ t_field_total_imag,
    int64_t cell_count,
    int coherent,
    int scattering_coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < cell_count;
         cell += stride) {
        T real_sum = T(0);
        T imag_sum = T(0);
        T t_real_sum = T(0);
        T t_imag_sum = T(0);
        T t_power_sum = T(0);
        T t_scattering_power = T(0);
        for (int slot = 0; slot < kAccumSlotCount; ++slot) {
            const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
            const T t_real = t_component_field_real[out];
            const T t_imag = t_component_field_imag[out];
            if (coherent) {
                if (slot == kScatteringSlot) {
                    if (scattering_coherent) {
                        // ADR-021 D3: the scattering power is |S|^2 of the
                        // summed field, so its tangent is the linearized
                        // square 2 Re(conj(S) t_S), folded into the total as
                        // a power term (excluded from the coherent field sum).
                        const T real = component_field_real[out];
                        const T imag = component_field_imag[out];
                        const T t_power =
                            T(2) * (real * t_real + imag * t_imag);
                        t_component_power[out] = t_power;
                        t_scattering_power += t_power;
                    } else {
                        // Power-domain slot: its tangent power is the
                        // scattered gain tangents and its field tangent
                        // reaches no total.
                        t_scattering_power += t_component_power[out];
                    }
                    continue;
                }
                const T real = component_field_real[out];
                const T imag = component_field_imag[out];
                t_component_power[out] = T(2) * (real * t_real + imag * t_imag);
                real_sum += real;
                imag_sum += imag;
            } else {
                if (slot == kScatteringSlot && scattering_coherent) {
                    const T real = component_field_real[out];
                    const T imag = component_field_imag[out];
                    const T t_power = T(2) * (real * t_real + imag * t_imag);
                    t_component_power[out] = t_power;
                    t_power_sum += t_power;
                } else {
                    t_power_sum += t_component_power[out];
                }
            }
            t_real_sum += t_real;
            t_imag_sum += t_imag;
        }
        if (coherent) {
            t_field_total_real[cell] = t_real_sum;
            t_field_total_imag[cell] = t_imag_sum;
            t_power_total[cell] =
                T(2) * (real_sum * t_real_sum + imag_sum * t_imag_sum) +
                t_scattering_power;
        } else {
            t_power_total[cell] = t_power_sum;
            const T total = power_total[cell];
            t_field_total_real[cell] =
                total > T(0) ? t_power_sum / (T(2) * accum_sqrt(total)) : T(0);
            t_field_total_imag[cell] = T(0);
        }
    }
}

__global__ void deterministic_component_counts_kernel(
    const int *__restrict__ component_id,
    int64_t path_count,
    unsigned long long *__restrict__ counts) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        const int cid = component_id[idx];
        if (cid >= 0 && cid < kComponentCount) {
            atomicAdd(counts + cid, 1ULL);
        }
    }
}

__global__ void deterministic_edge_flags_kernel(
    const int *__restrict__ edge_id,
    int64_t path_count,
    int *__restrict__ flags) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        flags[idx] = edge_id[idx] >= 0 ? 1 : 0;
    }
}

__global__ void deterministic_compact_edges_kernel(
    const int *__restrict__ edge_id,
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    int64_t path_count,
    int *__restrict__ compacted) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        if (flags[idx] == 0) {
            continue;
        }
        compacted[offsets[idx]] = edge_id[idx];
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

// Tangent accumulators must start at zero; allocate raw and memset on the
// current stream (same pattern as field_transport_ad.cu) instead of ATen
// zero-fill.
at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions &options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

const at::Tensor *optional_grad(
    pybind11::object value,
    at::Tensor &storage,
    const char *name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor &reference) {
    if (value.is_none()) {
        return nullptr;
    }
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
const T *grad_ptr(const at::Tensor *tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

void check_cell_tensor(
    const at::Tensor &tensor,
    const char *name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor &reference) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.get_device() == reference.get_device(),
        name, " must share the index device");
}

void check_accumulate_indices(
    const at::Tensor &tx_id,
    const at::Tensor &rx_id,
    const at::Tensor &component_id,
    int64_t num_tx,
    int64_t num_rx) {
    check_flat_tensor(tx_id, "tx_id", at::kInt);
    check_flat_tensor(rx_id, "rx_id", at::kInt);
    check_flat_tensor(component_id, "component_id", at::kInt);
    TORCH_CHECK(rx_id.sizes() == tx_id.sizes(), "rx_id must match tx_id");
    TORCH_CHECK(component_id.sizes() == tx_id.sizes(), "component_id must match tx_id");
    TORCH_CHECK(num_tx >= 0, "num_tx must be non-negative");
    TORCH_CHECK(num_rx >= 0, "num_rx must be non-negative");
}

template <typename T>
pybind11::dict accumulate_flat_launch(
    const at::Tensor &tx_id,
    const at::Tensor &rx_id,
    const at::Tensor &component_id,
    const at::Tensor &path_gain,
    const at::Tensor &field_real,
    const at::Tensor &field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int scattering_coherent) {
    auto fopts = path_gain.options();
    at::Tensor component_power = at::empty({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor component_field_real = at::empty({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor component_field_imag = at::empty({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor power_total = at::empty({num_tx, num_rx}, fopts);
    at::Tensor field_total_real = at::empty({num_tx, num_rx}, fopts);
    at::Tensor field_total_imag = at::empty({num_tx, num_rx}, fopts);

    const int64_t path_count = tx_id.numel();
    const int64_t cell_count = num_tx * num_rx;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(path_gain.get_device()).stream();
    const int64_t component_element_count = static_cast<int64_t>(kAccumSlotCount) * cell_count;
    if (component_element_count > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            component_power.data_ptr<T>(),
            0,
            component_element_count * sizeof(T),
            stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            component_field_real.data_ptr<T>(),
            0,
            component_element_count * sizeof(T),
            stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            component_field_imag.data_ptr<T>(),
            0,
            component_element_count * sizeof(T),
            stream));
    }
    if (path_count > 0) {
        deterministic_accumulate_paths_kernel<T>
            <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                path_gain.data_ptr<T>(),
                field_real.data_ptr<T>(),
                field_imag.data_ptr<T>(),
                component_power.data_ptr<T>(),
                component_field_real.data_ptr<T>(),
                component_field_imag.data_ptr<T>(),
                path_count,
                num_tx,
                num_rx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (cell_count > 0) {
        deterministic_finalize_accumulation_kernel<T>
            <<<launch_blocks(cell_count), kBlockSize, 0, stream>>>(
                component_power.data_ptr<T>(),
                component_field_real.data_ptr<T>(),
                component_field_imag.data_ptr<T>(),
                power_total.data_ptr<T>(),
                field_total_real.data_ptr<T>(),
                field_total_imag.data_ptr<T>(),
                cell_count,
                coherent ? 1 : 0,
                scattering_coherent);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["power_total"] = power_total;
    out["field_total_real"] = field_total_real;
    out["field_total_imag"] = field_total_imag;
    out["component_power"] = component_power;
    out["component_field_real"] = component_field_real;
    out["component_field_imag"] = component_field_imag;
    return out;
}

pybind11::dict accumulate_flat_checked(
    const at::Tensor &tx_id,
    const at::Tensor &rx_id,
    const at::Tensor &component_id,
    const at::Tensor &path_gain,
    const at::Tensor &field_real,
    const at::Tensor &field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain,
    c10::ScalarType real_dtype) {
    check_accumulate_indices(tx_id, rx_id, component_id, num_tx, num_rx);
    check_flat_tensor(path_gain, "path_gain", real_dtype);
    check_flat_tensor(field_real, "field_real", real_dtype);
    check_flat_tensor(field_imag, "field_imag", real_dtype);
    TORCH_CHECK(path_gain.sizes() == tx_id.sizes(), "path_gain must match tx_id");
    TORCH_CHECK(field_real.sizes() == tx_id.sizes(), "field_real must match tx_id");
    TORCH_CHECK(field_imag.sizes() == tx_id.sizes(), "field_imag must match tx_id");
    TORCH_CHECK(
        scattering_combine_domain == 0 || scattering_combine_domain == 1,
        "scattering_combine_domain must be 0 (power) or 1 (coherent)");
    const int scattering_coherent = static_cast<int>(scattering_combine_domain);
    if (real_dtype == at::kFloat) {
        return accumulate_flat_launch<float>(
            tx_id, rx_id, component_id, path_gain, field_real, field_imag,
            num_tx, num_rx, coherent, scattering_coherent);
    }
    return accumulate_flat_launch<double>(
        tx_id, rx_id, component_id, path_gain, field_real, field_imag,
        num_tx, num_rx, coherent, scattering_coherent);
}

}  // namespace

pybind11::dict cn_deterministic_accumulate_flat(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor path_gain,
    at::Tensor field_real,
    at::Tensor field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain) {
    return accumulate_flat_checked(
        tx_id, rx_id, component_id, path_gain, field_real, field_imag,
        num_tx, num_rx, coherent, scattering_combine_domain, at::kFloat);
}

pybind11::dict cn_deterministic_accumulate_flat_fwd64(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor path_gain,
    at::Tensor field_real,
    at::Tensor field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain) {
    return accumulate_flat_checked(
        tx_id, rx_id, component_id, path_gain, field_real, field_imag,
        num_tx, num_rx, coherent, scattering_combine_domain, at::kDouble);
}

pybind11::dict cn_deterministic_accumulate_flat_backward(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor component_field_real,
    at::Tensor component_field_imag,
    at::Tensor field_total_real,
    at::Tensor field_total_imag,
    at::Tensor power_total,
    pybind11::object grad_power_total,
    pybind11::object grad_field_total_real,
    pybind11::object grad_field_total_imag,
    pybind11::object grad_component_power,
    pybind11::object grad_component_field_real,
    pybind11::object grad_component_field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain) {
    check_accumulate_indices(tx_id, rx_id, component_id, num_tx, num_rx);
    const c10::ScalarType real_dtype = component_field_real.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "deterministic_accumulate_flat_backward requires float32 or float64 cells");
    TORCH_CHECK(
        scattering_combine_domain == 0 || scattering_combine_domain == 1,
        "scattering_combine_domain must be 0 (power) or 1 (coherent)");
    const int scattering_coherent = static_cast<int>(scattering_combine_domain);
    const std::array<int64_t, 3> component_sizes{kAccumSlotCount, num_tx, num_rx};
    const std::array<int64_t, 2> total_sizes{num_tx, num_rx};
    check_cell_tensor(
        component_field_real, "component_field_real", real_dtype,
        component_sizes, tx_id);
    check_cell_tensor(
        component_field_imag, "component_field_imag", real_dtype,
        component_sizes, tx_id);
    check_cell_tensor(
        field_total_real, "field_total_real", real_dtype, total_sizes, tx_id);
    check_cell_tensor(
        field_total_imag, "field_total_imag", real_dtype, total_sizes, tx_id);
    check_cell_tensor(power_total, "power_total", real_dtype, total_sizes, tx_id);
    at::Tensor gpt_storage;
    at::Tensor gftr_storage;
    at::Tensor gfti_storage;
    at::Tensor gcp_storage;
    at::Tensor gcfr_storage;
    at::Tensor gcfi_storage;
    const at::Tensor *gpt = optional_grad(
        std::move(grad_power_total), gpt_storage, "grad_power_total",
        real_dtype, total_sizes, tx_id);
    const at::Tensor *gftr = optional_grad(
        std::move(grad_field_total_real), gftr_storage, "grad_field_total_real",
        real_dtype, total_sizes, tx_id);
    const at::Tensor *gfti = optional_grad(
        std::move(grad_field_total_imag), gfti_storage, "grad_field_total_imag",
        real_dtype, total_sizes, tx_id);
    const at::Tensor *gcp = optional_grad(
        std::move(grad_component_power), gcp_storage, "grad_component_power",
        real_dtype, component_sizes, tx_id);
    const at::Tensor *gcfr = optional_grad(
        std::move(grad_component_field_real), gcfr_storage,
        "grad_component_field_real", real_dtype, component_sizes, tx_id);
    const at::Tensor *gcfi = optional_grad(
        std::move(grad_component_field_imag), gcfi_storage,
        "grad_component_field_imag", real_dtype, component_sizes, tx_id);

    const int64_t path_count = tx_id.numel();
    auto fopts = component_field_real.options();
    at::Tensor grad_path_gain = at::empty({path_count}, fopts);
    at::Tensor grad_field_real = at::empty({path_count}, fopts);
    at::Tensor grad_field_imag = at::empty({path_count}, fopts);
    if (path_count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tx_id.get_device()).stream();
        if (real_dtype == at::kFloat) {
            deterministic_accumulate_backward_kernel<float>
                <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    component_field_real.data_ptr<float>(),
                    component_field_imag.data_ptr<float>(),
                    field_total_real.data_ptr<float>(),
                    field_total_imag.data_ptr<float>(),
                    power_total.data_ptr<float>(),
                    grad_ptr<float>(gpt),
                    grad_ptr<float>(gftr),
                    grad_ptr<float>(gfti),
                    grad_ptr<float>(gcp),
                    grad_ptr<float>(gcfr),
                    grad_ptr<float>(gcfi),
                    grad_path_gain.data_ptr<float>(),
                    grad_field_real.data_ptr<float>(),
                    grad_field_imag.data_ptr<float>(),
                    path_count,
                    num_tx,
                    num_rx,
                    coherent ? 1 : 0,
                    scattering_coherent);
        } else {
            deterministic_accumulate_backward_kernel<double>
                <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    component_field_real.data_ptr<double>(),
                    component_field_imag.data_ptr<double>(),
                    field_total_real.data_ptr<double>(),
                    field_total_imag.data_ptr<double>(),
                    power_total.data_ptr<double>(),
                    grad_ptr<double>(gpt),
                    grad_ptr<double>(gftr),
                    grad_ptr<double>(gfti),
                    grad_ptr<double>(gcp),
                    grad_ptr<double>(gcfr),
                    grad_ptr<double>(gcfi),
                    grad_path_gain.data_ptr<double>(),
                    grad_field_real.data_ptr<double>(),
                    grad_field_imag.data_ptr<double>(),
                    path_count,
                    num_tx,
                    num_rx,
                    coherent ? 1 : 0,
                    scattering_coherent);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_path_gain"] = grad_path_gain;
    out["grad_field_real"] = grad_field_real;
    out["grad_field_imag"] = grad_field_imag;
    return out;
}

pybind11::dict cn_deterministic_accumulate_flat_jvp(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor component_field_real,
    at::Tensor component_field_imag,
    at::Tensor power_total,
    pybind11::object tangent_path_gain,
    pybind11::object tangent_field_real,
    pybind11::object tangent_field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain) {
    check_accumulate_indices(tx_id, rx_id, component_id, num_tx, num_rx);
    const c10::ScalarType real_dtype = component_field_real.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "deterministic_accumulate_flat_jvp requires float32 or float64 cells");
    TORCH_CHECK(
        scattering_combine_domain == 0 || scattering_combine_domain == 1,
        "scattering_combine_domain must be 0 (power) or 1 (coherent)");
    const int scattering_coherent = static_cast<int>(scattering_combine_domain);
    const std::array<int64_t, 3> component_sizes{kAccumSlotCount, num_tx, num_rx};
    const std::array<int64_t, 2> total_sizes{num_tx, num_rx};
    const at::IntArrayRef path_sizes = tx_id.sizes();
    check_cell_tensor(
        component_field_real, "component_field_real", real_dtype,
        component_sizes, tx_id);
    check_cell_tensor(
        component_field_imag, "component_field_imag", real_dtype,
        component_sizes, tx_id);
    check_cell_tensor(power_total, "power_total", real_dtype, total_sizes, tx_id);
    at::Tensor tpg_storage;
    at::Tensor tfr_storage;
    at::Tensor tfi_storage;
    const at::Tensor *tpg = optional_grad(
        std::move(tangent_path_gain), tpg_storage, "tangent_path_gain",
        real_dtype, path_sizes, tx_id);
    const at::Tensor *tfr = optional_grad(
        std::move(tangent_field_real), tfr_storage, "tangent_field_real",
        real_dtype, path_sizes, tx_id);
    const at::Tensor *tfi = optional_grad(
        std::move(tangent_field_imag), tfi_storage, "tangent_field_imag",
        real_dtype, path_sizes, tx_id);

    auto fopts = component_field_real.options();
    at::Tensor t_component_power = zero_filled({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor t_component_field_real = zero_filled({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor t_component_field_imag = zero_filled({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor t_power_total = at::empty({num_tx, num_rx}, fopts);
    at::Tensor t_field_total_real = at::empty({num_tx, num_rx}, fopts);
    at::Tensor t_field_total_imag = at::empty({num_tx, num_rx}, fopts);

    const int64_t path_count = tx_id.numel();
    const int64_t cell_count = num_tx * num_rx;
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(tx_id.get_device()).stream();
    const bool any_tangent = tpg != nullptr || tfr != nullptr || tfi != nullptr;
    if (path_count > 0 && any_tangent) {
        if (real_dtype == at::kFloat) {
            deterministic_accumulate_tangent_scatter_kernel<float>
                <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    grad_ptr<float>(tpg),
                    grad_ptr<float>(tfr),
                    grad_ptr<float>(tfi),
                    t_component_power.data_ptr<float>(),
                    t_component_field_real.data_ptr<float>(),
                    t_component_field_imag.data_ptr<float>(),
                    path_count,
                    num_tx,
                    num_rx,
                    coherent ? 1 : 0,
                    scattering_coherent);
        } else {
            deterministic_accumulate_tangent_scatter_kernel<double>
                <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    grad_ptr<double>(tpg),
                    grad_ptr<double>(tfr),
                    grad_ptr<double>(tfi),
                    t_component_power.data_ptr<double>(),
                    t_component_field_real.data_ptr<double>(),
                    t_component_field_imag.data_ptr<double>(),
                    path_count,
                    num_tx,
                    num_rx,
                    coherent ? 1 : 0,
                    scattering_coherent);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (cell_count > 0) {
        if (real_dtype == at::kFloat) {
            deterministic_accumulate_jvp_finalize_kernel<float>
                <<<launch_blocks(cell_count), kBlockSize, 0, stream>>>(
                    component_field_real.data_ptr<float>(),
                    component_field_imag.data_ptr<float>(),
                    power_total.data_ptr<float>(),
                    t_component_power.data_ptr<float>(),
                    t_component_field_real.data_ptr<float>(),
                    t_component_field_imag.data_ptr<float>(),
                    t_power_total.data_ptr<float>(),
                    t_field_total_real.data_ptr<float>(),
                    t_field_total_imag.data_ptr<float>(),
                    cell_count,
                    coherent ? 1 : 0,
                    scattering_coherent);
        } else {
            deterministic_accumulate_jvp_finalize_kernel<double>
                <<<launch_blocks(cell_count), kBlockSize, 0, stream>>>(
                    component_field_real.data_ptr<double>(),
                    component_field_imag.data_ptr<double>(),
                    power_total.data_ptr<double>(),
                    t_component_power.data_ptr<double>(),
                    t_component_field_real.data_ptr<double>(),
                    t_component_field_imag.data_ptr<double>(),
                    t_power_total.data_ptr<double>(),
                    t_field_total_real.data_ptr<double>(),
                    t_field_total_imag.data_ptr<double>(),
                    cell_count,
                    coherent ? 1 : 0,
                    scattering_coherent);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["power_total"] = t_power_total;
    out["field_total_real"] = t_field_total_real;
    out["field_total_imag"] = t_field_total_imag;
    out["component_power"] = t_component_power;
    out["component_field_real"] = t_component_field_real;
    out["component_field_imag"] = t_component_field_imag;
    return out;
}

pybind11::dict cn_deterministic_component_counts(at::Tensor component_id) {
    check_flat_tensor(component_id, "component_id", at::kInt);

    at::Tensor counts = at::empty({kComponentCount}, component_id.options().dtype(at::kLong));
    const int64_t path_count = component_id.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(component_id.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int64_t>(), 0, kComponentCount * sizeof(int64_t), stream));
    if (path_count > 0) {
        const int block_count = static_cast<int>((path_count + kBlockSize - 1) / kBlockSize);
        deterministic_component_counts_kernel<<<block_count, kBlockSize, 0, stream>>>(
            component_id.data_ptr<int>(),
            path_count,
            reinterpret_cast<unsigned long long *>(counts.data_ptr<int64_t>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int64_t data[kComponentCount] = {0, 0, 0};
    C10_CUDA_CHECK(cudaMemcpyAsync(
        data,
        counts.data_ptr<int64_t>(),
        kComponentCount * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    pybind11::dict out;
    out["los"] = data[0];
    out["reflection"] = data[1];
    out["diffraction"] = data[2];
    return out;
}

int64_t cn_deterministic_selected_edge_count(at::Tensor edge_id) {
    check_flat_tensor(edge_id, "edge_id", at::kInt);

    const int64_t path_count = edge_id.numel();
    if (path_count == 0) {
        return 0;
    }
    auto int_options = edge_id.options().dtype(at::kInt);
    auto flags = at::empty({path_count}, int_options);
    auto offsets = at::empty({path_count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_id.get_device()).stream();
    const int block_count = static_cast<int>((path_count + kBlockSize - 1) / kBlockSize);
    deterministic_edge_flags_kernel<<<block_count, kBlockSize, 0, stream>>>(
        edge_id.data_ptr<int>(),
        path_count,
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + path_count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));

    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + path_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + path_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t selected_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
    if (selected_count == 0) {
        return 0;
    }

    auto compacted = at::empty({selected_count}, int_options);
    deterministic_compact_edges_kernel<<<block_count, kBlockSize, 0, stream>>>(
        edge_id.data_ptr<int>(),
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
        path_count,
        compacted.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto begin = thrust::device_pointer_cast(compacted.data_ptr<int>());
    auto end = begin + selected_count;
    thrust::sort(thrust::cuda::par.on(stream), begin, end);
    auto unique_end = thrust::unique(thrust::cuda::par.on(stream), begin, end);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return static_cast<int64_t>(unique_end - begin);
}
