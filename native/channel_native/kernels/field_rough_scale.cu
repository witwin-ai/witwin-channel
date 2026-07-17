// ADR-010 op 3: native rough-surface coherent attenuation C_r and its
// application onto the reflection field outputs.
//
// C_r = prod_b att_b with att_b = exp(-2*(k0*cos_b*sigma_b)^2) on rough
// bounces (else 1), cos_b = |dot(seg_dir_b, n_b)|, seg_dir_b the unit
// direction of the incoming segment (pos_b - prev_b, prev_0 = source). The
// factor is real, so the four reflection outputs scale by C_r (path_gain by
// C_r^2). Rows flagged ``replaced`` (a realization phase screen replaces the
// delta specular) are zeroed. One forward launch scales all four outputs; the
// backward/jvp companions differentiate frequency and the hit geometry
// (positions, normals, source), matching the input set the previous Torch
// implementation reached under the fixed-topology contract.
//
// Elementwise over rows; per-row bounce loop (depth <= 5). No float atomics
// except the scalar frequency-gradient reduction (same convention as the
// other field backward kernels).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../tensor_checks.h"

namespace {

constexpr int kBlockSize = 128;
constexpr int kMaxDepth = 5;
constexpr double kPi = 3.14159265358979323846;
constexpr double kC0 = 299792458.0;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
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

const at::Tensor* optional_arg(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none())
        return nullptr;
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
const T* opt_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

using cfloat = c10::complex<float>;

__device__ __forceinline__ cfloat cscale(cfloat value, float s) {
    return cfloat(value.real() * s, value.imag() * s);
}

// Per-bounce recomputation shared by forward, backward and jvp. Fills the
// bounce arrays and returns the product factor (before the replaced mask).
__device__ __forceinline__ float rough_bounces(
    int64_t row, int depth, float k0,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ source,
    const float* __restrict__ sigma_b,
    const bool* __restrict__ rough_b,
    float seg_dir[kMaxDepth][3],
    float normal[kMaxDepth][3],
    float sign[kMaxDepth],
    float cos_b[kMaxDepth],
    float inv_len[kMaxDepth],
    bool rough[kMaxDepth],
    float sigma[kMaxDepth]) {
    float factor = 1.0f;
    for (int b = 0; b < depth; ++b) {
        const int64_t pb = (row * depth + b) * 3;
        float sx, sy, sz;
        if (b == 0) {
            sx = positions[pb + 0] - source[row * 3 + 0];
            sy = positions[pb + 1] - source[row * 3 + 1];
            sz = positions[pb + 2] - source[row * 3 + 2];
        } else {
            const int64_t prev = (row * depth + b - 1) * 3;
            sx = positions[pb + 0] - positions[prev + 0];
            sy = positions[pb + 1] - positions[prev + 1];
            sz = positions[pb + 2] - positions[prev + 2];
        }
        const float len = sqrtf(sx * sx + sy * sy + sz * sz);
        // Match Torch's ``seg / norm.clamp_min(1e-9)`` (division, not a
        // reciprocal multiply) so the coherent factor is float-faithful.
        const float denom = fmaxf(len, 1.0e-9f);
        const float dx = sx / denom, dy = sy / denom, dz = sz / denom;
        const float inv = 1.0f / denom;
        const float nx = normals[pb + 0];
        const float ny = normals[pb + 1];
        const float nz = normals[pb + 2];
        // Torch evaluates (seg_dir*normal).sum(-1) as three rounded products
        // then a left-associated sum; keep the products in registers so the
        // compiler cannot fuse them into an fma with different rounding.
        const float p0 = __fmul_rn(dx, nx);
        const float p1 = __fmul_rn(dy, ny);
        const float p2 = __fmul_rn(dz, nz);
        const float dot = __fadd_rn(__fadd_rn(p0, p1), p2);
        const float cb = fabsf(dot);
        seg_dir[b][0] = dx; seg_dir[b][1] = dy; seg_dir[b][2] = dz;
        normal[b][0] = nx; normal[b][1] = ny; normal[b][2] = nz;
        sign[b] = dot > 0.0f ? 1.0f : (dot < 0.0f ? -1.0f : 0.0f);
        cos_b[b] = cb;
        inv_len[b] = inv;
        const bool r = rough_b[row * depth + b];
        rough[b] = r;
        const float s = sigma_b[row * depth + b];
        sigma[b] = s;
        if (r) {
            // Match the Torch association exp(-2 * (k0*cos*sigma).square()):
            // square first, then scale by -2.
            const float u = k0 * cb * s;
            factor *= expf(-2.0f * (u * u));
        }
    }
    return factor;
}

__global__ void rough_scale_forward_kernel(
    int64_t count, int depth, float k0,
    const cfloat* __restrict__ field_vector,
    const cfloat* __restrict__ coefficient,
    const cfloat* __restrict__ path_field,
    const float* __restrict__ path_gain,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ source,
    const float* __restrict__ sigma_b,
    const bool* __restrict__ rough_b,
    const bool* __restrict__ replaced,
    cfloat* __restrict__ out_field_vector,
    cfloat* __restrict__ out_coefficient,
    cfloat* __restrict__ out_path_field,
    float* __restrict__ out_path_gain,
    float* __restrict__ out_factor) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float seg_dir[kMaxDepth][3], normal[kMaxDepth][3];
        float sign[kMaxDepth], cos_b[kMaxDepth], inv_len[kMaxDepth], sigma[kMaxDepth];
        bool rough[kMaxDepth];
        float factor = rough_bounces(
            row, depth, k0, positions, normals, source, sigma_b, rough_b,
            seg_dir, normal, sign, cos_b, inv_len, rough, sigma);
        if (replaced[row]) factor = 0.0f;
        out_factor[row] = factor;
        for (int c = 0; c < 3; ++c)
            out_field_vector[row * 3 + c] = cscale(field_vector[row * 3 + c], factor);
        out_coefficient[row] = cscale(coefficient[row], factor);
        out_path_field[row] = cscale(path_field[row], factor);
        out_path_gain[row] = path_gain[row] * factor * factor;
    }
}

__global__ void rough_scale_backward_kernel(
    int64_t count, int depth, float k0, float dk0_df,
    const cfloat* __restrict__ field_vector,
    const cfloat* __restrict__ coefficient,
    const cfloat* __restrict__ path_field,
    const float* __restrict__ path_gain,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ source,
    const float* __restrict__ sigma_b,
    const bool* __restrict__ rough_b,
    const bool* __restrict__ replaced,
    const cfloat* __restrict__ grad_field_vector,
    const cfloat* __restrict__ grad_coefficient,
    const cfloat* __restrict__ grad_path_field,
    const float* __restrict__ grad_path_gain,
    cfloat* __restrict__ out_grad_field_vector,
    cfloat* __restrict__ out_grad_coefficient,
    cfloat* __restrict__ out_grad_path_field,
    float* __restrict__ out_grad_path_gain,
    float* __restrict__ out_grad_positions,
    float* __restrict__ out_grad_normals,
    float* __restrict__ out_grad_source,
    float* __restrict__ out_grad_frequency,
    bool need_field, bool need_geometry, bool need_frequency) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float seg_dir[kMaxDepth][3], normal[kMaxDepth][3];
        float sign[kMaxDepth], cos_b[kMaxDepth], inv_len[kMaxDepth], sigma[kMaxDepth];
        bool rough[kMaxDepth];
        float factor = rough_bounces(
            row, depth, k0, positions, normals, source, sigma_b, rough_b,
            seg_dir, normal, sign, cos_b, inv_len, rough, sigma);
        const bool rep = replaced[row];
        if (rep) factor = 0.0f;

        // grad_factor = sum_outputs Re(conj(cotangent) * primal); path_gain
        // (out = pg*factor^2) adds cotangent * 2 * factor * pg.
        float grad_factor = 0.0f;
        for (int c = 0; c < 3; ++c) {
            const cfloat in = field_vector[row * 3 + c];
            const cfloat g = grad_field_vector != nullptr ? grad_field_vector[row * 3 + c]
                                                          : cfloat(0.0f, 0.0f);
            grad_factor += g.real() * in.real() + g.imag() * in.imag();
            if (need_field)
                out_grad_field_vector[row * 3 + c] = cscale(g, factor);
        }
        {
            const cfloat in = coefficient[row];
            const cfloat g = grad_coefficient != nullptr ? grad_coefficient[row]
                                                         : cfloat(0.0f, 0.0f);
            grad_factor += g.real() * in.real() + g.imag() * in.imag();
            if (need_field) out_grad_coefficient[row] = cscale(g, factor);
        }
        {
            const cfloat in = path_field[row];
            const cfloat g = grad_path_field != nullptr ? grad_path_field[row]
                                                        : cfloat(0.0f, 0.0f);
            grad_factor += g.real() * in.real() + g.imag() * in.imag();
            if (need_field) out_grad_path_field[row] = cscale(g, factor);
        }
        {
            const float in = path_gain[row];
            const float g = grad_path_gain != nullptr ? grad_path_gain[row] : 0.0f;
            grad_factor += g * 2.0f * factor * in;
            if (need_field) out_grad_path_gain[row] = g * factor * factor;
        }

        if (need_geometry) {
            float gpos[kMaxDepth][3];
            for (int b = 0; b < depth; ++b) {
                gpos[b][0] = gpos[b][1] = gpos[b][2] = 0.0f;
            }
            float gsrc[3] = {0.0f, 0.0f, 0.0f};
            for (int b = 0; b < depth; ++b) {
                float gn0 = 0.0f, gn1 = 0.0f, gn2 = 0.0f;
                if (!rep && rough[b]) {
                    // A_b = grad_factor * d(factor)/d(cos_b)
                    //     = grad_factor * factor * (-4 k0^2 sigma_b^2 cos_b).
                    const float A = grad_factor * factor *
                        (-4.0f * k0 * k0 * sigma[b] * sigma[b] * cos_b[b]);
                    const float s = sign[b];
                    // grad_normal_b = A * sign * seg_dir_b.
                    gn0 = A * s * seg_dir[b][0];
                    gn1 = A * s * seg_dir[b][1];
                    gn2 = A * s * seg_dir[b][2];
                    // d(cos)/d(seg) = sign*(n - dot*seg_dir)*inv (len>clamp) or
                    // sign*n*inv when the norm was clamped.
                    const float dot = cos_b[b] * s;  // signed dot
                    float dsx, dsy, dsz;
                    if (inv_len[b] < (1.0f / 1.0e-9f)) {
                        dsx = s * (normal[b][0] - dot * seg_dir[b][0]) * inv_len[b];
                        dsy = s * (normal[b][1] - dot * seg_dir[b][1]) * inv_len[b];
                        dsz = s * (normal[b][2] - dot * seg_dir[b][2]) * inv_len[b];
                    } else {
                        dsx = s * normal[b][0] * inv_len[b];
                        dsy = s * normal[b][1] * inv_len[b];
                        dsz = s * normal[b][2] * inv_len[b];
                    }
                    const float gsx = A * dsx, gsy = A * dsy, gsz = A * dsz;
                    gpos[b][0] += gsx; gpos[b][1] += gsy; gpos[b][2] += gsz;
                    if (b == 0) {
                        gsrc[0] -= gsx; gsrc[1] -= gsy; gsrc[2] -= gsz;
                    } else {
                        gpos[b - 1][0] -= gsx; gpos[b - 1][1] -= gsy; gpos[b - 1][2] -= gsz;
                    }
                }
                const int64_t nb = (row * depth + b) * 3;
                out_grad_normals[nb + 0] = gn0;
                out_grad_normals[nb + 1] = gn1;
                out_grad_normals[nb + 2] = gn2;
            }
            for (int b = 0; b < depth; ++b) {
                const int64_t pb = (row * depth + b) * 3;
                out_grad_positions[pb + 0] = gpos[b][0];
                out_grad_positions[pb + 1] = gpos[b][1];
                out_grad_positions[pb + 2] = gpos[b][2];
            }
            out_grad_source[row * 3 + 0] = gsrc[0];
            out_grad_source[row * 3 + 1] = gsrc[1];
            out_grad_source[row * 3 + 2] = gsrc[2];
        }

        if (need_frequency && !rep) {
            float df = 0.0f;
            for (int b = 0; b < depth; ++b) {
                if (rough[b]) {
                    // d(factor)/d(k0) contribution of bounce b, times dk0/df.
                    df += factor *
                        (-4.0f * k0 * cos_b[b] * cos_b[b] * sigma[b] * sigma[b]) * dk0_df;
                }
            }
            atomicAdd(out_grad_frequency, grad_factor * df);
        }
    }
}

__global__ void rough_scale_jvp_kernel(
    int64_t count, int depth, float k0, float dk0_df,
    const cfloat* __restrict__ field_vector,
    const cfloat* __restrict__ coefficient,
    const cfloat* __restrict__ path_field,
    const float* __restrict__ path_gain,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ source,
    const float* __restrict__ sigma_b,
    const bool* __restrict__ rough_b,
    const bool* __restrict__ replaced,
    const cfloat* __restrict__ t_field_vector,
    const cfloat* __restrict__ t_coefficient,
    const cfloat* __restrict__ t_path_field,
    const float* __restrict__ t_path_gain,
    const float* __restrict__ t_positions,
    const float* __restrict__ t_normals,
    const float* __restrict__ t_source,
    float t_frequency,
    cfloat* __restrict__ out_t_field_vector,
    cfloat* __restrict__ out_t_coefficient,
    cfloat* __restrict__ out_t_path_field,
    float* __restrict__ out_t_path_gain) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float seg_dir[kMaxDepth][3], normal[kMaxDepth][3];
        float sign[kMaxDepth], cos_b[kMaxDepth], inv_len[kMaxDepth], sigma[kMaxDepth];
        bool rough[kMaxDepth];
        float factor = rough_bounces(
            row, depth, k0, positions, normals, source, sigma_b, rough_b,
            seg_dir, normal, sign, cos_b, inv_len, rough, sigma);
        const bool rep = replaced[row];
        float dfactor = 0.0f;
        if (!rep) {
            for (int b = 0; b < depth; ++b) {
                if (!rough[b]) continue;
                const float s = sign[b];
                const float dot = cos_b[b] * s;
                // d(cos_b) along geometry tangents.
                float tsegx = 0.0f, tsegy = 0.0f, tsegz = 0.0f;
                const int64_t pb = (row * depth + b) * 3;
                if (t_positions != nullptr) {
                    tsegx = t_positions[pb + 0];
                    tsegy = t_positions[pb + 1];
                    tsegz = t_positions[pb + 2];
                }
                if (b == 0) {
                    if (t_source != nullptr) {
                        tsegx -= t_source[row * 3 + 0];
                        tsegy -= t_source[row * 3 + 1];
                        tsegz -= t_source[row * 3 + 2];
                    }
                } else if (t_positions != nullptr) {
                    const int64_t prev = (row * depth + b - 1) * 3;
                    tsegx -= t_positions[prev + 0];
                    tsegy -= t_positions[prev + 1];
                    tsegz -= t_positions[prev + 2];
                }
                float dcos = 0.0f;
                if (inv_len[b] < (1.0f / 1.0e-9f)) {
                    dcos = s * (
                        (normal[b][0] - dot * seg_dir[b][0]) * tsegx +
                        (normal[b][1] - dot * seg_dir[b][1]) * tsegy +
                        (normal[b][2] - dot * seg_dir[b][2]) * tsegz) * inv_len[b];
                } else {
                    dcos = s * (normal[b][0] * tsegx + normal[b][1] * tsegy +
                                normal[b][2] * tsegz) * inv_len[b];
                }
                if (t_normals != nullptr) {
                    dcos += s * (seg_dir[b][0] * t_normals[pb + 0] +
                                 seg_dir[b][1] * t_normals[pb + 1] +
                                 seg_dir[b][2] * t_normals[pb + 2]);
                }
                // d(factor)/d(cos_b) * dcos.
                dfactor += factor *
                    (-4.0f * k0 * k0 * sigma[b] * sigma[b] * cos_b[b]) * dcos;
                // d(factor)/d(f) * t_frequency.
                dfactor += factor *
                    (-4.0f * k0 * cos_b[b] * cos_b[b] * sigma[b] * sigma[b]) *
                    dk0_df * t_frequency;
            }
        } else {
            factor = 0.0f;
        }
        for (int c = 0; c < 3; ++c) {
            const cfloat in = field_vector[row * 3 + c];
            cfloat t = cscale(in, dfactor);
            if (t_field_vector != nullptr)
                t += cscale(t_field_vector[row * 3 + c], factor);
            out_t_field_vector[row * 3 + c] = t;
        }
        {
            const cfloat in = coefficient[row];
            cfloat t = cscale(in, dfactor);
            if (t_coefficient != nullptr) t += cscale(t_coefficient[row], factor);
            out_t_coefficient[row] = t;
        }
        {
            const cfloat in = path_field[row];
            cfloat t = cscale(in, dfactor);
            if (t_path_field != nullptr) t += cscale(t_path_field[row], factor);
            out_t_path_field[row] = t;
        }
        {
            const float in = path_gain[row];
            float t = in * 2.0f * factor * dfactor;
            if (t_path_gain != nullptr) t += t_path_gain[row] * factor * factor;
            out_t_path_gain[row] = t;
        }
    }
}

void check_inputs(
    const at::Tensor& field_vector,
    const at::Tensor& coefficient,
    const at::Tensor& path_field,
    const at::Tensor& path_gain,
    const at::Tensor& positions,
    const at::Tensor& normals,
    const at::Tensor& source,
    const at::Tensor& sigma_b,
    const at::Tensor& rough_b,
    const at::Tensor& replaced,
    int64_t& count,
    int& depth) {
    using channel_native::check_tensor;
    check_tensor(field_vector, "field_vector", at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, "field_vector must have shape (R, 3)");
    count = field_vector.size(0);
    check_tensor(coefficient, "coefficient", at::kComplexFloat, 1);
    check_tensor(path_field, "path_field", at::kComplexFloat, 1);
    check_tensor(path_gain, "path_gain", at::kFloat, 1);
    check_tensor(positions, "positions", at::kFloat, 3);
    check_tensor(normals, "normals", at::kFloat, 3);
    check_tensor(source, "source", at::kFloat, 2);
    check_tensor(sigma_b, "sigma_b", at::kFloat, 2);
    check_tensor(rough_b, "rough_b", at::kBool, 2);
    check_tensor(replaced, "replaced", at::kBool, 1);
    depth = static_cast<int>(positions.size(1));
    TORCH_CHECK(depth >= 1 && depth <= kMaxDepth, "depth must be in [1, 5]");
    TORCH_CHECK(
        coefficient.size(0) == count && path_field.size(0) == count &&
            path_gain.size(0) == count && positions.size(0) == count &&
            normals.size(0) == count && source.size(0) == count &&
            sigma_b.size(0) == count && rough_b.size(0) == count &&
            replaced.size(0) == count,
        "rough-scale row counts must match field_vector");
    TORCH_CHECK(
        positions.size(2) == 3 && normals.size(1) == depth &&
            normals.size(2) == 3 && source.size(1) == 3 &&
            sigma_b.size(1) == depth && rough_b.size(1) == depth,
        "rough-scale per-bounce shapes are inconsistent");
    for (const auto& t : {coefficient, path_field, path_gain, positions, normals,
                          source, sigma_b, rough_b, replaced}) {
        TORCH_CHECK(t.get_device() == field_vector.get_device(),
                    "rough-scale tensors must share device");
    }
}

}  // namespace

pybind11::dict cn_field_rough_reflection_scale(
    at::Tensor field_vector,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor positions,
    at::Tensor normals,
    at::Tensor source,
    at::Tensor sigma_b,
    at::Tensor rough_b,
    at::Tensor replaced,
    double frequency_hz) {
    int64_t count = 0;
    int depth = 0;
    check_inputs(field_vector, coefficient, path_field, path_gain, positions,
                 normals, source, sigma_b, rough_b, replaced, count, depth);
    const float k0 = static_cast<float>(2.0 * kPi * frequency_hz / kC0);
    auto out_field_vector = at::empty_like(field_vector);
    auto out_coefficient = at::empty_like(coefficient);
    auto out_path_field = at::empty_like(path_field);
    auto out_path_gain = at::empty_like(path_gain);
    auto out_factor = at::empty({count}, path_gain.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(field_vector.get_device()).stream();
        rough_scale_forward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, k0,
            field_vector.data_ptr<cfloat>(), coefficient.data_ptr<cfloat>(),
            path_field.data_ptr<cfloat>(), path_gain.data_ptr<float>(),
            positions.data_ptr<float>(), normals.data_ptr<float>(),
            source.data_ptr<float>(), sigma_b.data_ptr<float>(),
            rough_b.data_ptr<bool>(), replaced.data_ptr<bool>(),
            out_field_vector.data_ptr<cfloat>(), out_coefficient.data_ptr<cfloat>(),
            out_path_field.data_ptr<cfloat>(), out_path_gain.data_ptr<float>(),
            out_factor.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = out_field_vector;
    out["coefficient"] = out_coefficient;
    out["path_field"] = out_path_field;
    out["path_gain"] = out_path_gain;
    out["factor"] = out_factor;
    return out;
}

pybind11::dict cn_field_rough_reflection_scale_backward(
    at::Tensor field_vector,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor positions,
    at::Tensor normals,
    at::Tensor source,
    at::Tensor sigma_b,
    at::Tensor rough_b,
    at::Tensor replaced,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    bool need_field,
    bool need_geometry,
    bool need_frequency) {
    int64_t count = 0;
    int depth = 0;
    check_inputs(field_vector, coefficient, path_field, path_gain, positions,
                 normals, source, sigma_b, rough_b, replaced, count, depth);
    const float k0 = static_cast<float>(2.0 * kPi * frequency_hz / kC0);
    const float dk0_df = static_cast<float>(2.0 * kPi / kC0);
    at::Tensor storage[4];
    const at::Tensor* g_fv = optional_arg(
        std::move(grad_field_vector), storage[0], "grad_field_vector",
        at::kComplexFloat, {count, 3}, field_vector);
    const at::Tensor* g_coef = optional_arg(
        std::move(grad_coefficient), storage[1], "grad_coefficient",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* g_pf = optional_arg(
        std::move(grad_path_field), storage[2], "grad_path_field",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* g_pg = optional_arg(
        std::move(grad_path_gain), storage[3], "grad_path_gain",
        at::kFloat, {count}, field_vector);

    at::Tensor grad_field_vector_out, grad_coefficient_out, grad_path_field_out,
        grad_path_gain_out, grad_positions, grad_normals, grad_source,
        grad_frequency;
    if (need_field) {
        grad_field_vector_out = at::empty_like(field_vector);
        grad_coefficient_out = at::empty_like(coefficient);
        grad_path_field_out = at::empty_like(path_field);
        grad_path_gain_out = at::empty_like(path_gain);
    }
    if (need_geometry) {
        grad_positions = zero_filled({count, depth, 3}, positions.options());
        grad_normals = zero_filled({count, depth, 3}, normals.options());
        grad_source = zero_filled({count, 3}, source.options());
    }
    if (need_frequency) {
        grad_frequency = zero_filled({1}, path_gain.options());
    }
    const bool any_grad =
        g_fv != nullptr || g_coef != nullptr || g_pf != nullptr || g_pg != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(field_vector.get_device()).stream();
        rough_scale_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, k0, dk0_df,
            field_vector.data_ptr<cfloat>(), coefficient.data_ptr<cfloat>(),
            path_field.data_ptr<cfloat>(), path_gain.data_ptr<float>(),
            positions.data_ptr<float>(), normals.data_ptr<float>(),
            source.data_ptr<float>(), sigma_b.data_ptr<float>(),
            rough_b.data_ptr<bool>(), replaced.data_ptr<bool>(),
            opt_ptr<cfloat>(g_fv), opt_ptr<cfloat>(g_coef), opt_ptr<cfloat>(g_pf),
            opt_ptr<float>(g_pg),
            need_field ? grad_field_vector_out.data_ptr<cfloat>() : nullptr,
            need_field ? grad_coefficient_out.data_ptr<cfloat>() : nullptr,
            need_field ? grad_path_field_out.data_ptr<cfloat>() : nullptr,
            need_field ? grad_path_gain_out.data_ptr<float>() : nullptr,
            need_geometry ? grad_positions.data_ptr<float>() : nullptr,
            need_geometry ? grad_normals.data_ptr<float>() : nullptr,
            need_geometry ? grad_source.data_ptr<float>() : nullptr,
            need_frequency ? grad_frequency.data_ptr<float>() : nullptr,
            need_field, need_geometry, need_frequency);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_field_vector"] =
        need_field ? pybind11::cast(grad_field_vector_out) : pybind11::object(pybind11::none());
    out["grad_coefficient"] =
        need_field ? pybind11::cast(grad_coefficient_out) : pybind11::object(pybind11::none());
    out["grad_path_field"] =
        need_field ? pybind11::cast(grad_path_field_out) : pybind11::object(pybind11::none());
    out["grad_path_gain"] =
        need_field ? pybind11::cast(grad_path_gain_out) : pybind11::object(pybind11::none());
    out["grad_positions"] =
        need_geometry ? pybind11::cast(grad_positions) : pybind11::object(pybind11::none());
    out["grad_normals"] =
        need_geometry ? pybind11::cast(grad_normals) : pybind11::object(pybind11::none());
    out["grad_source"] =
        need_geometry ? pybind11::cast(grad_source) : pybind11::object(pybind11::none());
    out["grad_frequency"] =
        need_frequency ? pybind11::cast(grad_frequency) : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_field_rough_reflection_scale_jvp(
    at::Tensor field_vector,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor positions,
    at::Tensor normals,
    at::Tensor source,
    at::Tensor sigma_b,
    at::Tensor rough_b,
    at::Tensor replaced,
    double frequency_hz,
    pybind11::object tangent_field_vector,
    pybind11::object tangent_coefficient,
    pybind11::object tangent_path_field,
    pybind11::object tangent_path_gain,
    pybind11::object tangent_positions,
    pybind11::object tangent_normals,
    pybind11::object tangent_source,
    double tangent_frequency) {
    int64_t count = 0;
    int depth = 0;
    check_inputs(field_vector, coefficient, path_field, path_gain, positions,
                 normals, source, sigma_b, rough_b, replaced, count, depth);
    const float k0 = static_cast<float>(2.0 * kPi * frequency_hz / kC0);
    const float dk0_df = static_cast<float>(2.0 * kPi / kC0);
    at::Tensor storage[7];
    const at::Tensor* t_fv = optional_arg(
        std::move(tangent_field_vector), storage[0], "tangent_field_vector",
        at::kComplexFloat, {count, 3}, field_vector);
    const at::Tensor* t_coef = optional_arg(
        std::move(tangent_coefficient), storage[1], "tangent_coefficient",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* t_pf = optional_arg(
        std::move(tangent_path_field), storage[2], "tangent_path_field",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* t_pg = optional_arg(
        std::move(tangent_path_gain), storage[3], "tangent_path_gain",
        at::kFloat, {count}, field_vector);
    const at::Tensor* t_pos = optional_arg(
        std::move(tangent_positions), storage[4], "tangent_positions",
        at::kFloat, {count, depth, 3}, field_vector);
    const at::Tensor* t_nrm = optional_arg(
        std::move(tangent_normals), storage[5], "tangent_normals",
        at::kFloat, {count, depth, 3}, field_vector);
    const at::Tensor* t_src = optional_arg(
        std::move(tangent_source), storage[6], "tangent_source",
        at::kFloat, {count, 3}, field_vector);

    auto out_field_vector = at::empty_like(field_vector);
    auto out_coefficient = at::empty_like(coefficient);
    auto out_path_field = at::empty_like(path_field);
    auto out_path_gain = at::empty_like(path_gain);
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(field_vector.get_device()).stream();
        rough_scale_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, k0, dk0_df,
            field_vector.data_ptr<cfloat>(), coefficient.data_ptr<cfloat>(),
            path_field.data_ptr<cfloat>(), path_gain.data_ptr<float>(),
            positions.data_ptr<float>(), normals.data_ptr<float>(),
            source.data_ptr<float>(), sigma_b.data_ptr<float>(),
            rough_b.data_ptr<bool>(), replaced.data_ptr<bool>(),
            opt_ptr<cfloat>(t_fv), opt_ptr<cfloat>(t_coef), opt_ptr<cfloat>(t_pf),
            opt_ptr<float>(t_pg), opt_ptr<float>(t_pos), opt_ptr<float>(t_nrm),
            opt_ptr<float>(t_src), static_cast<float>(tangent_frequency),
            out_field_vector.data_ptr<cfloat>(), out_coefficient.data_ptr<cfloat>(),
            out_path_field.data_ptr<cfloat>(), out_path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_field_vector"] = out_field_vector;
    out["tangent_coefficient"] = out_coefficient;
    out["tangent_path_field"] = out_path_field;
    out["tangent_path_gain"] = out_path_gain;
    return out;
}
