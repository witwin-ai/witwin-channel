// ISB boundary taper (ADR-017), LoS member. Native, channel-native owned.
//
// Two per-(tx,rx) CUDA operations that implement the DEFAULT-OFF joint ISB
// taper's line-of-sight half:
//
//   cn_los_silhouette_clearance:  for each (source, target) segment, find the
//     nearest occluding axis-aligned box the segment grazes, and return the
//     C1 membership factor tau(c_plane / (width * w_F)) where
//       - c   = signed clearance of the segment past the box silhouette
//               (positive when the segment clears the box / lit, negative when
//               it penetrates / shadow), taken as the signed AABB distance at
//               the segment's closest-approach sample (measured at the
//               occluder);
//       - c_plane = c * (d1 + d2) / d1 magnifies that occluder-plane miss
//               distance into the receiver plane, where the point-source shadow
//               of the silhouette edge is enlarged by the same factor. The
//               accepted projection (artifacts/isb-taper/stage2.py) scores the
//               clearance as an in-receiver-plane distance transform, so the
//               native band must cover the same receiver-plane extent;
//       - w_F = sqrt(lambda * d1 * d2 / (d1 + d2)) is the Fresnel penumbra of
//               the grazed edge (d1 = |grazing - source|, d2 = |target -
//               grazing|); the exact form and the signed-distance / grazing
//               conventions match artifacts/isb-taper/common.py + stage1_geom.py;
//       - tau(w) = smoothstep01(0.5 * (w + 1)) is the C1 step through 1/2 at
//               c = 0 (artifacts/isb-taper/stage2.py tau_smoothstep).
//     A segment that grazes no box (empty scene) returns tau = 1 (fully lit).
//
//   cn_los_taper_apply: scale a LoS field bundle (complex3 vector, complex
//     coefficient, complex path_field, real path_gain) by the real per-row
//     factor tau. tau multiplies the field amplitude, so path_gain (a power)
//     is scaled by tau*tau. No torch hot-path math; the scale runs in-kernel.
//
// Both ops are only ever launched when isb_boundary_taper is on; the off path
// never reaches this translation unit, so the default solve is bit-identical.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <tuple>

namespace {

constexpr int kBlockSize = 256;
// Segment samples used to locate the closest-approach point to each box.
// Matches artifacts/isb-taper/stage1_geom.py occluding_edge_geom (400 samples).
constexpr int kSegmentSamples = 400;

__device__ __forceinline__ float smoothstep01(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Signed distance of point p to an axis-aligned box [bmin, bmax]:
//   q = max(bmin - p, p - bmax);  sd = ||max(q, 0)|| + min(max(q), 0)
// negative inside the box, positive outside. Matches common.py conventions.
__device__ __forceinline__ float aabb_signed_distance(
    float px, float py, float pz,
    float minx, float miny, float minz,
    float maxx, float maxy, float maxz) {
    const float qx = fmaxf(minx - px, px - maxx);
    const float qy = fmaxf(miny - py, py - maxy);
    const float qz = fmaxf(minz - pz, pz - maxz);
    const float ox = fmaxf(qx, 0.0f);
    const float oy = fmaxf(qy, 0.0f);
    const float oz = fmaxf(qz, 0.0f);
    const float outside = sqrtf(ox * ox + oy * oy + oz * oz);
    const float inside = fminf(fmaxf(fmaxf(qx, qy), qz), 0.0f);
    return outside + inside;
}

__global__ void los_silhouette_clearance_kernel(
    const float *__restrict__ source,
    const float *__restrict__ target,
    const float *__restrict__ box_min,
    const float *__restrict__ box_max,
    float *__restrict__ tau,
    int64_t pair_count,
    int64_t box_count,
    float wavelength,
    float width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < pair_count;
         idx += stride) {
        const float sx = source[idx * 3 + 0];
        const float sy = source[idx * 3 + 1];
        const float sz = source[idx * 3 + 2];
        const float tx = target[idx * 3 + 0];
        const float ty = target[idx * 3 + 1];
        const float tz = target[idx * 3 + 2];
        const float dx = tx - sx;
        const float dy = ty - sy;
        const float dz = tz - sz;

        // Nearest grazed box: minimize |signed AABB distance| along the segment.
        float best_slack = 3.4e38f;
        float best_c = 1.0e30f;     // signed clearance at the closest approach
        float best_d1 = 0.0f;       // |grazing - source|
        float best_d2 = 0.0f;       // |target  - grazing|
        bool found = false;
        for (int64_t b = 0; b < box_count; ++b) {
            const float minx = box_min[b * 3 + 0];
            const float miny = box_min[b * 3 + 1];
            const float minz = box_min[b * 3 + 2];
            const float maxx = box_max[b * 3 + 0];
            const float maxy = box_max[b * 3 + 1];
            const float maxz = box_max[b * 3 + 2];
            float box_slack = 3.4e38f;
            float box_c = 0.0f;
            float box_t = 0.0f;
            for (int s = 0; s < kSegmentSamples; ++s) {
                const float u = static_cast<float>(s) /
                                static_cast<float>(kSegmentSamples - 1);
                const float px = sx + u * dx;
                const float py = sy + u * dy;
                const float pz = sz + u * dz;
                const float sd = aabb_signed_distance(
                    px, py, pz, minx, miny, minz, maxx, maxy, maxz);
                const float slack = fabsf(sd);
                if (slack < box_slack) {
                    box_slack = slack;
                    box_c = sd;
                    box_t = u;
                }
            }
            if (box_slack < best_slack) {
                best_slack = box_slack;
                best_c = box_c;
                const float gx = sx + box_t * dx;
                const float gy = sy + box_t * dy;
                const float gz = sz + box_t * dz;
                best_d1 = sqrtf((gx - sx) * (gx - sx) + (gy - sy) * (gy - sy) +
                                (gz - sz) * (gz - sz));
                best_d2 = sqrtf((tx - gx) * (tx - gx) + (ty - gy) * (ty - gy) +
                                (tz - gz) * (tz - gz));
                found = true;
            }
        }

        if (!found) {
            // No occluder: the segment is fully lit.
            tau[idx] = 1.0f;
            continue;
        }
        // Shadow magnification: best_c is the 3D miss distance measured at the
        // occluder (closest-approach sample), but the accepted projection scores
        // the clearance in the RECEIVER PLANE, where the point-source shadow of
        // the silhouette edge is magnified by (d1 + d2) / d1. Convert so the
        // native taper band covers the same receiver-plane extent as the
        // projection (artifacts/isb-taper/stage2.py in-plane distance transform).
        const float mag = (best_d1 + best_d2) / fmaxf(best_d1, 1.0e-6f);
        const float c_plane = best_c * mag;
        const float w_F = sqrtf(fmaxf(
            wavelength * best_d1 * best_d2 / fmaxf(best_d1 + best_d2, 1.0e-12f),
            0.0f));
        const float w = fmaxf(width * w_F, 1.0e-6f);
        tau[idx] = smoothstep01(0.5f * (c_plane / w + 1.0f));
    }
}

__global__ void los_taper_apply_kernel(
    const c10::complex<float> *__restrict__ field_vector,
    const c10::complex<float> *__restrict__ coefficient,
    const c10::complex<float> *__restrict__ path_field,
    const float *__restrict__ path_gain,
    const float *__restrict__ tau,
    c10::complex<float> *__restrict__ out_field_vector,
    c10::complex<float> *__restrict__ out_coefficient,
    c10::complex<float> *__restrict__ out_path_field,
    float *__restrict__ out_path_gain,
    int64_t row_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < row_count;
         idx += stride) {
        const float s = tau[idx];
        out_field_vector[idx * 3 + 0] = field_vector[idx * 3 + 0] * s;
        out_field_vector[idx * 3 + 1] = field_vector[idx * 3 + 1] * s;
        out_field_vector[idx * 3 + 2] = field_vector[idx * 3 + 2] * s;
        out_coefficient[idx] = coefficient[idx] * s;
        out_path_field[idx] = path_field[idx] * s;
        // tau scales the field amplitude, so a power scales by tau^2.
        out_path_gain[idx] = path_gain[idx] * s * s;
    }
}

void check_cuda(const at::Tensor &t, const char *name, c10::ScalarType dtype,
                int64_t ndim) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(t.dim() == ndim, name, " has the wrong rank");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

at::Tensor cn_los_silhouette_clearance(
    at::Tensor source,
    at::Tensor target,
    at::Tensor box_min,
    at::Tensor box_max,
    double wavelength,
    double width) {
    check_cuda(source, "source", at::kFloat, 2);
    check_cuda(target, "target", at::kFloat, 2);
    check_cuda(box_min, "box_min", at::kFloat, 2);
    check_cuda(box_max, "box_max", at::kFloat, 2);
    TORCH_CHECK(source.size(1) == 3, "source must have shape (N, 3)");
    TORCH_CHECK(target.sizes() == source.sizes(), "target must match source");
    TORCH_CHECK(box_min.size(1) == 3, "box_min must have shape (B, 3)");
    TORCH_CHECK(box_max.sizes() == box_min.sizes(), "box_max must match box_min");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    TORCH_CHECK(width > 0.0, "width must be positive");
    TORCH_CHECK(
        source.get_device() == box_min.get_device() &&
            target.get_device() == source.get_device() &&
            box_max.get_device() == source.get_device(),
        "silhouette clearance tensors must share one CUDA device");

    const int64_t pair_count = source.size(0);
    const int64_t box_count = box_min.size(0);
    auto tau = at::empty({pair_count}, source.options());
    if (pair_count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        const int blocks =
            static_cast<int>((pair_count + kBlockSize - 1) / kBlockSize);
        los_silhouette_clearance_kernel<<<blocks, kBlockSize, 0, stream>>>(
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            box_min.data_ptr<float>(),
            box_max.data_ptr<float>(),
            tau.data_ptr<float>(),
            pair_count,
            box_count,
            static_cast<float>(wavelength),
            static_cast<float>(width));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return tau;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_los_taper_apply(
    at::Tensor field_vector,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor tau) {
    check_cuda(field_vector, "field_vector", at::kComplexFloat, 2);
    check_cuda(coefficient, "coefficient", at::kComplexFloat, 1);
    check_cuda(path_field, "path_field", at::kComplexFloat, 1);
    check_cuda(path_gain, "path_gain", at::kFloat, 1);
    check_cuda(tau, "tau", at::kFloat, 1);
    const int64_t row_count = tau.size(0);
    TORCH_CHECK(field_vector.size(0) == row_count && field_vector.size(1) == 3,
                "field_vector must have shape (N, 3)");
    TORCH_CHECK(coefficient.size(0) == row_count, "coefficient must match tau");
    TORCH_CHECK(path_field.size(0) == row_count, "path_field must match tau");
    TORCH_CHECK(path_gain.size(0) == row_count, "path_gain must match tau");

    auto out_field_vector = at::empty_like(field_vector);
    auto out_coefficient = at::empty_like(coefficient);
    auto out_path_field = at::empty_like(path_field);
    auto out_path_gain = at::empty_like(path_gain);
    if (row_count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tau.get_device()).stream();
        const int blocks =
            static_cast<int>((row_count + kBlockSize - 1) / kBlockSize);
        los_taper_apply_kernel<<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const c10::complex<float> *>(
                field_vector.data_ptr<c10::complex<float>>()),
            reinterpret_cast<const c10::complex<float> *>(
                coefficient.data_ptr<c10::complex<float>>()),
            reinterpret_cast<const c10::complex<float> *>(
                path_field.data_ptr<c10::complex<float>>()),
            path_gain.data_ptr<float>(),
            tau.data_ptr<float>(),
            reinterpret_cast<c10::complex<float> *>(
                out_field_vector.data_ptr<c10::complex<float>>()),
            reinterpret_cast<c10::complex<float> *>(
                out_coefficient.data_ptr<c10::complex<float>>()),
            reinterpret_cast<c10::complex<float> *>(
                out_path_field.data_ptr<c10::complex<float>>()),
            out_path_gain.data_ptr<float>(),
            row_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out_field_vector, out_coefficient, out_path_field, out_path_gain};
}
