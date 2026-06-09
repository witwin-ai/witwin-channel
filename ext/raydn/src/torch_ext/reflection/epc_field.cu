#include <raydn/reflection/epc_field.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

#include <raydn/common/complex.cuh>
#include <raydn/common/math.cuh>
#include <raydn/common/native_compat.h>



namespace raydn {

namespace {

constexpr float kReflEps = 1e-6f;
constexpr float kEpsilon0 = 8.854187817e-12f;

static __forceinline__ __device__ float3 fallback_axis(float3 direction) {
    return fabsf(direction.z) < 0.9f
               ? make_f3(0.f, 0.f, 1.f)
               : make_f3(0.f, 1.f, 0.f);
}

static __forceinline__ __device__ float3 stable_perpendicular(float3 direction,
                                                              float3 preferred) {
    const float3 dir = normalize3(direction);
    float3 projected = preferred - dot3(preferred, dir) * dir;
    if (dot3(projected, projected) > 1e-12f) {
        return normalize3(projected);
    }
    const float3 axis = fallback_axis(dir);
    projected = axis - dot3(axis, dir) * dir;
    return normalize3(projected);
}

static __forceinline__ __device__ bool slot_reflection_coefficients(
    const ReflEpcFieldParams params,
    int slot,
    float cos_theta,
    Complex &r_te,
    Complex &r_tm) {
    const float eta_r_value = params.slot_eta_r != nullptr ? params.slot_eta_r[slot] : 1.f;
    const float sigma_value = params.slot_sigma != nullptr ? params.slot_sigma[slot] : 0.f;
    const float gain = params.slot_gain != nullptr ? params.slot_gain[slot] : 1.f;
    const float mu_r_value = params.slot_mu_r != nullptr ? params.slot_mu_r[slot] : 1.f;
    const float eta_r = fmaxf(eta_r_value, kReflEps);
    const float sigma = fmaxf(sigma_value, 0.f);
    const float mu_r = fmaxf(mu_r_value, kReflEps);
    const float omega = fmaxf(params.omega, kReflEps);
    const Complex eta = c_make(eta_r, -sigma / (omega * kEpsilon0));
    const Complex mu = c_make(mu_r, 0.f);
    const float cos_clamped = fminf(fmaxf(fabsf(cos_theta), kReflEps), 1.f);
    const float sin2 = fmaxf(0.f, 1.f - cos_clamped * cos_clamped);
    const Complex a = c_sqrt(c_sub(c_mul(mu, eta), c_make(sin2, 0.f)));
    const Complex mu_cos = c_make(mu_r * cos_clamped, 0.f);
    const Complex eta_cos = c_make(eta.r * cos_clamped, eta.i * cos_clamped);
    r_te = c_scale(c_div(c_sub(mu_cos, a), c_add(mu_cos, a)), gain);
    r_tm = c_scale(c_div(c_sub(eta_cos, a), c_add(eta_cos, a)), gain);
    if (!isfinite(r_te.r) || !isfinite(r_te.i)) {
        r_te = c_make(0.f, 0.f);
    }
    if (!isfinite(r_tm.r) || !isfinite(r_tm.i)) {
        r_tm = c_make(0.f, 0.f);
    }
    return c_abs2(r_te) > 0.f || c_abs2(r_tm) > 0.f;
}

static __forceinline__ __device__ Complex3 reflect_field_vector(
    const ReflEpcFieldParams params,
    int slot,
    Complex3 field,
    float3 incident_dir) {
    const float3 incident_hat = normalize3(incident_dir);
    float3 normal_hat =
        normalize3(make_f3(params.slot_normal_x[slot],
                             params.slot_normal_y[slot],
                             params.slot_normal_z[slot]));
    if (dot3(normal_hat, normal_hat) <= 0.f) {
        return c3_zero();
    }
    if (dot3(incident_hat, normal_hat) > 0.f) {
        normal_hat = -1.f * normal_hat;
    }

    const float dot_dn = dot3(incident_hat, normal_hat);
    const float3 reflected_dir =
        normalize3(incident_hat - 2.f * dot_dn * normal_hat);

    float3 s_hat = cross(normal_hat, incident_hat);
    if (dot3(s_hat, s_hat) <= 1e-12f) {
        s_hat = stable_perpendicular(incident_hat, normal_hat);
    } else {
        s_hat = normalize3(s_hat);
    }
    float3 p_in_hat = cross(s_hat, incident_hat);
    if (dot3(p_in_hat, p_in_hat) <= 1e-12f) {
        p_in_hat = stable_perpendicular(incident_hat, normal_hat);
    } else {
        p_in_hat = normalize3(p_in_hat);
    }
    float3 p_out_hat = cross(s_hat, reflected_dir);
    if (dot3(p_out_hat, p_out_hat) <= 1e-12f) {
        p_out_hat = stable_perpendicular(reflected_dir, normal_hat);
    } else {
        p_out_hat = normalize3(p_out_hat);
    }

    Complex r_te;
    Complex r_tm;
    const float cos_theta = fabsf(dot3(incident_hat, normal_hat));
    if (!slot_reflection_coefficients(params, slot, cos_theta, r_te, r_tm)) {
        return c3_zero();
    }

    const Complex e_s = c3_dot_real(field, s_hat);
    const Complex e_p = c3_dot_real(field, p_in_hat);
    return c3_add(c3_scale_complex(s_hat, c_mul(r_te, e_s)),
                  c3_scale_complex(p_out_hat, c_mul(r_tm, e_p)));
}

static __forceinline__ __device__ void store_zero_field(
    const ReflEpcFieldParams params,
    int ray_index) {
    if (params.out_valid != nullptr) {
        params.out_valid[ray_index] = 0u;
    }
    if (params.out_field_x_re != nullptr) {
        params.out_field_x_re[ray_index] = 0.f;
        params.out_field_x_im[ray_index] = 0.f;
    }
    if (params.out_field_y_re != nullptr) {
        params.out_field_y_re[ray_index] = 0.f;
        params.out_field_y_im[ray_index] = 0.f;
    }
    if (params.out_field_z_re != nullptr) {
        params.out_field_z_re[ray_index] = 0.f;
        params.out_field_z_im[ray_index] = 0.f;
    }
}

__global__ void reflection_epc_forward_setup_kernel(ReflEpcForwardSetupParams params) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int slot_count = params.n_rays * params.max_bounces;
    const int total = params.n_rays > slot_count ? params.n_rays : slot_count;
    if (idx >= total) {
        return;
    }

    if (idx < params.n_rays) {
        const int base3 = idx * 3;
        const float sx = params.source_aos[base3 + 0];
        const float sy = params.source_aos[base3 + 1];
        const float sz = params.source_aos[base3 + 2];
        const float rx = params.receiver_aos[base3 + 0];
        const float ry = params.receiver_aos[base3 + 1];
        const float rz = params.receiver_aos[base3 + 2];
        const float dx = rx - sx;
        const float dy = ry - sy;
        const float dz = rz - sz;

        params.source_x[idx] = sx;
        params.source_y[idx] = sy;
        params.source_z[idx] = sz;
        params.receiver_x[idx] = rx;
        params.receiver_y[idx] = ry;
        params.receiver_z[idx] = rz;
        params.ray_dx[idx] = dx;
        params.ray_dy[idx] = dy;
        params.ray_dz[idx] = dz;
        params.ray_tmax[idx] = sqrtf(dx * dx + dy * dy + dz * dz);

        params.epc_valid[idx] = 0u;
        params.epc_bounce_count[idx] = 0;
        params.epc_path_length[idx] = __uint_as_float(0x7f800000u);
        params.first_blocked_segment[idx] = -1;
        params.first_blocked_prim[idx] = -1;
        params.first_blocked_group[idx] = -1;

        const int bary = idx * 3;
        params.tape_barycentric[bary + 0] = 0.f;
        params.tape_barycentric[bary + 1] = 0.f;
        params.tape_barycentric[bary + 2] = 0.f;
    }

    if (idx < slot_count) {
        params.point_x[idx] = 0.f;
        params.point_y[idx] = 0.f;
        params.point_z[idx] = 0.f;
        params.trace_prim_ids[idx] = -1;
        params.resolved_prim_ids[idx] = -1;
        params.surface_group_ids[idx] = -1;
        params.plane_normal_x[idx] = 0.f;
        params.plane_normal_y[idx] = 0.f;
        params.plane_normal_z[idx] = 0.f;
    }
}

/// One ray per thread (blockIdx.x * blockDim.x + threadIdx.x, bounds-checked); evaluates the
/// complex reflected field from the ray's precomputed EPC geometry and writes the per-ray outputs.
__global__ void reflection_epc_field_kernel(ReflEpcFieldParams params) {
    const int ray_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (ray_index >= params.n_rays) {
        return;
    }

    const int base = ray_index * params.max_bounces;
    if (params.out_first_resolved_prim_id != nullptr) {
        params.out_first_resolved_prim_id[ray_index] =
            params.resolved_prim_ids != nullptr ? params.resolved_prim_ids[base] : -1;
    }
    if (params.out_first_trace_prim_id != nullptr) {
        params.out_first_trace_prim_id[ray_index] =
            params.trace_prim_ids != nullptr ? params.trace_prim_ids[base] : -1;
    }
    const bool epc_valid =
        params.epc_valid != nullptr && params.epc_valid[ray_index] != 0u;
    const int bounce_count =
        params.epc_bounce_count != nullptr ? params.epc_bounce_count[ray_index] : 0;
    const int clamped_bounce_count =
        min(max(bounce_count, 0), params.max_bounces);

    if (params.out_bounce_count != nullptr) {
        params.out_bounce_count[ray_index] = bounce_count;
    }
    if (params.out_path_length != nullptr) {
        params.out_path_length[ray_index] =
            params.epc_path_length != nullptr ? params.epc_path_length[ray_index]
                                              : __uint_as_float(0x7f800000u);
    }

    if (params.out_hit_x != nullptr || params.out_normal_x != nullptr ||
        params.out_resolved_prim_ids != nullptr ||
        params.out_surface_group_ids != nullptr) {
        for (int b = 0; b < params.max_bounces; ++b) {
            const int slot = base + b;
            if (params.out_hit_x != nullptr) {
                params.out_hit_x[slot] = params.hit_x[slot];
                params.out_hit_y[slot] = params.hit_y[slot];
                params.out_hit_z[slot] = params.hit_z[slot];
            }
            if (params.out_normal_x != nullptr) {
                params.out_normal_x[slot] = params.epc_normal_x[slot];
                params.out_normal_y[slot] = params.epc_normal_y[slot];
                params.out_normal_z[slot] = params.epc_normal_z[slot];
            }
            if (params.out_resolved_prim_ids != nullptr &&
                params.resolved_prim_ids != nullptr) {
                params.out_resolved_prim_ids[slot] =
                    params.resolved_prim_ids[slot];
            }
            if (params.out_surface_group_ids != nullptr &&
                params.surface_group_ids != nullptr) {
                params.out_surface_group_ids[slot] =
                    params.surface_group_ids[slot];
            }
        }
    }

    const float3 tx = make_f3(params.ray_ox[ray_index],
                                params.ray_oy[ray_index],
                                params.ray_oz[ray_index]);
    if (params.out_tx_x != nullptr) {
        params.out_tx_x[ray_index] = tx.x;
        params.out_tx_y[ray_index] = tx.y;
        params.out_tx_z[ray_index] = tx.z;
        const float3 zero = make_f3(0.f, 0.f, 0.f);
        float3 first = zero;
        float3 last = zero;
        if (clamped_bounce_count > 0) {
            first = make_f3(params.hit_x[base],
                              params.hit_y[base],
                              params.hit_z[base]);
            const int last_slot = base + clamped_bounce_count - 1;
            last = make_f3(params.hit_x[last_slot],
                             params.hit_y[last_slot],
                             params.hit_z[last_slot]);
        }
        params.out_first_hit_x[ray_index] = first.x;
        params.out_first_hit_y[ray_index] = first.y;
        params.out_first_hit_z[ray_index] = first.z;
        params.out_last_hit_x[ray_index] = last.x;
        params.out_last_hit_y[ray_index] = last.y;
        params.out_last_hit_z[ray_index] = last.z;
    }

    if (!epc_valid || params.max_bounces <= 0) {
        store_zero_field(params, ray_index);
        return;
    }

    float3 previous = tx;
    const float3 first_hit = make_f3(params.hit_x[base],
                                       params.hit_y[base],
                                       params.hit_z[base]);
    const float3 first_dir = normalize3(first_hit - previous);
    if (dot3(first_dir, first_dir) <= 0.f) {
        store_zero_field(params, ray_index);
        return;
    }

    float3 tx_polarization = make_f3(1.f, 0.f, 0.f);
    if (params.tx_pol_x != nullptr) {
        const int tx_pol_index = params.tx_pol_count == 1 ? 0 : ray_index;
        tx_polarization = make_f3(params.tx_pol_x[tx_pol_index],
                                  params.tx_pol_y[tx_pol_index],
                                  params.tx_pol_z[tx_pol_index]);
    }
    float3 transverse_polarization =
        tx_polarization - dot3(tx_polarization, first_dir) * first_dir;
    if (dot3(transverse_polarization, transverse_polarization) <= 1e-12f) {
        transverse_polarization = stable_perpendicular(first_dir, tx_polarization);
    } else {
        transverse_polarization = normalize3(transverse_polarization);
    }
    Complex3 field = c3_from_real(transverse_polarization);

    for (int b = 0; b < params.max_bounces; ++b) {
        const int slot = base + b;
        const float3 hit = make_f3(params.hit_x[slot],
                                     params.hit_y[slot],
                                     params.hit_z[slot]);
        const float3 incident_dir = normalize3(hit - previous);
        if (dot3(incident_dir, incident_dir) <= 0.f) {
            store_zero_field(params, ray_index);
            return;
        }
        field = reflect_field_vector(params, slot, field, incident_dir);
        if (!finite_complex3(field)) {
            store_zero_field(params, ray_index);
            return;
        }
        previous = hit;
    }

    const int rx_id = params.rx_count == 1 ? 0 : ray_index;
    const float3 rx = make_f3(params.rx_x[rx_id],
                                params.rx_y[rx_id],
                                params.rx_z[rx_id]);
    const float final_segment_length = norm3(rx - previous);
    const float path_length =
        params.epc_path_length != nullptr ? params.epc_path_length[ray_index]
                                          : final_segment_length;
    if (!(path_length > kReflEps) || !isfinite(path_length)) {
        store_zero_field(params, ray_index);
        return;
    }

    const float wave_k = 2.f * kPi / fmaxf(params.wavelength, kReflEps);
    const Complex phase = c_exp_neg_i(wave_k * path_length);
    const float amplitude =
        params.wavelength / (4.f * kPi * fmaxf(path_length, kReflEps));
    field = c3_mul_complex(field, c_scale(phase, amplitude));
    const float power = c3_power(field);
    if (!finite_complex3(field) || !isfinite(power)) {
        store_zero_field(params, ray_index);
        return;
    }

    if (params.out_valid != nullptr) {
        params.out_valid[ray_index] = 1u;
    }
    if (params.out_field_x_re != nullptr) {
        params.out_field_x_re[ray_index] = field.x.r;
        params.out_field_x_im[ray_index] = field.x.i;
    }
    if (params.out_field_y_re != nullptr) {
        params.out_field_y_re[ray_index] = field.y.r;
        params.out_field_y_im[ray_index] = field.y.i;
    }
    if (params.out_field_z_re != nullptr) {
        params.out_field_z_re[ray_index] = field.z.r;
        params.out_field_z_im[ray_index] = field.z.i;
    }
}

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

} // namespace

void reflection_epc_forward_setup_gpu(const ReflEpcForwardSetupParams &params) {
    require(params.n_rays >= 0,
            "reflection_epc_forward_setup_gpu(): n_rays must be non-negative.");
    require(params.max_bounces > 0,
            "reflection_epc_forward_setup_gpu(): max_bounces must be positive.");
    if (params.n_rays == 0) {
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    const int slot_count = params.n_rays * params.max_bounces;
    const int total = std::max(params.n_rays, slot_count);
    const int block_size = 128;
    const int block_count = (total + block_size - 1) / block_size;
    audit_cuda_kernel_launch("reflection_epc_forward_setup_kernel",
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(total));
    reflection_epc_forward_setup_kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error(
        "reflection_epc_forward_setup_gpu(): failed to launch setup kernel");
}

void reflection_epc_field_gpu(const ReflEpcFieldParams &params) {
    require(params.n_rays >= 0,
            "reflection_epc_field_gpu(): n_rays must be non-negative.");
    require(params.max_bounces > 0,
            "reflection_epc_field_gpu(): max_bounces must be positive.");
    if (params.n_rays == 0) {
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());

    const int block_size = 128;
    const int block_count = (params.n_rays + block_size - 1) / block_size;
    audit_cuda_kernel_launch("reflection_epc_field_kernel",
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(params.n_rays));
    reflection_epc_field_kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error(
        "reflection_epc_field_gpu(): failed to launch field kernel");
}

} // namespace raydn
