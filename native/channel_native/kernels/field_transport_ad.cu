#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../field_transport_ad.cuh"
#include "../tensor_checks.h"

// Backward / JVP companion kernels for the field transport forwards
// (plan 07 AD-1 materials/frequency, AD-2 geometry). Fixed-topology contract:
// the discrete winner (face sequence, validity, normal flips, polarizations,
// tx_power, material ids) is constant; the differentiable inputs are
// eps_r / sigma_e / gain / thickness (per bounce or CSR layer), the carrier
// frequency, and the continuous hit geometry (source, target,
// interaction_positions, interaction_normals) behind need_grad_geometry.
// path_length_m / delay_s are differentiable outputs of the geometry alone
// (their material/frequency cotangent is exactly zero).

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxAdDepth = 8;
namespace field = witwin::channel::native_ext;
namespace em = channel_native::em;
namespace transport = channel_native::field_transport;
namespace ad = channel_native::field_transport_ad;

using ad::DualC;
using ad::adj_dot;

__device__ __forceinline__ field::float3a load3f(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::float3a load_sequence3f(
    const float* values, int64_t index, int64_t bounce, int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::Complex complex_of(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

__device__ __forceinline__ c10::complex<float> to_c10(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

// Cotangent of the final complex3 field value, folded from the output
// cotangents (field_vector, coefficient, path_field, path_gain). The scalar
// chain is coefficient = <value, rx_axis>, path_field = coefficient * a,
// path_gain = |path_field|^2 with a = sqrt(max(tx_power, 0)); all real-linear
// maps except path_gain, whose real-pair adjoint at pf is 2*g*(pf.re, pf.im).
// The folded scalar cotangent is also emitted: it is the coefficient-level
// cotangent the geometry adjoint needs for the rx_axis (plan 07 AD-2).
__device__ __forceinline__ field::Complex3 fold_output_cotangents(
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    int64_t index,
    field::float3a rx_axis,
    field::Complex path_field_value,
    float amplitude_scale,
    field::Complex& g_scalar_out) {
    field::Complex g_scalar = field::cplx_zero();
    if (grad_coefficient != nullptr)
        g_scalar = field::cplx_add(g_scalar, complex_of(grad_coefficient[index]));
    if (grad_path_field != nullptr)
        g_scalar = field::cplx_add(
            g_scalar,
            field::cplx_mul_real(
                complex_of(grad_path_field[index]), amplitude_scale));
    if (grad_path_gain != nullptr) {
        const float g_gain = grad_path_gain[index];
        g_scalar = field::cplx_add(
            g_scalar,
            field::cplx_mul_real(
                path_field_value, 2.0f * g_gain * amplitude_scale));
    }
    g_scalar_out = g_scalar;
    field::Complex3 g_value = field::c3_zero();
    g_value.x = field::cplx_mul_real(g_scalar, rx_axis.x);
    g_value.y = field::cplx_mul_real(g_scalar, rx_axis.y);
    g_value.z = field::cplx_mul_real(g_scalar, rx_axis.z);
    if (grad_field_vector != nullptr) {
        const int64_t base = index * 3;
        g_value.x = field::cplx_add(g_value.x, complex_of(grad_field_vector[base]));
        g_value.y = field::cplx_add(g_value.y, complex_of(grad_field_vector[base + 1]));
        g_value.z = field::cplx_add(g_value.z, complex_of(grad_field_vector[base + 2]));
    }
    return g_value;
}

// Forward-mode dual of the shared scalar output chain. Writes the six
// differentiable output tangents; d_rx_axis and d_length carry the geometry
// tangents (zero under material/frequency-only seeds).
__device__ __forceinline__ void write_output_tangents(
    int64_t index,
    field::Complex3 value,
    field::Complex3 d_value,
    field::float3a rx_axis,
    field::float3a d_rx_axis,
    float amplitude_scale,
    float d_length,
    c10::complex<float>* t_field_vector,
    c10::complex<float>* t_coefficient,
    c10::complex<float>* t_path_field,
    float* t_path_gain,
    float* t_path_length,
    float* t_delay) {
    const int64_t base = index * 3;
    t_field_vector[base] = to_c10(d_value.x);
    t_field_vector[base + 1] = to_c10(d_value.y);
    t_field_vector[base + 2] = to_c10(d_value.z);
    const field::Complex scalar = transport::complex3_dot_real(value, rx_axis);
    const field::Complex d_scalar = field::cplx_add(
        transport::complex3_dot_real(d_value, rx_axis),
        transport::complex3_dot_real(value, d_rx_axis));
    t_coefficient[index] = to_c10(d_scalar);
    const field::Complex path_field = field::cplx_mul_real(scalar, amplitude_scale);
    const field::Complex d_path_field = field::cplx_mul_real(d_scalar, amplitude_scale);
    t_path_field[index] = to_c10(d_path_field);
    t_path_gain[index] =
        2.0f * (path_field.re * d_path_field.re + path_field.im * d_path_field.im);
    t_path_length[index] = d_length;
    t_delay[index] = d_length / transport::kSpeedOfLight;
}

// ---------------------------------------------------------------------------
// Free space (frequency is the only differentiable input in AD-1).
// ---------------------------------------------------------------------------

template <typename T>
__global__ void free_space_fwd_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    c10::complex<T>* field_vector,
    c10::complex<T>* coefficient,
    c10::complex<T>* path_field,
    T* path_gain,
    T* path_length,
    T* delay,
    T* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            ad::v3_load(source, index),
            ad::v3_load(target, index),
            ad::v3_load(tx_polarization, index),
            ad::v3_load(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const int64_t base = index * 3;
        field_vector[base] = eval.carrier * eval.tx_axis.x;
        field_vector[base + 1] = eval.carrier * eval.tx_axis.y;
        field_vector[base + 2] = eval.carrier * eval.tx_axis.z;
        const c10::complex<T> scalar = eval.carrier * eval.projection;
        coefficient[index] = scalar;
        const c10::complex<T> received = scalar * eval.amplitude_scale;
        path_field[index] = received;
        path_gain[index] = received.real() * received.real() +
                           received.imag() * received.imag();
        path_length[index] = eval.distance;
        delay[index] = eval.distance / T(ad::kSpeedOfLight);
        direction_out[base] = eval.direction.x;
        direction_out[base + 1] = eval.direction.y;
        direction_out[base + 2] = eval.direction.z;
    }
}

template <typename T>
__global__ void free_space_backward_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    const c10::complex<T>* grad_field_vector,
    const c10::complex<T>* grad_coefficient,
    const c10::complex<T>* grad_path_field,
    const T* grad_path_gain,
    const T* grad_path_length,
    const T* grad_delay,
    T* grad_frequency,
    T* grad_source,
    T* grad_target) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            ad::v3_load(source, index),
            ad::v3_load(target, index),
            ad::v3_load(tx_polarization, index),
            ad::v3_load(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const int64_t base = index * 3;
        // Fold the field-output cotangents onto the carrier P; the geometry
        // enters through the carrier distance, the tx/rx bases and the raw
        // straight length (path_length_m / delay_s).
        const c10::complex<T> path_field_value =
            eval.carrier * eval.projection * eval.amplitude_scale;
        c10::complex<T> g_scalar(T(0), T(0));
        if (grad_coefficient != nullptr)
            g_scalar += grad_coefficient[index];
        if (grad_path_field != nullptr)
            g_scalar += grad_path_field[index] * eval.amplitude_scale;
        if (grad_path_gain != nullptr)
            g_scalar += path_field_value *
                        (T(2) * grad_path_gain[index] * eval.amplitude_scale);
        c10::complex<T> g_carrier = g_scalar * eval.projection;
        if (grad_field_vector != nullptr) {
            g_carrier += grad_field_vector[base] * eval.tx_axis.x;
            g_carrier += grad_field_vector[base + 1] * eval.tx_axis.y;
            g_carrier += grad_field_vector[base + 2] * eval.tx_axis.z;
        }
        if (grad_frequency != nullptr) {
            const T g_freq = g_carrier.real() * eval.carrier_dfreq.real() +
                             g_carrier.imag() * eval.carrier_dfreq.imag();
            atomicAdd(grad_frequency, g_freq);
        }
        if (grad_source == nullptr && grad_target == nullptr)
            continue;
        // Real-pair cotangents of the tx/rx bases: coefficient = P *
        // <tx_axis, rx_axis> and field_vector = P * tx_axis.
        const T g_projection = g_scalar.real() * eval.carrier.real() +
                               g_scalar.imag() * eval.carrier.imag();
        ad::Vec3<T> g_tx_axis = {T(0), T(0), T(0)};
        ad::Vec3<T> g_rx_axis = {T(0), T(0), T(0)};
        g_tx_axis = ad::v3_add(g_tx_axis, ad::v3_scale(eval.rx_axis, g_projection));
        g_rx_axis = ad::v3_add(g_rx_axis, ad::v3_scale(eval.tx_axis, g_projection));
        if (grad_field_vector != nullptr) {
            g_tx_axis.x += grad_field_vector[base].real() * eval.carrier.real() +
                           grad_field_vector[base].imag() * eval.carrier.imag();
            g_tx_axis.y +=
                grad_field_vector[base + 1].real() * eval.carrier.real() +
                grad_field_vector[base + 1].imag() * eval.carrier.imag();
            g_tx_axis.z +=
                grad_field_vector[base + 2].real() * eval.carrier.real() +
                grad_field_vector[base + 2].imag() * eval.carrier.imag();
        }
        T g_distance = g_carrier.real() * eval.carrier_ddist.real() +
                       g_carrier.imag() * eval.carrier_ddist.imag();
        if (grad_path_length != nullptr)
            g_distance += grad_path_length[index];
        if (grad_delay != nullptr)
            g_distance += grad_delay[index] / T(ad::kSpeedOfLight);
        const ad::Vec3<T> offset =
            ad::v3_sub(ad::v3_load(target, index), ad::v3_load(source, index));
        ad::Vec3<T> g_direction = {T(0), T(0), T(0)};
        ad::adj_v3_stable_perp_basis(
            eval.direction, ad::v3_load(tx_polarization, index), g_tx_axis,
            g_direction);
        ad::adj_v3_stable_perp_basis(
            eval.direction, ad::v3_load(rx_polarization, index), g_rx_axis,
            g_direction);
        ad::Vec3<T> g_offset = {T(0), T(0), T(0)};
        ad::Vec3<T> g_alternate = {T(0), T(0), T(0)};
        ad::adj_v3_safe_normalize(
            offset, ad::Vec3<T>{T(0), T(0), T(1)}, g_direction, g_offset,
            g_alternate);
        ad::adj_v3_length(offset, g_distance, g_offset);
        if (grad_target != nullptr) {
            grad_target[base] = g_offset.x;
            grad_target[base + 1] = g_offset.y;
            grad_target[base + 2] = g_offset.z;
        }
        if (grad_source != nullptr) {
            grad_source[base] = -g_offset.x;
            grad_source[base + 1] = -g_offset.y;
            grad_source[base + 2] = -g_offset.z;
        }
    }
}

template <typename T>
__global__ void free_space_jvp_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    T tangent_frequency,
    const T* tangent_source,
    const T* tangent_target,
    c10::complex<T>* t_field_vector,
    c10::complex<T>* t_coefficient,
    c10::complex<T>* t_path_field,
    T* t_path_gain,
    T* t_path_length,
    T* t_delay) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            ad::v3_load(source, index),
            ad::v3_load(target, index),
            ad::v3_load(tx_polarization, index),
            ad::v3_load(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const ad::Vec3<T> zero3 = {T(0), T(0), T(0)};
        const ad::Vec3<T> d_source =
            tangent_source != nullptr ? ad::v3_load(tangent_source, index) : zero3;
        const ad::Vec3<T> d_target =
            tangent_target != nullptr ? ad::v3_load(tangent_target, index) : zero3;
        const ad::DualV3<T> offset = {
            ad::v3_sub(ad::v3_load(target, index), ad::v3_load(source, index)),
            ad::v3_sub(d_target, d_source)};
        T d_distance = T(0);
        (void)ad::dual_v3_length(offset, d_distance);
        const ad::DualV3<T> direction = ad::dual_v3_safe_normalize(
            offset, ad::dv3_const(ad::Vec3<T>{T(0), T(0), T(1)}));
        const ad::DualV3<T> tx_axis = ad::dual_v3_stable_perp_basis(
            direction, ad::v3_load(tx_polarization, index));
        const ad::DualV3<T> rx_axis = ad::dual_v3_stable_perp_basis(
            direction, ad::v3_load(rx_polarization, index));
        const c10::complex<T> d_carrier =
            eval.carrier_dfreq * tangent_frequency +
            eval.carrier_ddist * d_distance;
        const int64_t base = index * 3;
        t_field_vector[base] =
            d_carrier * eval.tx_axis.x + eval.carrier * tx_axis.d.x;
        t_field_vector[base + 1] =
            d_carrier * eval.tx_axis.y + eval.carrier * tx_axis.d.y;
        t_field_vector[base + 2] =
            d_carrier * eval.tx_axis.z + eval.carrier * tx_axis.d.z;
        const T d_projection = ad::v3_dot(tx_axis.d, eval.rx_axis) +
                               ad::v3_dot(eval.tx_axis, rx_axis.d);
        const c10::complex<T> d_scalar =
            d_carrier * eval.projection + eval.carrier * d_projection;
        t_coefficient[index] = d_scalar;
        const c10::complex<T> d_path_field = d_scalar * eval.amplitude_scale;
        t_path_field[index] = d_path_field;
        const c10::complex<T> path_field_value =
            eval.carrier * eval.projection * eval.amplitude_scale;
        t_path_gain[index] =
            T(2) * (path_field_value.real() * d_path_field.real() +
                    path_field_value.imag() * d_path_field.imag());
        t_path_length[index] = d_distance;
        t_delay[index] = d_distance / T(ad::kSpeedOfLight);
    }
}

// ---------------------------------------------------------------------------
// Reflection sequence (per-bounce eps_r / sigma_e / gain / thickness and
// frequency are differentiable).
// ---------------------------------------------------------------------------

// Fixed per-path chain state recomputed from the primal inputs; shared by the
// backward and jvp kernels so both differentiate the identical forward chain.
struct ReflectionChain {
    transport::ReflectFrame frames[kMaxAdDepth];
    field::Complex3 value_in[kMaxAdDepth];  // field entering each bounce
    field::Complex e_s[kMaxAdDepth];
    field::Complex e_p[kMaxAdDepth];
    field::Complex r_te[kMaxAdDepth];
    field::Complex r_tm[kMaxAdDepth];
    field::Complex3 value_chain;  // pre-propagation
    field::float3a rx_axis;
    field::Complex propagation;
    field::Complex propagation_dfreq;
    field::Complex propagation_dlength;
    float total_length;
    float amplitude_scale;
};

__device__ void reflection_chain_eval(
    int64_t index,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_positions,
    const float* interaction_normals,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    float frequency_hz,
    ReflectionChain& chain) {
    field::float3a previous = load3f(source, index);
    const field::float3a first_hit = load_sequence3f(
        interaction_positions, index, 0, depth);
    field::float3a incident = field::safe_normalize(
        field::f3_sub(first_hit, previous), field::make_f3(0.0f, 0.0f, 1.0f));
    const field::float3a tx_axis = field::stable_perp_basis(
        incident, load3f(tx_polarization, index));
    field::Complex3 value = field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f));
    float total_length = 0.0f;
    field::float3a outgoing = incident;
    for (int64_t bounce = 0; bounce < depth; ++bounce) {
        const field::float3a hit = load_sequence3f(
            interaction_positions, index, bounce, depth);
        incident = field::safe_normalize(field::f3_sub(hit, previous), outgoing);
        total_length += field::safe_length(field::f3_sub(hit, previous));
        const int64_t scalar = index * depth + bounce;
        const transport::ReflectFrame frame = transport::reflect_frame(
            incident, load_sequence3f(interaction_normals, index, bounce, depth));
        field::Complex r_te;
        field::Complex r_tm;
        transport::slab_fresnel(
            frame.cos_theta,
            eps_r[scalar],
            sigma_e[scalar],
            mu_r[scalar],
            gain[scalar],
            thickness[scalar],
            frequency_hz,
            r_te,
            r_tm);
        const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(value, frame.p_in);
        chain.frames[bounce] = frame;
        chain.value_in[bounce] = value;
        chain.e_s[bounce] = e_s;
        chain.e_p[bounce] = e_p;
        chain.r_te[bounce] = r_te;
        chain.r_tm[bounce] = r_tm;
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis, field::cplx_mul(r_te, e_s)),
            field::cplx_scale_real(frame.p_out, field::cplx_mul(r_tm, e_p)));
        outgoing = frame.reflected_direction;
        previous = hit;
    }
    const field::float3a target_value = load3f(target, index);
    const field::float3a final_offset = field::f3_sub(target_value, previous);
    const field::float3a final_direction = field::safe_normalize(
        final_offset, outgoing);
    total_length += field::safe_length(final_offset);
    const float wave_number =
        2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
    const float amplitude = 1.0f /
                            (2.0f * wave_number *
                             fmaxf(total_length, field::UTD_EPS));
    const field::Complex propagation = field::cplx_mul_real(
        field::cplx_exp_phase(
            transport::precise_neg_kd(wave_number, total_length)),
        amplitude);
    chain.value_chain = value;
    chain.rx_axis = field::stable_perp_basis(
        final_direction, load3f(rx_polarization, index));
    chain.propagation = propagation;
    // dP/df = P * (-1/k - j*L) * (2*pi/c); the phase term uses the raw length
    // (fmod has unit slope) and the amplitude term has no length dependence.
    const field::Complex dlog = field::cplx(-1.0f / wave_number, -total_length);
    chain.propagation_dfreq = field::cplx_mul_real(
        field::cplx_mul(propagation, dlog),
        2.0f * field::UTD_PI / transport::kSpeedOfLight);
    // dP/dL = P * (-[L >= EPS]/L_clamped - j*k): phase over the raw length,
    // amplitude over the clamped length (clamp_min subgradient convention).
    const float length_gate = total_length >= field::UTD_EPS
                                  ? 1.0f / fmaxf(total_length, field::UTD_EPS)
                                  : 0.0f;
    chain.propagation_dlength = field::cplx_mul(
        propagation, field::cplx(-length_gate, -wave_number));
    chain.total_length = total_length;
    chain.amplitude_scale = sqrtf(fmaxf(tx_power[index], 0.0f));
}

__global__ void reflection_sequence_backward_kernel(
    int64_t count,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_positions,
    const float* interaction_normals,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    float frequency_hz,
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    const float* grad_path_length,
    const float* grad_delay,
    float* grad_eps_r,
    float* grad_sigma_e,
    float* grad_gain,
    float* grad_thickness,
    float* grad_frequency,
    float* grad_source,
    float* grad_target,
    float* grad_positions,
    float* grad_normals) {
    const bool need_geometry = grad_source != nullptr;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        ReflectionChain chain;
        reflection_chain_eval(
            index, depth, source, target, interaction_positions,
            interaction_normals, tx_power, tx_polarization, rx_polarization,
            eps_r, sigma_e, mu_r, gain, thickness, frequency_hz, chain);

        const field::Complex3 value_final = field::c3_scale(
            chain.value_chain, chain.propagation);
        const field::Complex scalar = transport::complex3_dot_real(
            value_final, chain.rx_axis);
        const field::Complex path_field_value = field::cplx_mul_real(
            scalar, chain.amplitude_scale);
        field::Complex g_scalar = field::cplx_zero();
        field::Complex3 g_value = fold_output_cotangents(
            grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
            index, chain.rx_axis, path_field_value, chain.amplitude_scale,
            g_scalar);

        // Split value_final = P * value_chain.
        field::Complex g_propagation = field::cplx_zero();
        field::Complex3 g_chain = field::c3_zero();
        field::adj_cplx_mul(
            chain.value_chain.x, chain.propagation, g_value.x,
            g_chain.x, g_propagation);
        field::adj_cplx_mul(
            chain.value_chain.y, chain.propagation, g_value.y,
            g_chain.y, g_propagation);
        field::adj_cplx_mul(
            chain.value_chain.z, chain.propagation, g_value.z,
            g_chain.z, g_propagation);
        float g_freq = adj_dot(g_propagation, chain.propagation_dfreq);

        // Geometry adjoint state (plan 07 AD-2): the total path length
        // cotangent, the cotangent flowing onto the `previous` endpoint of
        // the segment being processed, the alternate-branch cotangent on the
        // bounce's outgoing direction, and the final receive-segment terms.
        const field::float3a e_z = field::make_f3(0.0f, 0.0f, 1.0f);
        float g_length = 0.0f;
        field::float3a g_carry = field::f3_zero();
        field::float3a g_outgoing = field::f3_zero();
        field::float3a g_hit_first = field::f3_zero();
        if (need_geometry) {
            g_length = adj_dot(g_propagation, chain.propagation_dlength);
            if (grad_path_length != nullptr)
                g_length += grad_path_length[index];
            if (grad_delay != nullptr)
                g_length += grad_delay[index] / transport::kSpeedOfLight;
            // rx_axis cotangent from the folded scalar (coefficient =
            // <value_final, rx_axis> with a real axis).
            field::float3a g_rx_axis = field::make_f3(
                field::cplx_adj_dot(g_scalar, value_final.x),
                field::cplx_adj_dot(g_scalar, value_final.y),
                field::cplx_adj_dot(g_scalar, value_final.z));
            const field::float3a previous_last = load_sequence3f(
                interaction_positions, index, depth - 1, depth);
            const field::float3a final_offset = field::f3_sub(
                load3f(target, index), previous_last);
            const field::float3a outgoing_last =
                chain.frames[depth - 1].reflected_direction;
            const field::float3a final_direction = field::safe_normalize(
                final_offset, outgoing_last);
            field::float3a g_final_direction = field::f3_zero();
            field::float3a g_pol_dump = field::f3_zero();
            field::adj_stable_perp_basis(
                final_direction, load3f(rx_polarization, index), g_rx_axis,
                g_final_direction, g_pol_dump);
            field::float3a g_final_offset = field::f3_zero();
            field::adj_safe_normalize(
                final_offset, outgoing_last, g_final_direction,
                g_final_offset, g_outgoing);
            ad::adj_safe_length(final_offset, g_length, g_final_offset);
            const int64_t base = index * 3;
            grad_target[base] = g_final_offset.x;
            grad_target[base + 1] = g_final_offset.y;
            grad_target[base + 2] = g_final_offset.z;
            g_carry = field::f3_neg(g_final_offset);
        }

        for (int64_t bounce = depth - 1; bounce >= 0; --bounce) {
            const transport::ReflectFrame& frame = chain.frames[bounce];
            // value_out = s_axis*(r_te*e_s) + p_out*(r_tm*e_p): recover the
            // complex coefficient cotangents and (for geometry) the real
            // basis cotangents in one adjoint step each.
            field::float3a g_s_axis = field::f3_zero();
            field::float3a g_p_in = field::f3_zero();
            field::float3a g_p_out = field::f3_zero();
            field::Complex gs = field::cplx_zero();
            field::Complex gp = field::cplx_zero();
            field::adj_cplx_scale_real(
                frame.s_axis,
                field::cplx_mul(chain.r_te[bounce], chain.e_s[bounce]),
                g_chain, g_s_axis, gs);
            field::adj_cplx_scale_real(
                frame.p_out,
                field::cplx_mul(chain.r_tm[bounce], chain.e_p[bounce]),
                g_chain, g_p_out, gp);
            field::Complex g_r_te = field::cplx_zero();
            field::Complex g_r_tm = field::cplx_zero();
            field::Complex g_e_s = field::cplx_zero();
            field::Complex g_e_p = field::cplx_zero();
            field::adj_cplx_mul(
                chain.r_te[bounce], chain.e_s[bounce], gs, g_r_te, g_e_s);
            field::adj_cplx_mul(
                chain.r_tm[bounce], chain.e_p[bounce], gp, g_r_tm, g_e_p);
            // e_s = <value_in, s_axis>, e_p = <value_in, p_in>.
            field::Complex3 g_value_in = field::c3_zero();
            field::adj_cplx_dot_real(
                chain.value_in[bounce], frame.s_axis, g_e_s, g_value_in,
                g_s_axis);
            field::adj_cplx_dot_real(
                chain.value_in[bounce], frame.p_in, g_e_p, g_value_in, g_p_in);
            g_chain = g_value_in;

            // Convert (g_r_te, g_r_tm) to input gradients with one dual
            // slab_fresnel evaluation per requested basis direction.
            const int64_t scalar_index = index * depth + bounce;
            const float b_eps = eps_r[scalar_index];
            const float b_sigma = sigma_e[scalar_index];
            const float b_mu = mu_r[scalar_index];
            const float b_gain = gain[scalar_index];
            const float b_thickness = thickness[scalar_index];
            DualC r_te_dual;
            DualC r_tm_dual;
            if (grad_eps_r != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_eps_r[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_sigma_e != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_sigma_e[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_gain != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_gain[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_thickness != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_thickness[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_frequency != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                    r_te_dual, r_tm_dual);
                g_freq +=
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (!need_geometry)
                continue;

            // Geometry enters the Fresnel only through cos_theta.
            ad::slab_fresnel_dual(
                frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                frequency_hz, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                r_te_dual, r_tm_dual);
            const float g_cos_theta =
                adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);

            // Replay the segment geometry of this bounce.
            const field::float3a previous =
                bounce > 0 ? load_sequence3f(
                                 interaction_positions, index, bounce - 1, depth)
                           : load3f(source, index);
            const field::float3a hit = load_sequence3f(
                interaction_positions, index, bounce, depth);
            const field::float3a segment = field::f3_sub(hit, previous);
            const field::float3a outgoing_previous =
                bounce > 0 ? chain.frames[bounce - 1].reflected_direction
                           : field::safe_normalize(
                                 field::f3_sub(
                                     load_sequence3f(
                                         interaction_positions, index, 0, depth),
                                     load3f(source, index)),
                                 e_z);
            const field::float3a incident_argument = field::safe_normalize(
                segment, outgoing_previous);
            const field::float3a raw_normal = load_sequence3f(
                interaction_normals, index, bounce, depth);

            // Frame outputs -> incident direction + raw normal. The frozen
            // normal flip contributes only its sign; g_outgoing carries the
            // downstream alternate-branch cotangent of frame.reflected_direction.
            field::float3a g_incident = field::f3_zero();
            field::float3a g_normal_raw = field::f3_zero();
            ad::adj_reflect_frame(
                incident_argument, raw_normal, g_s_axis, g_p_in, g_p_out,
                g_outgoing, g_cos_theta, g_incident, g_normal_raw);
            const int64_t normal_base = scalar_index * 3;
            grad_normals[normal_base] = g_normal_raw.x;
            grad_normals[normal_base + 1] = g_normal_raw.y;
            grad_normals[normal_base + 2] = g_normal_raw.z;

            // incident = safe_normalize(segment, outgoing_previous); the
            // segment also carries the path-length cotangent.
            field::float3a g_segment = field::f3_zero();
            field::float3a g_outgoing_previous = field::f3_zero();
            field::adj_safe_normalize(
                segment, outgoing_previous, g_incident, g_segment,
                g_outgoing_previous);
            ad::adj_safe_length(segment, g_length, g_segment);
            field::float3a g_hit = field::f3_add(g_carry, g_segment);
            g_carry = field::f3_neg(g_segment);
            g_outgoing = g_outgoing_previous;
            if (bounce > 0) {
                const int64_t hit_base = scalar_index * 3;
                grad_positions[hit_base] = g_hit.x;
                grad_positions[hit_base + 1] = g_hit.y;
                grad_positions[hit_base + 2] = g_hit.z;
            } else {
                g_hit_first = g_hit;
            }
        }
        if (grad_frequency != nullptr)
            atomicAdd(grad_frequency, g_freq);
        if (!need_geometry)
            continue;

        // Launch segment: value_0 = tx_axis * (1 + 0j) with tx_axis built on
        // the pre-loop incident direction (alternate e_z); the bounce-0
        // normalize falls back onto this same direction, so g_outgoing now
        // holds that branch's cotangent.
        const field::float3a source_value = load3f(source, index);
        const field::float3a first_hit = load_sequence3f(
            interaction_positions, index, 0, depth);
        const field::float3a segment_pre = field::f3_sub(first_hit, source_value);
        const field::float3a incident_pre = field::safe_normalize(segment_pre, e_z);
        field::float3a g_tx_axis = field::make_f3(
            g_chain.x.re, g_chain.y.re, g_chain.z.re);
        field::float3a g_incident_pre = g_outgoing;
        field::float3a g_pol_dump = field::f3_zero();
        field::adj_stable_perp_basis(
            incident_pre, load3f(tx_polarization, index), g_tx_axis,
            g_incident_pre, g_pol_dump);
        field::float3a g_segment_pre = field::f3_zero();
        field::float3a g_ez_dump = field::f3_zero();
        field::adj_safe_normalize(
            segment_pre, e_z, g_incident_pre, g_segment_pre, g_ez_dump);
        g_hit_first = field::f3_add(g_hit_first, g_segment_pre);
        const field::float3a g_source_total = field::f3_sub(
            g_carry, g_segment_pre);
        const int64_t base = index * 3;
        const int64_t first_base = index * depth * 3;
        grad_positions[first_base] = g_hit_first.x;
        grad_positions[first_base + 1] = g_hit_first.y;
        grad_positions[first_base + 2] = g_hit_first.z;
        grad_source[base] = g_source_total.x;
        grad_source[base + 1] = g_source_total.y;
        grad_source[base + 2] = g_source_total.z;
    }
}

__device__ __forceinline__ ad::DualF3 load_dual3f(
    const float* values, const float* tangents, int64_t index) {
    return {
        load3f(values, index),
        tangents != nullptr ? load3f(tangents, index) : field::f3_zero()};
}

__device__ __forceinline__ ad::DualF3 load_dual_sequence3f(
    const float* values,
    const float* tangents,
    int64_t index,
    int64_t bounce,
    int64_t depth) {
    return {
        load_sequence3f(values, index, bounce, depth),
        tangents != nullptr ? load_sequence3f(tangents, index, bounce, depth)
                            : field::f3_zero()};
}

__global__ void reflection_sequence_jvp_kernel(
    int64_t count,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_positions,
    const float* interaction_normals,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    float frequency_hz,
    const float* tangent_eps_r,
    const float* tangent_sigma_e,
    const float* tangent_gain,
    const float* tangent_thickness,
    float tangent_frequency,
    const float* tangent_source,
    const float* tangent_target,
    const float* tangent_positions,
    const float* tangent_normals,
    c10::complex<float>* t_field_vector,
    c10::complex<float>* t_coefficient,
    c10::complex<float>* t_path_field,
    float* t_path_gain,
    float* t_path_length,
    float* t_delay) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        // Full forward-mode dual sweep mirroring reflection_sequence_kernel
        // (and reflection_chain_eval) step by step: the dual vector helpers
        // replay the same utd formulas, so material/frequency-only seeds
        // reduce exactly to the AD-1 tangent chain while geometry seeds move
        // the frames, the incidence cosines and the path length.
        const ad::DualF3 e_z = ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f));
        ad::DualF3 previous = load_dual3f(source, tangent_source, index);
        const ad::DualF3 first_hit = load_dual_sequence3f(
            interaction_positions, tangent_positions, index, 0, depth);
        const ad::DualF3 incident_pre = ad::dual_safe_normalize(
            ad::df3_sub(first_hit, previous), e_z);
        const ad::DualF3 tx_axis = ad::dual_stable_perp_basis(
            incident_pre, ad::df3_const(load3f(tx_polarization, index)));
        field::Complex3 value = field::cplx_scale_real(
            tx_axis.v, field::cplx(1.0f, 0.0f));
        field::Complex3 d_value = field::cplx_scale_real(
            tx_axis.d, field::cplx(1.0f, 0.0f));
        ad::DualF total_length = {0.0f, 0.0f};
        ad::DualF3 outgoing = incident_pre;
        for (int64_t bounce = 0; bounce < depth; ++bounce) {
            const ad::DualF3 hit = load_dual_sequence3f(
                interaction_positions, tangent_positions, index, bounce, depth);
            const ad::DualF3 segment = ad::df3_sub(hit, previous);
            const ad::DualF3 incident = ad::dual_safe_normalize(
                segment, outgoing);
            const ad::DualF segment_length = ad::dual_safe_length(segment);
            total_length.v += segment_length.v;
            total_length.d += segment_length.d;
            const ad::DualF3 raw_normal = load_dual_sequence3f(
                interaction_normals, tangent_normals, index, bounce, depth);
            const ad::DualReflectFrame frame = ad::dual_reflect_frame(
                incident, raw_normal);
            const int64_t scalar_index = index * depth + bounce;
            DualC r_te_dual;
            DualC r_tm_dual;
            ad::slab_fresnel_dual(
                frame.cos_theta.v,
                eps_r[scalar_index],
                sigma_e[scalar_index],
                mu_r[scalar_index],
                gain[scalar_index],
                thickness[scalar_index],
                frequency_hz,
                frame.cos_theta.d,
                tangent_eps_r != nullptr ? tangent_eps_r[scalar_index] : 0.0f,
                tangent_sigma_e != nullptr ? tangent_sigma_e[scalar_index] : 0.0f,
                tangent_gain != nullptr ? tangent_gain[scalar_index] : 0.0f,
                tangent_thickness != nullptr ? tangent_thickness[scalar_index]
                                             : 0.0f,
                tangent_frequency,
                r_te_dual,
                r_tm_dual);
            const field::Complex e_s = transport::complex3_dot_real(
                value, frame.s_axis.v);
            const field::Complex e_p = transport::complex3_dot_real(
                value, frame.p_in.v);
            const field::Complex d_e_s = field::cplx_add(
                transport::complex3_dot_real(d_value, frame.s_axis.v),
                transport::complex3_dot_real(value, frame.s_axis.d));
            const field::Complex d_e_p = field::cplx_add(
                transport::complex3_dot_real(d_value, frame.p_in.v),
                transport::complex3_dot_real(value, frame.p_in.d));
            const field::Complex w_te = field::cplx_mul(r_te_dual.v, e_s);
            const field::Complex w_tm = field::cplx_mul(r_tm_dual.v, e_p);
            const field::Complex d_w_te = field::cplx_add(
                field::cplx_mul(r_te_dual.d, e_s),
                field::cplx_mul(r_te_dual.v, d_e_s));
            const field::Complex d_w_tm = field::cplx_add(
                field::cplx_mul(r_tm_dual.d, e_p),
                field::cplx_mul(r_tm_dual.v, d_e_p));
            value = field::c3_add(
                field::cplx_scale_real(frame.s_axis.v, w_te),
                field::cplx_scale_real(frame.p_out.v, w_tm));
            d_value = field::c3_add(
                field::c3_add(
                    field::cplx_scale_real(frame.s_axis.d, w_te),
                    field::cplx_scale_real(frame.s_axis.v, d_w_te)),
                field::c3_add(
                    field::cplx_scale_real(frame.p_out.d, w_tm),
                    field::cplx_scale_real(frame.p_out.v, d_w_tm)));
            outgoing = frame.reflected_direction;
            previous = hit;
        }
        const ad::DualF3 target_dual = load_dual3f(target, tangent_target, index);
        const ad::DualF3 final_offset = ad::df3_sub(target_dual, previous);
        const ad::DualF3 final_direction = ad::dual_safe_normalize(
            final_offset, outgoing);
        const ad::DualF final_length = ad::dual_safe_length(final_offset);
        total_length.v += final_length.v;
        total_length.d += final_length.d;
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
        const float amplitude = 1.0f /
                                (2.0f * wave_number *
                                 fmaxf(total_length.v, field::UTD_EPS));
        const field::Complex propagation = field::cplx_mul_real(
            field::cplx_exp_phase(
                transport::precise_neg_kd(wave_number, total_length.v)),
            amplitude);
        const field::Complex dlog_freq = field::cplx(
            -1.0f / wave_number, -total_length.v);
        const field::Complex propagation_dfreq = field::cplx_mul_real(
            field::cplx_mul(propagation, dlog_freq),
            2.0f * field::UTD_PI / transport::kSpeedOfLight);
        const float length_gate =
            total_length.v >= field::UTD_EPS
                ? 1.0f / fmaxf(total_length.v, field::UTD_EPS)
                : 0.0f;
        const field::Complex propagation_dlength = field::cplx_mul(
            propagation, field::cplx(-length_gate, -wave_number));
        const field::Complex d_propagation = field::cplx_add(
            field::cplx_mul_real(propagation_dfreq, tangent_frequency),
            field::cplx_mul_real(propagation_dlength, total_length.d));
        const ad::DualF3 rx_axis = ad::dual_stable_perp_basis(
            final_direction, ad::df3_const(load3f(rx_polarization, index)));
        const field::Complex3 value_final = field::c3_scale(value, propagation);
        const field::Complex3 d_final = field::c3_add(
            field::c3_scale(d_value, propagation),
            field::c3_scale(value, d_propagation));
        write_output_tangents(
            index, value_final, d_final, rx_axis.v, rx_axis.d,
            sqrtf(fmaxf(tx_power[index], 0.0f)), total_length.d,
            t_field_vector, t_coefficient, t_path_field, t_path_gain,
            t_path_length, t_delay);
    }
}

// ---------------------------------------------------------------------------
// Transmission sequence (CSR layer eps_r / sigma_e / thickness and frequency
// are differentiable; gradients on the shared layer store accumulate
// atomically across paths).
// ---------------------------------------------------------------------------

struct TransmissionChain {
    transport::WallFrame frames[kMaxAdDepth];
    field::Complex3 value_in[kMaxAdDepth];  // field entering each wall
    field::Complex e_s[kMaxAdDepth];
    field::Complex e_p[kMaxAdDepth];
    field::Complex t_te[kMaxAdDepth];
    field::Complex t_tm[kMaxAdDepth];
    float wall_thickness[kMaxAdDepth];
    int wall_material[kMaxAdDepth];  // -1 for skipped slots
    field::float3a direction;
    field::float3a rx_axis;
    field::Complex3 value_chain;
    field::Complex propagation;
    field::Complex propagation_dfreq;
    field::Complex propagation_dcarrier;
    field::Complex propagation_dtotal;  // amplitude spread over the raw length
    float total_length;
    float carrier_length;
    float amplitude_scale;
    bool path_valid;
};

__device__ void transmission_chain_eval(
    int64_t index,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_normals,
    const int* interaction_material_id,
    const bool* interaction_valid,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const em::LayerView& layers_base,
    int64_t material_count,
    float frequency_hz,
    TransmissionChain& chain) {
    const field::float3a source_value = load3f(source, index);
    const field::float3a target_value = load3f(target, index);
    const field::float3a offset = field::f3_sub(target_value, source_value);
    chain.total_length = field::safe_length(offset);
    chain.direction = field::safe_normalize(
        offset, field::make_f3(0.0f, 0.0f, 1.0f));
    const field::float3a tx_axis = field::stable_perp_basis(
        chain.direction, load3f(tx_polarization, index));
    field::Complex3 value = field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f));
    float carrier_length = chain.total_length;
    chain.path_valid = true;
    for (int64_t wall = 0; wall < depth; ++wall) {
        chain.wall_material[wall] = -1;
        const int64_t scalar = index * depth + wall;
        if (!interaction_valid[scalar])
            continue;
        const int material = interaction_material_id[scalar];
        if (material < 0 || static_cast<int64_t>(material) >= material_count) {
            chain.path_valid = false;
            break;
        }
        const transport::WallFrame frame = transport::wall_frame(
            chain.direction,
            load_sequence3f(interaction_normals, index, wall, depth));
        em::LayerView layers = layers_base;
        layers.material = material;
        const em::StackRT te = em::stack_rt(
            frame.cos_theta, layers, frequency_hz, em::kPolTE);
        const em::StackRT tm = em::stack_rt(
            frame.cos_theta, layers, frequency_hz, em::kPolTM);
        const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(value, frame.p_axis);
        chain.frames[wall] = frame;
        chain.value_in[wall] = value;
        chain.e_s[wall] = e_s;
        chain.e_p[wall] = e_p;
        chain.t_te[wall] = te.t;
        chain.t_tm[wall] = tm.t;
        chain.wall_material[wall] = material;
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis, field::cplx_mul(te.t, e_s)),
            field::cplx_scale_real(frame.p_axis, field::cplx_mul(tm.t, e_p)));
        float wall_thickness = 0.0f;
        const int first = layers_base.layer_offset[material];
        const int layers_in_wall = layers_base.layer_count[material];
        for (int layer = 0; layer < layers_in_wall; ++layer)
            wall_thickness += fmaxf(
                layers_base.layer_thickness_m[first + layer], 0.0f);
        chain.wall_thickness[wall] = wall_thickness;
        carrier_length -= wall_thickness * frame.cos_theta;
    }
    const float wave_number =
        2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
    const float amplitude = 1.0f /
                            (2.0f * wave_number *
                             fmaxf(chain.total_length, field::UTD_EPS));
    const field::Complex propagation = field::cplx_mul_real(
        field::cplx_exp_phase(
            transport::precise_neg_kd(wave_number, carrier_length)),
        amplitude);
    chain.value_chain = value;
    chain.rx_axis = field::stable_perp_basis(
        chain.direction, load3f(rx_polarization, index));
    chain.propagation = propagation;
    chain.carrier_length = carrier_length;
    // dP/df = P * (-1/k - j*carrier) * (2*pi/c); the amplitude spreads over
    // the full straight length (geometry, handled via dP/dtotal), the phase
    // runs over the carrier length (thickness and cos_theta dependent,
    // handled via dP/dcarrier = -j*k*P).
    const field::Complex dlog = field::cplx(-1.0f / wave_number, -carrier_length);
    chain.propagation_dfreq = field::cplx_mul_real(
        field::cplx_mul(propagation, dlog),
        2.0f * field::UTD_PI / transport::kSpeedOfLight);
    chain.propagation_dcarrier = field::cplx(
        propagation.im * wave_number, -propagation.re * wave_number);
    const float length_gate =
        chain.total_length >= field::UTD_EPS
            ? 1.0f / fmaxf(chain.total_length, field::UTD_EPS)
            : 0.0f;
    chain.propagation_dtotal = field::cplx_mul_real(propagation, -length_gate);
    chain.amplitude_scale = sqrtf(fmaxf(tx_power[index], 0.0f));
}

struct ZeroSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};

struct BasisSeed {
    int slot;
    int param;  // 0 thickness, 1 eps, 2 sigma
    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed seed{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (param == 0)
                seed.d_thickness = 1.0f;
            else if (param == 1)
                seed.d_eps = 1.0f;
            else
                seed.d_sigma = 1.0f;
        }
        return seed;
    }
};

struct TangentSeed {
    const float* t_thickness;
    const float* t_eps;
    const float* t_sigma;
    __device__ ad::LayerSeed operator()(int query) const {
        return {
            t_thickness != nullptr ? t_thickness[query] : 0.0f,
            t_eps != nullptr ? t_eps[query] : 0.0f,
            t_sigma != nullptr ? t_sigma[query] : 0.0f};
    }
};

__global__ void transmission_sequence_backward_kernel(
    int64_t count,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_normals,
    const int* interaction_material_id,
    const bool* interaction_valid,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const int* layer_offset,
    const int* layer_count,
    const float* layer_thickness_m,
    const float* layer_eps_r,
    const float* layer_sigma_e,
    const float* layer_mu_r,
    int64_t material_count,
    float frequency_hz,
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    const float* grad_path_length,
    const float* grad_delay,
    float* grad_layer_thickness,
    float* grad_layer_eps_r,
    float* grad_layer_sigma_e,
    float* grad_frequency,
    float* grad_source,
    float* grad_target,
    float* grad_normals) {
    const em::LayerView layers_base{
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        0,
    };
    const bool need_geometry = grad_source != nullptr;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        TransmissionChain chain;
        transmission_chain_eval(
            index, depth, source, target, interaction_normals,
            interaction_material_id, interaction_valid, tx_power,
            tx_polarization, rx_polarization, layers_base, material_count,
            frequency_hz, chain);
        if (!chain.path_valid)
            continue;  // forward zeroed the outputs; every gradient is zero

        const field::Complex3 value_final = field::c3_scale(
            chain.value_chain, chain.propagation);
        const field::Complex scalar = transport::complex3_dot_real(
            value_final, chain.rx_axis);
        const field::Complex path_field_value = field::cplx_mul_real(
            scalar, chain.amplitude_scale);
        field::Complex g_scalar = field::cplx_zero();
        field::Complex3 g_value = fold_output_cotangents(
            grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
            index, chain.rx_axis, path_field_value, chain.amplitude_scale,
            g_scalar);

        field::Complex g_propagation = field::cplx_zero();
        field::Complex3 g_chain = field::c3_zero();
        field::adj_cplx_mul(
            chain.value_chain.x, chain.propagation, g_value.x,
            g_chain.x, g_propagation);
        field::adj_cplx_mul(
            chain.value_chain.y, chain.propagation, g_value.y,
            g_chain.y, g_propagation);
        field::adj_cplx_mul(
            chain.value_chain.z, chain.propagation, g_value.z,
            g_chain.z, g_propagation);
        float g_freq = adj_dot(g_propagation, chain.propagation_dfreq);
        const float g_carrier = adj_dot(
            g_propagation, chain.propagation_dcarrier);

        // Geometry cotangents (plan 07 AD-2). The straight length L feeds the
        // amplitude spread, the carrier start (carrier = L - sum_w d_w*cos_w)
        // and the path_length_m / delay_s outputs; the shared ray direction
        // feeds the tx/rx bases and every wall frame.
        field::float3a g_direction = field::f3_zero();
        float g_total_length = 0.0f;
        if (need_geometry) {
            g_total_length = adj_dot(g_propagation, chain.propagation_dtotal) +
                             g_carrier;
            if (grad_path_length != nullptr)
                g_total_length += grad_path_length[index];
            if (grad_delay != nullptr)
                g_total_length += grad_delay[index] / transport::kSpeedOfLight;
            const field::float3a g_rx_axis = field::make_f3(
                field::cplx_adj_dot(g_scalar, value_final.x),
                field::cplx_adj_dot(g_scalar, value_final.y),
                field::cplx_adj_dot(g_scalar, value_final.z));
            field::float3a g_pol_dump = field::f3_zero();
            field::adj_stable_perp_basis(
                chain.direction, load3f(rx_polarization, index), g_rx_axis,
                g_direction, g_pol_dump);
        }

        for (int64_t wall = depth - 1; wall >= 0; --wall) {
            const int material = chain.wall_material[wall];
            if (material < 0)
                continue;
            const transport::WallFrame& frame = chain.frames[wall];
            field::float3a g_s_axis = field::f3_zero();
            field::float3a g_p_axis = field::f3_zero();
            field::Complex gs = field::cplx_zero();
            field::Complex gp = field::cplx_zero();
            field::adj_cplx_scale_real(
                frame.s_axis,
                field::cplx_mul(chain.t_te[wall], chain.e_s[wall]),
                g_chain, g_s_axis, gs);
            field::adj_cplx_scale_real(
                frame.p_axis,
                field::cplx_mul(chain.t_tm[wall], chain.e_p[wall]),
                g_chain, g_p_axis, gp);
            field::Complex g_t_te = field::cplx_zero();
            field::Complex g_t_tm = field::cplx_zero();
            field::Complex g_e_s = field::cplx_zero();
            field::Complex g_e_p = field::cplx_zero();
            field::adj_cplx_mul(
                chain.t_te[wall], chain.e_s[wall], gs, g_t_te, g_e_s);
            field::adj_cplx_mul(
                chain.t_tm[wall], chain.e_p[wall], gp, g_t_tm, g_e_p);
            field::Complex3 g_value_in = field::c3_zero();
            field::adj_cplx_dot_real(
                chain.value_in[wall], frame.s_axis, g_e_s, g_value_in, g_s_axis);
            field::adj_cplx_dot_real(
                chain.value_in[wall], frame.p_axis, g_e_p, g_value_in, g_p_axis);
            g_chain = g_value_in;

            em::LayerView layers = layers_base;
            layers.material = material;
            const int first = layer_offset[material];
            const int layers_in_wall = layer_count[material];
            for (int layer = 0; layer < layers_in_wall; ++layer) {
                const int slot = first + layer;
                for (int param = 0; param < 3; ++param) {
                    float* destination = param == 0 ? grad_layer_thickness
                                         : param == 1 ? grad_layer_eps_r
                                                      : grad_layer_sigma_e;
                    if (destination == nullptr)
                        continue;
                    const BasisSeed seed{slot, param};
                    const ad::DualStackRT te = ad::stack_rt_dual(
                        frame.cos_theta, layers, frequency_hz, 0.0f, 0.0f,
                        em::kPolTE, seed);
                    const ad::DualStackRT tm = ad::stack_rt_dual(
                        frame.cos_theta, layers, frequency_hz, 0.0f, 0.0f,
                        em::kPolTM, seed);
                    float grad = adj_dot(g_t_te, te.t.d) + adj_dot(g_t_tm, tm.t.d);
                    if (param == 0 && layer_thickness_m[slot] >= 0.0f) {
                        // Carrier phase runs over L - sum_w d_w * cos(theta_w).
                        grad += g_carrier * (-frame.cos_theta);
                    }
                    atomicAdd(destination + slot, grad);
                }
            }
            if (grad_frequency != nullptr) {
                const ZeroSeed zero_seed;
                const ad::DualStackRT te = ad::stack_rt_dual(
                    frame.cos_theta, layers, frequency_hz, 0.0f, 1.0f,
                    em::kPolTE, zero_seed);
                const ad::DualStackRT tm = ad::stack_rt_dual(
                    frame.cos_theta, layers, frequency_hz, 0.0f, 1.0f,
                    em::kPolTM, zero_seed);
                g_freq += adj_dot(g_t_te, te.t.d) + adj_dot(g_t_tm, tm.t.d);
            }
            if (!need_geometry)
                continue;

            // Geometry enters this wall through cos_theta (Fresnel stack and
            // carrier chord) and through the s/p frame.
            const ZeroSeed zero_seed;
            const ad::DualStackRT te_cos = ad::stack_rt_dual(
                frame.cos_theta, layers, frequency_hz, 1.0f, 0.0f,
                em::kPolTE, zero_seed);
            const ad::DualStackRT tm_cos = ad::stack_rt_dual(
                frame.cos_theta, layers, frequency_hz, 1.0f, 0.0f,
                em::kPolTM, zero_seed);
            float g_cos_theta = adj_dot(g_t_te, te_cos.t.d) +
                                adj_dot(g_t_tm, tm_cos.t.d);
            g_cos_theta += g_carrier * (-chain.wall_thickness[wall]);
            field::float3a g_normal_raw = field::f3_zero();
            ad::adj_wall_frame(
                chain.direction,
                load_sequence3f(interaction_normals, index, wall, depth),
                g_s_axis, g_p_axis, g_cos_theta, g_direction, g_normal_raw);
            const int64_t normal_base = (index * depth + wall) * 3;
            grad_normals[normal_base] = g_normal_raw.x;
            grad_normals[normal_base + 1] = g_normal_raw.y;
            grad_normals[normal_base + 2] = g_normal_raw.z;
        }
        if (grad_frequency != nullptr)
            atomicAdd(grad_frequency, g_freq);
        if (!need_geometry)
            continue;

        // tx_axis cotangent (value_0 = tx_axis * (1 + 0j)), then the shared
        // straight offset: direction = safe_normalize(target - source, e_z)
        // and L = safe_length(target - source).
        const field::float3a g_tx_axis = field::make_f3(
            g_chain.x.re, g_chain.y.re, g_chain.z.re);
        field::float3a g_pol_dump = field::f3_zero();
        field::adj_stable_perp_basis(
            chain.direction, load3f(tx_polarization, index), g_tx_axis,
            g_direction, g_pol_dump);
        const field::float3a offset = field::f3_sub(
            load3f(target, index), load3f(source, index));
        field::float3a g_offset = field::f3_zero();
        field::float3a g_ez_dump = field::f3_zero();
        field::adj_safe_normalize(
            offset, field::make_f3(0.0f, 0.0f, 1.0f), g_direction, g_offset,
            g_ez_dump);
        ad::adj_safe_length(offset, g_total_length, g_offset);
        const int64_t base = index * 3;
        grad_target[base] = g_offset.x;
        grad_target[base + 1] = g_offset.y;
        grad_target[base + 2] = g_offset.z;
        grad_source[base] = -g_offset.x;
        grad_source[base + 1] = -g_offset.y;
        grad_source[base + 2] = -g_offset.z;
    }
}

__global__ void transmission_sequence_jvp_kernel(
    int64_t count,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_normals,
    const int* interaction_material_id,
    const bool* interaction_valid,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const int* layer_offset,
    const int* layer_count,
    const float* layer_thickness_m,
    const float* layer_eps_r,
    const float* layer_sigma_e,
    const float* layer_mu_r,
    int64_t material_count,
    float frequency_hz,
    const float* tangent_layer_thickness,
    const float* tangent_layer_eps_r,
    const float* tangent_layer_sigma_e,
    float tangent_frequency,
    const float* tangent_source,
    const float* tangent_target,
    const float* tangent_normals,
    c10::complex<float>* t_field_vector,
    c10::complex<float>* t_coefficient,
    c10::complex<float>* t_path_field,
    float* t_path_gain,
    float* t_path_length,
    float* t_delay) {
    const em::LayerView layers_base{
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        0,
    };
    const TangentSeed tangent_seed{
        tangent_layer_thickness, tangent_layer_eps_r, tangent_layer_sigma_e};
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        TransmissionChain chain;
        transmission_chain_eval(
            index, depth, source, target, interaction_normals,
            interaction_material_id, interaction_valid, tx_power,
            tx_polarization, rx_polarization, layers_base, material_count,
            frequency_hz, chain);
        if (!chain.path_valid) {
            const int64_t base = index * 3;
            const c10::complex<float> zero(0.0f, 0.0f);
            t_field_vector[base] = zero;
            t_field_vector[base + 1] = zero;
            t_field_vector[base + 2] = zero;
            t_coefficient[index] = zero;
            t_path_field[index] = zero;
            t_path_gain[index] = 0.0f;
            t_path_length[index] = 0.0f;
            t_delay[index] = 0.0f;
            continue;
        }

        // Straight-ray geometry duals shared by every wall: the offset feeds
        // the direction (tx/rx bases and wall frames) and the raw length
        // (amplitude spread, carrier start, path_length_m / delay_s).
        const ad::DualF3 e_z = ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f));
        const ad::DualF3 offset = ad::df3_sub(
            load_dual3f(target, tangent_target, index),
            load_dual3f(source, tangent_source, index));
        const ad::DualF total_length = ad::dual_safe_length(offset);
        const ad::DualF3 direction = ad::dual_safe_normalize(offset, e_z);
        const ad::DualF3 tx_axis = ad::dual_stable_perp_basis(
            direction, ad::df3_const(load3f(tx_polarization, index)));
        const ad::DualF3 rx_axis = ad::dual_stable_perp_basis(
            direction, ad::df3_const(load3f(rx_polarization, index)));
        field::Complex3 d_value = field::cplx_scale_real(
            tx_axis.d, field::cplx(1.0f, 0.0f));
        float d_carrier = total_length.d;
        for (int64_t wall = 0; wall < depth; ++wall) {
            const int material = chain.wall_material[wall];
            if (material < 0)
                continue;
            const ad::DualWallFrame frame = ad::dual_wall_frame(
                direction,
                load_dual_sequence3f(
                    interaction_normals, tangent_normals, index, wall, depth));
            em::LayerView layers = layers_base;
            layers.material = material;
            const ad::DualStackRT te = ad::stack_rt_dual(
                frame.cos_theta.v, layers, frequency_hz, frame.cos_theta.d,
                tangent_frequency, em::kPolTE, tangent_seed);
            const ad::DualStackRT tm = ad::stack_rt_dual(
                frame.cos_theta.v, layers, frequency_hz, frame.cos_theta.d,
                tangent_frequency, em::kPolTM, tangent_seed);
            const field::Complex e_s = chain.e_s[wall];
            const field::Complex e_p = chain.e_p[wall];
            const field::Complex3 value_in = chain.value_in[wall];
            const field::Complex d_e_s = field::cplx_add(
                transport::complex3_dot_real(d_value, frame.s_axis.v),
                transport::complex3_dot_real(value_in, frame.s_axis.d));
            const field::Complex d_e_p = field::cplx_add(
                transport::complex3_dot_real(d_value, frame.p_axis.v),
                transport::complex3_dot_real(value_in, frame.p_axis.d));
            const field::Complex w_te = field::cplx_mul(te.t.v, e_s);
            const field::Complex w_tm = field::cplx_mul(tm.t.v, e_p);
            const field::Complex d_w_te = field::cplx_add(
                field::cplx_mul(te.t.d, e_s), field::cplx_mul(te.t.v, d_e_s));
            const field::Complex d_w_tm = field::cplx_add(
                field::cplx_mul(tm.t.d, e_p), field::cplx_mul(tm.t.v, d_e_p));
            d_value = field::c3_add(
                field::c3_add(
                    field::cplx_scale_real(frame.s_axis.d, w_te),
                    field::cplx_scale_real(frame.s_axis.v, d_w_te)),
                field::c3_add(
                    field::cplx_scale_real(frame.p_axis.d, w_tm),
                    field::cplx_scale_real(frame.p_axis.v, d_w_tm)));
            // Carrier chord: d(wall_thickness * cos_theta) with the clamped
            // per-layer thickness gates of the primal accumulation.
            float d_wall_thickness = 0.0f;
            if (tangent_layer_thickness != nullptr) {
                const int first = layer_offset[material];
                const int layers_in_wall = layer_count[material];
                for (int layer = 0; layer < layers_in_wall; ++layer) {
                    const int slot = first + layer;
                    if (layer_thickness_m[slot] >= 0.0f)
                        d_wall_thickness += tangent_layer_thickness[slot];
                }
            }
            d_carrier -= d_wall_thickness * frame.cos_theta.v +
                         chain.wall_thickness[wall] * frame.cos_theta.d;
        }
        const field::Complex d_propagation = field::cplx_add(
            field::cplx_add(
                field::cplx_mul_real(chain.propagation_dfreq, tangent_frequency),
                field::cplx_mul_real(chain.propagation_dcarrier, d_carrier)),
            field::cplx_mul_real(chain.propagation_dtotal, total_length.d));
        const field::Complex3 value_final = field::c3_scale(
            chain.value_chain, chain.propagation);
        const field::Complex3 d_final = field::c3_add(
            field::c3_scale(d_value, chain.propagation),
            field::c3_scale(chain.value_chain, d_propagation));
        write_output_tangents(
            index, value_final, d_final, chain.rx_axis, rx_axis.d,
            chain.amplitude_scale, total_length.d,
            t_field_vector, t_coefficient, t_path_field, t_path_gain,
            t_path_length, t_delay);
    }
}

// ---------------------------------------------------------------------------
// Host entries.
// ---------------------------------------------------------------------------

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

// Gradient accumulators must start at zero; allocate raw and memset on the
// current stream (same pattern as los.cu) instead of ATen zero-fill.
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

const at::Tensor* optional_grad(
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
const T* grad_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

void check_free_space_primal(
    const at::Tensor& source,
    const at::Tensor& target,
    const at::Tensor& tx_power,
    const at::Tensor& tx_polarization,
    const at::Tensor& rx_polarization,
    double frequency_hz,
    c10::ScalarType real_dtype) {
    using channel_native::check_tensor;
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {target, "target"},
             {tx_polarization, "tx_polarization"},
             {rx_polarization, "rx_polarization"}}) {
        check_tensor(named.first, named.second, real_dtype, 2);
        TORCH_CHECK(
            named.first.size(1) == 3, named.second, " must have shape (N, 3)");
    }
    check_tensor(tx_power, "tx_power", real_dtype, 1);
    const int64_t count = source.size(0);
    TORCH_CHECK(
        target.size(0) == count && tx_power.size(0) == count &&
            tx_polarization.size(0) == count && rx_polarization.size(0) == count,
        "free-space field tensors must have matching rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
}

}  // namespace

pybind11::dict cn_field_free_space_fwd64(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz) {
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, at::kDouble);
    const int64_t count = source.size(0);
    auto complex_options = source.options().dtype(at::kComplexDouble);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        free_space_fwd_kernel<double><<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<double>(),
            target.data_ptr<double>(),
            tx_power.data_ptr<double>(),
            tx_polarization.data_ptr<double>(),
            rx_polarization.data_ptr<double>(),
            frequency_hz,
            field_vector.data_ptr<c10::complex<double>>(),
            coefficient.data_ptr<c10::complex<double>>(),
            path_field.data_ptr<c10::complex<double>>(),
            path_gain.data_ptr<double>(),
            path_length.data_ptr<double>(),
            delay.data_ptr<double>(),
            direction.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["direction"] = direction;
    return out;
}

pybind11::dict cn_field_free_space_backward(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    bool need_grad_frequency,
    bool need_grad_geometry) {
    const c10::ScalarType real_dtype = source.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "field_free_space_backward supports float32 and float64");
    const c10::ScalarType complex_dtype =
        real_dtype == at::kFloat ? at::kComplexFloat : at::kComplexDouble;
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, real_dtype);
    const int64_t count = source.size(0);
    at::Tensor gfv_storage;
    at::Tensor gc_storage;
    at::Tensor gpf_storage;
    at::Tensor gpg_storage;
    at::Tensor gpl_storage;
    at::Tensor gd_storage;
    const at::Tensor* gfv = optional_grad(
        std::move(grad_field_vector), gfv_storage, "grad_field_vector",
        complex_dtype, {count, 3}, source);
    const at::Tensor* gc = optional_grad(
        std::move(grad_coefficient), gc_storage, "grad_coefficient",
        complex_dtype, {count}, source);
    const at::Tensor* gpf = optional_grad(
        std::move(grad_path_field), gpf_storage, "grad_path_field",
        complex_dtype, {count}, source);
    const at::Tensor* gpg = optional_grad(
        std::move(grad_path_gain), gpg_storage, "grad_path_gain",
        real_dtype, {count}, source);
    const at::Tensor* gpl = optional_grad(
        std::move(grad_path_length), gpl_storage, "grad_path_length",
        real_dtype, {count}, source);
    const at::Tensor* gd = optional_grad(
        std::move(grad_delay), gd_storage, "grad_delay",
        real_dtype, {count}, source);

    pybind11::dict out;
    if (!need_grad_frequency && !need_grad_geometry) {
        out["grad_frequency"] = pybind11::none();
        out["grad_source"] = pybind11::none();
        out["grad_target"] = pybind11::none();
        return out;
    }
    at::Tensor grad_frequency = need_grad_frequency
                                    ? zero_filled({1}, source.options())
                                    : at::Tensor();
    at::Tensor grad_source = need_grad_geometry
                                 ? zero_filled({count, 3}, source.options())
                                 : at::Tensor();
    at::Tensor grad_target = need_grad_geometry
                                 ? zero_filled({count, 3}, source.options())
                                 : at::Tensor();
    const bool any_grad = gfv != nullptr || gc != nullptr || gpf != nullptr ||
                          gpg != nullptr || gpl != nullptr || gd != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        if (real_dtype == at::kFloat) {
            free_space_backward_kernel<float>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<float>(),
                    target.data_ptr<float>(),
                    tx_power.data_ptr<float>(),
                    tx_polarization.data_ptr<float>(),
                    rx_polarization.data_ptr<float>(),
                    static_cast<float>(frequency_hz),
                    gfv ? gfv->data_ptr<c10::complex<float>>() : nullptr,
                    gc ? gc->data_ptr<c10::complex<float>>() : nullptr,
                    gpf ? gpf->data_ptr<c10::complex<float>>() : nullptr,
                    grad_ptr<float>(gpg),
                    grad_ptr<float>(gpl),
                    grad_ptr<float>(gd),
                    need_grad_frequency ? grad_frequency.data_ptr<float>()
                                        : nullptr,
                    need_grad_geometry ? grad_source.data_ptr<float>() : nullptr,
                    need_grad_geometry ? grad_target.data_ptr<float>() : nullptr);
        } else {
            free_space_backward_kernel<double>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<double>(),
                    target.data_ptr<double>(),
                    tx_power.data_ptr<double>(),
                    tx_polarization.data_ptr<double>(),
                    rx_polarization.data_ptr<double>(),
                    frequency_hz,
                    gfv ? gfv->data_ptr<c10::complex<double>>() : nullptr,
                    gc ? gc->data_ptr<c10::complex<double>>() : nullptr,
                    gpf ? gpf->data_ptr<c10::complex<double>>() : nullptr,
                    grad_ptr<double>(gpg),
                    grad_ptr<double>(gpl),
                    grad_ptr<double>(gd),
                    need_grad_frequency ? grad_frequency.data_ptr<double>()
                                        : nullptr,
                    need_grad_geometry ? grad_source.data_ptr<double>() : nullptr,
                    need_grad_geometry ? grad_target.data_ptr<double>()
                                       : nullptr);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    out["grad_frequency"] = need_grad_frequency
                                ? pybind11::cast(grad_frequency)
                                : pybind11::object(pybind11::none());
    out["grad_source"] = need_grad_geometry
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_target"] = need_grad_geometry
                             ? pybind11::cast(grad_target)
                             : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_field_free_space_jvp(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target) {
    const c10::ScalarType real_dtype = source.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "field_free_space_jvp supports float32 and float64");
    const c10::ScalarType complex_dtype =
        real_dtype == at::kFloat ? at::kComplexFloat : at::kComplexDouble;
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, real_dtype);
    const int64_t count = source.size(0);
    at::Tensor ts_storage;
    at::Tensor tt_storage;
    const at::Tensor* t_source = optional_grad(
        std::move(tangent_source), ts_storage, "tangent_source",
        real_dtype, {count, 3}, source);
    const at::Tensor* t_target = optional_grad(
        std::move(tangent_target), tt_storage, "tangent_target",
        real_dtype, {count, 3}, source);
    auto complex_options = source.options().dtype(complex_dtype);
    auto t_field_vector = at::empty({count, 3}, complex_options);
    auto t_coefficient = at::empty({count}, complex_options);
    auto t_path_field = at::empty({count}, complex_options);
    auto t_path_gain = at::empty({count}, source.options());
    auto t_path_length = at::empty({count}, source.options());
    auto t_delay = at::empty({count}, source.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        if (real_dtype == at::kFloat) {
            free_space_jvp_kernel<float>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<float>(),
                    target.data_ptr<float>(),
                    tx_power.data_ptr<float>(),
                    tx_polarization.data_ptr<float>(),
                    rx_polarization.data_ptr<float>(),
                    static_cast<float>(frequency_hz),
                    static_cast<float>(tangent_frequency),
                    grad_ptr<float>(t_source),
                    grad_ptr<float>(t_target),
                    t_field_vector.data_ptr<c10::complex<float>>(),
                    t_coefficient.data_ptr<c10::complex<float>>(),
                    t_path_field.data_ptr<c10::complex<float>>(),
                    t_path_gain.data_ptr<float>(),
                    t_path_length.data_ptr<float>(),
                    t_delay.data_ptr<float>());
        } else {
            free_space_jvp_kernel<double>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<double>(),
                    target.data_ptr<double>(),
                    tx_power.data_ptr<double>(),
                    tx_polarization.data_ptr<double>(),
                    rx_polarization.data_ptr<double>(),
                    frequency_hz,
                    tangent_frequency,
                    grad_ptr<double>(t_source),
                    grad_ptr<double>(t_target),
                    t_field_vector.data_ptr<c10::complex<double>>(),
                    t_coefficient.data_ptr<c10::complex<double>>(),
                    t_path_field.data_ptr<c10::complex<double>>(),
                    t_path_gain.data_ptr<double>(),
                    t_path_length.data_ptr<double>(),
                    t_delay.data_ptr<double>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = t_field_vector;
    out["coefficient"] = t_coefficient;
    out["path_field"] = t_path_field;
    out["path_gain"] = t_path_gain;
    out["path_length_m"] = t_path_length;
    out["delay_s"] = t_delay;
    return out;
}

namespace {

int64_t check_reflection_primal(
    const at::Tensor& source,
    const at::Tensor& target,
    const at::Tensor& interaction_positions,
    const at::Tensor& interaction_normals,
    const at::Tensor& tx_power,
    const at::Tensor& tx_polarization,
    const at::Tensor& rx_polarization,
    const at::Tensor& eps_r,
    const at::Tensor& sigma_e,
    const at::Tensor& mu_r,
    const at::Tensor& gain,
    const at::Tensor& thickness,
    double frequency_hz) {
    using channel_native::check_flat_tensor;
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(target, "target");
    check_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = source.size(0);
    const int64_t depth = interaction_positions.size(1);
    TORCH_CHECK(
        depth > 0 && depth <= kMaxAdDepth && interaction_positions.size(2) == 3,
        "interaction_positions must have shape (N, D, 3) with 0 < D <= ",
        kMaxAdDepth);
    TORCH_CHECK(
        interaction_positions.size(0) == count &&
            interaction_normals.sizes() == interaction_positions.sizes(),
        "interaction tensors must match source rows");
    for (const auto& tensor : {eps_r, sigma_e, mu_r, gain, thickness})
        TORCH_CHECK(
            tensor.scalar_type() == at::kFloat && tensor.is_contiguous() &&
                tensor.dim() == 2 && tensor.size(0) == count &&
                tensor.size(1) == depth,
            "reflection material tensors must be contiguous f32 (N, D)");
    TORCH_CHECK(
        target.size(0) == count && tx_power.size(0) == count &&
            tx_polarization.size(0) == count && rx_polarization.size(0) == count,
        "reflection endpoint tensors must match source rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    return depth;
}

}  // namespace

pybind11::dict cn_field_reflection_sequence_backward(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    at::Tensor thickness,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    bool need_grad_eps_r,
    bool need_grad_sigma_e,
    bool need_grad_gain,
    bool need_grad_thickness,
    bool need_grad_frequency,
    bool need_grad_geometry) {
    const int64_t depth = check_reflection_primal(
        source, target, interaction_positions, interaction_normals, tx_power,
        tx_polarization, rx_polarization, eps_r, sigma_e, mu_r, gain, thickness,
        frequency_hz);
    const int64_t count = source.size(0);
    at::Tensor gfv_storage;
    at::Tensor gc_storage;
    at::Tensor gpf_storage;
    at::Tensor gpg_storage;
    at::Tensor gpl_storage;
    at::Tensor gd_storage;
    const at::Tensor* gfv = optional_grad(
        std::move(grad_field_vector), gfv_storage, "grad_field_vector",
        at::kComplexFloat, {count, 3}, source);
    const at::Tensor* gc = optional_grad(
        std::move(grad_coefficient), gc_storage, "grad_coefficient",
        at::kComplexFloat, {count}, source);
    const at::Tensor* gpf = optional_grad(
        std::move(grad_path_field), gpf_storage, "grad_path_field",
        at::kComplexFloat, {count}, source);
    const at::Tensor* gpg = optional_grad(
        std::move(grad_path_gain), gpg_storage, "grad_path_gain",
        at::kFloat, {count}, source);
    const at::Tensor* gpl = optional_grad(
        std::move(grad_path_length), gpl_storage, "grad_path_length",
        at::kFloat, {count}, source);
    const at::Tensor* gd = optional_grad(
        std::move(grad_delay), gd_storage, "grad_delay",
        at::kFloat, {count}, source);

    auto material_grad = [&](bool needed) {
        return needed ? zero_filled({count, depth}, source.options()) : at::Tensor();
    };
    at::Tensor grad_eps = material_grad(need_grad_eps_r);
    at::Tensor grad_sigma = material_grad(need_grad_sigma_e);
    at::Tensor grad_gain_out = material_grad(need_grad_gain);
    at::Tensor grad_thickness_out = material_grad(need_grad_thickness);
    at::Tensor grad_frequency = need_grad_frequency
                                    ? zero_filled({1}, source.options())
                                    : at::Tensor();
    at::Tensor grad_source;
    at::Tensor grad_target;
    at::Tensor grad_positions;
    at::Tensor grad_normals;
    if (need_grad_geometry) {
        grad_source = zero_filled({count, 3}, source.options());
        grad_target = zero_filled({count, 3}, source.options());
        grad_positions = zero_filled({count, depth, 3}, source.options());
        grad_normals = zero_filled({count, depth, 3}, source.options());
    }
    const bool any_grad_in = gfv != nullptr || gc != nullptr || gpf != nullptr ||
                             gpg != nullptr || gpl != nullptr || gd != nullptr;
    const bool any_grad_out = need_grad_eps_r || need_grad_sigma_e ||
                              need_grad_gain || need_grad_thickness ||
                              need_grad_frequency || need_grad_geometry;
    if (count > 0 && any_grad_in && any_grad_out) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        reflection_sequence_backward_kernel
            <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                source.data_ptr<float>(),
                target.data_ptr<float>(),
                interaction_positions.data_ptr<float>(),
                interaction_normals.data_ptr<float>(),
                tx_power.data_ptr<float>(),
                tx_polarization.data_ptr<float>(),
                rx_polarization.data_ptr<float>(),
                eps_r.data_ptr<float>(),
                sigma_e.data_ptr<float>(),
                mu_r.data_ptr<float>(),
                gain.data_ptr<float>(),
                thickness.data_ptr<float>(),
                static_cast<float>(frequency_hz),
                gfv ? gfv->data_ptr<c10::complex<float>>() : nullptr,
                gc ? gc->data_ptr<c10::complex<float>>() : nullptr,
                gpf ? gpf->data_ptr<c10::complex<float>>() : nullptr,
                grad_ptr<float>(gpg),
                grad_ptr<float>(gpl),
                grad_ptr<float>(gd),
                need_grad_eps_r ? grad_eps.data_ptr<float>() : nullptr,
                need_grad_sigma_e ? grad_sigma.data_ptr<float>() : nullptr,
                need_grad_gain ? grad_gain_out.data_ptr<float>() : nullptr,
                need_grad_thickness ? grad_thickness_out.data_ptr<float>()
                                    : nullptr,
                need_grad_frequency ? grad_frequency.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_source.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_target.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_positions.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_normals.data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_eps_r"] = need_grad_eps_r ? pybind11::cast(grad_eps)
                                        : pybind11::object(pybind11::none());
    out["grad_sigma_e"] = need_grad_sigma_e ? pybind11::cast(grad_sigma)
                                            : pybind11::object(pybind11::none());
    out["grad_gain"] = need_grad_gain ? pybind11::cast(grad_gain_out)
                                      : pybind11::object(pybind11::none());
    out["grad_thickness"] = need_grad_thickness
                                ? pybind11::cast(grad_thickness_out)
                                : pybind11::object(pybind11::none());
    out["grad_frequency"] = need_grad_frequency
                                ? pybind11::cast(grad_frequency)
                                : pybind11::object(pybind11::none());
    out["grad_source"] = need_grad_geometry
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_target"] = need_grad_geometry
                             ? pybind11::cast(grad_target)
                             : pybind11::object(pybind11::none());
    out["grad_interaction_positions"] =
        need_grad_geometry ? pybind11::cast(grad_positions)
                           : pybind11::object(pybind11::none());
    out["grad_interaction_normals"] =
        need_grad_geometry ? pybind11::cast(grad_normals)
                           : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_field_reflection_sequence_jvp(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    at::Tensor thickness,
    double frequency_hz,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_interaction_positions,
    pybind11::object tangent_interaction_normals) {
    const int64_t depth = check_reflection_primal(
        source, target, interaction_positions, interaction_normals, tx_power,
        tx_polarization, rx_polarization, eps_r, sigma_e, mu_r, gain, thickness,
        frequency_hz);
    const int64_t count = source.size(0);
    at::Tensor te_storage;
    at::Tensor ts_storage;
    at::Tensor tg_storage;
    at::Tensor tt_storage;
    at::Tensor tsrc_storage;
    at::Tensor ttgt_storage;
    at::Tensor tpos_storage;
    at::Tensor tnrm_storage;
    const at::Tensor* t_eps = optional_grad(
        std::move(tangent_eps_r), te_storage, "tangent_eps_r",
        at::kFloat, {count, depth}, source);
    const at::Tensor* t_sigma = optional_grad(
        std::move(tangent_sigma_e), ts_storage, "tangent_sigma_e",
        at::kFloat, {count, depth}, source);
    const at::Tensor* t_gain = optional_grad(
        std::move(tangent_gain), tg_storage, "tangent_gain",
        at::kFloat, {count, depth}, source);
    const at::Tensor* t_thickness = optional_grad(
        std::move(tangent_thickness), tt_storage, "tangent_thickness",
        at::kFloat, {count, depth}, source);
    const at::Tensor* t_source = optional_grad(
        std::move(tangent_source), tsrc_storage, "tangent_source",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_target = optional_grad(
        std::move(tangent_target), ttgt_storage, "tangent_target",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_positions = optional_grad(
        std::move(tangent_interaction_positions), tpos_storage,
        "tangent_interaction_positions", at::kFloat, {count, depth, 3}, source);
    const at::Tensor* t_normals = optional_grad(
        std::move(tangent_interaction_normals), tnrm_storage,
        "tangent_interaction_normals", at::kFloat, {count, depth, 3}, source);

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto t_field_vector = at::empty({count, 3}, complex_options);
    auto t_coefficient = at::empty({count}, complex_options);
    auto t_path_field = at::empty({count}, complex_options);
    auto t_path_gain = at::empty({count}, source.options());
    auto t_path_length = at::empty({count}, source.options());
    auto t_delay = at::empty({count}, source.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        reflection_sequence_jvp_kernel
            <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                source.data_ptr<float>(),
                target.data_ptr<float>(),
                interaction_positions.data_ptr<float>(),
                interaction_normals.data_ptr<float>(),
                tx_power.data_ptr<float>(),
                tx_polarization.data_ptr<float>(),
                rx_polarization.data_ptr<float>(),
                eps_r.data_ptr<float>(),
                sigma_e.data_ptr<float>(),
                mu_r.data_ptr<float>(),
                gain.data_ptr<float>(),
                thickness.data_ptr<float>(),
                static_cast<float>(frequency_hz),
                grad_ptr<float>(t_eps),
                grad_ptr<float>(t_sigma),
                grad_ptr<float>(t_gain),
                grad_ptr<float>(t_thickness),
                static_cast<float>(tangent_frequency),
                grad_ptr<float>(t_source),
                grad_ptr<float>(t_target),
                grad_ptr<float>(t_positions),
                grad_ptr<float>(t_normals),
                t_field_vector.data_ptr<c10::complex<float>>(),
                t_coefficient.data_ptr<c10::complex<float>>(),
                t_path_field.data_ptr<c10::complex<float>>(),
                t_path_gain.data_ptr<float>(),
                t_path_length.data_ptr<float>(),
                t_delay.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = t_field_vector;
    out["coefficient"] = t_coefficient;
    out["path_field"] = t_path_field;
    out["path_gain"] = t_path_gain;
    out["path_length_m"] = t_path_length;
    out["delay_s"] = t_delay;
    return out;
}

namespace {

std::pair<int64_t, int64_t> check_transmission_primal(
    const at::Tensor& source,
    const at::Tensor& target,
    const at::Tensor& interaction_positions,
    const at::Tensor& interaction_normals,
    const at::Tensor& interaction_material_id,
    const at::Tensor& interaction_valid,
    const at::Tensor& tx_power,
    const at::Tensor& tx_polarization,
    const at::Tensor& rx_polarization,
    const at::Tensor& layer_offset,
    const at::Tensor& layer_count,
    const at::Tensor& layer_thickness_m,
    const at::Tensor& layer_eps_r,
    const at::Tensor& layer_sigma_e,
    const at::Tensor& layer_mu_r,
    double frequency_hz) {
    using channel_native::check_flat_tensor;
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(target, "target");
    check_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
    check_tensor(interaction_material_id, "interaction_material_id", at::kInt, 2);
    check_tensor(interaction_valid, "interaction_valid", at::kBool, 2);
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    const int64_t count = source.size(0);
    const int64_t depth = interaction_positions.size(1);
    TORCH_CHECK(
        depth > 0 && depth <= kMaxAdDepth && interaction_positions.size(2) == 3,
        "interaction_positions must have shape (N, D, 3) with 0 < D <= ",
        kMaxAdDepth);
    TORCH_CHECK(
        interaction_positions.size(0) == count &&
            interaction_normals.sizes() == interaction_positions.sizes(),
        "interaction tensors must match source rows");
    TORCH_CHECK(
        interaction_material_id.size(0) == count &&
            interaction_material_id.size(1) == depth &&
            interaction_valid.size(0) == count &&
            interaction_valid.size(1) == depth,
        "transmission event tensors must have shape (N, D)");
    TORCH_CHECK(
        target.size(0) == count && tx_power.size(0) == count &&
            tx_polarization.size(0) == count && rx_polarization.size(0) == count,
        "transmission endpoint tensors must match source rows");
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    TORCH_CHECK(
        layer_count.size(0) == material_count,
        "layer_count must match layer_offset rows");
    for (const auto& tensor : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(
            tensor.size(0) == layer_total,
            "layer parameter tensors must match layer_thickness_m rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    return {depth, material_count};
}

}  // namespace

pybind11::dict cn_field_transmission_sequence_backward(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor interaction_material_id,
    at::Tensor interaction_valid,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    bool need_grad_layer_thickness,
    bool need_grad_layer_eps_r,
    bool need_grad_layer_sigma_e,
    bool need_grad_frequency,
    bool need_grad_geometry) {
    const auto [depth, material_count] = check_transmission_primal(
        source, target, interaction_positions, interaction_normals,
        interaction_material_id, interaction_valid, tx_power, tx_polarization,
        rx_polarization, layer_offset, layer_count, layer_thickness_m,
        layer_eps_r, layer_sigma_e, layer_mu_r, frequency_hz);
    const int64_t count = source.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    at::Tensor gfv_storage;
    at::Tensor gc_storage;
    at::Tensor gpf_storage;
    at::Tensor gpg_storage;
    at::Tensor gpl_storage;
    at::Tensor gd_storage;
    const at::Tensor* gfv = optional_grad(
        std::move(grad_field_vector), gfv_storage, "grad_field_vector",
        at::kComplexFloat, {count, 3}, source);
    const at::Tensor* gc = optional_grad(
        std::move(grad_coefficient), gc_storage, "grad_coefficient",
        at::kComplexFloat, {count}, source);
    const at::Tensor* gpf = optional_grad(
        std::move(grad_path_field), gpf_storage, "grad_path_field",
        at::kComplexFloat, {count}, source);
    const at::Tensor* gpg = optional_grad(
        std::move(grad_path_gain), gpg_storage, "grad_path_gain",
        at::kFloat, {count}, source);
    const at::Tensor* gpl = optional_grad(
        std::move(grad_path_length), gpl_storage, "grad_path_length",
        at::kFloat, {count}, source);
    const at::Tensor* gd = optional_grad(
        std::move(grad_delay), gd_storage, "grad_delay",
        at::kFloat, {count}, source);

    auto layer_grad = [&](bool needed) {
        return needed ? zero_filled({layer_total}, source.options()) : at::Tensor();
    };
    at::Tensor grad_thickness = layer_grad(need_grad_layer_thickness);
    at::Tensor grad_eps = layer_grad(need_grad_layer_eps_r);
    at::Tensor grad_sigma = layer_grad(need_grad_layer_sigma_e);
    at::Tensor grad_frequency = need_grad_frequency
                                    ? zero_filled({1}, source.options())
                                    : at::Tensor();
    at::Tensor grad_source;
    at::Tensor grad_target;
    at::Tensor grad_normals;
    if (need_grad_geometry) {
        grad_source = zero_filled({count, 3}, source.options());
        grad_target = zero_filled({count, 3}, source.options());
        grad_normals = zero_filled({count, depth, 3}, source.options());
    }
    const bool any_grad_in = gfv != nullptr || gc != nullptr || gpf != nullptr ||
                             gpg != nullptr || gpl != nullptr || gd != nullptr;
    const bool any_grad_out = need_grad_layer_thickness ||
                              need_grad_layer_eps_r || need_grad_layer_sigma_e ||
                              need_grad_frequency || need_grad_geometry;
    if (count > 0 && any_grad_in && any_grad_out) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        transmission_sequence_backward_kernel
            <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                source.data_ptr<float>(),
                target.data_ptr<float>(),
                interaction_normals.data_ptr<float>(),
                interaction_material_id.data_ptr<int>(),
                interaction_valid.data_ptr<bool>(),
                tx_power.data_ptr<float>(),
                tx_polarization.data_ptr<float>(),
                rx_polarization.data_ptr<float>(),
                layer_offset.data_ptr<int>(),
                layer_count.data_ptr<int>(),
                layer_thickness_m.data_ptr<float>(),
                layer_eps_r.data_ptr<float>(),
                layer_sigma_e.data_ptr<float>(),
                layer_mu_r.data_ptr<float>(),
                material_count,
                static_cast<float>(frequency_hz),
                gfv ? gfv->data_ptr<c10::complex<float>>() : nullptr,
                gc ? gc->data_ptr<c10::complex<float>>() : nullptr,
                gpf ? gpf->data_ptr<c10::complex<float>>() : nullptr,
                grad_ptr<float>(gpg),
                grad_ptr<float>(gpl),
                grad_ptr<float>(gd),
                need_grad_layer_thickness ? grad_thickness.data_ptr<float>()
                                          : nullptr,
                need_grad_layer_eps_r ? grad_eps.data_ptr<float>() : nullptr,
                need_grad_layer_sigma_e ? grad_sigma.data_ptr<float>() : nullptr,
                need_grad_frequency ? grad_frequency.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_source.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_target.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_normals.data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_layer_thickness_m"] =
        need_grad_layer_thickness ? pybind11::cast(grad_thickness)
                                  : pybind11::object(pybind11::none());
    out["grad_layer_eps_r"] = need_grad_layer_eps_r
                                  ? pybind11::cast(grad_eps)
                                  : pybind11::object(pybind11::none());
    out["grad_layer_sigma_e"] = need_grad_layer_sigma_e
                                    ? pybind11::cast(grad_sigma)
                                    : pybind11::object(pybind11::none());
    out["grad_frequency"] = need_grad_frequency
                                ? pybind11::cast(grad_frequency)
                                : pybind11::object(pybind11::none());
    out["grad_source"] = need_grad_geometry
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_target"] = need_grad_geometry
                             ? pybind11::cast(grad_target)
                             : pybind11::object(pybind11::none());
    // The straight-path transmission field is independent of the crossing
    // points themselves (the ray is source->target); their gradient is
    // exactly zero, reported as None so autograd materializes nothing.
    out["grad_interaction_positions"] = pybind11::none();
    out["grad_interaction_normals"] =
        need_grad_geometry ? pybind11::cast(grad_normals)
                           : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_field_transmission_sequence_jvp(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor interaction_material_id,
    at::Tensor interaction_valid,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object tangent_layer_thickness_m,
    pybind11::object tangent_layer_eps_r,
    pybind11::object tangent_layer_sigma_e,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_interaction_positions,
    pybind11::object tangent_interaction_normals) {
    const auto [depth, material_count] = check_transmission_primal(
        source, target, interaction_positions, interaction_normals,
        interaction_material_id, interaction_valid, tx_power, tx_polarization,
        rx_polarization, layer_offset, layer_count, layer_thickness_m,
        layer_eps_r, layer_sigma_e, layer_mu_r, frequency_hz);
    const int64_t count = source.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    at::Tensor tt_storage;
    at::Tensor te_storage;
    at::Tensor ts_storage;
    at::Tensor tsrc_storage;
    at::Tensor ttgt_storage;
    at::Tensor tpos_storage;
    at::Tensor tnrm_storage;
    const at::Tensor* t_thickness = optional_grad(
        std::move(tangent_layer_thickness_m), tt_storage,
        "tangent_layer_thickness_m", at::kFloat, {layer_total}, source);
    const at::Tensor* t_eps = optional_grad(
        std::move(tangent_layer_eps_r), te_storage, "tangent_layer_eps_r",
        at::kFloat, {layer_total}, source);
    const at::Tensor* t_sigma = optional_grad(
        std::move(tangent_layer_sigma_e), ts_storage, "tangent_layer_sigma_e",
        at::kFloat, {layer_total}, source);
    const at::Tensor* t_source = optional_grad(
        std::move(tangent_source), tsrc_storage, "tangent_source",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_target = optional_grad(
        std::move(tangent_target), ttgt_storage, "tangent_target",
        at::kFloat, {count, 3}, source);
    // The crossing points do not enter the straight-path transmission field;
    // their tangent is validated for ABI symmetry and contributes zero.
    (void)optional_grad(
        std::move(tangent_interaction_positions), tpos_storage,
        "tangent_interaction_positions", at::kFloat, {count, depth, 3}, source);
    const at::Tensor* t_normals = optional_grad(
        std::move(tangent_interaction_normals), tnrm_storage,
        "tangent_interaction_normals", at::kFloat, {count, depth, 3}, source);

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto t_field_vector = at::empty({count, 3}, complex_options);
    auto t_coefficient = at::empty({count}, complex_options);
    auto t_path_field = at::empty({count}, complex_options);
    auto t_path_gain = at::empty({count}, source.options());
    auto t_path_length = at::empty({count}, source.options());
    auto t_delay = at::empty({count}, source.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        transmission_sequence_jvp_kernel
            <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                source.data_ptr<float>(),
                target.data_ptr<float>(),
                interaction_normals.data_ptr<float>(),
                interaction_material_id.data_ptr<int>(),
                interaction_valid.data_ptr<bool>(),
                tx_power.data_ptr<float>(),
                tx_polarization.data_ptr<float>(),
                rx_polarization.data_ptr<float>(),
                layer_offset.data_ptr<int>(),
                layer_count.data_ptr<int>(),
                layer_thickness_m.data_ptr<float>(),
                layer_eps_r.data_ptr<float>(),
                layer_sigma_e.data_ptr<float>(),
                layer_mu_r.data_ptr<float>(),
                material_count,
                static_cast<float>(frequency_hz),
                grad_ptr<float>(t_thickness),
                grad_ptr<float>(t_eps),
                grad_ptr<float>(t_sigma),
                static_cast<float>(tangent_frequency),
                grad_ptr<float>(t_source),
                grad_ptr<float>(t_target),
                grad_ptr<float>(t_normals),
                t_field_vector.data_ptr<c10::complex<float>>(),
                t_coefficient.data_ptr<c10::complex<float>>(),
                t_path_field.data_ptr<c10::complex<float>>(),
                t_path_gain.data_ptr<float>(),
                t_path_length.data_ptr<float>(),
                t_delay.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = t_field_vector;
    out["coefficient"] = t_coefficient;
    out["path_field"] = t_path_field;
    out["path_gain"] = t_path_gain;
    out["path_length_m"] = t_path_length;
    out["delay_s"] = t_delay;
    return out;
}
