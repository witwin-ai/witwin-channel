#include "field_transport_ad_common.cuh"

namespace {

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
    // F1: unnormalized transverse projection of the transmit polarization.
    const field::float3a tx_axis = field::project_to_wedge_plane(
        load3f(tx_polarization, index), incident);
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
    // F1: receiver scalar = p_rx . E via the unnormalized transverse of p_rx.
    chain.rx_axis = field::project_to_wedge_plane(
        load3f(rx_polarization, index), final_direction);
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
            ad::adj_transverse_project(
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
        ad::adj_transverse_project(
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
        const ad::DualF3 tx_axis = ad::dual_transverse_project(
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
        const ad::DualF3 rx_axis = ad::dual_transverse_project(
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
    using channel::check_flat_tensor;
    using channel::check_tensor;
    using channel::check_vec3_table;
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

pybind11::dict channel_field_reflection_sequence_backward(
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

pybind11::dict channel_field_reflection_sequence_jvp(
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
