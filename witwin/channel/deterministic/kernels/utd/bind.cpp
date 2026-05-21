#include "drjit_common.h"
#include <utd/bind.h>

#include <utd/utd_types.h>
#include <utd/utd_accumulate.h>

#include <stdexcept>
#include <vector>

#define UTD_STATE_ARRAY_FIELDS \
    edge_pos_x, edge_pos_y, edge_pos_z, \
    edge_dir_x, edge_dir_y, edge_dir_z, \
    n0_x, n0_y, n0_z, \
    nn_x, nn_y, nn_z, \
    wedge_n, edge_line_min, edge_line_max, \
    source_pos_x, source_pos_y, source_pos_z, \
    incident_field_re, incident_field_im, \
    incident_nderiv_re, incident_nderiv_im, \
    r0_re, r0_im, rn_re, rn_im, \
    inc_vec_x_re, inc_vec_x_im, \
    inc_vec_y_re, inc_vec_y_im, \
    inc_vec_z_re, inc_vec_z_im, \
    inc_dvec_x_re, inc_dvec_x_im, \
    inc_dvec_y_re, inc_dvec_y_im, \
    inc_dvec_z_re, inc_dvec_z_im, \
    inc_jones_u_re, inc_jones_u_im, \
    inc_jones_v_re, inc_jones_v_im, \
    inc_djones_u_re, inc_djones_u_im, \
    inc_djones_v_re, inc_djones_v_im, \
    inc_basis_u_x, inc_basis_u_y, inc_basis_u_z, \
    inc_basis_v_x, inc_basis_v_y, inc_basis_v_z, \
    inc_basis_k_x, inc_basis_k_y, inc_basis_k_z, \
    face0_op_m00_re, face0_op_m00_im, \
    face0_op_m01_re, face0_op_m01_im, \
    face0_op_m10_re, face0_op_m10_im, \
    face0_op_m11_re, face0_op_m11_im, \
    face1_op_m00_re, face1_op_m00_im, \
    face1_op_m01_re, face1_op_m01_im, \
    face1_op_m10_re, face1_op_m10_im, \
    face1_op_m11_re, face1_op_m11_im, \
    face0_eta_r, face0_mu_r, face0_sigma, face0_gain, face0_use_fresnel, face0_present, \
    face1_eta_r, face1_mu_r, face1_sigma, face1_gain, face1_use_fresnel, face1_present, \
    select_stationary_point

#define UTD_FOR_EACH_STATE_FIELD(M) \
    M(edge_pos_x) M(edge_pos_y) M(edge_pos_z) \
    M(edge_dir_x) M(edge_dir_y) M(edge_dir_z) \
    M(n0_x) M(n0_y) M(n0_z) \
    M(nn_x) M(nn_y) M(nn_z) \
    M(wedge_n) M(edge_line_min) M(edge_line_max) \
    M(source_pos_x) M(source_pos_y) M(source_pos_z) \
    M(incident_field_re) M(incident_field_im) \
    M(incident_nderiv_re) M(incident_nderiv_im) \
    M(r0_re) M(r0_im) M(rn_re) M(rn_im) \
    M(inc_vec_x_re) M(inc_vec_x_im) \
    M(inc_vec_y_re) M(inc_vec_y_im) \
    M(inc_vec_z_re) M(inc_vec_z_im) \
    M(inc_dvec_x_re) M(inc_dvec_x_im) \
    M(inc_dvec_y_re) M(inc_dvec_y_im) \
    M(inc_dvec_z_re) M(inc_dvec_z_im) \
    M(inc_jones_u_re) M(inc_jones_u_im) \
    M(inc_jones_v_re) M(inc_jones_v_im) \
    M(inc_djones_u_re) M(inc_djones_u_im) \
    M(inc_djones_v_re) M(inc_djones_v_im) \
    M(inc_basis_u_x) M(inc_basis_u_y) M(inc_basis_u_z) \
    M(inc_basis_v_x) M(inc_basis_v_y) M(inc_basis_v_z) \
    M(inc_basis_k_x) M(inc_basis_k_y) M(inc_basis_k_z) \
    M(face0_op_m00_re) M(face0_op_m00_im) \
    M(face0_op_m01_re) M(face0_op_m01_im) \
    M(face0_op_m10_re) M(face0_op_m10_im) \
    M(face0_op_m11_re) M(face0_op_m11_im) \
    M(face1_op_m00_re) M(face1_op_m00_im) \
    M(face1_op_m01_re) M(face1_op_m01_im) \
    M(face1_op_m10_re) M(face1_op_m10_im) \
    M(face1_op_m11_re) M(face1_op_m11_im) \
    M(face0_eta_r) M(face0_mu_r) M(face0_sigma) M(face0_gain) M(face0_use_fresnel) M(face0_present) \
    M(face1_eta_r) M(face1_mu_r) M(face1_sigma) M(face1_gain) M(face1_use_fresnel) M(face1_present) \
    M(select_stationary_point)

struct UTDTiledStateArrays {
    DiffFloat edge_pos_x, edge_pos_y, edge_pos_z;
    DiffFloat edge_dir_x, edge_dir_y, edge_dir_z;
    DiffFloat n0_x, n0_y, n0_z;
    DiffFloat nn_x, nn_y, nn_z;
    DiffFloat wedge_n, edge_line_min, edge_line_max;
    DiffFloat source_pos_x, source_pos_y, source_pos_z;
    DiffFloat incident_field_re, incident_field_im;
    DiffFloat incident_nderiv_re, incident_nderiv_im;
    DiffFloat r0_re, r0_im, rn_re, rn_im;
    DiffFloat inc_vec_x_re, inc_vec_x_im;
    DiffFloat inc_vec_y_re, inc_vec_y_im;
    DiffFloat inc_vec_z_re, inc_vec_z_im;
    DiffFloat inc_dvec_x_re, inc_dvec_x_im;
    DiffFloat inc_dvec_y_re, inc_dvec_y_im;
    DiffFloat inc_dvec_z_re, inc_dvec_z_im;
    DiffFloat inc_jones_u_re, inc_jones_u_im;
    DiffFloat inc_jones_v_re, inc_jones_v_im;
    DiffFloat inc_djones_u_re, inc_djones_u_im;
    DiffFloat inc_djones_v_re, inc_djones_v_im;
    DiffFloat inc_basis_u_x, inc_basis_u_y, inc_basis_u_z;
    DiffFloat inc_basis_v_x, inc_basis_v_y, inc_basis_v_z;
    DiffFloat inc_basis_k_x, inc_basis_k_y, inc_basis_k_z;
    DiffFloat face0_op_m00_re, face0_op_m00_im;
    DiffFloat face0_op_m01_re, face0_op_m01_im;
    DiffFloat face0_op_m10_re, face0_op_m10_im;
    DiffFloat face0_op_m11_re, face0_op_m11_im;
    DiffFloat face1_op_m00_re, face1_op_m00_im;
    DiffFloat face1_op_m01_re, face1_op_m01_im;
    DiffFloat face1_op_m10_re, face1_op_m10_im;
    DiffFloat face1_op_m11_re, face1_op_m11_im;
    DiffFloat face0_eta_r, face0_mu_r, face0_sigma, face0_gain, face0_use_fresnel, face0_present;
    DiffFloat face1_eta_r, face1_mu_r, face1_sigma, face1_gain, face1_use_fresnel, face1_present;
    DiffFloat select_stationary_point;

    DRJIT_STRUCT(UTDTiledStateArrays, UTD_STATE_ARRAY_FIELDS);
};

struct UTDTiledReceiverArrays {
    DiffFloat rx_x, rx_y, rx_z;

    DRJIT_STRUCT(UTDTiledReceiverArrays, rx_x, rx_y, rx_z);
};

struct UTDTiledOpInput {
    Int32 state_idx;
    Int32 rx_idx;
    Int32 valid_mask;
    Int32 ownership_code;
    UTDTiledStateArrays state;
    UTDTiledReceiverArrays rx;
    witwin::channel::native_ext::MaterialParams material;
    int n_local_states;
    int n_local_receivers;
    float k;

    DRJIT_STRUCT(
        UTDTiledOpInput,
        state_idx,
        rx_idx,
        valid_mask,
        ownership_code,
        state,
        rx,
        material,
        n_local_states,
        n_local_receivers,
        k
    );
};

struct UTDTiledOpOutput {
    DiffFloat direct_vec_x_re, direct_vec_x_im;
    DiffFloat direct_vec_y_re, direct_vec_y_im;
    DiffFloat direct_vec_z_re, direct_vec_z_im;
    DiffFloat multi_vec_x_re, multi_vec_x_im;
    DiffFloat multi_vec_y_re, multi_vec_y_im;
    DiffFloat multi_vec_z_re, multi_vec_z_im;

    DRJIT_STRUCT(
        UTDTiledOpOutput,
        direct_vec_x_re, direct_vec_x_im,
        direct_vec_y_re, direct_vec_y_im,
        direct_vec_z_re, direct_vec_z_im,
        multi_vec_x_re, multi_vec_x_im,
        multi_vec_y_re, multi_vec_y_im,
        multi_vec_z_re, multi_vec_z_im
    );
};

struct UTDPairOpInput {
    Int32 state_idx;
    Int32 rx_idx;
    Int32 ownership_code;
    UTDTiledStateArrays state;
    UTDTiledReceiverArrays rx;
    witwin::channel::native_ext::MaterialParams material;
    int n_pairs;
    float k;

    DRJIT_STRUCT(
        UTDPairOpInput,
        state_idx,
        rx_idx,
        ownership_code,
        state,
        rx,
        material,
        n_pairs,
        k
    );
};

struct UTDPairOpOutput {
    DiffFloat direct_re, direct_im;
    DiffFloat multi_re, multi_im;
    DiffFloat direct_vec_x_re, direct_vec_x_im;
    DiffFloat direct_vec_y_re, direct_vec_y_im;
    DiffFloat direct_vec_z_re, direct_vec_z_im;
    DiffFloat multi_vec_x_re, multi_vec_x_im;
    DiffFloat multi_vec_y_re, multi_vec_y_im;
    DiffFloat multi_vec_z_re, multi_vec_z_im;

    DRJIT_STRUCT(
        UTDPairOpOutput,
        direct_re,
        direct_im,
        multi_re,
        multi_im,
        direct_vec_x_re, direct_vec_x_im,
        direct_vec_y_re, direct_vec_y_im,
        direct_vec_z_re, direct_vec_z_im,
        multi_vec_x_re, multi_vec_x_im,
        multi_vec_y_re, multi_vec_y_im,
        multi_vec_z_re, multi_vec_z_im
    );
};

UTDTiledStateArrays make_utd_tiled_state_arrays(nb::tuple state_soa, const char *label) {
    if (nb::len(state_soa) != 84) {
        throw std::runtime_error(std::string(label) + " expected 84 state arrays");
    }
    auto state = [&](size_t index) -> DiffFloat {
        return nb::cast<DiffFloat>(state_soa[index]);
    };
    return {
        state(0), state(1), state(2),
        state(3), state(4), state(5),
        state(6), state(7), state(8),
        state(9), state(10), state(11),
        state(12), state(13), state(14),
        state(15), state(16), state(17),
        state(18), state(19),
        state(20), state(21),
        state(22), state(23), state(24), state(25),
        state(26), state(27),
        state(28), state(29),
        state(30), state(31),
        state(32), state(33),
        state(34), state(35),
        state(36), state(37),
        state(38), state(39),
        state(40), state(41),
        state(42), state(43),
        state(44), state(45),
        state(46), state(47), state(48),
        state(49), state(50), state(51),
        state(52), state(53), state(54),
        state(55), state(56),
        state(57), state(58),
        state(59), state(60),
        state(61), state(62),
        state(63), state(64),
        state(65), state(66),
        state(67), state(68),
        state(69), state(70),
        state(71), state(72), state(73), state(74), state(75),
        state(76), state(77), state(78), state(79), state(80), state(81),
        state(82), state(83),
    };
}

std::vector<const float*> utd_state_slot_ptrs(const UTDTiledStateArrays &state) {
    return {
        drjit_data_ptr(state.edge_pos_x), drjit_data_ptr(state.edge_pos_y), drjit_data_ptr(state.edge_pos_z),
        drjit_data_ptr(state.edge_dir_x), drjit_data_ptr(state.edge_dir_y), drjit_data_ptr(state.edge_dir_z),
        drjit_data_ptr(state.n0_x), drjit_data_ptr(state.n0_y), drjit_data_ptr(state.n0_z),
        drjit_data_ptr(state.nn_x), drjit_data_ptr(state.nn_y), drjit_data_ptr(state.nn_z),
        drjit_data_ptr(state.wedge_n),
        drjit_data_ptr(state.edge_line_min), drjit_data_ptr(state.edge_line_max),
        drjit_data_ptr(state.source_pos_x), drjit_data_ptr(state.source_pos_y), drjit_data_ptr(state.source_pos_z),
        drjit_data_ptr(state.incident_field_re), drjit_data_ptr(state.incident_field_im),
        drjit_data_ptr(state.incident_nderiv_re), drjit_data_ptr(state.incident_nderiv_im),
        drjit_data_ptr(state.r0_re), drjit_data_ptr(state.r0_im),
        drjit_data_ptr(state.rn_re), drjit_data_ptr(state.rn_im),
        drjit_data_ptr(state.inc_vec_x_re), drjit_data_ptr(state.inc_vec_x_im),
        drjit_data_ptr(state.inc_vec_y_re), drjit_data_ptr(state.inc_vec_y_im),
        drjit_data_ptr(state.inc_vec_z_re), drjit_data_ptr(state.inc_vec_z_im),
        drjit_data_ptr(state.inc_dvec_x_re), drjit_data_ptr(state.inc_dvec_x_im),
        drjit_data_ptr(state.inc_dvec_y_re), drjit_data_ptr(state.inc_dvec_y_im),
        drjit_data_ptr(state.inc_dvec_z_re), drjit_data_ptr(state.inc_dvec_z_im),
        drjit_data_ptr(state.inc_jones_u_re), drjit_data_ptr(state.inc_jones_u_im),
        drjit_data_ptr(state.inc_jones_v_re), drjit_data_ptr(state.inc_jones_v_im),
        drjit_data_ptr(state.inc_djones_u_re), drjit_data_ptr(state.inc_djones_u_im),
        drjit_data_ptr(state.inc_djones_v_re), drjit_data_ptr(state.inc_djones_v_im),
        drjit_data_ptr(state.inc_basis_u_x), drjit_data_ptr(state.inc_basis_u_y), drjit_data_ptr(state.inc_basis_u_z),
        drjit_data_ptr(state.inc_basis_v_x), drjit_data_ptr(state.inc_basis_v_y), drjit_data_ptr(state.inc_basis_v_z),
        drjit_data_ptr(state.inc_basis_k_x), drjit_data_ptr(state.inc_basis_k_y), drjit_data_ptr(state.inc_basis_k_z),
        drjit_data_ptr(state.face0_op_m00_re), drjit_data_ptr(state.face0_op_m00_im),
        drjit_data_ptr(state.face0_op_m01_re), drjit_data_ptr(state.face0_op_m01_im),
        drjit_data_ptr(state.face0_op_m10_re), drjit_data_ptr(state.face0_op_m10_im),
        drjit_data_ptr(state.face0_op_m11_re), drjit_data_ptr(state.face0_op_m11_im),
        drjit_data_ptr(state.face1_op_m00_re), drjit_data_ptr(state.face1_op_m00_im),
        drjit_data_ptr(state.face1_op_m01_re), drjit_data_ptr(state.face1_op_m01_im),
        drjit_data_ptr(state.face1_op_m10_re), drjit_data_ptr(state.face1_op_m10_im),
        drjit_data_ptr(state.face1_op_m11_re), drjit_data_ptr(state.face1_op_m11_im),
        drjit_data_ptr(state.face0_eta_r), drjit_data_ptr(state.face0_mu_r),
        drjit_data_ptr(state.face0_sigma), drjit_data_ptr(state.face0_gain), drjit_data_ptr(state.face0_use_fresnel),
        drjit_data_ptr(state.face0_present),
        drjit_data_ptr(state.face1_eta_r), drjit_data_ptr(state.face1_mu_r),
        drjit_data_ptr(state.face1_sigma), drjit_data_ptr(state.face1_gain), drjit_data_ptr(state.face1_use_fresnel),
        drjit_data_ptr(state.face1_present),
        drjit_data_ptr(state.select_stationary_point),
    };
}

void eval_utd_tiled_state_arrays(const UTDTiledStateArrays &state) {
    drjit::eval(
        state.edge_pos_x, state.edge_pos_y, state.edge_pos_z,
        state.edge_dir_x, state.edge_dir_y, state.edge_dir_z,
        state.n0_x, state.n0_y, state.n0_z,
        state.nn_x, state.nn_y, state.nn_z,
        state.wedge_n, state.edge_line_min, state.edge_line_max,
        state.source_pos_x, state.source_pos_y, state.source_pos_z
    );
    drjit::eval(
        state.incident_field_re, state.incident_field_im,
        state.incident_nderiv_re, state.incident_nderiv_im,
        state.r0_re, state.r0_im, state.rn_re, state.rn_im,
        state.inc_vec_x_re, state.inc_vec_x_im,
        state.inc_vec_y_re, state.inc_vec_y_im,
        state.inc_vec_z_re, state.inc_vec_z_im,
        state.inc_dvec_x_re, state.inc_dvec_x_im,
        state.inc_dvec_y_re, state.inc_dvec_y_im,
        state.inc_dvec_z_re, state.inc_dvec_z_im
    );
    drjit::eval(
        state.inc_jones_u_re, state.inc_jones_u_im,
        state.inc_jones_v_re, state.inc_jones_v_im,
        state.inc_djones_u_re, state.inc_djones_u_im,
        state.inc_djones_v_re, state.inc_djones_v_im,
        state.inc_basis_u_x, state.inc_basis_u_y, state.inc_basis_u_z,
        state.inc_basis_v_x, state.inc_basis_v_y, state.inc_basis_v_z,
        state.inc_basis_k_x, state.inc_basis_k_y, state.inc_basis_k_z
    );
    drjit::eval(
        state.face0_op_m00_re, state.face0_op_m00_im,
        state.face0_op_m01_re, state.face0_op_m01_im,
        state.face0_op_m10_re, state.face0_op_m10_im,
        state.face0_op_m11_re, state.face0_op_m11_im,
        state.face1_op_m00_re, state.face1_op_m00_im,
        state.face1_op_m01_re, state.face1_op_m01_im,
        state.face1_op_m10_re, state.face1_op_m10_im,
        state.face1_op_m11_re, state.face1_op_m11_im
    );
    drjit::eval(
        state.face0_eta_r, state.face0_mu_r, state.face0_sigma, state.face0_gain,
        state.face0_use_fresnel, state.face0_present,
        state.face1_eta_r, state.face1_mu_r, state.face1_sigma, state.face1_gain,
        state.face1_use_fresnel, state.face1_present,
        state.select_stationary_point
    );
}

UTDTiledOpOutput zero_utd_tiled_output(size_t width) {
    return {
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
    };
}

drjit::detached_t<UTDTiledOpOutput> zero_utd_tiled_output_grad(size_t width) {
    return {
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
    };
}

void set_utd_tiled_output_grad(
    UTDTiledOpOutput &registered_output,
    const drjit::detached_t<UTDTiledOpOutput> &grad_output)
{
    drjit::set_grad(registered_output.direct_vec_x_re, grad_output.direct_vec_x_re);
    drjit::set_grad(registered_output.direct_vec_x_im, grad_output.direct_vec_x_im);
    drjit::set_grad(registered_output.direct_vec_y_re, grad_output.direct_vec_y_re);
    drjit::set_grad(registered_output.direct_vec_y_im, grad_output.direct_vec_y_im);
    drjit::set_grad(registered_output.direct_vec_z_re, grad_output.direct_vec_z_re);
    drjit::set_grad(registered_output.direct_vec_z_im, grad_output.direct_vec_z_im);
    drjit::set_grad(registered_output.multi_vec_x_re, grad_output.multi_vec_x_re);
    drjit::set_grad(registered_output.multi_vec_x_im, grad_output.multi_vec_x_im);
    drjit::set_grad(registered_output.multi_vec_y_re, grad_output.multi_vec_y_re);
    drjit::set_grad(registered_output.multi_vec_y_im, grad_output.multi_vec_y_im);
    drjit::set_grad(registered_output.multi_vec_z_re, grad_output.multi_vec_z_re);
    drjit::set_grad(registered_output.multi_vec_z_im, grad_output.multi_vec_z_im);
}

UTDPairOpOutput zero_utd_pair_output(size_t width) {
    return {
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),
    };
}

drjit::detached_t<UTDPairOpOutput> zero_utd_pair_output_grad(size_t width) {
    return {
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
        drjit::zeros<Float>(width), drjit::zeros<Float>(width),
    };
}

void set_utd_pair_output_grad(
    UTDPairOpOutput &registered_output,
    const drjit::detached_t<UTDPairOpOutput> &grad_output)
{
    drjit::set_grad(registered_output.direct_re, grad_output.direct_re);
    drjit::set_grad(registered_output.direct_im, grad_output.direct_im);
    drjit::set_grad(registered_output.multi_re, grad_output.multi_re);
    drjit::set_grad(registered_output.multi_im, grad_output.multi_im);
    drjit::set_grad(registered_output.direct_vec_x_re, grad_output.direct_vec_x_re);
    drjit::set_grad(registered_output.direct_vec_x_im, grad_output.direct_vec_x_im);
    drjit::set_grad(registered_output.direct_vec_y_re, grad_output.direct_vec_y_re);
    drjit::set_grad(registered_output.direct_vec_y_im, grad_output.direct_vec_y_im);
    drjit::set_grad(registered_output.direct_vec_z_re, grad_output.direct_vec_z_re);
    drjit::set_grad(registered_output.direct_vec_z_im, grad_output.direct_vec_z_im);
    drjit::set_grad(registered_output.multi_vec_x_re, grad_output.multi_vec_x_re);
    drjit::set_grad(registered_output.multi_vec_x_im, grad_output.multi_vec_x_im);
    drjit::set_grad(registered_output.multi_vec_y_re, grad_output.multi_vec_y_re);
    drjit::set_grad(registered_output.multi_vec_y_im, grad_output.multi_vec_y_im);
    drjit::set_grad(registered_output.multi_vec_z_re, grad_output.multi_vec_z_re);
    drjit::set_grad(registered_output.multi_vec_z_im, grad_output.multi_vec_z_im);
}

struct MaterializedTangentSlots {
    std::vector<Float> arrays;
    std::vector<const float*> ptrs;
};

template <typename Array>
void add_tangent_slot(MaterializedTangentSlots &slots, const Array &value, size_t width) {
    slots.arrays.push_back(
        value.size() == width
            ? drjit::detach<false>(value)
            : drjit::zeros<Float>(width)
    );
    drjit::eval(slots.arrays.back());
    slots.ptrs.push_back(drjit_data_ptr(slots.arrays.back()));
}

#define UTD_ADD_TANGENT_SLOT(name) add_tangent_slot(result, tangent.name, primal.name.size())

MaterializedTangentSlots materialize_utd_tangent_slots(
    const UTDTiledStateArrays &primal,
    const drjit::detached_t<UTDTiledStateArrays> &tangent)
{
    MaterializedTangentSlots result;
    result.arrays.reserve(84);
    result.ptrs.reserve(84);
    UTD_ADD_TANGENT_SLOT(edge_pos_x);
    UTD_ADD_TANGENT_SLOT(edge_pos_y);
    UTD_ADD_TANGENT_SLOT(edge_pos_z);
    UTD_ADD_TANGENT_SLOT(edge_dir_x);
    UTD_ADD_TANGENT_SLOT(edge_dir_y);
    UTD_ADD_TANGENT_SLOT(edge_dir_z);
    UTD_ADD_TANGENT_SLOT(n0_x);
    UTD_ADD_TANGENT_SLOT(n0_y);
    UTD_ADD_TANGENT_SLOT(n0_z);
    UTD_ADD_TANGENT_SLOT(nn_x);
    UTD_ADD_TANGENT_SLOT(nn_y);
    UTD_ADD_TANGENT_SLOT(nn_z);
    UTD_ADD_TANGENT_SLOT(wedge_n);
    UTD_ADD_TANGENT_SLOT(edge_line_min);
    UTD_ADD_TANGENT_SLOT(edge_line_max);
    UTD_ADD_TANGENT_SLOT(source_pos_x);
    UTD_ADD_TANGENT_SLOT(source_pos_y);
    UTD_ADD_TANGENT_SLOT(source_pos_z);
    UTD_ADD_TANGENT_SLOT(incident_field_re);
    UTD_ADD_TANGENT_SLOT(incident_field_im);
    UTD_ADD_TANGENT_SLOT(incident_nderiv_re);
    UTD_ADD_TANGENT_SLOT(incident_nderiv_im);
    UTD_ADD_TANGENT_SLOT(r0_re);
    UTD_ADD_TANGENT_SLOT(r0_im);
    UTD_ADD_TANGENT_SLOT(rn_re);
    UTD_ADD_TANGENT_SLOT(rn_im);
    UTD_ADD_TANGENT_SLOT(inc_vec_x_re);
    UTD_ADD_TANGENT_SLOT(inc_vec_x_im);
    UTD_ADD_TANGENT_SLOT(inc_vec_y_re);
    UTD_ADD_TANGENT_SLOT(inc_vec_y_im);
    UTD_ADD_TANGENT_SLOT(inc_vec_z_re);
    UTD_ADD_TANGENT_SLOT(inc_vec_z_im);
    UTD_ADD_TANGENT_SLOT(inc_dvec_x_re);
    UTD_ADD_TANGENT_SLOT(inc_dvec_x_im);
    UTD_ADD_TANGENT_SLOT(inc_dvec_y_re);
    UTD_ADD_TANGENT_SLOT(inc_dvec_y_im);
    UTD_ADD_TANGENT_SLOT(inc_dvec_z_re);
    UTD_ADD_TANGENT_SLOT(inc_dvec_z_im);
    UTD_ADD_TANGENT_SLOT(inc_jones_u_re);
    UTD_ADD_TANGENT_SLOT(inc_jones_u_im);
    UTD_ADD_TANGENT_SLOT(inc_jones_v_re);
    UTD_ADD_TANGENT_SLOT(inc_jones_v_im);
    UTD_ADD_TANGENT_SLOT(inc_djones_u_re);
    UTD_ADD_TANGENT_SLOT(inc_djones_u_im);
    UTD_ADD_TANGENT_SLOT(inc_djones_v_re);
    UTD_ADD_TANGENT_SLOT(inc_djones_v_im);
    UTD_ADD_TANGENT_SLOT(inc_basis_u_x);
    UTD_ADD_TANGENT_SLOT(inc_basis_u_y);
    UTD_ADD_TANGENT_SLOT(inc_basis_u_z);
    UTD_ADD_TANGENT_SLOT(inc_basis_v_x);
    UTD_ADD_TANGENT_SLOT(inc_basis_v_y);
    UTD_ADD_TANGENT_SLOT(inc_basis_v_z);
    UTD_ADD_TANGENT_SLOT(inc_basis_k_x);
    UTD_ADD_TANGENT_SLOT(inc_basis_k_y);
    UTD_ADD_TANGENT_SLOT(inc_basis_k_z);
    UTD_ADD_TANGENT_SLOT(face0_op_m00_re);
    UTD_ADD_TANGENT_SLOT(face0_op_m00_im);
    UTD_ADD_TANGENT_SLOT(face0_op_m01_re);
    UTD_ADD_TANGENT_SLOT(face0_op_m01_im);
    UTD_ADD_TANGENT_SLOT(face0_op_m10_re);
    UTD_ADD_TANGENT_SLOT(face0_op_m10_im);
    UTD_ADD_TANGENT_SLOT(face0_op_m11_re);
    UTD_ADD_TANGENT_SLOT(face0_op_m11_im);
    UTD_ADD_TANGENT_SLOT(face1_op_m00_re);
    UTD_ADD_TANGENT_SLOT(face1_op_m00_im);
    UTD_ADD_TANGENT_SLOT(face1_op_m01_re);
    UTD_ADD_TANGENT_SLOT(face1_op_m01_im);
    UTD_ADD_TANGENT_SLOT(face1_op_m10_re);
    UTD_ADD_TANGENT_SLOT(face1_op_m10_im);
    UTD_ADD_TANGENT_SLOT(face1_op_m11_re);
    UTD_ADD_TANGENT_SLOT(face1_op_m11_im);
    UTD_ADD_TANGENT_SLOT(face0_eta_r);
    UTD_ADD_TANGENT_SLOT(face0_mu_r);
    UTD_ADD_TANGENT_SLOT(face0_sigma);
    UTD_ADD_TANGENT_SLOT(face0_gain);
    UTD_ADD_TANGENT_SLOT(face0_use_fresnel);
    UTD_ADD_TANGENT_SLOT(face0_present);
    UTD_ADD_TANGENT_SLOT(face1_eta_r);
    UTD_ADD_TANGENT_SLOT(face1_mu_r);
    UTD_ADD_TANGENT_SLOT(face1_sigma);
    UTD_ADD_TANGENT_SLOT(face1_gain);
    UTD_ADD_TANGENT_SLOT(face1_use_fresnel);
    UTD_ADD_TANGENT_SLOT(face1_present);
    UTD_ADD_TANGENT_SLOT(select_stationary_point);
    return result;
}

#undef UTD_ADD_TANGENT_SLOT

struct MaterializedGradientSlots {
    std::vector<Float> arrays;
    std::vector<float*> ptrs;
};

void add_gradient_slot(MaterializedGradientSlots &slots, size_t width) {
    slots.arrays.push_back(drjit::zeros<Float>(width));
    drjit::eval(slots.arrays.back());
    slots.ptrs.push_back(drjit_data_ptr_mut(slots.arrays.back()));
}

#define UTD_ADD_GRADIENT_SLOT(name) add_gradient_slot(result, primal.name.size());

MaterializedGradientSlots materialize_utd_gradient_slots(
    const UTDTiledStateArrays &primal)
{
    MaterializedGradientSlots result;
    result.arrays.reserve(84);
    result.ptrs.reserve(84);
    UTD_FOR_EACH_STATE_FIELD(UTD_ADD_GRADIENT_SLOT)
    return result;
}

#undef UTD_ADD_GRADIENT_SLOT

void accum_utd_state_grads(
    UTDTiledStateArrays &registered_state,
    const MaterializedGradientSlots &grads)
{
    size_t slot = 0;
#define UTD_ACCUM_STATE_GRAD(name) \
    drjit::accum_grad(registered_state.name, drjit::detach<false>(grads.arrays[slot++]));
    UTD_FOR_EACH_STATE_FIELD(UTD_ACCUM_STATE_GRAD)
#undef UTD_ACCUM_STATE_GRAD
}

void accum_utd_receiver_grads(
    UTDTiledReceiverArrays &registered_rx,
    const Float &grad_rx_x,
    const Float &grad_rx_y,
    const Float &grad_rx_z)
{
    drjit::accum_grad(registered_rx.rx_x, drjit::detach<false>(grad_rx_x));
    drjit::accum_grad(registered_rx.rx_y, drjit::detach<false>(grad_rx_y));
    drjit::accum_grad(registered_rx.rx_z, drjit::detach<false>(grad_rx_z));
}

void launch_utd_tiled_forward(
    const drjit::detached_t<UTDTiledOpInput> &input,
    UTDTiledOpOutput &output)
{
    Float direct_re = drjit::zeros<Float>(input.rx.rx_x.size());
    Float direct_im = drjit::zeros<Float>(input.rx.rx_x.size());
    Float multi_re = drjit::zeros<Float>(input.rx.rx_x.size());
    Float multi_im = drjit::zeros<Float>(input.rx.rx_x.size());
    drjit::eval(input.state_idx, input.rx_idx, input.valid_mask, input.ownership_code);
    drjit::eval(input.rx.rx_x, input.rx.rx_y, input.rx.rx_z);
    eval_utd_tiled_state_arrays(input.state);
    drjit::eval(
        output.direct_vec_x_re, output.direct_vec_x_im,
        output.direct_vec_y_re, output.direct_vec_y_im,
        output.direct_vec_z_re, output.direct_vec_z_im,
        output.multi_vec_x_re, output.multi_vec_x_im,
        output.multi_vec_y_re, output.multi_vec_y_im,
        output.multi_vec_z_re, output.multi_vec_z_im,
        direct_re, direct_im, multi_re, multi_im
    );
    std::vector<const float*> state_slots = utd_state_slot_ptrs(input.state);
    witwin::channel::native_ext::utd_accumulate_tiled_forward_slots(
        drjit_data_ptr(input.state_idx),
        drjit_data_ptr(input.rx_idx),
        drjit_data_ptr(input.valid_mask),
        drjit_data_ptr(input.ownership_code),
        state_slots.data(),
        drjit_data_ptr(input.rx.rx_x),
        drjit_data_ptr(input.rx.rx_y),
        drjit_data_ptr(input.rx.rx_z),
        drjit_data_ptr_mut(direct_re),
        drjit_data_ptr_mut(direct_im),
        drjit_data_ptr_mut(multi_re),
        drjit_data_ptr_mut(multi_im),
        drjit_data_ptr_mut(output.direct_vec_x_re),
        drjit_data_ptr_mut(output.direct_vec_x_im),
        drjit_data_ptr_mut(output.direct_vec_y_re),
        drjit_data_ptr_mut(output.direct_vec_y_im),
        drjit_data_ptr_mut(output.direct_vec_z_re),
        drjit_data_ptr_mut(output.direct_vec_z_im),
        drjit_data_ptr_mut(output.multi_vec_x_re),
        drjit_data_ptr_mut(output.multi_vec_x_im),
        drjit_data_ptr_mut(output.multi_vec_y_re),
        drjit_data_ptr_mut(output.multi_vec_y_im),
        drjit_data_ptr_mut(output.multi_vec_z_re),
        drjit_data_ptr_mut(output.multi_vec_z_im),
        input.n_local_states,
        input.n_local_receivers,
        input.k,
        input.material
    );
}

void launch_utd_tiled_jvp(
    const drjit::detached_t<UTDTiledOpInput> &input,
    const drjit::detached_t<UTDTiledOpInput> &tangent,
    drjit::detached_t<UTDTiledOpOutput> &output)
{
    auto coerce = [](const auto &value, size_t width) -> Float {
        return value.size() == width
            ? drjit::detach<false>(value)
            : drjit::zeros<Float>(width);
    };
    MaterializedTangentSlots tangent_slots =
        materialize_utd_tangent_slots(input.state, tangent.state);
    Float tangent_rx_x = coerce(tangent.rx.rx_x, input.rx.rx_x.size());
    Float tangent_rx_y = coerce(tangent.rx.rx_y, input.rx.rx_y.size());
    Float tangent_rx_z = coerce(tangent.rx.rx_z, input.rx.rx_z.size());
    drjit::eval(tangent_rx_x, tangent_rx_y, tangent_rx_z);
    drjit::eval(
        output.direct_vec_x_re, output.direct_vec_x_im,
        output.direct_vec_y_re, output.direct_vec_y_im,
        output.direct_vec_z_re, output.direct_vec_z_im,
        output.multi_vec_x_re, output.multi_vec_x_im,
        output.multi_vec_y_re, output.multi_vec_y_im,
        output.multi_vec_z_re, output.multi_vec_z_im
    );
    std::vector<const float*> state_slots = utd_state_slot_ptrs(input.state);
    witwin::channel::native_ext::utd_accumulate_tiled_jvp_slots(
        drjit_data_ptr(input.state_idx),
        drjit_data_ptr(input.rx_idx),
        drjit_data_ptr(input.valid_mask),
        drjit_data_ptr(input.ownership_code),
        state_slots.data(),
        drjit_data_ptr(input.rx.rx_x),
        drjit_data_ptr(input.rx.rx_y),
        drjit_data_ptr(input.rx.rx_z),
        tangent_slots.ptrs.data(),
        drjit_data_ptr(tangent_rx_x),
        drjit_data_ptr(tangent_rx_y),
        drjit_data_ptr(tangent_rx_z),
        drjit_data_ptr_mut(output.direct_vec_x_re),
        drjit_data_ptr_mut(output.direct_vec_x_im),
        drjit_data_ptr_mut(output.direct_vec_y_re),
        drjit_data_ptr_mut(output.direct_vec_y_im),
        drjit_data_ptr_mut(output.direct_vec_z_re),
        drjit_data_ptr_mut(output.direct_vec_z_im),
        drjit_data_ptr_mut(output.multi_vec_x_re),
        drjit_data_ptr_mut(output.multi_vec_x_im),
        drjit_data_ptr_mut(output.multi_vec_y_re),
        drjit_data_ptr_mut(output.multi_vec_y_im),
        drjit_data_ptr_mut(output.multi_vec_z_re),
        drjit_data_ptr_mut(output.multi_vec_z_im),
        input.n_local_states,
        input.n_local_receivers,
        input.k,
        input.material
    );
}

void launch_utd_pair_forward(
    const drjit::detached_t<UTDPairOpInput> &input,
    UTDPairOpOutput &output)
{
    drjit::eval(input.state_idx, input.rx_idx, input.ownership_code);
    drjit::eval(input.rx.rx_x, input.rx.rx_y, input.rx.rx_z);
    eval_utd_tiled_state_arrays(input.state);
    drjit::eval(
        output.direct_re, output.direct_im,
        output.multi_re, output.multi_im,
        output.direct_vec_x_re, output.direct_vec_x_im,
        output.direct_vec_y_re, output.direct_vec_y_im,
        output.direct_vec_z_re, output.direct_vec_z_im,
        output.multi_vec_x_re, output.multi_vec_x_im,
        output.multi_vec_y_re, output.multi_vec_y_im,
        output.multi_vec_z_re, output.multi_vec_z_im
    );
    std::vector<const float*> state_slots = utd_state_slot_ptrs(input.state);
    witwin::channel::native_ext::utd_pair_forward_slots(
        drjit_data_ptr(input.state_idx),
        drjit_data_ptr(input.rx_idx),
        drjit_data_ptr(input.ownership_code),
        state_slots.data(),
        drjit_data_ptr(input.rx.rx_x),
        drjit_data_ptr(input.rx.rx_y),
        drjit_data_ptr(input.rx.rx_z),
        drjit_data_ptr_mut(output.direct_re),
        drjit_data_ptr_mut(output.direct_im),
        drjit_data_ptr_mut(output.multi_re),
        drjit_data_ptr_mut(output.multi_im),
        drjit_data_ptr_mut(output.direct_vec_x_re),
        drjit_data_ptr_mut(output.direct_vec_x_im),
        drjit_data_ptr_mut(output.direct_vec_y_re),
        drjit_data_ptr_mut(output.direct_vec_y_im),
        drjit_data_ptr_mut(output.direct_vec_z_re),
        drjit_data_ptr_mut(output.direct_vec_z_im),
        drjit_data_ptr_mut(output.multi_vec_x_re),
        drjit_data_ptr_mut(output.multi_vec_x_im),
        drjit_data_ptr_mut(output.multi_vec_y_re),
        drjit_data_ptr_mut(output.multi_vec_y_im),
        drjit_data_ptr_mut(output.multi_vec_z_re),
        drjit_data_ptr_mut(output.multi_vec_z_im),
        input.n_pairs,
        input.k,
        input.material
    );
}

void launch_utd_pair_jvp(
    const drjit::detached_t<UTDPairOpInput> &input,
    const drjit::detached_t<UTDPairOpInput> &tangent,
    drjit::detached_t<UTDPairOpOutput> &output)
{
    auto coerce = [](const auto &value, size_t width) -> Float {
        return value.size() == width
            ? drjit::detach<false>(value)
            : drjit::zeros<Float>(width);
    };
    MaterializedTangentSlots tangent_slots =
        materialize_utd_tangent_slots(input.state, tangent.state);
    Float tangent_rx_x = coerce(tangent.rx.rx_x, input.rx.rx_x.size());
    Float tangent_rx_y = coerce(tangent.rx.rx_y, input.rx.rx_y.size());
    Float tangent_rx_z = coerce(tangent.rx.rx_z, input.rx.rx_z.size());
    drjit::eval(tangent_rx_x, tangent_rx_y, tangent_rx_z);
    drjit::eval(
        output.direct_re, output.direct_im,
        output.multi_re, output.multi_im,
        output.direct_vec_x_re, output.direct_vec_x_im,
        output.direct_vec_y_re, output.direct_vec_y_im,
        output.direct_vec_z_re, output.direct_vec_z_im,
        output.multi_vec_x_re, output.multi_vec_x_im,
        output.multi_vec_y_re, output.multi_vec_y_im,
        output.multi_vec_z_re, output.multi_vec_z_im
    );
    std::vector<const float*> state_slots = utd_state_slot_ptrs(input.state);
    witwin::channel::native_ext::utd_pair_jvp_slots(
        drjit_data_ptr(input.state_idx),
        drjit_data_ptr(input.rx_idx),
        drjit_data_ptr(input.ownership_code),
        state_slots.data(),
        drjit_data_ptr(input.rx.rx_x),
        drjit_data_ptr(input.rx.rx_y),
        drjit_data_ptr(input.rx.rx_z),
        tangent_slots.ptrs.data(),
        drjit_data_ptr(tangent_rx_x),
        drjit_data_ptr(tangent_rx_y),
        drjit_data_ptr(tangent_rx_z),
        drjit_data_ptr_mut(output.direct_re),
        drjit_data_ptr_mut(output.direct_im),
        drjit_data_ptr_mut(output.multi_re),
        drjit_data_ptr_mut(output.multi_im),
        drjit_data_ptr_mut(output.direct_vec_x_re),
        drjit_data_ptr_mut(output.direct_vec_x_im),
        drjit_data_ptr_mut(output.direct_vec_y_re),
        drjit_data_ptr_mut(output.direct_vec_y_im),
        drjit_data_ptr_mut(output.direct_vec_z_re),
        drjit_data_ptr_mut(output.direct_vec_z_im),
        drjit_data_ptr_mut(output.multi_vec_x_re),
        drjit_data_ptr_mut(output.multi_vec_x_im),
        drjit_data_ptr_mut(output.multi_vec_y_re),
        drjit_data_ptr_mut(output.multi_vec_y_im),
        drjit_data_ptr_mut(output.multi_vec_z_re),
        drjit_data_ptr_mut(output.multi_vec_z_im),
        input.n_pairs,
        input.k,
        input.material
    );
}

void launch_utd_tiled_vjp(
    const drjit::detached_t<UTDTiledOpInput> &input,
    const drjit::detached_t<UTDTiledOpOutput> &grad_output,
    MaterializedGradientSlots &grad_slots,
    Float &grad_rx_x,
    Float &grad_rx_y,
    Float &grad_rx_z)
{
    drjit::eval(input.state_idx, input.rx_idx, input.valid_mask, input.ownership_code);
    drjit::eval(input.rx.rx_x, input.rx.rx_y, input.rx.rx_z);
    eval_utd_tiled_state_arrays(input.state);
    drjit::eval(
        grad_output.direct_vec_x_re, grad_output.direct_vec_x_im,
        grad_output.direct_vec_y_re, grad_output.direct_vec_y_im,
        grad_output.direct_vec_z_re, grad_output.direct_vec_z_im,
        grad_output.multi_vec_x_re, grad_output.multi_vec_x_im,
        grad_output.multi_vec_y_re, grad_output.multi_vec_y_im,
        grad_output.multi_vec_z_re, grad_output.multi_vec_z_im,
        grad_rx_x, grad_rx_y, grad_rx_z
    );
    std::vector<const float*> state_slots = utd_state_slot_ptrs(input.state);
    witwin::channel::native_ext::utd_accumulate_tiled_vjp_slots(
        drjit_data_ptr(input.state_idx),
        drjit_data_ptr(input.rx_idx),
        drjit_data_ptr(input.valid_mask),
        drjit_data_ptr(input.ownership_code),
        state_slots.data(),
        drjit_data_ptr(input.rx.rx_x),
        drjit_data_ptr(input.rx.rx_y),
        drjit_data_ptr(input.rx.rx_z),
        drjit_data_ptr(grad_output.direct_vec_x_re),
        drjit_data_ptr(grad_output.direct_vec_x_im),
        drjit_data_ptr(grad_output.direct_vec_y_re),
        drjit_data_ptr(grad_output.direct_vec_y_im),
        drjit_data_ptr(grad_output.direct_vec_z_re),
        drjit_data_ptr(grad_output.direct_vec_z_im),
        drjit_data_ptr(grad_output.multi_vec_x_re),
        drjit_data_ptr(grad_output.multi_vec_x_im),
        drjit_data_ptr(grad_output.multi_vec_y_re),
        drjit_data_ptr(grad_output.multi_vec_y_im),
        drjit_data_ptr(grad_output.multi_vec_z_re),
        drjit_data_ptr(grad_output.multi_vec_z_im),
        grad_slots.ptrs.data(),
        drjit_data_ptr_mut(grad_rx_x),
        drjit_data_ptr_mut(grad_rx_y),
        drjit_data_ptr_mut(grad_rx_z),
        input.n_local_states,
        input.n_local_receivers,
        input.k,
        input.material
    );
}

void launch_utd_pair_vjp(
    const drjit::detached_t<UTDPairOpInput> &input,
    const drjit::detached_t<UTDPairOpOutput> &grad_output,
    MaterializedGradientSlots &grad_slots,
    Float &grad_rx_x,
    Float &grad_rx_y,
    Float &grad_rx_z)
{
    drjit::eval(input.state_idx, input.rx_idx, input.ownership_code);
    drjit::eval(input.rx.rx_x, input.rx.rx_y, input.rx.rx_z);
    eval_utd_tiled_state_arrays(input.state);
    drjit::eval(
        grad_output.direct_re, grad_output.direct_im,
        grad_output.multi_re, grad_output.multi_im,
        grad_output.direct_vec_x_re, grad_output.direct_vec_x_im,
        grad_output.direct_vec_y_re, grad_output.direct_vec_y_im,
        grad_output.direct_vec_z_re, grad_output.direct_vec_z_im,
        grad_output.multi_vec_x_re, grad_output.multi_vec_x_im,
        grad_output.multi_vec_y_re, grad_output.multi_vec_y_im,
        grad_output.multi_vec_z_re, grad_output.multi_vec_z_im,
        grad_rx_x, grad_rx_y, grad_rx_z
    );
    std::vector<const float*> state_slots = utd_state_slot_ptrs(input.state);
    witwin::channel::native_ext::utd_pair_vjp_slots(
        drjit_data_ptr(input.state_idx),
        drjit_data_ptr(input.rx_idx),
        drjit_data_ptr(input.ownership_code),
        state_slots.data(),
        drjit_data_ptr(input.rx.rx_x),
        drjit_data_ptr(input.rx.rx_y),
        drjit_data_ptr(input.rx.rx_z),
        drjit_data_ptr(grad_output.direct_re),
        drjit_data_ptr(grad_output.direct_im),
        drjit_data_ptr(grad_output.multi_re),
        drjit_data_ptr(grad_output.multi_im),
        drjit_data_ptr(grad_output.direct_vec_x_re),
        drjit_data_ptr(grad_output.direct_vec_x_im),
        drjit_data_ptr(grad_output.direct_vec_y_re),
        drjit_data_ptr(grad_output.direct_vec_y_im),
        drjit_data_ptr(grad_output.direct_vec_z_re),
        drjit_data_ptr(grad_output.direct_vec_z_im),
        drjit_data_ptr(grad_output.multi_vec_x_re),
        drjit_data_ptr(grad_output.multi_vec_x_im),
        drjit_data_ptr(grad_output.multi_vec_y_re),
        drjit_data_ptr(grad_output.multi_vec_y_im),
        drjit_data_ptr(grad_output.multi_vec_z_re),
        drjit_data_ptr(grad_output.multi_vec_z_im),
        grad_slots.ptrs.data(),
        drjit_data_ptr_mut(grad_rx_x),
        drjit_data_ptr_mut(grad_rx_y),
        drjit_data_ptr_mut(grad_rx_z),
        input.n_pairs,
        input.k,
        input.material
    );
}

class UTDTiledAccumulateOp
    : public WitwinCustomOp<UTDTiledOpOutput, UTDTiledOpInput> {
public:
    using Base = WitwinCustomOp<UTDTiledOpOutput, UTDTiledOpInput>;
    using OutputType = typename Base::OutputType;

    explicit UTDTiledAccumulateOp(const UTDTiledOpInput &input)
        : Base(input) {}

    OutputType eval(drjit::detached_t<UTDTiledOpInput> input) {
        m_input = input;
        OutputType output = zero_utd_tiled_output(input.rx.rx_x.size());
        if (input.n_local_states > 0 && input.n_local_receivers > 0) {
            launch_utd_tiled_forward(input, output);
        }
        return output;
    }

    void forward() override {
        auto output = zero_utd_tiled_output_grad(m_input.rx.rx_x.size());
        if (m_input.n_local_states > 0 && m_input.n_local_receivers > 0) {
            launch_utd_tiled_jvp(
                m_input,
                drjit::grad<false>(this->m_registered_input),
                output
            );
        }
        set_utd_tiled_output_grad(this->m_registered_output, output);
    }

    void backward() override {
        if (m_input.n_local_states <= 0 || m_input.n_local_receivers <= 0) {
            return;
        }
        auto grad_output = drjit::grad<false>(this->m_registered_output);
        MaterializedGradientSlots grad_slots = materialize_utd_gradient_slots(m_input.state);
        Float grad_rx_x = drjit::zeros<Float>(m_input.rx.rx_x.size());
        Float grad_rx_y = drjit::zeros<Float>(m_input.rx.rx_y.size());
        Float grad_rx_z = drjit::zeros<Float>(m_input.rx.rx_z.size());
        drjit::eval(grad_rx_x, grad_rx_y, grad_rx_z);
        launch_utd_tiled_vjp(
            m_input,
            grad_output,
            grad_slots,
            grad_rx_x,
            grad_rx_y,
            grad_rx_z
        );
        accum_utd_state_grads(this->m_registered_input.state, grad_slots);
        accum_utd_receiver_grads(
            this->m_registered_input.rx,
            grad_rx_x,
            grad_rx_y,
            grad_rx_z
        );
    }

    const char *name() const override { return "UTDTiledAccumulate"; }

private:
    drjit::detached_t<UTDTiledOpInput> m_input;
};

class UTDPairAccumulateOp
    : public WitwinCustomOp<UTDPairOpOutput, UTDPairOpInput> {
public:
    using Base = WitwinCustomOp<UTDPairOpOutput, UTDPairOpInput>;
    using OutputType = typename Base::OutputType;

    explicit UTDPairAccumulateOp(const UTDPairOpInput &input)
        : Base(input) {}

    OutputType eval(drjit::detached_t<UTDPairOpInput> input) {
        m_input = input;
        OutputType output = zero_utd_pair_output(input.rx.rx_x.size());
        if (input.n_pairs > 0) {
            launch_utd_pair_forward(input, output);
        }
        return output;
    }

    void forward() override {
        auto output = zero_utd_pair_output_grad(m_input.rx.rx_x.size());
        if (m_input.n_pairs > 0) {
            launch_utd_pair_jvp(
                m_input,
                drjit::grad<false>(this->m_registered_input),
                output
            );
        }
        set_utd_pair_output_grad(this->m_registered_output, output);
    }

    void backward() override {
        if (m_input.n_pairs <= 0) {
            return;
        }
        auto grad_output = drjit::grad<false>(this->m_registered_output);
        MaterializedGradientSlots grad_slots = materialize_utd_gradient_slots(m_input.state);
        Float grad_rx_x = drjit::zeros<Float>(m_input.rx.rx_x.size());
        Float grad_rx_y = drjit::zeros<Float>(m_input.rx.rx_y.size());
        Float grad_rx_z = drjit::zeros<Float>(m_input.rx.rx_z.size());
        drjit::eval(grad_rx_x, grad_rx_y, grad_rx_z);
        launch_utd_pair_vjp(
            m_input,
            grad_output,
            grad_slots,
            grad_rx_x,
            grad_rx_y,
            grad_rx_z
        );
        accum_utd_state_grads(this->m_registered_input.state, grad_slots);
        accum_utd_receiver_grads(
            this->m_registered_input.rx,
            grad_rx_x,
            grad_rx_y,
            grad_rx_z
        );
    }

    const char *name() const override { return "UTDPairAccumulate"; }

private:
    drjit::detached_t<UTDPairOpInput> m_input;
};


void register_utd_bindings(nb::module_ &m) {

    // MaterialParams struct binding

    nb::class_<witwin::channel::native_ext::MaterialParams>(m, "MaterialParams")

        .def(nb::init<>())

        .def(nb::init<int, float, float, float, float, float>(),

             nb::arg("use_fresnel"), nb::arg("eta_r"),

             nb::arg("mu_r"), nb::arg("sigma"), nb::arg("gain"), nb::arg("omega"))

        .def_rw("use_fresnel", &witwin::channel::native_ext::MaterialParams::useFresnel)

        .def_rw("eta_r",       &witwin::channel::native_ext::MaterialParams::etaR)

        .def_rw("mu_r",        &witwin::channel::native_ext::MaterialParams::muR)

        .def_rw("sigma",       &witwin::channel::native_ext::MaterialParams::sigma)

        .def_rw("gain",        &witwin::channel::native_ext::MaterialParams::gain)

        .def_rw("omega",       &witwin::channel::native_ext::MaterialParams::omega)

        .def_rw("tx_pol_x",    &witwin::channel::native_ext::MaterialParams::txPolX)

        .def_rw("tx_pol_y",    &witwin::channel::native_ext::MaterialParams::txPolY)

        .def_rw("tx_pol_z",    &witwin::channel::native_ext::MaterialParams::txPolZ);

    m.def(
        "utd_accumulate_tiled_vectors",
        [](
            Int32 state_idx,
            Int32 rx_idx,
            Int32 valid_mask,
            Int32 ownership_code,
            nb::tuple state_soa,
            nb::tuple rx_arrays,
            witwin::channel::native_ext::MaterialParams material,
            int n_local_states,
            int n_local_receivers,
            float k
        ) {
            if (nb::len(rx_arrays) != 3) {
                throw std::runtime_error("utd_accumulate_tiled_vectors expected 3 receiver arrays");
            }
            UTDTiledOpInput input{
                state_idx,
                rx_idx,
                valid_mask,
                ownership_code,
                make_utd_tiled_state_arrays(state_soa, "utd_accumulate_tiled_vectors"),
                {
                    nb::cast<DiffFloat>(rx_arrays[0]),
                    nb::cast<DiffFloat>(rx_arrays[1]),
                    nb::cast<DiffFloat>(rx_arrays[2]),
                },
                material,
                n_local_states,
                n_local_receivers,
                k,
            };
            UTDTiledOpOutput output = witwin_custom_op<UTDTiledAccumulateOp>(input);
            return nb::make_tuple(
                output.direct_vec_x_re,
                output.direct_vec_x_im,
                output.direct_vec_y_re,
                output.direct_vec_y_im,
                output.direct_vec_z_re,
                output.direct_vec_z_im,
                output.multi_vec_x_re,
                output.multi_vec_x_im,
                output.multi_vec_y_re,
                output.multi_vec_y_im,
                output.multi_vec_z_re,
                output.multi_vec_z_im
            );
        },
        "Launch finite-wedge UTD tiled vector accumulation with Dr.Jit forward-mode AD support."
    );


    m.def(
        "utd_pair_vectors",
        [](
            Int32 state_idx,
            Int32 rx_idx,
            Int32 ownership_code,
            nb::tuple state_soa,
            nb::tuple rx_arrays,
            witwin::channel::native_ext::MaterialParams material,
            int n_pairs,
            float k
        ) {
            if (nb::len(rx_arrays) != 3) {
                throw std::runtime_error("utd_pair_vectors expected 3 receiver arrays");
            }
            UTDPairOpInput input{
                state_idx,
                rx_idx,
                ownership_code,
                make_utd_tiled_state_arrays(state_soa, "utd_pair_vectors"),
                {
                    nb::cast<DiffFloat>(rx_arrays[0]),
                    nb::cast<DiffFloat>(rx_arrays[1]),
                    nb::cast<DiffFloat>(rx_arrays[2]),
                },
                material,
                n_pairs,
                k,
            };
            UTDPairOpOutput output = witwin_custom_op<UTDPairAccumulateOp>(input);
            return nb::make_tuple(
                output.direct_re,
                output.direct_im,
                output.multi_re,
                output.multi_im,
                output.direct_vec_x_re,
                output.direct_vec_x_im,
                output.direct_vec_y_re,
                output.direct_vec_y_im,
                output.direct_vec_z_re,
                output.direct_vec_z_im,
                output.multi_vec_x_re,
                output.multi_vec_x_im,
                output.multi_vec_y_re,
                output.multi_vec_y_im,
                output.multi_vec_z_re,
                output.multi_vec_z_im
            );
        },
        "Launch one-to-one finite-wedge UTD pair vector evaluation with Dr.Jit forward-mode AD support."
    );




}

