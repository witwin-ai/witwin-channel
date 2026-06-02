#include "drjit_common.h"
#include <trace/utd/bind.h>

#include <trace/utd/utd_types.h>
#include <trace/utd/utd_math.h>
#include <trace/utd/utd_accumulate.h>
#include <trace/utd/utd_jvp.h>
#include <monitors/common/utd_state_tiles/utd_state_tiles.h>
#include <common/cuda_check.h>


// ---------------------------------------------------------------------------

// Python dict helpers for debug terms

// ---------------------------------------------------------------------------



inline nb::dict pack_complex_dict(witwin::channel::native_ext::Complex value) {

    nb::dict result;

    result["re"] = value.re;

    result["im"] = value.im;

    return result;

}



inline nb::dict pack_float3_dict(witwin::channel::native_ext::float3a value) {

    nb::dict result;

    result["x"] = value.x;

    result["y"] = value.y;

    result["z"] = value.z;

    return result;

}



inline nb::dict pack_complex3_dict(witwin::channel::native_ext::Complex3 value) {

    nb::dict result;

    result["x"] = pack_complex_dict(value.x);

    result["y"] = pack_complex_dict(value.y);

    result["z"] = pack_complex_dict(value.z);

    return result;

}



inline nb::dict pack_jones_dict(witwin::channel::native_ext::Jones2 value) {

    nb::dict result;

    result["u"] = pack_complex_dict(value.u);

    result["v"] = pack_complex_dict(value.v);

    return result;

}



inline nb::dict pack_jones_operator_dict(witwin::channel::native_ext::JonesOperator value) {

    nb::dict result;

    result["m00"] = pack_complex_dict(value.m00);

    result["m01"] = pack_complex_dict(value.m01);

    result["m10"] = pack_complex_dict(value.m10);

    result["m11"] = pack_complex_dict(value.m11);

    return result;

}



inline nb::dict pack_basis_dict(witwin::channel::native_ext::Basis3 value) {

    nb::dict result;

    result["u"] = pack_float3_dict(value.u);

    result["v"] = pack_float3_dict(value.v);

    result["k"] = pack_float3_dict(value.k);

    return result;

}



inline witwin::channel::native_ext::Complex read_complex_dict(nb::handle obj) {

    nb::dict value = nb::cast<nb::dict>(obj);

    return witwin::channel::native_ext::cplx(
        nb::cast<float>(value["re"]),
        nb::cast<float>(value["im"])
    );

}



inline witwin::channel::native_ext::float3a read_float3_dict(nb::handle obj) {

    nb::dict value = nb::cast<nb::dict>(obj);

    return witwin::channel::native_ext::make_f3(
        nb::cast<float>(value["x"]),
        nb::cast<float>(value["y"]),
        nb::cast<float>(value["z"])
    );

}



inline witwin::channel::native_ext::Jones2 read_jones_dict(nb::handle obj) {

    nb::dict value = nb::cast<nb::dict>(obj);

    return {
        read_complex_dict(value["u"]),
        read_complex_dict(value["v"])
    };

}



inline witwin::channel::native_ext::JonesOperator read_jones_operator_dict(nb::handle obj) {

    nb::dict value = nb::cast<nb::dict>(obj);

    return {
        read_complex_dict(value["m00"]),
        read_complex_dict(value["m01"]),
        read_complex_dict(value["m10"]),
        read_complex_dict(value["m11"])
    };

}



inline witwin::channel::native_ext::Basis3 read_basis_dict(nb::handle obj) {

    nb::dict value = nb::cast<nb::dict>(obj);

    return {
        read_float3_dict(value["u"]),
        read_float3_dict(value["v"]),
        read_float3_dict(value["k"])
    };

}



inline witwin::channel::native_ext::FaceMaterialParams read_face_material_dict(nb::handle obj) {

    nb::dict value = nb::cast<nb::dict>(obj);

    return {
        nb::cast<float>(value["eta_r"]),
        nb::cast<float>(value["sigma"]),
        nb::cast<float>(value["gain"]),
        nb::cast<float>(value["use_fresnel"]),
        nb::cast<float>(value["present"])
    };

}



inline float read_device_float_at(const DiffFloat &arr, size_t index, const char *label) {

    drjit::eval(arr);
    float value = 0.0f;
    witwin::channel::native_ext::common::throw_cuda(
        cudaMemcpy(
            &value,
            drjit_data_ptr(arr) + index,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        label
    );
    return value;

}


inline nb::dict pack_state_soa_debug_dict(nb::tuple state_soa, size_t state_index) {

    if (nb::len(state_soa) != 81) {
        throw std::runtime_error("utd_debug_state_soa_at expected 81 state arrays");
    }

    std::vector<DiffFloat> arrays;
    arrays.reserve(81);
    for (size_t i = 0; i < 81; ++i) {
        arrays.emplace_back(nb::cast<DiffFloat>(state_soa[i]));
    }

    auto read = [&](size_t slot) -> float {
        return read_device_float_at(arrays[slot], state_index, "utd_debug_state_soa_at");
    };

    nb::dict result;
    auto pack_float3 = [&](size_t x_slot, size_t y_slot, size_t z_slot) {
        nb::dict value;
        value["x"] = read(x_slot);
        value["y"] = read(y_slot);
        value["z"] = read(z_slot);
        return value;
    };
    auto pack_complex = [&](size_t re_slot, size_t im_slot) {
        nb::dict value;
        value["re"] = read(re_slot);
        value["im"] = read(im_slot);
        return value;
    };
    auto pack_jones = [&](size_t u_re, size_t u_im, size_t v_re, size_t v_im) {
        nb::dict value;
        value["u"] = pack_complex(u_re, u_im);
        value["v"] = pack_complex(v_re, v_im);
        return value;
    };
    auto pack_basis = [&](size_t u0, size_t u1, size_t u2, size_t v0, size_t v1, size_t v2, size_t k0, size_t k1, size_t k2) {
        nb::dict value;
        value["u"] = pack_float3(u0, u1, u2);
        value["v"] = pack_float3(v0, v1, v2);
        value["k"] = pack_float3(k0, k1, k2);
        return value;
    };
    auto pack_operator = [&](size_t m00_re, size_t m00_im, size_t m01_re, size_t m01_im, size_t m10_re, size_t m10_im, size_t m11_re, size_t m11_im) {
        nb::dict value;
        value["m00"] = pack_complex(m00_re, m00_im);
        value["m01"] = pack_complex(m01_re, m01_im);
        value["m10"] = pack_complex(m10_re, m10_im);
        value["m11"] = pack_complex(m11_re, m11_im);
        return value;
    };
    auto pack_material = [&](size_t eta_slot, size_t sigma_slot, size_t gain_slot, size_t fresnel_slot, size_t present_slot) {
        nb::dict value;
        value["eta_r"] = read(eta_slot);
        value["sigma"] = read(sigma_slot);
        value["gain"] = read(gain_slot);
        value["use_fresnel"] = read(fresnel_slot);
        value["present"] = read(present_slot);
        return value;
    };

    result["edge_pos"] = pack_float3(0, 1, 2);
    result["edge_dir"] = pack_float3(3, 4, 5);
    result["n0"] = pack_float3(6, 7, 8);
    result["nn"] = pack_float3(9, 10, 11);
    result["wedge_n"] = read(12);
    result["edge_line_min"] = read(13);
    result["edge_line_max"] = read(14);
    result["source_pos"] = pack_float3(15, 16, 17);
    result["incident_field"] = pack_complex(18, 19);
    result["incident_normal_derivative"] = pack_complex(20, 21);
    result["r0"] = pack_complex(22, 23);
    result["rn"] = pack_complex(24, 25);
    result["incident_vector_x"] = pack_complex(26, 27);
    result["incident_vector_y"] = pack_complex(28, 29);
    result["incident_vector_z"] = pack_complex(30, 31);
    result["incident_derivative_vector_x"] = pack_complex(32, 33);
    result["incident_derivative_vector_y"] = pack_complex(34, 35);
    result["incident_derivative_vector_z"] = pack_complex(36, 37);
    result["incident_jones"] = pack_jones(38, 39, 40, 41);
    result["incident_derivative_jones"] = pack_jones(42, 43, 44, 45);
    result["incident_basis"] = pack_basis(46, 47, 48, 49, 50, 51, 52, 53, 54);
    result["face0_operator"] = pack_operator(55, 56, 57, 58, 59, 60, 61, 62);
    result["face1_operator"] = pack_operator(63, 64, 65, 66, 67, 68, 69, 70);
    result["face0_material"] = pack_material(71, 72, 73, 74, 75);
    result["face1_material"] = pack_material(76, 77, 78, 79, 80);
    return result;

}


struct MaterializedStateSlots {
    std::vector<DiffFloat> arrays;
    std::vector<const float*> ptrs;
};


inline MaterializedStateSlots materialize_state_slot_pointers(nb::tuple state_soa, const char *label) {

    if (nb::len(state_soa) != 81) {
        throw std::runtime_error(std::string(label) + " expected 81 state arrays");
    }

    MaterializedStateSlots materialized;
    materialized.arrays.reserve(81);
    materialized.ptrs.reserve(81);
    for (size_t i = 0; i < 81; ++i) {
        materialized.arrays.emplace_back(nb::cast<DiffFloat>(state_soa[i]));
        drjit::eval(materialized.arrays.back());
        materialized.ptrs.push_back(drjit_data_ptr(materialized.arrays.back()));
    }
    return materialized;

}


inline witwin::channel::native_ext::PairInputs read_pair_inputs_dict(nb::dict state) {

    using namespace witwin::channel::native_ext;

    PairInputs pair_state;
    pair_state.edgePos = read_float3_dict(state["edge_pos"]);
    pair_state.edgeDir = read_float3_dict(state["edge_dir"]);
    pair_state.n0 = read_float3_dict(state["n0"]);
    pair_state.nn = read_float3_dict(state["nn"]);
    pair_state.wedgeN = nb::cast<float>(state["wedge_n"]);
    pair_state.edgeLineMin = nb::cast<float>(state["edge_line_min"]);
    pair_state.edgeLineMax = nb::cast<float>(state["edge_line_max"]);
    pair_state.sourcePos = read_float3_dict(state["source_pos"]);
    pair_state.incidentField = read_complex_dict(state["incident_field"]);
    pair_state.incidentNormalDerivative = read_complex_dict(state["incident_normal_derivative"]);
    pair_state.r0 = read_complex_dict(state["r0"]);
    pair_state.rn = read_complex_dict(state["rn"]);
    pair_state.incidentVector = {
        read_complex_dict(state["incident_vector_x"]),
        read_complex_dict(state["incident_vector_y"]),
        read_complex_dict(state["incident_vector_z"])
    };
    pair_state.incidentDerivativeVector = {
        read_complex_dict(state["incident_derivative_vector_x"]),
        read_complex_dict(state["incident_derivative_vector_y"]),
        read_complex_dict(state["incident_derivative_vector_z"])
    };
    pair_state.incidentJones = read_jones_dict(state["incident_jones"]);
    pair_state.incidentDerivativeJones = read_jones_dict(state["incident_derivative_jones"]);
    pair_state.incidentBasis = read_basis_dict(state["incident_basis"]);
    pair_state.face0Operator = read_jones_operator_dict(state["face0_operator"]);
    pair_state.face1Operator = read_jones_operator_dict(state["face1_operator"]);
    pair_state.face0Material = read_face_material_dict(state["face0_material"]);
    pair_state.face1Material = read_face_material_dict(state["face1_material"]);
    return pair_state;

}


inline nb::dict pack_pair_contribution_debug_dict(
    const witwin::channel::native_ext::PairContributionDebug &value
) {

    nb::dict result;
    result["src_ext"] = value.srcExt != 0;
    result["tgt_ext"] = value.tgtExt != 0;
    result["geom_valid"] = value.geomValid != 0;
    result["pole_safe"] = value.poleSafe != 0;
    result["slope_safe"] = value.slopeSafe != 0;
    result["phi"] = value.phi;
    result["phi_prime"] = value.phiPrime;
    result["s"] = value.s;
    result["s_prime"] = value.sPrime;
    result["sin_beta0"] = value.sinBeta0;
    result["finite_factor"] = pack_complex_dict(value.finiteFactor);
    result["field"] = pack_complex_dict(value.field);
    result["direct_gain"] = pack_complex_dict(value.directGain);
    result["derivative_gain"] = pack_complex_dict(value.derivativeGain);
    result["vector_field"] = pack_complex3_dict(value.vectorField);
    return result;

}


inline nb::dict pack_diffraction_terms_dict(

    const witwin::channel::native_ext::DiffractionOperatorTerms &terms

) {

    nb::dict result;

    result["direct"] = pack_complex_dict(terms.direct);

    result["face0"] = pack_complex_dict(terms.face0);

    result["face1"] = pack_complex_dict(terms.face1);

    result["direct_dphi_prime"] = pack_complex_dict(terms.directDphiPrime);

    result["face0_dphi_prime"] = pack_complex_dict(terms.face0DphiPrime);

    result["face1_dphi_prime"] = pack_complex_dict(terms.face1DphiPrime);

    return result;

}



// ---------------------------------------------------------------------------

// DRJIT_STRUCT types for UTD accumulation

// ---------------------------------------------------------------------------



struct UTDStateArrays {

    DiffFloat edge_pos_x;

    DiffFloat edge_pos_y;

    DiffFloat edge_pos_z;

    DiffFloat edge_dir_x;

    DiffFloat edge_dir_y;

    DiffFloat edge_dir_z;

    DiffFloat n0_x;

    DiffFloat n0_y;

    DiffFloat n0_z;

    DiffFloat nn_x;

    DiffFloat nn_y;

    DiffFloat nn_z;

    DiffFloat wedge_n;

    DiffFloat source_pos_x;

    DiffFloat source_pos_y;

    DiffFloat source_pos_z;

    DiffFloat incident_field_re;

    DiffFloat incident_field_im;

    DiffFloat incident_nderiv_re;

    DiffFloat incident_nderiv_im;

    DiffFloat r0_re;

    DiffFloat r0_im;

    DiffFloat rn_re;

    DiffFloat rn_im;

    DiffFloat inc_vec_x_re;

    DiffFloat inc_vec_x_im;

    DiffFloat inc_vec_y_re;

    DiffFloat inc_vec_y_im;

    DiffFloat inc_vec_z_re;

    DiffFloat inc_vec_z_im;

    DiffFloat inc_dvec_x_re;

    DiffFloat inc_dvec_x_im;

    DiffFloat inc_dvec_y_re;

    DiffFloat inc_dvec_y_im;

    DiffFloat inc_dvec_z_re;

    DiffFloat inc_dvec_z_im;

    DiffFloat inc_jones_u_re;

    DiffFloat inc_jones_u_im;

    DiffFloat inc_jones_v_re;

    DiffFloat inc_jones_v_im;

    DiffFloat inc_djones_u_re;

    DiffFloat inc_djones_u_im;

    DiffFloat inc_djones_v_re;

    DiffFloat inc_djones_v_im;

    DiffFloat inc_basis_u_x;

    DiffFloat inc_basis_u_y;

    DiffFloat inc_basis_u_z;

    DiffFloat inc_basis_v_x;

    DiffFloat inc_basis_v_y;

    DiffFloat inc_basis_v_z;

    DiffFloat inc_basis_k_x;

    DiffFloat inc_basis_k_y;

    DiffFloat inc_basis_k_z;

    DiffFloat face0_op_m00_re;

    DiffFloat face0_op_m00_im;

    DiffFloat face0_op_m01_re;

    DiffFloat face0_op_m01_im;

    DiffFloat face0_op_m10_re;

    DiffFloat face0_op_m10_im;

    DiffFloat face0_op_m11_re;

    DiffFloat face0_op_m11_im;

    DiffFloat face1_op_m00_re;

    DiffFloat face1_op_m00_im;

    DiffFloat face1_op_m01_re;

    DiffFloat face1_op_m01_im;

    DiffFloat face1_op_m10_re;

    DiffFloat face1_op_m10_im;

    DiffFloat face1_op_m11_re;

    DiffFloat face1_op_m11_im;

    DiffFloat face0_eta_r;

    DiffFloat face0_sigma;

    DiffFloat face0_gain;

    DiffFloat face0_use_fresnel;

    DiffFloat face0_present;

    DiffFloat face1_eta_r;

    DiffFloat face1_sigma;

    DiffFloat face1_gain;

    DiffFloat face1_use_fresnel;

    DiffFloat face1_present;



    DRJIT_STRUCT(

        UTDStateArrays,

        edge_pos_x, edge_pos_y, edge_pos_z,

        edge_dir_x, edge_dir_y, edge_dir_z,

        n0_x, n0_y, n0_z,

        nn_x, nn_y, nn_z,

        wedge_n,

        source_pos_x, source_pos_y, source_pos_z,

        incident_field_re, incident_field_im,

        incident_nderiv_re, incident_nderiv_im,

        r0_re, r0_im, rn_re, rn_im,

        inc_vec_x_re, inc_vec_x_im,

        inc_vec_y_re, inc_vec_y_im,

        inc_vec_z_re, inc_vec_z_im,

        inc_dvec_x_re, inc_dvec_x_im,

        inc_dvec_y_re, inc_dvec_y_im,

        inc_dvec_z_re, inc_dvec_z_im,

        inc_jones_u_re, inc_jones_u_im,

        inc_jones_v_re, inc_jones_v_im,

        inc_djones_u_re, inc_djones_u_im,

        inc_djones_v_re, inc_djones_v_im,

        inc_basis_u_x, inc_basis_u_y, inc_basis_u_z,

        inc_basis_v_x, inc_basis_v_y, inc_basis_v_z,

        inc_basis_k_x, inc_basis_k_y, inc_basis_k_z,

        face0_op_m00_re, face0_op_m00_im,

        face0_op_m01_re, face0_op_m01_im,

        face0_op_m10_re, face0_op_m10_im,

        face0_op_m11_re, face0_op_m11_im,

        face1_op_m00_re, face1_op_m00_im,

        face1_op_m01_re, face1_op_m01_im,

        face1_op_m10_re, face1_op_m10_im,

        face1_op_m11_re, face1_op_m11_im,

        face0_eta_r, face0_sigma, face0_gain, face0_use_fresnel, face0_present,

        face1_eta_r, face1_sigma, face1_gain, face1_use_fresnel, face1_present

    );

};



struct UTDReceiverArrays {

    DiffFloat rx_x;

    DiffFloat rx_y;

    DiffFloat rx_z;



    DRJIT_STRUCT(UTDReceiverArrays, rx_x, rx_y, rx_z);

};



struct UTDOpInput {

    Int32 state_idx;

    Int32 rx_idx;

    Int32 ownership_code;

    UTDStateArrays state;

    UTDReceiverArrays rx;

    witwin::channel::native_ext::MaterialParams material;

    int n_pairs;

    float k;



    DRJIT_STRUCT(

        UTDOpInput,

        state_idx, rx_idx, ownership_code,

        state, rx, material,

        n_pairs, k

    );

};



struct UTDOpOutput {

    DiffFloat direct_vec_x_re;

    DiffFloat direct_vec_x_im;

    DiffFloat direct_vec_y_re;

    DiffFloat direct_vec_y_im;

    DiffFloat direct_vec_z_re;

    DiffFloat direct_vec_z_im;

    DiffFloat multi_vec_x_re;

    DiffFloat multi_vec_x_im;

    DiffFloat multi_vec_y_re;

    DiffFloat multi_vec_y_im;

    DiffFloat multi_vec_z_re;

    DiffFloat multi_vec_z_im;



    DRJIT_STRUCT(

        UTDOpOutput,

        direct_vec_x_re, direct_vec_x_im,

        direct_vec_y_re, direct_vec_y_im,

        direct_vec_z_re, direct_vec_z_im,

        multi_vec_x_re, multi_vec_x_im,

        multi_vec_y_re, multi_vec_y_im,

        multi_vec_z_re, multi_vec_z_im

    );

};



// ---------------------------------------------------------------------------

// Zero / grad helpers

// ---------------------------------------------------------------------------



inline UTDOpOutput zero_utd_output(size_t width) {

    return {

        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),

        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),

        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),

        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),

        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),

        drjit::zeros<DiffFloat>(width), drjit::zeros<DiffFloat>(width),

    };

}



inline drjit::detached_t<UTDOpOutput> zero_utd_output_grad(size_t width) {

    return {

        drjit::zeros<Float>(width), drjit::zeros<Float>(width),

        drjit::zeros<Float>(width), drjit::zeros<Float>(width),

        drjit::zeros<Float>(width), drjit::zeros<Float>(width),

        drjit::zeros<Float>(width), drjit::zeros<Float>(width),

        drjit::zeros<Float>(width), drjit::zeros<Float>(width),

        drjit::zeros<Float>(width), drjit::zeros<Float>(width),

    };

}



inline void set_utd_output_grad(

    UTDOpOutput &registered_output,

    const drjit::detached_t<UTDOpOutput> &grad_output

) {

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



struct UTDBackwardGrads {

    DiffFloat edge_pos_x;

    DiffFloat edge_pos_y;

    DiffFloat edge_pos_z;

    DiffFloat edge_dir_x;

    DiffFloat edge_dir_y;

    DiffFloat edge_dir_z;

    DiffFloat n0_x;

    DiffFloat n0_y;

    DiffFloat n0_z;

    DiffFloat nn_x;

    DiffFloat nn_y;

    DiffFloat nn_z;

    DiffFloat wedge_n;

    DiffFloat source_pos_x;

    DiffFloat source_pos_y;

    DiffFloat source_pos_z;

    DiffFloat incident_field_re;

    DiffFloat incident_field_im;

    DiffFloat incident_nderiv_re;

    DiffFloat incident_nderiv_im;

    DiffFloat r0_re;

    DiffFloat r0_im;

    DiffFloat rn_re;

    DiffFloat rn_im;

    DiffFloat inc_vec_x_re;

    DiffFloat inc_vec_x_im;

    DiffFloat inc_vec_y_re;

    DiffFloat inc_vec_y_im;

    DiffFloat inc_vec_z_re;

    DiffFloat inc_vec_z_im;

    DiffFloat inc_dvec_x_re;

    DiffFloat inc_dvec_x_im;

    DiffFloat inc_dvec_y_re;

    DiffFloat inc_dvec_y_im;

    DiffFloat inc_dvec_z_re;

    DiffFloat inc_dvec_z_im;
    DiffFloat face0_op_m00_re;
    DiffFloat face0_op_m00_im;
    DiffFloat face0_op_m01_re;
    DiffFloat face0_op_m01_im;
    DiffFloat face0_op_m10_re;
    DiffFloat face0_op_m10_im;
    DiffFloat face0_op_m11_re;
    DiffFloat face0_op_m11_im;
    DiffFloat face1_op_m00_re;
    DiffFloat face1_op_m00_im;
    DiffFloat face1_op_m01_re;
    DiffFloat face1_op_m01_im;
    DiffFloat face1_op_m10_re;
    DiffFloat face1_op_m10_im;
    DiffFloat face1_op_m11_re;
    DiffFloat face1_op_m11_im;
    DiffFloat face0_eta_r;
    DiffFloat face0_sigma;
    DiffFloat face0_gain;
    DiffFloat face1_eta_r;
    DiffFloat face1_sigma;
    DiffFloat face1_gain;
    DiffFloat rx_x;
    DiffFloat rx_y;

    DiffFloat rx_z;

};



inline UTDBackwardGrads zero_utd_backward_grads(size_t n_states, size_t n_rx) {
    UTDBackwardGrads result{};

    result.edge_pos_x = drjit::zeros<DiffFloat>(n_states);
    result.edge_pos_y = drjit::zeros<DiffFloat>(n_states);
    result.edge_pos_z = drjit::zeros<DiffFloat>(n_states);
    result.edge_dir_x = drjit::zeros<DiffFloat>(n_states);
    result.edge_dir_y = drjit::zeros<DiffFloat>(n_states);
    result.edge_dir_z = drjit::zeros<DiffFloat>(n_states);
    result.n0_x = drjit::zeros<DiffFloat>(n_states);
    result.n0_y = drjit::zeros<DiffFloat>(n_states);
    result.n0_z = drjit::zeros<DiffFloat>(n_states);
    result.nn_x = drjit::zeros<DiffFloat>(n_states);
    result.nn_y = drjit::zeros<DiffFloat>(n_states);
    result.nn_z = drjit::zeros<DiffFloat>(n_states);
    result.wedge_n = drjit::zeros<DiffFloat>(n_states);
    result.source_pos_x = drjit::zeros<DiffFloat>(n_states);
    result.source_pos_y = drjit::zeros<DiffFloat>(n_states);
    result.source_pos_z = drjit::zeros<DiffFloat>(n_states);
    result.incident_field_re = drjit::zeros<DiffFloat>(n_states);
    result.incident_field_im = drjit::zeros<DiffFloat>(n_states);
    result.incident_nderiv_re = drjit::zeros<DiffFloat>(n_states);
    result.incident_nderiv_im = drjit::zeros<DiffFloat>(n_states);
    result.r0_re = drjit::zeros<DiffFloat>(n_states);
    result.r0_im = drjit::zeros<DiffFloat>(n_states);
    result.rn_re = drjit::zeros<DiffFloat>(n_states);
    result.rn_im = drjit::zeros<DiffFloat>(n_states);
    result.inc_vec_x_re = drjit::zeros<DiffFloat>(n_states);
    result.inc_vec_x_im = drjit::zeros<DiffFloat>(n_states);
    result.inc_vec_y_re = drjit::zeros<DiffFloat>(n_states);
    result.inc_vec_y_im = drjit::zeros<DiffFloat>(n_states);
    result.inc_vec_z_re = drjit::zeros<DiffFloat>(n_states);
    result.inc_vec_z_im = drjit::zeros<DiffFloat>(n_states);
    result.inc_dvec_x_re = drjit::zeros<DiffFloat>(n_states);
    result.inc_dvec_x_im = drjit::zeros<DiffFloat>(n_states);
    result.inc_dvec_y_re = drjit::zeros<DiffFloat>(n_states);
    result.inc_dvec_y_im = drjit::zeros<DiffFloat>(n_states);
    result.inc_dvec_z_re = drjit::zeros<DiffFloat>(n_states);
    result.inc_dvec_z_im = drjit::zeros<DiffFloat>(n_states);
    result.face0_op_m00_re = drjit::zeros<DiffFloat>(n_states);
    result.face0_op_m00_im = drjit::zeros<DiffFloat>(n_states);
    result.face0_op_m01_re = drjit::zeros<DiffFloat>(n_states);
    result.face0_op_m01_im = drjit::zeros<DiffFloat>(n_states);
    result.face0_op_m10_re = drjit::zeros<DiffFloat>(n_states);
    result.face0_op_m10_im = drjit::zeros<DiffFloat>(n_states);
    result.face0_op_m11_re = drjit::zeros<DiffFloat>(n_states);
    result.face0_op_m11_im = drjit::zeros<DiffFloat>(n_states);
    result.face1_op_m00_re = drjit::zeros<DiffFloat>(n_states);
    result.face1_op_m00_im = drjit::zeros<DiffFloat>(n_states);
    result.face1_op_m01_re = drjit::zeros<DiffFloat>(n_states);
    result.face1_op_m01_im = drjit::zeros<DiffFloat>(n_states);
    result.face1_op_m10_re = drjit::zeros<DiffFloat>(n_states);
    result.face1_op_m10_im = drjit::zeros<DiffFloat>(n_states);
    result.face1_op_m11_re = drjit::zeros<DiffFloat>(n_states);
    result.face1_op_m11_im = drjit::zeros<DiffFloat>(n_states);
    result.face0_eta_r = drjit::zeros<DiffFloat>(n_states);
    result.face0_sigma = drjit::zeros<DiffFloat>(n_states);
    result.face0_gain = drjit::zeros<DiffFloat>(n_states);
    result.face1_eta_r = drjit::zeros<DiffFloat>(n_states);
    result.face1_sigma = drjit::zeros<DiffFloat>(n_states);
    result.face1_gain = drjit::zeros<DiffFloat>(n_states);
    result.rx_x = drjit::zeros<DiffFloat>(n_rx);
    result.rx_y = drjit::zeros<DiffFloat>(n_rx);
    result.rx_z = drjit::zeros<DiffFloat>(n_rx);

    return result;
}

inline UTDBackwardGrads launch_utd_backward_grads(
    const drjit::detached_t<UTDOpInput> &input,

    const drjit::detached_t<UTDOpOutput> &grad_output

) {

    size_t n_states = input.state.edge_pos_x.size();

    size_t n_rx = input.rx.rx_x.size();

    UTDBackwardGrads grads = zero_utd_backward_grads(n_states, n_rx);

    Float zero_scalar = drjit::zeros<Float>(n_rx);



    drjit::eval(

        input.state_idx, input.rx_idx, input.ownership_code,

        input.state.edge_pos_x, input.state.edge_pos_y, input.state.edge_pos_z,

        input.state.edge_dir_x, input.state.edge_dir_y, input.state.edge_dir_z,

        input.state.n0_x, input.state.n0_y, input.state.n0_z,

        input.state.nn_x, input.state.nn_y, input.state.nn_z,

        input.state.wedge_n,

        input.state.source_pos_x, input.state.source_pos_y, input.state.source_pos_z,

        input.state.incident_field_re, input.state.incident_field_im,

        input.state.incident_nderiv_re, input.state.incident_nderiv_im,

        input.state.r0_re, input.state.r0_im,

        input.state.rn_re, input.state.rn_im,

        input.state.inc_vec_x_re, input.state.inc_vec_x_im,

        input.state.inc_vec_y_re, input.state.inc_vec_y_im,

        input.state.inc_vec_z_re, input.state.inc_vec_z_im,

        input.state.inc_dvec_x_re, input.state.inc_dvec_x_im,

        input.state.inc_dvec_y_re, input.state.inc_dvec_y_im,

        input.state.inc_dvec_z_re, input.state.inc_dvec_z_im,

        input.state.inc_jones_u_re, input.state.inc_jones_u_im,

        input.state.inc_jones_v_re, input.state.inc_jones_v_im,

        input.state.inc_djones_u_re, input.state.inc_djones_u_im,

        input.state.inc_djones_v_re, input.state.inc_djones_v_im,

        input.state.inc_basis_u_x, input.state.inc_basis_u_y, input.state.inc_basis_u_z,

        input.state.inc_basis_v_x, input.state.inc_basis_v_y, input.state.inc_basis_v_z,

        input.state.inc_basis_k_x, input.state.inc_basis_k_y, input.state.inc_basis_k_z,

        input.state.face0_op_m00_re, input.state.face0_op_m00_im,

        input.state.face0_op_m01_re, input.state.face0_op_m01_im,

        input.state.face0_op_m10_re, input.state.face0_op_m10_im,

        input.state.face0_op_m11_re, input.state.face0_op_m11_im,

        input.state.face1_op_m00_re, input.state.face1_op_m00_im,

        input.state.face1_op_m01_re, input.state.face1_op_m01_im,

        input.state.face1_op_m10_re, input.state.face1_op_m10_im,

        input.state.face1_op_m11_re, input.state.face1_op_m11_im,

        input.state.face0_eta_r, input.state.face0_sigma, input.state.face0_gain,

        input.state.face0_use_fresnel, input.state.face0_present,

        input.state.face1_eta_r, input.state.face1_sigma, input.state.face1_gain,

        input.state.face1_use_fresnel, input.state.face1_present,

        input.rx.rx_x, input.rx.rx_y, input.rx.rx_z,

        zero_scalar, zero_scalar, zero_scalar, zero_scalar,

        grad_output.direct_vec_x_re, grad_output.direct_vec_x_im,

        grad_output.direct_vec_y_re, grad_output.direct_vec_y_im,

        grad_output.direct_vec_z_re, grad_output.direct_vec_z_im,

        grad_output.multi_vec_x_re, grad_output.multi_vec_x_im,

        grad_output.multi_vec_y_re, grad_output.multi_vec_y_im,

        grad_output.multi_vec_z_re, grad_output.multi_vec_z_im,

        grads.edge_pos_x, grads.edge_pos_y, grads.edge_pos_z,

        grads.edge_dir_x, grads.edge_dir_y, grads.edge_dir_z,

        grads.n0_x, grads.n0_y, grads.n0_z,

        grads.nn_x, grads.nn_y, grads.nn_z,

        grads.wedge_n,

        grads.source_pos_x, grads.source_pos_y, grads.source_pos_z,

        grads.incident_field_re, grads.incident_field_im,

        grads.incident_nderiv_re, grads.incident_nderiv_im,

        grads.r0_re, grads.r0_im,

        grads.rn_re, grads.rn_im,

        grads.inc_vec_x_re, grads.inc_vec_x_im,

        grads.inc_vec_y_re, grads.inc_vec_y_im,

        grads.inc_vec_z_re, grads.inc_vec_z_im,

        grads.inc_dvec_x_re, grads.inc_dvec_x_im,

        grads.inc_dvec_y_re, grads.inc_dvec_y_im,

        grads.inc_dvec_z_re, grads.inc_dvec_z_im,
        grads.face0_op_m00_re, grads.face0_op_m00_im,
        grads.face0_op_m01_re, grads.face0_op_m01_im,
        grads.face0_op_m10_re, grads.face0_op_m10_im,
        grads.face0_op_m11_re, grads.face0_op_m11_im,
        grads.face1_op_m00_re, grads.face1_op_m00_im,
        grads.face1_op_m01_re, grads.face1_op_m01_im,
        grads.face1_op_m10_re, grads.face1_op_m10_im,
        grads.face1_op_m11_re, grads.face1_op_m11_im,
        grads.face0_eta_r, grads.face0_sigma, grads.face0_gain,
        grads.face1_eta_r, grads.face1_sigma, grads.face1_gain,
        grads.rx_x, grads.rx_y, grads.rx_z
    );



    witwin::channel::native_ext::utd_accumulate_backward(

        drjit_data_ptr(input.state_idx),

        drjit_data_ptr(input.rx_idx),

        drjit_data_ptr(input.ownership_code),

        drjit_data_ptr(input.state.edge_pos_x), drjit_data_ptr(input.state.edge_pos_y), drjit_data_ptr(input.state.edge_pos_z),

        drjit_data_ptr(input.state.edge_dir_x), drjit_data_ptr(input.state.edge_dir_y), drjit_data_ptr(input.state.edge_dir_z),

        drjit_data_ptr(input.state.n0_x), drjit_data_ptr(input.state.n0_y), drjit_data_ptr(input.state.n0_z),

        drjit_data_ptr(input.state.nn_x), drjit_data_ptr(input.state.nn_y), drjit_data_ptr(input.state.nn_z),

        drjit_data_ptr(input.state.wedge_n),

        drjit_data_ptr(input.state.source_pos_x), drjit_data_ptr(input.state.source_pos_y), drjit_data_ptr(input.state.source_pos_z),

        drjit_data_ptr(input.state.incident_field_re), drjit_data_ptr(input.state.incident_field_im),

        drjit_data_ptr(input.state.incident_nderiv_re), drjit_data_ptr(input.state.incident_nderiv_im),

        drjit_data_ptr(input.state.r0_re), drjit_data_ptr(input.state.r0_im),

        drjit_data_ptr(input.state.rn_re), drjit_data_ptr(input.state.rn_im),

        drjit_data_ptr(input.state.inc_vec_x_re), drjit_data_ptr(input.state.inc_vec_x_im),

        drjit_data_ptr(input.state.inc_vec_y_re), drjit_data_ptr(input.state.inc_vec_y_im),

        drjit_data_ptr(input.state.inc_vec_z_re), drjit_data_ptr(input.state.inc_vec_z_im),

        drjit_data_ptr(input.state.inc_dvec_x_re), drjit_data_ptr(input.state.inc_dvec_x_im),

        drjit_data_ptr(input.state.inc_dvec_y_re), drjit_data_ptr(input.state.inc_dvec_y_im),

        drjit_data_ptr(input.state.inc_dvec_z_re), drjit_data_ptr(input.state.inc_dvec_z_im),

        drjit_data_ptr(input.state.inc_jones_u_re), drjit_data_ptr(input.state.inc_jones_u_im),

        drjit_data_ptr(input.state.inc_jones_v_re), drjit_data_ptr(input.state.inc_jones_v_im),

        drjit_data_ptr(input.state.inc_djones_u_re), drjit_data_ptr(input.state.inc_djones_u_im),

        drjit_data_ptr(input.state.inc_djones_v_re), drjit_data_ptr(input.state.inc_djones_v_im),

        drjit_data_ptr(input.state.inc_basis_u_x), drjit_data_ptr(input.state.inc_basis_u_y), drjit_data_ptr(input.state.inc_basis_u_z),

        drjit_data_ptr(input.state.inc_basis_v_x), drjit_data_ptr(input.state.inc_basis_v_y), drjit_data_ptr(input.state.inc_basis_v_z),

        drjit_data_ptr(input.state.inc_basis_k_x), drjit_data_ptr(input.state.inc_basis_k_y), drjit_data_ptr(input.state.inc_basis_k_z),

        drjit_data_ptr(input.state.face0_op_m00_re), drjit_data_ptr(input.state.face0_op_m00_im),

        drjit_data_ptr(input.state.face0_op_m01_re), drjit_data_ptr(input.state.face0_op_m01_im),

        drjit_data_ptr(input.state.face0_op_m10_re), drjit_data_ptr(input.state.face0_op_m10_im),

        drjit_data_ptr(input.state.face0_op_m11_re), drjit_data_ptr(input.state.face0_op_m11_im),

        drjit_data_ptr(input.state.face1_op_m00_re), drjit_data_ptr(input.state.face1_op_m00_im),

        drjit_data_ptr(input.state.face1_op_m01_re), drjit_data_ptr(input.state.face1_op_m01_im),

        drjit_data_ptr(input.state.face1_op_m10_re), drjit_data_ptr(input.state.face1_op_m10_im),

        drjit_data_ptr(input.state.face1_op_m11_re), drjit_data_ptr(input.state.face1_op_m11_im),

        drjit_data_ptr(input.state.face0_eta_r), drjit_data_ptr(input.state.face0_sigma),

        drjit_data_ptr(input.state.face0_gain), drjit_data_ptr(input.state.face0_use_fresnel),

        drjit_data_ptr(input.state.face0_present),

        drjit_data_ptr(input.state.face1_eta_r), drjit_data_ptr(input.state.face1_sigma),

        drjit_data_ptr(input.state.face1_gain), drjit_data_ptr(input.state.face1_use_fresnel),

        drjit_data_ptr(input.state.face1_present),

        drjit_data_ptr(input.rx.rx_x), drjit_data_ptr(input.rx.rx_y), drjit_data_ptr(input.rx.rx_z),

        drjit_data_ptr(zero_scalar), drjit_data_ptr(zero_scalar),

        drjit_data_ptr(zero_scalar), drjit_data_ptr(zero_scalar),

        drjit_data_ptr(grad_output.direct_vec_x_re), drjit_data_ptr(grad_output.direct_vec_x_im),

        drjit_data_ptr(grad_output.direct_vec_y_re), drjit_data_ptr(grad_output.direct_vec_y_im),

        drjit_data_ptr(grad_output.direct_vec_z_re), drjit_data_ptr(grad_output.direct_vec_z_im),

        drjit_data_ptr(grad_output.multi_vec_x_re), drjit_data_ptr(grad_output.multi_vec_x_im),

        drjit_data_ptr(grad_output.multi_vec_y_re), drjit_data_ptr(grad_output.multi_vec_y_im),

        drjit_data_ptr(grad_output.multi_vec_z_re), drjit_data_ptr(grad_output.multi_vec_z_im),

        drjit_data_ptr_mut(grads.edge_pos_x), drjit_data_ptr_mut(grads.edge_pos_y), drjit_data_ptr_mut(grads.edge_pos_z),

        drjit_data_ptr_mut(grads.edge_dir_x), drjit_data_ptr_mut(grads.edge_dir_y), drjit_data_ptr_mut(grads.edge_dir_z),

        drjit_data_ptr_mut(grads.n0_x), drjit_data_ptr_mut(grads.n0_y), drjit_data_ptr_mut(grads.n0_z),

        drjit_data_ptr_mut(grads.nn_x), drjit_data_ptr_mut(grads.nn_y), drjit_data_ptr_mut(grads.nn_z),

        drjit_data_ptr_mut(grads.wedge_n),

        drjit_data_ptr_mut(grads.source_pos_x), drjit_data_ptr_mut(grads.source_pos_y), drjit_data_ptr_mut(grads.source_pos_z),

        drjit_data_ptr_mut(grads.incident_field_re), drjit_data_ptr_mut(grads.incident_field_im),

        drjit_data_ptr_mut(grads.incident_nderiv_re), drjit_data_ptr_mut(grads.incident_nderiv_im),

        drjit_data_ptr_mut(grads.r0_re), drjit_data_ptr_mut(grads.r0_im),

        drjit_data_ptr_mut(grads.rn_re), drjit_data_ptr_mut(grads.rn_im),

        drjit_data_ptr_mut(grads.inc_vec_x_re), drjit_data_ptr_mut(grads.inc_vec_x_im),

        drjit_data_ptr_mut(grads.inc_vec_y_re), drjit_data_ptr_mut(grads.inc_vec_y_im),

        drjit_data_ptr_mut(grads.inc_vec_z_re), drjit_data_ptr_mut(grads.inc_vec_z_im),

        drjit_data_ptr_mut(grads.inc_dvec_x_re), drjit_data_ptr_mut(grads.inc_dvec_x_im),

        drjit_data_ptr_mut(grads.inc_dvec_y_re), drjit_data_ptr_mut(grads.inc_dvec_y_im),

        drjit_data_ptr_mut(grads.inc_dvec_z_re), drjit_data_ptr_mut(grads.inc_dvec_z_im),
        drjit_data_ptr_mut(grads.face0_op_m00_re), drjit_data_ptr_mut(grads.face0_op_m00_im),
        drjit_data_ptr_mut(grads.face0_op_m01_re), drjit_data_ptr_mut(grads.face0_op_m01_im),
        drjit_data_ptr_mut(grads.face0_op_m10_re), drjit_data_ptr_mut(grads.face0_op_m10_im),
        drjit_data_ptr_mut(grads.face0_op_m11_re), drjit_data_ptr_mut(grads.face0_op_m11_im),
        drjit_data_ptr_mut(grads.face1_op_m00_re), drjit_data_ptr_mut(grads.face1_op_m00_im),
        drjit_data_ptr_mut(grads.face1_op_m01_re), drjit_data_ptr_mut(grads.face1_op_m01_im),
        drjit_data_ptr_mut(grads.face1_op_m10_re), drjit_data_ptr_mut(grads.face1_op_m10_im),
        drjit_data_ptr_mut(grads.face1_op_m11_re), drjit_data_ptr_mut(grads.face1_op_m11_im),
        drjit_data_ptr_mut(grads.face0_eta_r), drjit_data_ptr_mut(grads.face0_sigma), drjit_data_ptr_mut(grads.face0_gain),
        drjit_data_ptr_mut(grads.face1_eta_r), drjit_data_ptr_mut(grads.face1_sigma), drjit_data_ptr_mut(grads.face1_gain),
        drjit_data_ptr_mut(grads.rx_x), drjit_data_ptr_mut(grads.rx_y), drjit_data_ptr_mut(grads.rx_z),
        input.n_pairs,

        input.k,

        input.material

    );



    return grads;

}



inline void accum_utd_input_grads(

    UTDOpInput &registered_input,

    const UTDBackwardGrads &grads

) {

    auto detach = [](const DiffFloat &value) -> Float {

        return drjit::detach<false>(value);

    };



    drjit::accum_grad(registered_input.state.edge_pos_x, detach(grads.edge_pos_x));

    drjit::accum_grad(registered_input.state.edge_pos_y, detach(grads.edge_pos_y));

    drjit::accum_grad(registered_input.state.edge_pos_z, detach(grads.edge_pos_z));

    drjit::accum_grad(registered_input.state.edge_dir_x, detach(grads.edge_dir_x));

    drjit::accum_grad(registered_input.state.edge_dir_y, detach(grads.edge_dir_y));

    drjit::accum_grad(registered_input.state.edge_dir_z, detach(grads.edge_dir_z));

    drjit::accum_grad(registered_input.state.n0_x, detach(grads.n0_x));

    drjit::accum_grad(registered_input.state.n0_y, detach(grads.n0_y));

    drjit::accum_grad(registered_input.state.n0_z, detach(grads.n0_z));

    drjit::accum_grad(registered_input.state.nn_x, detach(grads.nn_x));

    drjit::accum_grad(registered_input.state.nn_y, detach(grads.nn_y));

    drjit::accum_grad(registered_input.state.nn_z, detach(grads.nn_z));

    drjit::accum_grad(registered_input.state.wedge_n, detach(grads.wedge_n));

    drjit::accum_grad(registered_input.state.source_pos_x, detach(grads.source_pos_x));

    drjit::accum_grad(registered_input.state.source_pos_y, detach(grads.source_pos_y));

    drjit::accum_grad(registered_input.state.source_pos_z, detach(grads.source_pos_z));

    drjit::accum_grad(registered_input.state.incident_field_re, detach(grads.incident_field_re));

    drjit::accum_grad(registered_input.state.incident_field_im, detach(grads.incident_field_im));

    drjit::accum_grad(registered_input.state.incident_nderiv_re, detach(grads.incident_nderiv_re));

    drjit::accum_grad(registered_input.state.incident_nderiv_im, detach(grads.incident_nderiv_im));

    drjit::accum_grad(registered_input.state.r0_re, detach(grads.r0_re));

    drjit::accum_grad(registered_input.state.r0_im, detach(grads.r0_im));

    drjit::accum_grad(registered_input.state.rn_re, detach(grads.rn_re));

    drjit::accum_grad(registered_input.state.rn_im, detach(grads.rn_im));

    drjit::accum_grad(registered_input.state.inc_vec_x_re, detach(grads.inc_vec_x_re));

    drjit::accum_grad(registered_input.state.inc_vec_x_im, detach(grads.inc_vec_x_im));

    drjit::accum_grad(registered_input.state.inc_vec_y_re, detach(grads.inc_vec_y_re));

    drjit::accum_grad(registered_input.state.inc_vec_y_im, detach(grads.inc_vec_y_im));

    drjit::accum_grad(registered_input.state.inc_vec_z_re, detach(grads.inc_vec_z_re));

    drjit::accum_grad(registered_input.state.inc_vec_z_im, detach(grads.inc_vec_z_im));

    drjit::accum_grad(registered_input.state.inc_dvec_x_re, detach(grads.inc_dvec_x_re));

    drjit::accum_grad(registered_input.state.inc_dvec_x_im, detach(grads.inc_dvec_x_im));

    drjit::accum_grad(registered_input.state.inc_dvec_y_re, detach(grads.inc_dvec_y_re));

    drjit::accum_grad(registered_input.state.inc_dvec_y_im, detach(grads.inc_dvec_y_im));

    drjit::accum_grad(registered_input.state.inc_dvec_z_re, detach(grads.inc_dvec_z_re));

    drjit::accum_grad(registered_input.state.inc_dvec_z_im, detach(grads.inc_dvec_z_im));
    drjit::accum_grad(registered_input.state.face0_op_m00_re, detach(grads.face0_op_m00_re));
    drjit::accum_grad(registered_input.state.face0_op_m00_im, detach(grads.face0_op_m00_im));
    drjit::accum_grad(registered_input.state.face0_op_m01_re, detach(grads.face0_op_m01_re));
    drjit::accum_grad(registered_input.state.face0_op_m01_im, detach(grads.face0_op_m01_im));
    drjit::accum_grad(registered_input.state.face0_op_m10_re, detach(grads.face0_op_m10_re));
    drjit::accum_grad(registered_input.state.face0_op_m10_im, detach(grads.face0_op_m10_im));
    drjit::accum_grad(registered_input.state.face0_op_m11_re, detach(grads.face0_op_m11_re));
    drjit::accum_grad(registered_input.state.face0_op_m11_im, detach(grads.face0_op_m11_im));
    drjit::accum_grad(registered_input.state.face1_op_m00_re, detach(grads.face1_op_m00_re));
    drjit::accum_grad(registered_input.state.face1_op_m00_im, detach(grads.face1_op_m00_im));
    drjit::accum_grad(registered_input.state.face1_op_m01_re, detach(grads.face1_op_m01_re));
    drjit::accum_grad(registered_input.state.face1_op_m01_im, detach(grads.face1_op_m01_im));
    drjit::accum_grad(registered_input.state.face1_op_m10_re, detach(grads.face1_op_m10_re));
    drjit::accum_grad(registered_input.state.face1_op_m10_im, detach(grads.face1_op_m10_im));
    drjit::accum_grad(registered_input.state.face1_op_m11_re, detach(grads.face1_op_m11_re));
    drjit::accum_grad(registered_input.state.face1_op_m11_im, detach(grads.face1_op_m11_im));
    drjit::accum_grad(registered_input.state.face0_eta_r, detach(grads.face0_eta_r));
    drjit::accum_grad(registered_input.state.face0_sigma, detach(grads.face0_sigma));
    drjit::accum_grad(registered_input.state.face0_gain, detach(grads.face0_gain));
    drjit::accum_grad(registered_input.state.face1_eta_r, detach(grads.face1_eta_r));
    drjit::accum_grad(registered_input.state.face1_sigma, detach(grads.face1_sigma));
    drjit::accum_grad(registered_input.state.face1_gain, detach(grads.face1_gain));
    drjit::accum_grad(registered_input.rx.rx_x, detach(grads.rx_x));
    drjit::accum_grad(registered_input.rx.rx_y, detach(grads.rx_y));

    drjit::accum_grad(registered_input.rx.rx_z, detach(grads.rx_z));

}



// ---------------------------------------------------------------------------

// CUDA kernel launchers (forward / JVP)

// ---------------------------------------------------------------------------



inline void launch_utd_forward(

    const drjit::detached_t<UTDOpInput> &input,

    UTDOpOutput &output

) {

    Float direct_re = drjit::zeros<Float>(input.rx.rx_x.size());

    Float direct_im = drjit::zeros<Float>(input.rx.rx_x.size());

    Float multi_re  = drjit::zeros<Float>(input.rx.rx_x.size());

    Float multi_im  = drjit::zeros<Float>(input.rx.rx_x.size());



    drjit::eval(

        input.state_idx, input.rx_idx, input.ownership_code,

        input.state.edge_pos_x, input.state.edge_pos_y, input.state.edge_pos_z,

        input.state.edge_dir_x, input.state.edge_dir_y, input.state.edge_dir_z,

        input.state.n0_x, input.state.n0_y, input.state.n0_z,

        input.state.nn_x, input.state.nn_y, input.state.nn_z,

        input.state.wedge_n,

        input.state.source_pos_x, input.state.source_pos_y, input.state.source_pos_z,

        input.state.incident_field_re, input.state.incident_field_im,

        input.state.incident_nderiv_re, input.state.incident_nderiv_im,

        input.state.r0_re, input.state.r0_im,

        input.state.rn_re, input.state.rn_im

    );

    drjit::eval(

        input.state.inc_vec_x_re, input.state.inc_vec_x_im,

        input.state.inc_vec_y_re, input.state.inc_vec_y_im,

        input.state.inc_vec_z_re, input.state.inc_vec_z_im,

        input.state.inc_dvec_x_re, input.state.inc_dvec_x_im,

        input.state.inc_dvec_y_re, input.state.inc_dvec_y_im,

        input.state.inc_dvec_z_re, input.state.inc_dvec_z_im,

        input.state.inc_jones_u_re, input.state.inc_jones_u_im,

        input.state.inc_jones_v_re, input.state.inc_jones_v_im,

        input.state.inc_djones_u_re, input.state.inc_djones_u_im,

        input.state.inc_djones_v_re, input.state.inc_djones_v_im

    );

    drjit::eval(

        input.state.inc_basis_u_x, input.state.inc_basis_u_y, input.state.inc_basis_u_z,

        input.state.inc_basis_v_x, input.state.inc_basis_v_y, input.state.inc_basis_v_z,

        input.state.inc_basis_k_x, input.state.inc_basis_k_y, input.state.inc_basis_k_z,

        input.state.face0_op_m00_re, input.state.face0_op_m00_im,

        input.state.face0_op_m01_re, input.state.face0_op_m01_im,

        input.state.face0_op_m10_re, input.state.face0_op_m10_im,

        input.state.face0_op_m11_re, input.state.face0_op_m11_im,

        input.state.face1_op_m00_re, input.state.face1_op_m00_im,

        input.state.face1_op_m01_re, input.state.face1_op_m01_im,

        input.state.face1_op_m10_re, input.state.face1_op_m10_im,

        input.state.face1_op_m11_re, input.state.face1_op_m11_im

    );

    drjit::eval(

        input.state.face0_eta_r, input.state.face0_sigma, input.state.face0_gain,

        input.state.face0_use_fresnel, input.state.face0_present,

        input.state.face1_eta_r, input.state.face1_sigma, input.state.face1_gain,

        input.state.face1_use_fresnel, input.state.face1_present,

        input.rx.rx_x, input.rx.rx_y, input.rx.rx_z,

        direct_re, direct_im, multi_re, multi_im,

        output.direct_vec_x_re, output.direct_vec_x_im,

        output.direct_vec_y_re, output.direct_vec_y_im,

        output.direct_vec_z_re, output.direct_vec_z_im,

        output.multi_vec_x_re, output.multi_vec_x_im,

        output.multi_vec_y_re, output.multi_vec_y_im,

        output.multi_vec_z_re, output.multi_vec_z_im

    );



    witwin::channel::native_ext::utd_accumulate_forward(

        drjit_data_ptr(input.state_idx),

        drjit_data_ptr(input.rx_idx),

        drjit_data_ptr(input.ownership_code),

        drjit_data_ptr(input.state.edge_pos_x),

        drjit_data_ptr(input.state.edge_pos_y),

        drjit_data_ptr(input.state.edge_pos_z),

        drjit_data_ptr(input.state.edge_dir_x),

        drjit_data_ptr(input.state.edge_dir_y),

        drjit_data_ptr(input.state.edge_dir_z),

        drjit_data_ptr(input.state.n0_x),

        drjit_data_ptr(input.state.n0_y),

        drjit_data_ptr(input.state.n0_z),

        drjit_data_ptr(input.state.nn_x),

        drjit_data_ptr(input.state.nn_y),

        drjit_data_ptr(input.state.nn_z),

        drjit_data_ptr(input.state.wedge_n),

        drjit_data_ptr(input.state.source_pos_x),

        drjit_data_ptr(input.state.source_pos_y),

        drjit_data_ptr(input.state.source_pos_z),

        drjit_data_ptr(input.state.incident_field_re),

        drjit_data_ptr(input.state.incident_field_im),

        drjit_data_ptr(input.state.incident_nderiv_re),

        drjit_data_ptr(input.state.incident_nderiv_im),

        drjit_data_ptr(input.state.r0_re),

        drjit_data_ptr(input.state.r0_im),

        drjit_data_ptr(input.state.rn_re),

        drjit_data_ptr(input.state.rn_im),

        drjit_data_ptr(input.state.inc_vec_x_re),

        drjit_data_ptr(input.state.inc_vec_x_im),

        drjit_data_ptr(input.state.inc_vec_y_re),

        drjit_data_ptr(input.state.inc_vec_y_im),

        drjit_data_ptr(input.state.inc_vec_z_re),

        drjit_data_ptr(input.state.inc_vec_z_im),

        drjit_data_ptr(input.state.inc_dvec_x_re),

        drjit_data_ptr(input.state.inc_dvec_x_im),

        drjit_data_ptr(input.state.inc_dvec_y_re),

        drjit_data_ptr(input.state.inc_dvec_y_im),

        drjit_data_ptr(input.state.inc_dvec_z_re),

        drjit_data_ptr(input.state.inc_dvec_z_im),

        drjit_data_ptr(input.state.inc_jones_u_re),

        drjit_data_ptr(input.state.inc_jones_u_im),

        drjit_data_ptr(input.state.inc_jones_v_re),

        drjit_data_ptr(input.state.inc_jones_v_im),

        drjit_data_ptr(input.state.inc_djones_u_re),

        drjit_data_ptr(input.state.inc_djones_u_im),

        drjit_data_ptr(input.state.inc_djones_v_re),

        drjit_data_ptr(input.state.inc_djones_v_im),

        drjit_data_ptr(input.state.inc_basis_u_x),

        drjit_data_ptr(input.state.inc_basis_u_y),

        drjit_data_ptr(input.state.inc_basis_u_z),

        drjit_data_ptr(input.state.inc_basis_v_x),

        drjit_data_ptr(input.state.inc_basis_v_y),

        drjit_data_ptr(input.state.inc_basis_v_z),

        drjit_data_ptr(input.state.inc_basis_k_x),

        drjit_data_ptr(input.state.inc_basis_k_y),

        drjit_data_ptr(input.state.inc_basis_k_z),

        drjit_data_ptr(input.state.face0_op_m00_re),

        drjit_data_ptr(input.state.face0_op_m00_im),

        drjit_data_ptr(input.state.face0_op_m01_re),

        drjit_data_ptr(input.state.face0_op_m01_im),

        drjit_data_ptr(input.state.face0_op_m10_re),

        drjit_data_ptr(input.state.face0_op_m10_im),

        drjit_data_ptr(input.state.face0_op_m11_re),

        drjit_data_ptr(input.state.face0_op_m11_im),

        drjit_data_ptr(input.state.face1_op_m00_re),

        drjit_data_ptr(input.state.face1_op_m00_im),

        drjit_data_ptr(input.state.face1_op_m01_re),

        drjit_data_ptr(input.state.face1_op_m01_im),

        drjit_data_ptr(input.state.face1_op_m10_re),

        drjit_data_ptr(input.state.face1_op_m10_im),

        drjit_data_ptr(input.state.face1_op_m11_re),

        drjit_data_ptr(input.state.face1_op_m11_im),

        drjit_data_ptr(input.state.face0_eta_r),

        drjit_data_ptr(input.state.face0_sigma),

        drjit_data_ptr(input.state.face0_gain),

        drjit_data_ptr(input.state.face0_use_fresnel),

        drjit_data_ptr(input.state.face0_present),

        drjit_data_ptr(input.state.face1_eta_r),

        drjit_data_ptr(input.state.face1_sigma),

        drjit_data_ptr(input.state.face1_gain),

        drjit_data_ptr(input.state.face1_use_fresnel),

        drjit_data_ptr(input.state.face1_present),

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

        input.n_pairs,

        input.k,

        input.material

    );

}



inline void launch_utd_jvp(

    const drjit::detached_t<UTDOpInput> &input,

    const drjit::detached_t<UTDOpInput> &tangent,

    drjit::detached_t<UTDOpOutput> &output

) {

    auto coerce = [](const auto &value, size_t width) -> Float {

        return value.size() == width ? drjit::detach<false>(value) : drjit::zeros<Float>(width);

    };

    auto t_edge_pos_x = coerce(tangent.state.edge_pos_x, input.state.edge_pos_x.size());

    auto t_edge_pos_y = coerce(tangent.state.edge_pos_y, input.state.edge_pos_y.size());

    auto t_edge_pos_z = coerce(tangent.state.edge_pos_z, input.state.edge_pos_z.size());

    auto t_edge_dir_x = coerce(tangent.state.edge_dir_x, input.state.edge_dir_x.size());

    auto t_edge_dir_y = coerce(tangent.state.edge_dir_y, input.state.edge_dir_y.size());

    auto t_edge_dir_z = coerce(tangent.state.edge_dir_z, input.state.edge_dir_z.size());

    auto t_n0_x = coerce(tangent.state.n0_x, input.state.n0_x.size());

    auto t_n0_y = coerce(tangent.state.n0_y, input.state.n0_y.size());

    auto t_n0_z = coerce(tangent.state.n0_z, input.state.n0_z.size());

    auto t_nn_x = coerce(tangent.state.nn_x, input.state.nn_x.size());

    auto t_nn_y = coerce(tangent.state.nn_y, input.state.nn_y.size());

    auto t_nn_z = coerce(tangent.state.nn_z, input.state.nn_z.size());

    auto t_wedge_n = coerce(tangent.state.wedge_n, input.state.wedge_n.size());

    auto t_source_pos_x = coerce(tangent.state.source_pos_x, input.state.source_pos_x.size());

    auto t_source_pos_y = coerce(tangent.state.source_pos_y, input.state.source_pos_y.size());

    auto t_source_pos_z = coerce(tangent.state.source_pos_z, input.state.source_pos_z.size());

    auto t_incident_field_re = coerce(tangent.state.incident_field_re, input.state.incident_field_re.size());

    auto t_incident_field_im = coerce(tangent.state.incident_field_im, input.state.incident_field_im.size());

    auto t_incident_nderiv_re = coerce(tangent.state.incident_nderiv_re, input.state.incident_nderiv_re.size());

    auto t_incident_nderiv_im = coerce(tangent.state.incident_nderiv_im, input.state.incident_nderiv_im.size());

    auto t_r0_re = coerce(tangent.state.r0_re, input.state.r0_re.size());

    auto t_r0_im = coerce(tangent.state.r0_im, input.state.r0_im.size());

    auto t_rn_re = coerce(tangent.state.rn_re, input.state.rn_re.size());

    auto t_rn_im = coerce(tangent.state.rn_im, input.state.rn_im.size());

    auto t_inc_vec_x_re = coerce(tangent.state.inc_vec_x_re, input.state.inc_vec_x_re.size());

    auto t_inc_vec_x_im = coerce(tangent.state.inc_vec_x_im, input.state.inc_vec_x_im.size());

    auto t_inc_vec_y_re = coerce(tangent.state.inc_vec_y_re, input.state.inc_vec_y_re.size());

    auto t_inc_vec_y_im = coerce(tangent.state.inc_vec_y_im, input.state.inc_vec_y_im.size());

    auto t_inc_vec_z_re = coerce(tangent.state.inc_vec_z_re, input.state.inc_vec_z_re.size());

    auto t_inc_vec_z_im = coerce(tangent.state.inc_vec_z_im, input.state.inc_vec_z_im.size());

    auto t_inc_dvec_x_re = coerce(tangent.state.inc_dvec_x_re, input.state.inc_dvec_x_re.size());

    auto t_inc_dvec_x_im = coerce(tangent.state.inc_dvec_x_im, input.state.inc_dvec_x_im.size());

    auto t_inc_dvec_y_re = coerce(tangent.state.inc_dvec_y_re, input.state.inc_dvec_y_re.size());

    auto t_inc_dvec_y_im = coerce(tangent.state.inc_dvec_y_im, input.state.inc_dvec_y_im.size());

    auto t_inc_dvec_z_re = coerce(tangent.state.inc_dvec_z_re, input.state.inc_dvec_z_re.size());

    auto t_inc_dvec_z_im = coerce(tangent.state.inc_dvec_z_im, input.state.inc_dvec_z_im.size());
    auto t_face0_op_m00_re = coerce(tangent.state.face0_op_m00_re, input.state.face0_op_m00_re.size());
    auto t_face0_op_m00_im = coerce(tangent.state.face0_op_m00_im, input.state.face0_op_m00_im.size());
    auto t_face0_op_m01_re = coerce(tangent.state.face0_op_m01_re, input.state.face0_op_m01_re.size());
    auto t_face0_op_m01_im = coerce(tangent.state.face0_op_m01_im, input.state.face0_op_m01_im.size());
    auto t_face0_op_m10_re = coerce(tangent.state.face0_op_m10_re, input.state.face0_op_m10_re.size());
    auto t_face0_op_m10_im = coerce(tangent.state.face0_op_m10_im, input.state.face0_op_m10_im.size());
    auto t_face0_op_m11_re = coerce(tangent.state.face0_op_m11_re, input.state.face0_op_m11_re.size());
    auto t_face0_op_m11_im = coerce(tangent.state.face0_op_m11_im, input.state.face0_op_m11_im.size());
    auto t_face1_op_m00_re = coerce(tangent.state.face1_op_m00_re, input.state.face1_op_m00_re.size());
    auto t_face1_op_m00_im = coerce(tangent.state.face1_op_m00_im, input.state.face1_op_m00_im.size());
    auto t_face1_op_m01_re = coerce(tangent.state.face1_op_m01_re, input.state.face1_op_m01_re.size());
    auto t_face1_op_m01_im = coerce(tangent.state.face1_op_m01_im, input.state.face1_op_m01_im.size());
    auto t_face1_op_m10_re = coerce(tangent.state.face1_op_m10_re, input.state.face1_op_m10_re.size());
    auto t_face1_op_m10_im = coerce(tangent.state.face1_op_m10_im, input.state.face1_op_m10_im.size());
    auto t_face1_op_m11_re = coerce(tangent.state.face1_op_m11_re, input.state.face1_op_m11_re.size());
    auto t_face1_op_m11_im = coerce(tangent.state.face1_op_m11_im, input.state.face1_op_m11_im.size());
    auto t_face0_eta_r = coerce(tangent.state.face0_eta_r, input.state.face0_eta_r.size());
    auto t_face0_sigma = coerce(tangent.state.face0_sigma, input.state.face0_sigma.size());
    auto t_face0_gain = coerce(tangent.state.face0_gain, input.state.face0_gain.size());
    auto t_face1_eta_r = coerce(tangent.state.face1_eta_r, input.state.face1_eta_r.size());
    auto t_face1_sigma = coerce(tangent.state.face1_sigma, input.state.face1_sigma.size());
    auto t_face1_gain = coerce(tangent.state.face1_gain, input.state.face1_gain.size());
    auto t_rx_x = coerce(tangent.rx.rx_x, input.rx.rx_x.size());
    auto t_rx_y = coerce(tangent.rx.rx_y, input.rx.rx_y.size());

    auto t_rx_z = coerce(tangent.rx.rx_z, input.rx.rx_z.size());



    drjit::eval(

        t_edge_pos_x, t_edge_pos_y, t_edge_pos_z,

        t_edge_dir_x, t_edge_dir_y, t_edge_dir_z,

        t_n0_x, t_n0_y, t_n0_z,

        t_nn_x, t_nn_y, t_nn_z,

        t_wedge_n,

        t_source_pos_x, t_source_pos_y, t_source_pos_z,

        t_incident_field_re, t_incident_field_im,

        t_incident_nderiv_re, t_incident_nderiv_im,

        t_r0_re, t_r0_im,

        t_rn_re, t_rn_im,

        t_inc_vec_x_re, t_inc_vec_x_im,

        t_inc_vec_y_re, t_inc_vec_y_im,

        t_inc_vec_z_re, t_inc_vec_z_im,

        t_inc_dvec_x_re, t_inc_dvec_x_im,

        t_inc_dvec_y_re, t_inc_dvec_y_im,

        t_inc_dvec_z_re, t_inc_dvec_z_im,
        t_face0_op_m00_re, t_face0_op_m00_im,
        t_face0_op_m01_re, t_face0_op_m01_im,
        t_face0_op_m10_re, t_face0_op_m10_im,
        t_face0_op_m11_re, t_face0_op_m11_im,
        t_face1_op_m00_re, t_face1_op_m00_im,
        t_face1_op_m01_re, t_face1_op_m01_im,
        t_face1_op_m10_re, t_face1_op_m10_im,
        t_face1_op_m11_re, t_face1_op_m11_im,
        t_face0_eta_r, t_face0_sigma, t_face0_gain,
        t_face1_eta_r, t_face1_sigma, t_face1_gain,
        t_rx_x, t_rx_y, t_rx_z
    );

    drjit::eval(

        output.direct_vec_x_re, output.direct_vec_x_im,

        output.direct_vec_y_re, output.direct_vec_y_im,

        output.direct_vec_z_re, output.direct_vec_z_im,

        output.multi_vec_x_re, output.multi_vec_x_im,

        output.multi_vec_y_re, output.multi_vec_y_im,

        output.multi_vec_z_re, output.multi_vec_z_im

    );



    witwin::channel::native_ext::utd_accumulate_jvp(

        drjit_data_ptr(input.state_idx),

        drjit_data_ptr(input.rx_idx),

        drjit_data_ptr(input.ownership_code),

        drjit_data_ptr(input.state.edge_pos_x),

        drjit_data_ptr(input.state.edge_pos_y),

        drjit_data_ptr(input.state.edge_pos_z),

        drjit_data_ptr(input.state.edge_dir_x),

        drjit_data_ptr(input.state.edge_dir_y),

        drjit_data_ptr(input.state.edge_dir_z),

        drjit_data_ptr(input.state.n0_x),

        drjit_data_ptr(input.state.n0_y),

        drjit_data_ptr(input.state.n0_z),

        drjit_data_ptr(input.state.nn_x),

        drjit_data_ptr(input.state.nn_y),

        drjit_data_ptr(input.state.nn_z),

        drjit_data_ptr(input.state.wedge_n),

        drjit_data_ptr(input.state.source_pos_x),

        drjit_data_ptr(input.state.source_pos_y),

        drjit_data_ptr(input.state.source_pos_z),

        drjit_data_ptr(input.state.incident_field_re),

        drjit_data_ptr(input.state.incident_field_im),

        drjit_data_ptr(input.state.incident_nderiv_re),

        drjit_data_ptr(input.state.incident_nderiv_im),

        drjit_data_ptr(input.state.r0_re),

        drjit_data_ptr(input.state.r0_im),

        drjit_data_ptr(input.state.rn_re),

        drjit_data_ptr(input.state.rn_im),

        drjit_data_ptr(input.state.inc_vec_x_re),

        drjit_data_ptr(input.state.inc_vec_x_im),

        drjit_data_ptr(input.state.inc_vec_y_re),

        drjit_data_ptr(input.state.inc_vec_y_im),

        drjit_data_ptr(input.state.inc_vec_z_re),

        drjit_data_ptr(input.state.inc_vec_z_im),

        drjit_data_ptr(input.state.inc_dvec_x_re),

        drjit_data_ptr(input.state.inc_dvec_x_im),

        drjit_data_ptr(input.state.inc_dvec_y_re),

        drjit_data_ptr(input.state.inc_dvec_y_im),

        drjit_data_ptr(input.state.inc_dvec_z_re),

        drjit_data_ptr(input.state.inc_dvec_z_im),

        drjit_data_ptr(input.state.inc_jones_u_re),

        drjit_data_ptr(input.state.inc_jones_u_im),

        drjit_data_ptr(input.state.inc_jones_v_re),

        drjit_data_ptr(input.state.inc_jones_v_im),

        drjit_data_ptr(input.state.inc_djones_u_re),

        drjit_data_ptr(input.state.inc_djones_u_im),

        drjit_data_ptr(input.state.inc_djones_v_re),

        drjit_data_ptr(input.state.inc_djones_v_im),

        drjit_data_ptr(input.state.inc_basis_u_x),

        drjit_data_ptr(input.state.inc_basis_u_y),

        drjit_data_ptr(input.state.inc_basis_u_z),

        drjit_data_ptr(input.state.inc_basis_v_x),

        drjit_data_ptr(input.state.inc_basis_v_y),

        drjit_data_ptr(input.state.inc_basis_v_z),

        drjit_data_ptr(input.state.inc_basis_k_x),

        drjit_data_ptr(input.state.inc_basis_k_y),

        drjit_data_ptr(input.state.inc_basis_k_z),

        drjit_data_ptr(input.state.face0_op_m00_re),

        drjit_data_ptr(input.state.face0_op_m00_im),

        drjit_data_ptr(input.state.face0_op_m01_re),

        drjit_data_ptr(input.state.face0_op_m01_im),

        drjit_data_ptr(input.state.face0_op_m10_re),

        drjit_data_ptr(input.state.face0_op_m10_im),

        drjit_data_ptr(input.state.face0_op_m11_re),

        drjit_data_ptr(input.state.face0_op_m11_im),

        drjit_data_ptr(input.state.face1_op_m00_re),

        drjit_data_ptr(input.state.face1_op_m00_im),

        drjit_data_ptr(input.state.face1_op_m01_re),

        drjit_data_ptr(input.state.face1_op_m01_im),

        drjit_data_ptr(input.state.face1_op_m10_re),

        drjit_data_ptr(input.state.face1_op_m10_im),

        drjit_data_ptr(input.state.face1_op_m11_re),

        drjit_data_ptr(input.state.face1_op_m11_im),

        drjit_data_ptr(input.state.face0_eta_r),

        drjit_data_ptr(input.state.face0_sigma),

        drjit_data_ptr(input.state.face0_gain),

        drjit_data_ptr(input.state.face0_use_fresnel),

        drjit_data_ptr(input.state.face0_present),

        drjit_data_ptr(input.state.face1_eta_r),

        drjit_data_ptr(input.state.face1_sigma),

        drjit_data_ptr(input.state.face1_gain),

        drjit_data_ptr(input.state.face1_use_fresnel),

        drjit_data_ptr(input.state.face1_present),

        drjit_data_ptr(input.rx.rx_x),

        drjit_data_ptr(input.rx.rx_y),

        drjit_data_ptr(input.rx.rx_z),

        drjit_data_ptr(t_edge_pos_x),

        drjit_data_ptr(t_edge_pos_y),

        drjit_data_ptr(t_edge_pos_z),

        drjit_data_ptr(t_edge_dir_x),

        drjit_data_ptr(t_edge_dir_y),

        drjit_data_ptr(t_edge_dir_z),

        drjit_data_ptr(t_n0_x),

        drjit_data_ptr(t_n0_y),

        drjit_data_ptr(t_n0_z),

        drjit_data_ptr(t_nn_x),

        drjit_data_ptr(t_nn_y),

        drjit_data_ptr(t_nn_z),

        drjit_data_ptr(t_wedge_n),

        drjit_data_ptr(t_source_pos_x),

        drjit_data_ptr(t_source_pos_y),

        drjit_data_ptr(t_source_pos_z),

        drjit_data_ptr(t_incident_field_re),

        drjit_data_ptr(t_incident_field_im),

        drjit_data_ptr(t_incident_nderiv_re),

        drjit_data_ptr(t_incident_nderiv_im),

        drjit_data_ptr(t_r0_re),

        drjit_data_ptr(t_r0_im),

        drjit_data_ptr(t_rn_re),

        drjit_data_ptr(t_rn_im),

        drjit_data_ptr(t_inc_vec_x_re),

        drjit_data_ptr(t_inc_vec_x_im),

        drjit_data_ptr(t_inc_vec_y_re),

        drjit_data_ptr(t_inc_vec_y_im),

        drjit_data_ptr(t_inc_vec_z_re),

        drjit_data_ptr(t_inc_vec_z_im),

        drjit_data_ptr(t_inc_dvec_x_re),

        drjit_data_ptr(t_inc_dvec_x_im),

        drjit_data_ptr(t_inc_dvec_y_re),

        drjit_data_ptr(t_inc_dvec_y_im),

        drjit_data_ptr(t_inc_dvec_z_re),
        drjit_data_ptr(t_inc_dvec_z_im),
        drjit_data_ptr(t_face0_op_m00_re),
        drjit_data_ptr(t_face0_op_m00_im),
        drjit_data_ptr(t_face0_op_m01_re),
        drjit_data_ptr(t_face0_op_m01_im),
        drjit_data_ptr(t_face0_op_m10_re),
        drjit_data_ptr(t_face0_op_m10_im),
        drjit_data_ptr(t_face0_op_m11_re),
        drjit_data_ptr(t_face0_op_m11_im),
        drjit_data_ptr(t_face1_op_m00_re),
        drjit_data_ptr(t_face1_op_m00_im),
        drjit_data_ptr(t_face1_op_m01_re),
        drjit_data_ptr(t_face1_op_m01_im),
        drjit_data_ptr(t_face1_op_m10_re),
        drjit_data_ptr(t_face1_op_m10_im),
        drjit_data_ptr(t_face1_op_m11_re),
        drjit_data_ptr(t_face1_op_m11_im),
        drjit_data_ptr(t_face0_eta_r),
        drjit_data_ptr(t_face0_sigma),
        drjit_data_ptr(t_face0_gain),
        drjit_data_ptr(t_face1_eta_r),
        drjit_data_ptr(t_face1_sigma),
        drjit_data_ptr(t_face1_gain),
        drjit_data_ptr(t_rx_x),
        drjit_data_ptr(t_rx_y),

        drjit_data_ptr(t_rx_z),

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



// ---------------------------------------------------------------------------

// UTDAccumulateOp 芒鈧?DrJit custom AD operation

// ---------------------------------------------------------------------------



class UTDAccumulateOp : public WitwinCustomOp<UTDOpOutput, UTDOpInput> {

public:

    using Base = WitwinCustomOp<UTDOpOutput, UTDOpInput>;

    using OutputType = typename Base::OutputType;



    explicit UTDAccumulateOp(const UTDOpInput &input)

        : Base(input) {}



    OutputType eval(drjit::detached_t<UTDOpInput> input) {

        m_input = input;

        OutputType output = zero_utd_output(input.rx.rx_x.size());

        if (input.n_pairs > 0) {

            launch_utd_forward(input, output);

        }

        return output;

    }



    void forward() override {

        auto output = zero_utd_output_grad(m_input.rx.rx_x.size());

        if (m_input.n_pairs > 0) {

            launch_utd_jvp(m_input, drjit::grad<false>(this->m_registered_input), output);

        }

        set_utd_output_grad(this->m_registered_output, output);

    }



    void backward() override {

        if (m_input.n_pairs <= 0)

            return;

        auto grad_output = drjit::grad<false>(this->m_registered_output);

        auto grads = launch_utd_backward_grads(m_input, grad_output);

        accum_utd_input_grads(this->m_registered_input, grads);

    }



    const char *name() const override { return "UTDAccumulate"; }



private:

    drjit::detached_t<UTDOpInput> m_input;

};



// ---------------------------------------------------------------------------

// register_utd_bindings 芒鈧?debug terms, raw pointer kernels, DrJit AD op

// ---------------------------------------------------------------------------



void register_utd_bindings(nb::module_ &m) {

    // MaterialParams struct binding

    nb::class_<witwin::channel::native_ext::MaterialParams>(m, "MaterialParams")

        .def(nb::init<>())

        .def(nb::init<int, float, float, float, float>(),

             nb::arg("use_fresnel"), nb::arg("eta_r"),

             nb::arg("sigma"), nb::arg("gain"), nb::arg("omega"))

        .def_rw("use_fresnel", &witwin::channel::native_ext::MaterialParams::useFresnel)

        .def_rw("eta_r",       &witwin::channel::native_ext::MaterialParams::etaR)

        .def_rw("sigma",       &witwin::channel::native_ext::MaterialParams::sigma)

        .def_rw("gain",        &witwin::channel::native_ext::MaterialParams::gain)

        .def_rw("omega",       &witwin::channel::native_ext::MaterialParams::omega);



    // Debug helpers

    m.def(

        "utd_debug_terms_3d",

        [](float phi, float phi_prime, float wedge_n, float k, float s, float s_prime, float sin_beta0) {

            return pack_diffraction_terms_dict(

                witwin::channel::native_ext::compute_op_terms_3d(

                    phi, phi_prime, wedge_n, k, s, s_prime, sin_beta0

                )

            );

        },

        nb::arg("phi"), nb::arg("phi_prime"), nb::arg("wedge_n"),

        nb::arg("k"), nb::arg("s"), nb::arg("s_prime"), nb::arg("sin_beta0"),

        "Debug helper: evaluate 3D UTD operator terms on the host."

    );

    m.def(
        "utd_debug_terms_2d",
        [](float phi, float phi_prime, float wedge_n, float k, float s, float s_prime) {
            return pack_diffraction_terms_dict(
                witwin::channel::native_ext::compute_op_terms_2d(
                    phi, phi_prime, wedge_n, k, s, s_prime

                )

            );

        },

        nb::arg("phi"), nb::arg("phi_prime"), nb::arg("wedge_n"),
        nb::arg("k"), nb::arg("s"), nb::arg("s_prime"),
        "Debug helper: evaluate 2D UTD operator terms on the host."
    );

    m.def(
        "utd_debug_pair_contribution",
        [](
            nb::dict state,
            nb::dict target,
            float k,
            witwin::channel::native_ext::MaterialParams mat
        ) {
            using namespace witwin::channel::native_ext;

            PairInputs pair_state = read_pair_inputs_dict(state);

            float3a tgt_pos = read_float3_dict(target);

            bool src_ext = wedge_exterior_mask(
                f3_sub(pair_state.sourcePos, pair_state.edgePos),
                pair_state.edgeDir,
                pair_state.n0,
                pair_state.nn
            );
            bool tgt_ext = wedge_exterior_mask(
                f3_sub(tgt_pos, pair_state.edgePos),
                pair_state.edgeDir,
                pair_state.n0,
                pair_state.nn
            );

            float phi = 0.f;
            float phi_prime = 0.f;
            float s = 0.f;
            float s_prime = 0.f;
            float sin_beta0 = 0.f;
            compute_edge_geometry_3d(
                pair_state.sourcePos,
                pair_state.edgePos,
                pair_state.edgeDir,
                pair_state.n0,
                tgt_pos,
                phi,
                phi_prime,
                s,
                s_prime,
                sin_beta0
            );

            bool pole_safe = cot_pole_safe_mask(phi, phi_prime, pair_state.wedgeN, 1.0e-6f);
            float safe_phi = pole_safe ? phi : 0.5f * pair_state.wedgeN * UTD_PI;
            float safe_phi_prime = pole_safe ? phi_prime : 0.5f * pair_state.wedgeN * UTD_PI;
            bool slope_safe = slope_safe_mask(safe_phi, safe_phi_prime, pair_state.wedgeN, UTD_SLOPE_STEP);
            bool geom_valid = src_ext && tgt_ext && (s_prime > UTD_MIN_DISTANCE) && (s > UTD_MIN_DISTANCE);

            Complex field = cplx_zero();
            Complex direct_gain = cplx_zero();
            Complex derivative_gain = cplx_zero();
            compute_pair_field_terms(pair_state, tgt_pos, k, mat, geom_valid, field, direct_gain, derivative_gain);

            Complex finite_factor = finite_wedge_truncation_factor(pair_state, tgt_pos, k);
            Complex3 vector_field = compute_pair_vector_contribution(pair_state, tgt_pos, k, mat);

            Basis3 in_edge_basis = diffraction_edge_basis(
                f3_sub(pair_state.edgePos, pair_state.sourcePos),
                pair_state.edgeDir,
                false
            );
            Basis3 out_edge_basis = diffraction_edge_basis(
                f3_sub(tgt_pos, pair_state.edgePos),
                pair_state.edgeDir,
                true
            );

            Complex3 incident_vector = vector_from_jones(pair_state.incidentJones, pair_state.incidentBasis);
            Complex3 incident_derivative_vector = vector_from_jones(
                pair_state.incidentDerivativeJones,
                pair_state.incidentBasis
            );
            Jones2 incident_jones_edge = jones_from_vector(incident_vector, in_edge_basis);
            Jones2 incident_derivative_jones_edge = jones_from_vector(incident_derivative_vector, in_edge_basis);

            bool use_face = (pair_state.face0Material.present > 0.5f) || (pair_state.face1Material.present > 0.5f);
            bool f0_has_material = pair_state.face0Material.present > 0.5f;
            bool f1_has_material = pair_state.face1Material.present > 0.5f;
            bool use_stored_face_ops = mat.omega <= 0.f;

            JonesOperator face0_op = (f0_has_material && !use_stored_face_ops)
                ? face_reflection_operator(
                    pair_state.face0Material,
                    fminf(fmaxf(fabsf(sinf(phi_prime)), 1.0e-6f), 1.f),
                    pair_state.n0,
                    in_edge_basis.k,
                    out_edge_basis.k,
                    in_edge_basis,
                    out_edge_basis,
                    mat.omega
                )
                : fallback_face_operator(
                    pair_state.face0Operator,
                    pair_state.n0,
                    in_edge_basis.k,
                    out_edge_basis.k,
                    in_edge_basis,
                    out_edge_basis
                );

            JonesOperator face1_op = (f1_has_material && !use_stored_face_ops)
                ? face_reflection_operator(
                    pair_state.face1Material,
                    fminf(fmaxf(fabsf(sinf(pair_state.wedgeN * UTD_PI - phi)), 1.0e-6f), 1.f),
                    pair_state.nn,
                    in_edge_basis.k,
                    out_edge_basis.k,
                    in_edge_basis,
                    out_edge_basis,
                    mat.omega
                )
                : fallback_face_operator(
                    pair_state.face1Operator,
                    pair_state.nn,
                    in_edge_basis.k,
                    out_edge_basis.k,
                    in_edge_basis,
                    out_edge_basis
                );

            DiffractionOperatorTerms terms = use_face
                ? compute_op_terms_3d(phi, phi_prime, pair_state.wedgeN, k, s, s_prime, sin_beta0)
                : compute_op_terms_2d(phi, phi_prime, pair_state.wedgeN, k, s, s_prime);

            JonesOperator direct_op = assemble_diff_operator(
                cplx_mul_real(terms.direct, -1.f),
                terms.face0,
                terms.face1,
                face0_op,
                face1_op
            );

            Complex slope_factor = cplx(0.f, -1.f / k);
            JonesOperator slope_op = assemble_diff_operator(
                cplx_mul(slope_factor, cplx_mul_real(terms.directDphiPrime, -1.f)),
                cplx_mul(slope_factor, terms.face0DphiPrime),
                cplx_mul(slope_factor, terms.face1DphiPrime),
                face0_op,
                face1_op
            );

            Jones2 field_jones = jones_add(
                apply_jop(incident_jones_edge, direct_op),
                apply_jop(incident_derivative_jones_edge, slope_op)
            );
            Jones2 field_jones_finite = jones_scale(field_jones, finite_factor);

            nb::dict result;
            result["src_ext"] = src_ext;
            result["tgt_ext"] = tgt_ext;
            result["geom_valid"] = geom_valid;
            result["pole_safe"] = pole_safe;
            result["slope_safe"] = slope_safe;
            result["phi"] = phi;
            result["phi_prime"] = phi_prime;
            result["s"] = s;
            result["s_prime"] = s_prime;
            result["sin_beta0"] = sin_beta0;
            result["finite_factor"] = pack_complex_dict(finite_factor);
            result["direct_gain"] = pack_complex_dict(direct_gain);
            result["derivative_gain"] = pack_complex_dict(derivative_gain);
            result["field"] = pack_complex_dict(field);
            result["vector_field"] = pack_complex3_dict(vector_field);
            result["incident_jones_edge"] = pack_jones_dict(incident_jones_edge);
            result["incident_derivative_jones_edge"] = pack_jones_dict(incident_derivative_jones_edge);
            result["in_edge_basis"] = pack_basis_dict(in_edge_basis);
            result["out_edge_basis"] = pack_basis_dict(out_edge_basis);
            result["terms"] = pack_diffraction_terms_dict(terms);
            result["face0_op"] = pack_jones_operator_dict(face0_op);
            result["face1_op"] = pack_jones_operator_dict(face1_op);
            result["direct_op"] = pack_jones_operator_dict(direct_op);
            result["slope_op"] = pack_jones_operator_dict(slope_op);
            result["field_jones"] = pack_jones_dict(field_jones);
            result["field_jones_finite"] = pack_jones_dict(field_jones_finite);
            return result;
        },
        nb::arg("state"),
        nb::arg("target"),
        nb::arg("k"),
        nb::arg("material"),
        "Debug helper: evaluate one UTD pair contribution on the host."
    );

    m.def(
        "utd_debug_pair_device",
        [](
            nb::dict state,
            nb::dict target,
            float k,
            witwin::channel::native_ext::MaterialParams mat
        ) {
            witwin::channel::native_ext::PairInputs pair_state = read_pair_inputs_dict(state);
            witwin::channel::native_ext::float3a tgt_pos = read_float3_dict(target);
            return pack_pair_contribution_debug_dict(
                witwin::channel::native_ext::utd_debug_pair_device(pair_state, tgt_pos, k, mat)
            );
        },
        nb::arg("state"),
        nb::arg("target"),
        nb::arg("k"),
        nb::arg("material"),
        "Debug helper: evaluate one UTD pair contribution on CUDA from host-constructed inputs."
    );

    m.def(
        "utd_debug_pair_outputs_device",
        [](
            nb::dict state,
            nb::dict target,
            float k,
            witwin::channel::native_ext::MaterialParams mat
        ) {
            witwin::channel::native_ext::PairInputs pair_state = read_pair_inputs_dict(state);
            witwin::channel::native_ext::float3a tgt_pos = read_float3_dict(target);
            witwin::channel::native_ext::PairOutputs result =
                witwin::channel::native_ext::utd_debug_pair_outputs_device(pair_state, tgt_pos, k, mat);
            nb::dict packed;
            packed["field"] = pack_complex_dict(result.field);
            packed["vector_field"] = pack_complex3_dict(result.vectorField);
            return packed;
        },
        nb::arg("state"),
        nb::arg("target"),
        nb::arg("k"),
        nb::arg("material"),
        "Debug helper: evaluate compute_pair_contribution() on CUDA from host-constructed inputs."
    );

    m.def(
        "utd_debug_pair_from_state_soa",
        [](
            nb::tuple state_soa,
            int state_index,
            nb::dict target,
            float k,
            witwin::channel::native_ext::MaterialParams mat
        ) {
            if (nb::len(state_soa) != 81) {
                throw std::runtime_error("utd_debug_pair_from_state_soa expected 81 state arrays");
            }

            std::vector<DiffFloat> state_arrays;
            state_arrays.reserve(81);
            std::vector<const float*> state_slots;
            state_slots.reserve(81);
            for (size_t i = 0; i < 81; ++i) {
                state_arrays.emplace_back(nb::cast<DiffFloat>(state_soa[i]));
                drjit::eval(state_arrays.back());
                state_slots.push_back(drjit_data_ptr(state_arrays.back()));
            }

            witwin::channel::native_ext::float3a tgt_pos = read_float3_dict(target);
            return pack_pair_contribution_debug_dict(
                witwin::channel::native_ext::utd_debug_pair_from_state_slots(
                    state_slots.data(),
                    state_index,
                    tgt_pos,
                    k,
                    mat
                )
            );
        },
        nb::arg("state_soa"),
        nb::arg("state_index"),
        nb::arg("target"),
        nb::arg("k"),
        nb::arg("material"),
        "Debug helper: evaluate one UTD pair contribution on CUDA from packed state SoA slots."
    );

    m.def(
        "utd_debug_state_soa_at",
        [](nb::tuple state_soa, size_t state_index) {
            return pack_state_soa_debug_dict(state_soa, state_index);
        },
        nb::arg("state_soa"),
        nb::arg("state_index"),
        "Debug helper: copy one packed UTD state element from device memory to a host dict."
    );

    m.def(
        "utd_debug_read_float_array_at",
        [](const DiffFloat &array, size_t index) {
            return read_device_float_at(array, index, "utd_debug_read_float_array_at");
        },
        nb::arg("array"),
        nb::arg("index"),
        "Debug helper: copy one float element from a Dr.Jit CUDA array."
    );

    m.def(
        "utd_debug_write_float_array_at",
        [](DiffFloat &array, size_t index, float value) {
            drjit::eval(array);
            witwin::channel::native_ext::common::throw_cuda(
                cudaMemcpy(
                    drjit_data_ptr_mut(array) + index,
                    &value,
                    sizeof(float),
                    cudaMemcpyHostToDevice
                ),
                "utd_debug_write_float_array_at"
            );
        },
        nb::arg("array"),
        nb::arg("index"),
        nb::arg("value"),
        "Debug helper: write one float element into a Dr.Jit CUDA array."
    );

    m.def(
        "build_utd_state_tile_plan_raw",
        [](
            float support_eps,
            const Float &coeff0_0,
            const Float &coeff0_1,
            const Float &bias0,
            const Float &coeff1_0,
            const Float &coeff1_1,
            const Float &bias1,
            const Int32 &finite_mask,
            const Float &tile_coord_0_min,
            const Float &tile_coord_0_max,
            const Float &tile_coord_1_min,
            const Float &tile_coord_1_max
        ) {
            using witwin::channel::native_ext::common::throw_cuda;

            const int n_states = static_cast<int>(coeff0_0.size());
            const int n_tiles = static_cast<int>(tile_coord_0_min.size());
            UInt32 state_tile_counts = drjit::zeros<UInt32>(static_cast<size_t>(n_states));
            if (n_states <= 0 || n_tiles <= 0) {
                return nb::make_tuple(
                    state_tile_counts,
                    drjit::zeros<UInt32>(0),
                    drjit::zeros<UInt32>(0)
                );
            }

            drjit::eval(
                coeff0_0,
                coeff0_1,
                bias0,
                coeff1_0,
                coeff1_1,
                bias1,
                finite_mask,
                tile_coord_0_min,
                tile_coord_0_max,
                tile_coord_1_min,
                tile_coord_1_max,
                state_tile_counts
            );

            witwin::channel::native_ext::utd_state_tile_plan_count(
                support_eps,
                drjit_data_ptr(coeff0_0),
                drjit_data_ptr(coeff0_1),
                drjit_data_ptr(bias0),
                drjit_data_ptr(coeff1_0),
                drjit_data_ptr(coeff1_1),
                drjit_data_ptr(bias1),
                drjit_data_ptr(finite_mask),
                drjit_data_ptr(tile_coord_0_min),
                drjit_data_ptr(tile_coord_0_max),
                drjit_data_ptr(tile_coord_1_min),
                drjit_data_ptr(tile_coord_1_max),
                n_states,
                n_tiles,
                drjit_data_ptr_mut(state_tile_counts)
            );

            std::uint32_t *state_offsets = nullptr;
            throw_cuda(
                cudaMalloc(&state_offsets, static_cast<size_t>(n_states) * sizeof(std::uint32_t)),
                "build_utd_state_tile_plan_raw cudaMalloc(state_offsets)"
            );

            try {
                const std::uint32_t task_count = witwin::channel::native_ext::utd_state_tile_plan_scan(
                    drjit_data_ptr(state_tile_counts),
                    n_states,
                    state_offsets
                );

                UInt32 task_state_idx = drjit::zeros<UInt32>(static_cast<size_t>(task_count));
                UInt32 task_tile_idx = drjit::zeros<UInt32>(static_cast<size_t>(task_count));
                drjit::eval(task_state_idx, task_tile_idx);

                if (task_count > 0u) {
                    witwin::channel::native_ext::utd_state_tile_plan_write(
                        support_eps,
                        drjit_data_ptr(coeff0_0),
                        drjit_data_ptr(coeff0_1),
                        drjit_data_ptr(bias0),
                        drjit_data_ptr(coeff1_0),
                        drjit_data_ptr(coeff1_1),
                        drjit_data_ptr(bias1),
                        drjit_data_ptr(finite_mask),
                        drjit_data_ptr(tile_coord_0_min),
                        drjit_data_ptr(tile_coord_0_max),
                        drjit_data_ptr(tile_coord_1_min),
                        drjit_data_ptr(tile_coord_1_max),
                        n_states,
                        n_tiles,
                        state_offsets,
                        drjit_data_ptr_mut(task_state_idx),
                        drjit_data_ptr_mut(task_tile_idx)
                    );
                }

                throw_cuda(cudaFree(state_offsets), "build_utd_state_tile_plan_raw cudaFree(state_offsets)");
                return nb::make_tuple(
                    state_tile_counts,
                    task_state_idx,
                    task_tile_idx
                );
            } catch (...) {
                cudaFree(state_offsets);
                throw;
            }
        },
        "Build a UTD state-to-receiver-tile support plan on the GPU using exact half-space support tests."
    );

    // UTD forward raw launcher
    m.def(
        "utd_accumulate_forward_raw",
        [](nb::args, nb::kwargs) {
            throw std::runtime_error(
                "Legacy UTD raw forward launcher is unsupported. Finite-wedge UTD requires "
                "edge_line_min and edge_line_max."
            );
        },
        "Legacy UTD raw forward launcher. Always errors because finite-wedge bounds are required."
    );



    m.def(

        "utd_accumulate_tiled_arrays",

        [](nb::args, nb::kwargs) {
            throw std::runtime_error(
                "Legacy UTD tiled array launcher is unsupported. Finite-wedge UTD requires "
                "edge_line_min and edge_line_max."
            );
        },

        "Legacy UTD tiled array launcher. Always errors because finite-wedge bounds are required."

    );

#if 0
    m.def(

        "utd_accumulate_tiled_arrays",

        [](

            const Int32 &state_idx,

            const Int32 &rx_idx,

            nb::handle valid_mask_value,

            const Int32 &ownership_code,

            const DiffFloat &edge_pos_x, const DiffFloat &edge_pos_y, const DiffFloat &edge_pos_z,

            const DiffFloat &edge_dir_x, const DiffFloat &edge_dir_y, const DiffFloat &edge_dir_z,

            const DiffFloat &n0_x, const DiffFloat &n0_y, const DiffFloat &n0_z,

            const DiffFloat &nn_x, const DiffFloat &nn_y, const DiffFloat &nn_z,

            const DiffFloat &wedge_n,

            const DiffFloat &source_pos_x, const DiffFloat &source_pos_y, const DiffFloat &source_pos_z,

            const DiffFloat &incident_field_re, const DiffFloat &incident_field_im,

            const DiffFloat &incident_nderiv_re, const DiffFloat &incident_nderiv_im,

            const DiffFloat &r0_re, const DiffFloat &r0_im,

            const DiffFloat &rn_re, const DiffFloat &rn_im,

            const DiffFloat &inc_vec_x_re, const DiffFloat &inc_vec_x_im,

            const DiffFloat &inc_vec_y_re, const DiffFloat &inc_vec_y_im,

            const DiffFloat &inc_vec_z_re, const DiffFloat &inc_vec_z_im,

            const DiffFloat &inc_dvec_x_re, const DiffFloat &inc_dvec_x_im,

            const DiffFloat &inc_dvec_y_re, const DiffFloat &inc_dvec_y_im,

            const DiffFloat &inc_dvec_z_re, const DiffFloat &inc_dvec_z_im,

            const DiffFloat &inc_jones_u_re, const DiffFloat &inc_jones_u_im,

            const DiffFloat &inc_jones_v_re, const DiffFloat &inc_jones_v_im,

            const DiffFloat &inc_djones_u_re, const DiffFloat &inc_djones_u_im,

            const DiffFloat &inc_djones_v_re, const DiffFloat &inc_djones_v_im,

            const DiffFloat &inc_basis_u_x, const DiffFloat &inc_basis_u_y, const DiffFloat &inc_basis_u_z,

            const DiffFloat &inc_basis_v_x, const DiffFloat &inc_basis_v_y, const DiffFloat &inc_basis_v_z,

            const DiffFloat &inc_basis_k_x, const DiffFloat &inc_basis_k_y, const DiffFloat &inc_basis_k_z,

            const DiffFloat &face0_op_m00_re, const DiffFloat &face0_op_m00_im,

            const DiffFloat &face0_op_m01_re, const DiffFloat &face0_op_m01_im,

            const DiffFloat &face0_op_m10_re, const DiffFloat &face0_op_m10_im,

            const DiffFloat &face0_op_m11_re, const DiffFloat &face0_op_m11_im,

            const DiffFloat &face1_op_m00_re, const DiffFloat &face1_op_m00_im,

            const DiffFloat &face1_op_m01_re, const DiffFloat &face1_op_m01_im,

            const DiffFloat &face1_op_m10_re, const DiffFloat &face1_op_m10_im,

            const DiffFloat &face1_op_m11_re, const DiffFloat &face1_op_m11_im,

            const DiffFloat &face0_eta_r, const DiffFloat &face0_sigma,

            const DiffFloat &face0_gain, const DiffFloat &face0_use_fresnel,

            const DiffFloat &face0_present,

            const DiffFloat &face1_eta_r, const DiffFloat &face1_sigma,

            const DiffFloat &face1_gain, const DiffFloat &face1_use_fresnel,

            const DiffFloat &face1_present,

            const DiffFloat &rx_x, const DiffFloat &rx_y, const DiffFloat &rx_z,

            witwin::channel::native_ext::MaterialParams material,

            int n_local_states,

            int n_local_receivers,

            float k

        ) {

            size_t n_rx = rx_x.size();

            DiffFloat direct_re = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat direct_im = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat multi_re = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat multi_im = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat direct_vec_x_re = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat direct_vec_x_im = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat direct_vec_y_re = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat direct_vec_y_im = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat direct_vec_z_re = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat direct_vec_z_im = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat multi_vec_x_re = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat multi_vec_x_im = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat multi_vec_y_re = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat multi_vec_y_im = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat multi_vec_z_re = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat multi_vec_z_im = drjit::zeros<DiffFloat>(n_rx);



            Int32 valid_mask;

            const int* valid_mask_ptr = nullptr;

            if (
                valid_mask_value.is_valid()
                && !valid_mask_value.is_none()
                && !(PyBool_Check(valid_mask_value.ptr()) && !nb::cast<bool>(valid_mask_value))
            ) {

                valid_mask = nb::cast<Int32>(valid_mask_value);

                drjit::eval(valid_mask);

                valid_mask_ptr = drjit_data_ptr(valid_mask);

            }



            drjit::eval(

                state_idx, rx_idx, ownership_code,

                edge_pos_x, edge_pos_y, edge_pos_z,

                edge_dir_x, edge_dir_y, edge_dir_z,

                n0_x, n0_y, n0_z,

                nn_x, nn_y, nn_z,

                wedge_n,

                source_pos_x, source_pos_y, source_pos_z,

                incident_field_re, incident_field_im,

                incident_nderiv_re, incident_nderiv_im,

                r0_re, r0_im, rn_re, rn_im,

                inc_vec_x_re, inc_vec_x_im,

                inc_vec_y_re, inc_vec_y_im,

                inc_vec_z_re, inc_vec_z_im,

                inc_dvec_x_re, inc_dvec_x_im,

                inc_dvec_y_re, inc_dvec_y_im,

                inc_dvec_z_re, inc_dvec_z_im,

                inc_jones_u_re, inc_jones_u_im,

                inc_jones_v_re, inc_jones_v_im,

                inc_djones_u_re, inc_djones_u_im,

                inc_djones_v_re, inc_djones_v_im,

                inc_basis_u_x, inc_basis_u_y, inc_basis_u_z,

                inc_basis_v_x, inc_basis_v_y, inc_basis_v_z,

                inc_basis_k_x, inc_basis_k_y, inc_basis_k_z,

                face0_op_m00_re, face0_op_m00_im,

                face0_op_m01_re, face0_op_m01_im,

                face0_op_m10_re, face0_op_m10_im,

                face0_op_m11_re, face0_op_m11_im,

                face1_op_m00_re, face1_op_m00_im,

                face1_op_m01_re, face1_op_m01_im,

                face1_op_m10_re, face1_op_m10_im,

                face1_op_m11_re, face1_op_m11_im,

                face0_eta_r, face0_sigma, face0_gain, face0_use_fresnel, face0_present,

                face1_eta_r, face1_sigma, face1_gain, face1_use_fresnel, face1_present,

                rx_x, rx_y, rx_z,

                direct_re, direct_im, multi_re, multi_im,

                direct_vec_x_re, direct_vec_x_im,

                direct_vec_y_re, direct_vec_y_im,

                direct_vec_z_re, direct_vec_z_im,

                multi_vec_x_re, multi_vec_x_im,

                multi_vec_y_re, multi_vec_y_im,

                multi_vec_z_re, multi_vec_z_im

            );



            witwin::channel::native_ext::utd_accumulate_tiled_forward(

                drjit_data_ptr(state_idx),

                drjit_data_ptr(rx_idx),

                valid_mask_ptr,

                drjit_data_ptr(ownership_code),

                drjit_data_ptr(edge_pos_x), drjit_data_ptr(edge_pos_y), drjit_data_ptr(edge_pos_z),

                drjit_data_ptr(edge_dir_x), drjit_data_ptr(edge_dir_y), drjit_data_ptr(edge_dir_z),

                drjit_data_ptr(n0_x), drjit_data_ptr(n0_y), drjit_data_ptr(n0_z),

                drjit_data_ptr(nn_x), drjit_data_ptr(nn_y), drjit_data_ptr(nn_z),

                drjit_data_ptr(wedge_n),

                drjit_data_ptr(source_pos_x), drjit_data_ptr(source_pos_y), drjit_data_ptr(source_pos_z),

                drjit_data_ptr(incident_field_re), drjit_data_ptr(incident_field_im),

                drjit_data_ptr(incident_nderiv_re), drjit_data_ptr(incident_nderiv_im),

                drjit_data_ptr(r0_re), drjit_data_ptr(r0_im),

                drjit_data_ptr(rn_re), drjit_data_ptr(rn_im),

                drjit_data_ptr(inc_vec_x_re), drjit_data_ptr(inc_vec_x_im),

                drjit_data_ptr(inc_vec_y_re), drjit_data_ptr(inc_vec_y_im),

                drjit_data_ptr(inc_vec_z_re), drjit_data_ptr(inc_vec_z_im),

                drjit_data_ptr(inc_dvec_x_re), drjit_data_ptr(inc_dvec_x_im),

                drjit_data_ptr(inc_dvec_y_re), drjit_data_ptr(inc_dvec_y_im),

                drjit_data_ptr(inc_dvec_z_re), drjit_data_ptr(inc_dvec_z_im),

                drjit_data_ptr(inc_jones_u_re), drjit_data_ptr(inc_jones_u_im),

                drjit_data_ptr(inc_jones_v_re), drjit_data_ptr(inc_jones_v_im),

                drjit_data_ptr(inc_djones_u_re), drjit_data_ptr(inc_djones_u_im),

                drjit_data_ptr(inc_djones_v_re), drjit_data_ptr(inc_djones_v_im),

                drjit_data_ptr(inc_basis_u_x), drjit_data_ptr(inc_basis_u_y), drjit_data_ptr(inc_basis_u_z),

                drjit_data_ptr(inc_basis_v_x), drjit_data_ptr(inc_basis_v_y), drjit_data_ptr(inc_basis_v_z),

                drjit_data_ptr(inc_basis_k_x), drjit_data_ptr(inc_basis_k_y), drjit_data_ptr(inc_basis_k_z),

                drjit_data_ptr(face0_op_m00_re), drjit_data_ptr(face0_op_m00_im),

                drjit_data_ptr(face0_op_m01_re), drjit_data_ptr(face0_op_m01_im),

                drjit_data_ptr(face0_op_m10_re), drjit_data_ptr(face0_op_m10_im),

                drjit_data_ptr(face0_op_m11_re), drjit_data_ptr(face0_op_m11_im),

                drjit_data_ptr(face1_op_m00_re), drjit_data_ptr(face1_op_m00_im),

                drjit_data_ptr(face1_op_m01_re), drjit_data_ptr(face1_op_m01_im),

                drjit_data_ptr(face1_op_m10_re), drjit_data_ptr(face1_op_m10_im),

                drjit_data_ptr(face1_op_m11_re), drjit_data_ptr(face1_op_m11_im),

                drjit_data_ptr(face0_eta_r), drjit_data_ptr(face0_sigma),

                drjit_data_ptr(face0_gain), drjit_data_ptr(face0_use_fresnel),

                drjit_data_ptr(face0_present),

                drjit_data_ptr(face1_eta_r), drjit_data_ptr(face1_sigma),

                drjit_data_ptr(face1_gain), drjit_data_ptr(face1_use_fresnel),

                drjit_data_ptr(face1_present),

                drjit_data_ptr(rx_x), drjit_data_ptr(rx_y), drjit_data_ptr(rx_z),

                drjit_data_ptr_mut(direct_re), drjit_data_ptr_mut(direct_im),

                drjit_data_ptr_mut(multi_re), drjit_data_ptr_mut(multi_im),

                drjit_data_ptr_mut(direct_vec_x_re), drjit_data_ptr_mut(direct_vec_x_im),

                drjit_data_ptr_mut(direct_vec_y_re), drjit_data_ptr_mut(direct_vec_y_im),

                drjit_data_ptr_mut(direct_vec_z_re), drjit_data_ptr_mut(direct_vec_z_im),

                drjit_data_ptr_mut(multi_vec_x_re), drjit_data_ptr_mut(multi_vec_x_im),

                drjit_data_ptr_mut(multi_vec_y_re), drjit_data_ptr_mut(multi_vec_y_im),

                drjit_data_ptr_mut(multi_vec_z_re), drjit_data_ptr_mut(multi_vec_z_im),

                n_local_states,

                n_local_receivers,

                k,

                material

            );



            return nb::make_tuple(

                direct_vec_x_re, direct_vec_x_im,

                direct_vec_y_re, direct_vec_y_im,

                direct_vec_z_re, direct_vec_z_im,

                multi_vec_x_re, multi_vec_x_im,

                multi_vec_y_re, multi_vec_y_im,

                multi_vec_z_re, multi_vec_z_im

            );

        },

        "Launch the UTD tiled forward mega-kernel from compact tile descriptors."

    );



    #endif

    m.def(

        "utd_accumulate_tiled_arrays_v2",

        [](

            nb::object state_idx_value,

            nb::object rx_idx_value,

            nb::object valid_mask_value,

            nb::object ownership_code_value,

            nb::tuple state_soa,

            nb::tuple rx_arrays,

            witwin::channel::native_ext::MaterialParams material,

            int n_local_states,

            int n_local_receivers,

            float k

        ) {

            if (nb::len(rx_arrays) != 3)

                throw std::runtime_error("utd_accumulate_tiled_arrays_v2 expected 3 receiver arrays");

            MaterializedStateSlots state_slots =
                materialize_state_slot_pointers(state_soa, "utd_accumulate_tiled_arrays_v2");

            Int32 state_idx = nb::cast<Int32>(state_idx_value);
            Int32 rx_idx = nb::cast<Int32>(rx_idx_value);
            Int32 ownership_code = nb::cast<Int32>(ownership_code_value);
            DiffFloat rx_x = nb::cast<DiffFloat>(rx_arrays[0]);
            DiffFloat rx_y = nb::cast<DiffFloat>(rx_arrays[1]);
            DiffFloat rx_z = nb::cast<DiffFloat>(rx_arrays[2]);
            size_t n_rx = rx_x.size();

            DiffFloat direct_re = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat direct_im = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat multi_re = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat multi_im = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat direct_vec_x_re = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat direct_vec_x_im = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat direct_vec_y_re = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat direct_vec_y_im = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat direct_vec_z_re = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat direct_vec_z_im = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat multi_vec_x_re = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat multi_vec_x_im = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat multi_vec_y_re = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat multi_vec_y_im = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat multi_vec_z_re = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat multi_vec_z_im = drjit::zeros<DiffFloat>(n_rx);

            Int32 valid_mask;
            const int *valid_mask_ptr = nullptr;
            if (
                valid_mask_value.is_valid()
                && !valid_mask_value.is_none()
                && !(PyBool_Check(valid_mask_value.ptr()) && !nb::cast<bool>(valid_mask_value))
            ) {
                valid_mask = nb::cast<Int32>(valid_mask_value);
                drjit::eval(valid_mask);
                valid_mask_ptr = drjit_data_ptr(valid_mask);
            }

            drjit::eval(
                state_idx,
                rx_idx,
                ownership_code,
                rx_x,
                rx_y,
                rx_z,
                direct_re,
                direct_im,
                multi_re,
                multi_im,
                direct_vec_x_re,
                direct_vec_x_im,
                direct_vec_y_re,
                direct_vec_y_im,
                direct_vec_z_re,
                direct_vec_z_im,
                multi_vec_x_re,
                multi_vec_x_im,
                multi_vec_y_re,
                multi_vec_y_im,
                multi_vec_z_re,
                multi_vec_z_im
            );

            witwin::channel::native_ext::utd_accumulate_tiled_forward_slots(
                drjit_data_ptr(state_idx),
                drjit_data_ptr(rx_idx),
                valid_mask_ptr,
                drjit_data_ptr(ownership_code),
                state_slots.ptrs.data(),
                drjit_data_ptr(rx_x),
                drjit_data_ptr(rx_y),
                drjit_data_ptr(rx_z),
                drjit_data_ptr_mut(direct_re),
                drjit_data_ptr_mut(direct_im),
                drjit_data_ptr_mut(multi_re),
                drjit_data_ptr_mut(multi_im),
                drjit_data_ptr_mut(direct_vec_x_re),
                drjit_data_ptr_mut(direct_vec_x_im),
                drjit_data_ptr_mut(direct_vec_y_re),
                drjit_data_ptr_mut(direct_vec_y_im),
                drjit_data_ptr_mut(direct_vec_z_re),
                drjit_data_ptr_mut(direct_vec_z_im),
                drjit_data_ptr_mut(multi_vec_x_re),
                drjit_data_ptr_mut(multi_vec_x_im),
                drjit_data_ptr_mut(multi_vec_y_re),
                drjit_data_ptr_mut(multi_vec_y_im),
                drjit_data_ptr_mut(multi_vec_z_re),
                drjit_data_ptr_mut(multi_vec_z_im),
                n_local_states,
                n_local_receivers,
                k,
                material
            );

            return nb::make_tuple(
                direct_re,
                direct_im,
                multi_re,
                multi_im,
                direct_vec_x_re,
                direct_vec_x_im,
                direct_vec_y_re,
                direct_vec_y_im,
                direct_vec_z_re,
                direct_vec_z_im,
                multi_vec_x_re,
                multi_vec_x_im,
                multi_vec_y_re,
                multi_vec_y_im,
                multi_vec_z_re,
                multi_vec_z_im
            );

        },

        "Launch the UTD tiled forward kernel and return freshly allocated Dr.Jit output arrays."

    );

    m.def(

        "utd_accumulate_tiled_into",

        [](

            nb::object state_idx_value,

            nb::object rx_idx_value,

            nb::object valid_mask_value,

            nb::object ownership_code_value,

            nb::tuple state_soa,

            nb::tuple rx_arrays,

            nb::tuple output_arrays,

            witwin::channel::native_ext::MaterialParams material,

            int n_local_states,

            int n_local_receivers,

            float k

        ) {

            if (nb::len(state_soa) != 81)

                throw std::runtime_error("utd_accumulate_tiled_into expected 81 state arrays");

            if (nb::len(rx_arrays) != 3)

                throw std::runtime_error("utd_accumulate_tiled_into expected 3 receiver arrays");

            if (nb::len(output_arrays) != 16)

                throw std::runtime_error("utd_accumulate_tiled_into expected 16 output arrays");

            Int32 state_idx = nb::cast<Int32>(state_idx_value);
            Int32 rx_idx = nb::cast<Int32>(rx_idx_value);
            Int32 ownership_code = nb::cast<Int32>(ownership_code_value);



            auto state = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(state_soa[index]);

            };

            auto rx = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(rx_arrays[index]);

            };

            auto out = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(output_arrays[index]);

            };



            DiffFloat edge_pos_x = state(0), edge_pos_y = state(1), edge_pos_z = state(2);

            DiffFloat edge_dir_x = state(3), edge_dir_y = state(4), edge_dir_z = state(5);

            DiffFloat n0_x = state(6), n0_y = state(7), n0_z = state(8);

            DiffFloat nn_x = state(9), nn_y = state(10), nn_z = state(11);

            DiffFloat wedge_n = state(12);
            DiffFloat edge_line_min = state(13), edge_line_max = state(14);

            DiffFloat source_pos_x = state(15), source_pos_y = state(16), source_pos_z = state(17);

            DiffFloat incident_field_re = state(18), incident_field_im = state(19);

            DiffFloat incident_nderiv_re = state(20), incident_nderiv_im = state(21);

            DiffFloat r0_re = state(22), r0_im = state(23);

            DiffFloat rn_re = state(24), rn_im = state(25);

            DiffFloat inc_vec_x_re = state(26), inc_vec_x_im = state(27);

            DiffFloat inc_vec_y_re = state(28), inc_vec_y_im = state(29);

            DiffFloat inc_vec_z_re = state(30), inc_vec_z_im = state(31);

            DiffFloat inc_dvec_x_re = state(32), inc_dvec_x_im = state(33);

            DiffFloat inc_dvec_y_re = state(34), inc_dvec_y_im = state(35);

            DiffFloat inc_dvec_z_re = state(36), inc_dvec_z_im = state(37);

            DiffFloat inc_jones_u_re = state(38), inc_jones_u_im = state(39);

            DiffFloat inc_jones_v_re = state(40), inc_jones_v_im = state(41);

            DiffFloat inc_djones_u_re = state(42), inc_djones_u_im = state(43);

            DiffFloat inc_djones_v_re = state(44), inc_djones_v_im = state(45);

            DiffFloat inc_basis_u_x = state(46), inc_basis_u_y = state(47), inc_basis_u_z = state(48);

            DiffFloat inc_basis_v_x = state(49), inc_basis_v_y = state(50), inc_basis_v_z = state(51);

            DiffFloat inc_basis_k_x = state(52), inc_basis_k_y = state(53), inc_basis_k_z = state(54);

            DiffFloat face0_op_m00_re = state(55), face0_op_m00_im = state(56);

            DiffFloat face0_op_m01_re = state(57), face0_op_m01_im = state(58);

            DiffFloat face0_op_m10_re = state(59), face0_op_m10_im = state(60);

            DiffFloat face0_op_m11_re = state(61), face0_op_m11_im = state(62);

            DiffFloat face1_op_m00_re = state(63), face1_op_m00_im = state(64);

            DiffFloat face1_op_m01_re = state(65), face1_op_m01_im = state(66);

            DiffFloat face1_op_m10_re = state(67), face1_op_m10_im = state(68);

            DiffFloat face1_op_m11_re = state(69), face1_op_m11_im = state(70);

            DiffFloat face0_eta_r = state(71), face0_sigma = state(72);

            DiffFloat face0_gain = state(73), face0_use_fresnel = state(74), face0_present = state(75);

            DiffFloat face1_eta_r = state(76), face1_sigma = state(77);

            DiffFloat face1_gain = state(78), face1_use_fresnel = state(79), face1_present = state(80);



            DiffFloat rx_x = rx(0), rx_y = rx(1), rx_z = rx(2);

            DiffFloat direct_re = out(0), direct_im = out(1);

            DiffFloat multi_re = out(2), multi_im = out(3);

            DiffFloat direct_vec_x_re = out(4), direct_vec_x_im = out(5);

            DiffFloat direct_vec_y_re = out(6), direct_vec_y_im = out(7);

            DiffFloat direct_vec_z_re = out(8), direct_vec_z_im = out(9);

            DiffFloat multi_vec_x_re = out(10), multi_vec_x_im = out(11);

            DiffFloat multi_vec_y_re = out(12), multi_vec_y_im = out(13);

            DiffFloat multi_vec_z_re = out(14), multi_vec_z_im = out(15);



            Int32 valid_mask;

            const int *valid_mask_ptr = nullptr;

            if (
                valid_mask_value.is_valid()
                && !valid_mask_value.is_none()
                && !(PyBool_Check(valid_mask_value.ptr()) && !nb::cast<bool>(valid_mask_value))
            ) {

                valid_mask = nb::cast<Int32>(valid_mask_value);

                drjit::eval(valid_mask);

                valid_mask_ptr = drjit_data_ptr(valid_mask);

            }



            drjit::eval(

                state_idx, rx_idx, ownership_code,

                edge_pos_x, edge_pos_y, edge_pos_z,

                edge_dir_x, edge_dir_y, edge_dir_z,

                n0_x, n0_y, n0_z,

                nn_x, nn_y, nn_z,

                wedge_n,

                edge_line_min, edge_line_max,

                source_pos_x, source_pos_y, source_pos_z,

                incident_field_re, incident_field_im,

                incident_nderiv_re, incident_nderiv_im,

                r0_re, r0_im, rn_re, rn_im,

                inc_vec_x_re, inc_vec_x_im,

                inc_vec_y_re, inc_vec_y_im,

                inc_vec_z_re, inc_vec_z_im,

                inc_dvec_x_re, inc_dvec_x_im,

                inc_dvec_y_re, inc_dvec_y_im,

                inc_dvec_z_re, inc_dvec_z_im,

                inc_jones_u_re, inc_jones_u_im,

                inc_jones_v_re, inc_jones_v_im,

                inc_djones_u_re, inc_djones_u_im,

                inc_djones_v_re, inc_djones_v_im,

                inc_basis_u_x, inc_basis_u_y, inc_basis_u_z,

                inc_basis_v_x, inc_basis_v_y, inc_basis_v_z,

                inc_basis_k_x, inc_basis_k_y, inc_basis_k_z,

                face0_op_m00_re, face0_op_m00_im,

                face0_op_m01_re, face0_op_m01_im,

                face0_op_m10_re, face0_op_m10_im,

                face0_op_m11_re, face0_op_m11_im,

                face1_op_m00_re, face1_op_m00_im,

                face1_op_m01_re, face1_op_m01_im,

                face1_op_m10_re, face1_op_m10_im,

                face1_op_m11_re, face1_op_m11_im,

                face0_eta_r, face0_sigma, face0_gain, face0_use_fresnel, face0_present,

                face1_eta_r, face1_sigma, face1_gain, face1_use_fresnel, face1_present,

                rx_x, rx_y, rx_z,

                direct_re, direct_im, multi_re, multi_im,

                direct_vec_x_re, direct_vec_x_im,

                direct_vec_y_re, direct_vec_y_im,

                direct_vec_z_re, direct_vec_z_im,

                multi_vec_x_re, multi_vec_x_im,

                multi_vec_y_re, multi_vec_y_im,

                multi_vec_z_re, multi_vec_z_im

            );


            std::vector<const float*> state_slots = {
                drjit_data_ptr(edge_pos_x), drjit_data_ptr(edge_pos_y), drjit_data_ptr(edge_pos_z),
                drjit_data_ptr(edge_dir_x), drjit_data_ptr(edge_dir_y), drjit_data_ptr(edge_dir_z),
                drjit_data_ptr(n0_x), drjit_data_ptr(n0_y), drjit_data_ptr(n0_z),
                drjit_data_ptr(nn_x), drjit_data_ptr(nn_y), drjit_data_ptr(nn_z),
                drjit_data_ptr(wedge_n),
                drjit_data_ptr(edge_line_min), drjit_data_ptr(edge_line_max),
                drjit_data_ptr(source_pos_x), drjit_data_ptr(source_pos_y), drjit_data_ptr(source_pos_z),
                drjit_data_ptr(incident_field_re), drjit_data_ptr(incident_field_im),
                drjit_data_ptr(incident_nderiv_re), drjit_data_ptr(incident_nderiv_im),
                drjit_data_ptr(r0_re), drjit_data_ptr(r0_im),
                drjit_data_ptr(rn_re), drjit_data_ptr(rn_im),
                drjit_data_ptr(inc_vec_x_re), drjit_data_ptr(inc_vec_x_im),
                drjit_data_ptr(inc_vec_y_re), drjit_data_ptr(inc_vec_y_im),
                drjit_data_ptr(inc_vec_z_re), drjit_data_ptr(inc_vec_z_im),
                drjit_data_ptr(inc_dvec_x_re), drjit_data_ptr(inc_dvec_x_im),
                drjit_data_ptr(inc_dvec_y_re), drjit_data_ptr(inc_dvec_y_im),
                drjit_data_ptr(inc_dvec_z_re), drjit_data_ptr(inc_dvec_z_im),
                drjit_data_ptr(inc_jones_u_re), drjit_data_ptr(inc_jones_u_im),
                drjit_data_ptr(inc_jones_v_re), drjit_data_ptr(inc_jones_v_im),
                drjit_data_ptr(inc_djones_u_re), drjit_data_ptr(inc_djones_u_im),
                drjit_data_ptr(inc_djones_v_re), drjit_data_ptr(inc_djones_v_im),
                drjit_data_ptr(inc_basis_u_x), drjit_data_ptr(inc_basis_u_y), drjit_data_ptr(inc_basis_u_z),
                drjit_data_ptr(inc_basis_v_x), drjit_data_ptr(inc_basis_v_y), drjit_data_ptr(inc_basis_v_z),
                drjit_data_ptr(inc_basis_k_x), drjit_data_ptr(inc_basis_k_y), drjit_data_ptr(inc_basis_k_z),
                drjit_data_ptr(face0_op_m00_re), drjit_data_ptr(face0_op_m00_im),
                drjit_data_ptr(face0_op_m01_re), drjit_data_ptr(face0_op_m01_im),
                drjit_data_ptr(face0_op_m10_re), drjit_data_ptr(face0_op_m10_im),
                drjit_data_ptr(face0_op_m11_re), drjit_data_ptr(face0_op_m11_im),
                drjit_data_ptr(face1_op_m00_re), drjit_data_ptr(face1_op_m00_im),
                drjit_data_ptr(face1_op_m01_re), drjit_data_ptr(face1_op_m01_im),
                drjit_data_ptr(face1_op_m10_re), drjit_data_ptr(face1_op_m10_im),
                drjit_data_ptr(face1_op_m11_re), drjit_data_ptr(face1_op_m11_im),
                drjit_data_ptr(face0_eta_r), drjit_data_ptr(face0_sigma),
                drjit_data_ptr(face0_gain), drjit_data_ptr(face0_use_fresnel), drjit_data_ptr(face0_present),
                drjit_data_ptr(face1_eta_r), drjit_data_ptr(face1_sigma),
                drjit_data_ptr(face1_gain), drjit_data_ptr(face1_use_fresnel), drjit_data_ptr(face1_present),
            };

            witwin::channel::native_ext::utd_accumulate_tiled_forward_slots(
                drjit_data_ptr(state_idx),
                drjit_data_ptr(rx_idx),
                valid_mask_ptr,
                drjit_data_ptr(ownership_code),
                state_slots.data(),
                drjit_data_ptr(rx_x), drjit_data_ptr(rx_y), drjit_data_ptr(rx_z),
                drjit_data_ptr_mut(direct_re), drjit_data_ptr_mut(direct_im),
                drjit_data_ptr_mut(multi_re), drjit_data_ptr_mut(multi_im),
                drjit_data_ptr_mut(direct_vec_x_re), drjit_data_ptr_mut(direct_vec_x_im),
                drjit_data_ptr_mut(direct_vec_y_re), drjit_data_ptr_mut(direct_vec_y_im),
                drjit_data_ptr_mut(direct_vec_z_re), drjit_data_ptr_mut(direct_vec_z_im),
                drjit_data_ptr_mut(multi_vec_x_re), drjit_data_ptr_mut(multi_vec_x_im),
                drjit_data_ptr_mut(multi_vec_y_re), drjit_data_ptr_mut(multi_vec_y_im),
                drjit_data_ptr_mut(multi_vec_z_re), drjit_data_ptr_mut(multi_vec_z_im),
                n_local_states,
                n_local_receivers,
                k,
                material
            );

        },

        "Launch the UTD tiled forward mega-kernel into preallocated Dr.Jit output arrays."

    );



    m.def(

        "utd_accumulate_tiled_vector_power_into",

        [](

            nb::object state_idx_value,

            nb::object rx_idx_value,

            nb::object valid_mask_value,

            nb::object ownership_code_value,

            nb::tuple state_soa,

            nb::tuple rx_arrays,

            nb::tuple output_arrays,

            witwin::channel::native_ext::MaterialParams material,

            int n_local_states,

            int n_local_receivers,

            float k,

            float rx_pol_x,

            float rx_pol_y,

            float rx_pol_z

        ) {

            if (nb::len(state_soa) != 81)

                throw std::runtime_error("utd_accumulate_tiled_vector_power_into expected 81 state arrays");

            if (nb::len(rx_arrays) != 3)

                throw std::runtime_error("utd_accumulate_tiled_vector_power_into expected 3 receiver arrays");

            if (nb::len(output_arrays) != 18)

                throw std::runtime_error("utd_accumulate_tiled_vector_power_into expected 18 output arrays");

            Int32 state_idx = nb::cast<Int32>(state_idx_value);
            Int32 rx_idx = nb::cast<Int32>(rx_idx_value);
            Int32 ownership_code = nb::cast<Int32>(ownership_code_value);



            auto state = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(state_soa[index]);

            };

            auto rx = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(rx_arrays[index]);

            };

            auto out = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(output_arrays[index]);

            };



            DiffFloat edge_pos_x = state(0), edge_pos_y = state(1), edge_pos_z = state(2);

            DiffFloat edge_dir_x = state(3), edge_dir_y = state(4), edge_dir_z = state(5);

            DiffFloat n0_x = state(6), n0_y = state(7), n0_z = state(8);

            DiffFloat nn_x = state(9), nn_y = state(10), nn_z = state(11);

            DiffFloat wedge_n = state(12);
            DiffFloat edge_line_min = state(13), edge_line_max = state(14);

            DiffFloat source_pos_x = state(15), source_pos_y = state(16), source_pos_z = state(17);

            DiffFloat incident_field_re = state(18), incident_field_im = state(19);

            DiffFloat incident_nderiv_re = state(20), incident_nderiv_im = state(21);

            DiffFloat r0_re = state(22), r0_im = state(23);

            DiffFloat rn_re = state(24), rn_im = state(25);

            DiffFloat inc_vec_x_re = state(26), inc_vec_x_im = state(27);

            DiffFloat inc_vec_y_re = state(28), inc_vec_y_im = state(29);

            DiffFloat inc_vec_z_re = state(30), inc_vec_z_im = state(31);

            DiffFloat inc_dvec_x_re = state(32), inc_dvec_x_im = state(33);

            DiffFloat inc_dvec_y_re = state(34), inc_dvec_y_im = state(35);

            DiffFloat inc_dvec_z_re = state(36), inc_dvec_z_im = state(37);

            DiffFloat inc_jones_u_re = state(38), inc_jones_u_im = state(39);

            DiffFloat inc_jones_v_re = state(40), inc_jones_v_im = state(41);

            DiffFloat inc_djones_u_re = state(42), inc_djones_u_im = state(43);

            DiffFloat inc_djones_v_re = state(44), inc_djones_v_im = state(45);

            DiffFloat inc_basis_u_x = state(46), inc_basis_u_y = state(47), inc_basis_u_z = state(48);

            DiffFloat inc_basis_v_x = state(49), inc_basis_v_y = state(50), inc_basis_v_z = state(51);

            DiffFloat inc_basis_k_x = state(52), inc_basis_k_y = state(53), inc_basis_k_z = state(54);

            DiffFloat face0_op_m00_re = state(55), face0_op_m00_im = state(56);

            DiffFloat face0_op_m01_re = state(57), face0_op_m01_im = state(58);

            DiffFloat face0_op_m10_re = state(59), face0_op_m10_im = state(60);

            DiffFloat face0_op_m11_re = state(61), face0_op_m11_im = state(62);

            DiffFloat face1_op_m00_re = state(63), face1_op_m00_im = state(64);

            DiffFloat face1_op_m01_re = state(65), face1_op_m01_im = state(66);

            DiffFloat face1_op_m10_re = state(67), face1_op_m10_im = state(68);

            DiffFloat face1_op_m11_re = state(69), face1_op_m11_im = state(70);

            DiffFloat face0_eta_r = state(71), face0_sigma = state(72);

            DiffFloat face0_gain = state(73), face0_use_fresnel = state(74), face0_present = state(75);

            DiffFloat face1_eta_r = state(76), face1_sigma = state(77);

            DiffFloat face1_gain = state(78), face1_use_fresnel = state(79), face1_present = state(80);



            DiffFloat rx_x = rx(0), rx_y = rx(1), rx_z = rx(2);

            DiffFloat direct_re = out(0), direct_im = out(1);

            DiffFloat multi_re = out(2), multi_im = out(3);

            DiffFloat direct_vec_x_re = out(4), direct_vec_x_im = out(5);

            DiffFloat direct_vec_y_re = out(6), direct_vec_y_im = out(7);

            DiffFloat direct_vec_z_re = out(8), direct_vec_z_im = out(9);

            DiffFloat multi_vec_x_re = out(10), multi_vec_x_im = out(11);

            DiffFloat multi_vec_y_re = out(12), multi_vec_y_im = out(13);

            DiffFloat multi_vec_z_re = out(14), multi_vec_z_im = out(15);

            DiffFloat matched_power = out(16), valid_pair_count = out(17);



            Int32 valid_mask;

            const int *valid_mask_ptr = nullptr;

            if (
                valid_mask_value.is_valid()
                && !valid_mask_value.is_none()
                && !(PyBool_Check(valid_mask_value.ptr()) && !nb::cast<bool>(valid_mask_value))
            ) {

                valid_mask = nb::cast<Int32>(valid_mask_value);

                drjit::eval(valid_mask);

                valid_mask_ptr = drjit_data_ptr(valid_mask);

            }



            drjit::eval(

                state_idx, rx_idx, ownership_code,

                edge_pos_x, edge_pos_y, edge_pos_z,

                edge_dir_x, edge_dir_y, edge_dir_z,

                n0_x, n0_y, n0_z,

                nn_x, nn_y, nn_z,

                wedge_n,

                edge_line_min, edge_line_max,

                source_pos_x, source_pos_y, source_pos_z,

                incident_field_re, incident_field_im,

                incident_nderiv_re, incident_nderiv_im,

                r0_re, r0_im, rn_re, rn_im,

                inc_vec_x_re, inc_vec_x_im,

                inc_vec_y_re, inc_vec_y_im,

                inc_vec_z_re, inc_vec_z_im,

                inc_dvec_x_re, inc_dvec_x_im,

                inc_dvec_y_re, inc_dvec_y_im,

                inc_dvec_z_re, inc_dvec_z_im,

                inc_jones_u_re, inc_jones_u_im,

                inc_jones_v_re, inc_jones_v_im,

                inc_djones_u_re, inc_djones_u_im,

                inc_djones_v_re, inc_djones_v_im,

                inc_basis_u_x, inc_basis_u_y, inc_basis_u_z,

                inc_basis_v_x, inc_basis_v_y, inc_basis_v_z,

                inc_basis_k_x, inc_basis_k_y, inc_basis_k_z,

                face0_op_m00_re, face0_op_m00_im,

                face0_op_m01_re, face0_op_m01_im,

                face0_op_m10_re, face0_op_m10_im,

                face0_op_m11_re, face0_op_m11_im,

                face1_op_m00_re, face1_op_m00_im,

                face1_op_m01_re, face1_op_m01_im,

                face1_op_m10_re, face1_op_m10_im,

                face1_op_m11_re, face1_op_m11_im,

                face0_eta_r, face0_sigma, face0_gain, face0_use_fresnel, face0_present,

                face1_eta_r, face1_sigma, face1_gain, face1_use_fresnel, face1_present,

                rx_x, rx_y, rx_z,

                direct_re, direct_im, multi_re, multi_im,

                direct_vec_x_re, direct_vec_x_im,

                direct_vec_y_re, direct_vec_y_im,

                direct_vec_z_re, direct_vec_z_im,

                multi_vec_x_re, multi_vec_x_im,

                multi_vec_y_re, multi_vec_y_im,

                multi_vec_z_re, multi_vec_z_im,

                matched_power,

                valid_pair_count

            );


            std::vector<const float*> state_slots = {
                drjit_data_ptr(edge_pos_x), drjit_data_ptr(edge_pos_y), drjit_data_ptr(edge_pos_z),
                drjit_data_ptr(edge_dir_x), drjit_data_ptr(edge_dir_y), drjit_data_ptr(edge_dir_z),
                drjit_data_ptr(n0_x), drjit_data_ptr(n0_y), drjit_data_ptr(n0_z),
                drjit_data_ptr(nn_x), drjit_data_ptr(nn_y), drjit_data_ptr(nn_z),
                drjit_data_ptr(wedge_n),
                drjit_data_ptr(edge_line_min), drjit_data_ptr(edge_line_max),
                drjit_data_ptr(source_pos_x), drjit_data_ptr(source_pos_y), drjit_data_ptr(source_pos_z),
                drjit_data_ptr(incident_field_re), drjit_data_ptr(incident_field_im),
                drjit_data_ptr(incident_nderiv_re), drjit_data_ptr(incident_nderiv_im),
                drjit_data_ptr(r0_re), drjit_data_ptr(r0_im),
                drjit_data_ptr(rn_re), drjit_data_ptr(rn_im),
                drjit_data_ptr(inc_vec_x_re), drjit_data_ptr(inc_vec_x_im),
                drjit_data_ptr(inc_vec_y_re), drjit_data_ptr(inc_vec_y_im),
                drjit_data_ptr(inc_vec_z_re), drjit_data_ptr(inc_vec_z_im),
                drjit_data_ptr(inc_dvec_x_re), drjit_data_ptr(inc_dvec_x_im),
                drjit_data_ptr(inc_dvec_y_re), drjit_data_ptr(inc_dvec_y_im),
                drjit_data_ptr(inc_dvec_z_re), drjit_data_ptr(inc_dvec_z_im),
                drjit_data_ptr(inc_jones_u_re), drjit_data_ptr(inc_jones_u_im),
                drjit_data_ptr(inc_jones_v_re), drjit_data_ptr(inc_jones_v_im),
                drjit_data_ptr(inc_djones_u_re), drjit_data_ptr(inc_djones_u_im),
                drjit_data_ptr(inc_djones_v_re), drjit_data_ptr(inc_djones_v_im),
                drjit_data_ptr(inc_basis_u_x), drjit_data_ptr(inc_basis_u_y), drjit_data_ptr(inc_basis_u_z),
                drjit_data_ptr(inc_basis_v_x), drjit_data_ptr(inc_basis_v_y), drjit_data_ptr(inc_basis_v_z),
                drjit_data_ptr(inc_basis_k_x), drjit_data_ptr(inc_basis_k_y), drjit_data_ptr(inc_basis_k_z),
                drjit_data_ptr(face0_op_m00_re), drjit_data_ptr(face0_op_m00_im),
                drjit_data_ptr(face0_op_m01_re), drjit_data_ptr(face0_op_m01_im),
                drjit_data_ptr(face0_op_m10_re), drjit_data_ptr(face0_op_m10_im),
                drjit_data_ptr(face0_op_m11_re), drjit_data_ptr(face0_op_m11_im),
                drjit_data_ptr(face1_op_m00_re), drjit_data_ptr(face1_op_m00_im),
                drjit_data_ptr(face1_op_m01_re), drjit_data_ptr(face1_op_m01_im),
                drjit_data_ptr(face1_op_m10_re), drjit_data_ptr(face1_op_m10_im),
                drjit_data_ptr(face1_op_m11_re), drjit_data_ptr(face1_op_m11_im),
                drjit_data_ptr(face0_eta_r), drjit_data_ptr(face0_sigma),
                drjit_data_ptr(face0_gain), drjit_data_ptr(face0_use_fresnel), drjit_data_ptr(face0_present),
                drjit_data_ptr(face1_eta_r), drjit_data_ptr(face1_sigma),
                drjit_data_ptr(face1_gain), drjit_data_ptr(face1_use_fresnel), drjit_data_ptr(face1_present),
            };

            witwin::channel::native_ext::utd_accumulate_tiled_vector_power_forward_slots(
                drjit_data_ptr(state_idx),
                drjit_data_ptr(rx_idx),
                valid_mask_ptr,
                drjit_data_ptr(ownership_code),
                state_slots.data(),
                drjit_data_ptr(rx_x), drjit_data_ptr(rx_y), drjit_data_ptr(rx_z),
                drjit_data_ptr_mut(direct_re), drjit_data_ptr_mut(direct_im),
                drjit_data_ptr_mut(multi_re), drjit_data_ptr_mut(multi_im),
                drjit_data_ptr_mut(direct_vec_x_re), drjit_data_ptr_mut(direct_vec_x_im),
                drjit_data_ptr_mut(direct_vec_y_re), drjit_data_ptr_mut(direct_vec_y_im),
                drjit_data_ptr_mut(direct_vec_z_re), drjit_data_ptr_mut(direct_vec_z_im),
                drjit_data_ptr_mut(multi_vec_x_re), drjit_data_ptr_mut(multi_vec_x_im),
                drjit_data_ptr_mut(multi_vec_y_re), drjit_data_ptr_mut(multi_vec_y_im),
                drjit_data_ptr_mut(multi_vec_z_re), drjit_data_ptr_mut(multi_vec_z_im),
                drjit_data_ptr_mut(matched_power),
                drjit_data_ptr_mut(valid_pair_count),
                n_local_states,
                n_local_receivers,
                k,
                rx_pol_x,
                rx_pol_y,
                rx_pol_z,
                material
            );

        },

        "Launch the UTD tiled forward vector-power kernel into preallocated Dr.Jit output arrays."

    );



    m.def(

        "utd_accumulate_scalar_power_arrays",

        [](

            nb::handle rx_idx_value,

            nb::tuple state_soa,

            nb::tuple rx_arrays,

            witwin::channel::native_ext::MaterialParams material,

            int n_output_rx,

            int n_pairs,

            float k,

            float rx_pol_x,

            float rx_pol_y,

            float rx_pol_z

        ) {

            if (nb::len(rx_arrays) != 3)

                throw std::runtime_error("utd_accumulate_scalar_power_arrays expected 3 receiver arrays");

            MaterializedStateSlots state_slots =
                materialize_state_slot_pointers(state_soa, "utd_accumulate_scalar_power_arrays");

            Int32 rx_idx = nb::cast<Int32>(rx_idx_value);
            DiffFloat rx_x = nb::cast<DiffFloat>(rx_arrays[0]);
            DiffFloat rx_y = nb::cast<DiffFloat>(rx_arrays[1]);
            DiffFloat rx_z = nb::cast<DiffFloat>(rx_arrays[2]);

            DiffFloat coherent_re = drjit::zeros<DiffFloat>(static_cast<size_t>(n_output_rx > 0 ? n_output_rx : 0));
            DiffFloat coherent_im = drjit::zeros<DiffFloat>(static_cast<size_t>(n_output_rx > 0 ? n_output_rx : 0));
            DiffFloat power = drjit::zeros<DiffFloat>(static_cast<size_t>(n_output_rx > 0 ? n_output_rx : 0));
            DiffFloat valid_pair_count = drjit::zeros<DiffFloat>(1);

            drjit::eval(
                rx_idx,
                rx_x,
                rx_y,
                rx_z,
                coherent_re,
                coherent_im,
                power,
                valid_pair_count
            );

            witwin::channel::native_ext::utd_accumulate_scalar_power_forward_slots(
                state_slots.ptrs.data(),
                drjit_data_ptr(rx_idx),
                drjit_data_ptr(rx_x),
                drjit_data_ptr(rx_y),
                drjit_data_ptr(rx_z),
                drjit_data_ptr_mut(coherent_re),
                drjit_data_ptr_mut(coherent_im),
                drjit_data_ptr_mut(power),
                drjit_data_ptr_mut(valid_pair_count),
                n_pairs,
                k,
                rx_pol_x,
                rx_pol_y,
                rx_pol_z,
                material
            );

            return nb::make_tuple(
                coherent_re,
                coherent_im,
                power,
                valid_pair_count
            );

        },

        "Launch the UTD scalar-power forward kernel and return freshly allocated Dr.Jit output arrays."

    );

    m.def(

        "utd_accumulate_scalar_power_into",

        [](

            nb::handle rx_idx_value,

            nb::tuple state_soa,

            nb::tuple rx_arrays,

            nb::tuple output_arrays,

            witwin::channel::native_ext::MaterialParams material,

            int n_pairs,

            float k,

            float rx_pol_x,

            float rx_pol_y,

            float rx_pol_z

        ) {

            if (nb::len(state_soa) != 81)

                throw std::runtime_error("utd_accumulate_scalar_power_into expected 81 state arrays");

            if (nb::len(rx_arrays) != 3)

                throw std::runtime_error("utd_accumulate_scalar_power_into expected 3 receiver arrays");

            if (nb::len(output_arrays) != 4)

                throw std::runtime_error("utd_accumulate_scalar_power_into expected 4 output arrays");

            Int32 rx_idx = nb::cast<Int32>(rx_idx_value);



            auto state = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(state_soa[index]);

            };

            auto rx = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(rx_arrays[index]);

            };

            auto out = [&](size_t index) -> DiffFloat {

                return nb::cast<DiffFloat>(output_arrays[index]);

            };



            DiffFloat edge_pos_x = state(0), edge_pos_y = state(1), edge_pos_z = state(2);

            DiffFloat edge_dir_x = state(3), edge_dir_y = state(4), edge_dir_z = state(5);

            DiffFloat n0_x = state(6), n0_y = state(7), n0_z = state(8);

            DiffFloat nn_x = state(9), nn_y = state(10), nn_z = state(11);

            DiffFloat wedge_n = state(12);
            DiffFloat edge_line_min = state(13), edge_line_max = state(14);

            DiffFloat source_pos_x = state(15), source_pos_y = state(16), source_pos_z = state(17);

            DiffFloat incident_field_re = state(18), incident_field_im = state(19);

            DiffFloat incident_nderiv_re = state(20), incident_nderiv_im = state(21);

            DiffFloat r0_re = state(22), r0_im = state(23);

            DiffFloat rn_re = state(24), rn_im = state(25);

            DiffFloat inc_vec_x_re = state(26), inc_vec_x_im = state(27);

            DiffFloat inc_vec_y_re = state(28), inc_vec_y_im = state(29);

            DiffFloat inc_vec_z_re = state(30), inc_vec_z_im = state(31);

            DiffFloat inc_dvec_x_re = state(32), inc_dvec_x_im = state(33);

            DiffFloat inc_dvec_y_re = state(34), inc_dvec_y_im = state(35);

            DiffFloat inc_dvec_z_re = state(36), inc_dvec_z_im = state(37);

            DiffFloat inc_jones_u_re = state(38), inc_jones_u_im = state(39);

            DiffFloat inc_jones_v_re = state(40), inc_jones_v_im = state(41);

            DiffFloat inc_djones_u_re = state(42), inc_djones_u_im = state(43);

            DiffFloat inc_djones_v_re = state(44), inc_djones_v_im = state(45);

            DiffFloat inc_basis_u_x = state(46), inc_basis_u_y = state(47), inc_basis_u_z = state(48);

            DiffFloat inc_basis_v_x = state(49), inc_basis_v_y = state(50), inc_basis_v_z = state(51);

            DiffFloat inc_basis_k_x = state(52), inc_basis_k_y = state(53), inc_basis_k_z = state(54);

            DiffFloat face0_op_m00_re = state(55), face0_op_m00_im = state(56);

            DiffFloat face0_op_m01_re = state(57), face0_op_m01_im = state(58);

            DiffFloat face0_op_m10_re = state(59), face0_op_m10_im = state(60);

            DiffFloat face0_op_m11_re = state(61), face0_op_m11_im = state(62);

            DiffFloat face1_op_m00_re = state(63), face1_op_m00_im = state(64);

            DiffFloat face1_op_m01_re = state(65), face1_op_m01_im = state(66);

            DiffFloat face1_op_m10_re = state(67), face1_op_m10_im = state(68);

            DiffFloat face1_op_m11_re = state(69), face1_op_m11_im = state(70);

            DiffFloat face0_eta_r = state(71), face0_sigma = state(72);

            DiffFloat face0_gain = state(73), face0_use_fresnel = state(74), face0_present = state(75);

            DiffFloat face1_eta_r = state(76), face1_sigma = state(77);

            DiffFloat face1_gain = state(78), face1_use_fresnel = state(79), face1_present = state(80);



            DiffFloat rx_x = rx(0), rx_y = rx(1), rx_z = rx(2);

            DiffFloat coherent_re = out(0), coherent_im = out(1);

            DiffFloat power = out(2), valid_pair_count = out(3);



            drjit::eval(

                rx_idx,

                edge_pos_x, edge_pos_y, edge_pos_z,

                edge_dir_x, edge_dir_y, edge_dir_z,

                n0_x, n0_y, n0_z,

                nn_x, nn_y, nn_z,

                wedge_n,

                edge_line_min, edge_line_max,

                source_pos_x, source_pos_y, source_pos_z,

                incident_field_re, incident_field_im,

                incident_nderiv_re, incident_nderiv_im,

                r0_re, r0_im, rn_re, rn_im,

                inc_vec_x_re, inc_vec_x_im,

                inc_vec_y_re, inc_vec_y_im,

                inc_vec_z_re, inc_vec_z_im,

                inc_dvec_x_re, inc_dvec_x_im,

                inc_dvec_y_re, inc_dvec_y_im,

                inc_dvec_z_re, inc_dvec_z_im,

                inc_jones_u_re, inc_jones_u_im,

                inc_jones_v_re, inc_jones_v_im,

                inc_djones_u_re, inc_djones_u_im,

                inc_djones_v_re, inc_djones_v_im,

                inc_basis_u_x, inc_basis_u_y, inc_basis_u_z,

                inc_basis_v_x, inc_basis_v_y, inc_basis_v_z,

                inc_basis_k_x, inc_basis_k_y, inc_basis_k_z,

                face0_op_m00_re, face0_op_m00_im,

                face0_op_m01_re, face0_op_m01_im,

                face0_op_m10_re, face0_op_m10_im,

                face0_op_m11_re, face0_op_m11_im,

                face1_op_m00_re, face1_op_m00_im,

                face1_op_m01_re, face1_op_m01_im,

                face1_op_m10_re, face1_op_m10_im,

                face1_op_m11_re, face1_op_m11_im,

                face0_eta_r, face0_sigma, face0_gain, face0_use_fresnel, face0_present,

                face1_eta_r, face1_sigma, face1_gain, face1_use_fresnel, face1_present,

                rx_x, rx_y, rx_z,

                coherent_re, coherent_im, power, valid_pair_count

            );


            std::vector<const float*> state_slots = {
                drjit_data_ptr(edge_pos_x), drjit_data_ptr(edge_pos_y), drjit_data_ptr(edge_pos_z),
                drjit_data_ptr(edge_dir_x), drjit_data_ptr(edge_dir_y), drjit_data_ptr(edge_dir_z),
                drjit_data_ptr(n0_x), drjit_data_ptr(n0_y), drjit_data_ptr(n0_z),
                drjit_data_ptr(nn_x), drjit_data_ptr(nn_y), drjit_data_ptr(nn_z),
                drjit_data_ptr(wedge_n),
                drjit_data_ptr(edge_line_min), drjit_data_ptr(edge_line_max),
                drjit_data_ptr(source_pos_x), drjit_data_ptr(source_pos_y), drjit_data_ptr(source_pos_z),
                drjit_data_ptr(incident_field_re), drjit_data_ptr(incident_field_im),
                drjit_data_ptr(incident_nderiv_re), drjit_data_ptr(incident_nderiv_im),
                drjit_data_ptr(r0_re), drjit_data_ptr(r0_im),
                drjit_data_ptr(rn_re), drjit_data_ptr(rn_im),
                drjit_data_ptr(inc_vec_x_re), drjit_data_ptr(inc_vec_x_im),
                drjit_data_ptr(inc_vec_y_re), drjit_data_ptr(inc_vec_y_im),
                drjit_data_ptr(inc_vec_z_re), drjit_data_ptr(inc_vec_z_im),
                drjit_data_ptr(inc_dvec_x_re), drjit_data_ptr(inc_dvec_x_im),
                drjit_data_ptr(inc_dvec_y_re), drjit_data_ptr(inc_dvec_y_im),
                drjit_data_ptr(inc_dvec_z_re), drjit_data_ptr(inc_dvec_z_im),
                drjit_data_ptr(inc_jones_u_re), drjit_data_ptr(inc_jones_u_im),
                drjit_data_ptr(inc_jones_v_re), drjit_data_ptr(inc_jones_v_im),
                drjit_data_ptr(inc_djones_u_re), drjit_data_ptr(inc_djones_u_im),
                drjit_data_ptr(inc_djones_v_re), drjit_data_ptr(inc_djones_v_im),
                drjit_data_ptr(inc_basis_u_x), drjit_data_ptr(inc_basis_u_y), drjit_data_ptr(inc_basis_u_z),
                drjit_data_ptr(inc_basis_v_x), drjit_data_ptr(inc_basis_v_y), drjit_data_ptr(inc_basis_v_z),
                drjit_data_ptr(inc_basis_k_x), drjit_data_ptr(inc_basis_k_y), drjit_data_ptr(inc_basis_k_z),
                drjit_data_ptr(face0_op_m00_re), drjit_data_ptr(face0_op_m00_im),
                drjit_data_ptr(face0_op_m01_re), drjit_data_ptr(face0_op_m01_im),
                drjit_data_ptr(face0_op_m10_re), drjit_data_ptr(face0_op_m10_im),
                drjit_data_ptr(face0_op_m11_re), drjit_data_ptr(face0_op_m11_im),
                drjit_data_ptr(face1_op_m00_re), drjit_data_ptr(face1_op_m00_im),
                drjit_data_ptr(face1_op_m01_re), drjit_data_ptr(face1_op_m01_im),
                drjit_data_ptr(face1_op_m10_re), drjit_data_ptr(face1_op_m10_im),
                drjit_data_ptr(face1_op_m11_re), drjit_data_ptr(face1_op_m11_im),
                drjit_data_ptr(face0_eta_r), drjit_data_ptr(face0_sigma),
                drjit_data_ptr(face0_gain), drjit_data_ptr(face0_use_fresnel), drjit_data_ptr(face0_present),
                drjit_data_ptr(face1_eta_r), drjit_data_ptr(face1_sigma),
                drjit_data_ptr(face1_gain), drjit_data_ptr(face1_use_fresnel), drjit_data_ptr(face1_present),
            };

            witwin::channel::native_ext::utd_accumulate_scalar_power_forward_slots(
                state_slots.data(),
                drjit_data_ptr(rx_idx),
                drjit_data_ptr(rx_x), drjit_data_ptr(rx_y), drjit_data_ptr(rx_z),
                drjit_data_ptr_mut(coherent_re),
                drjit_data_ptr_mut(coherent_im),
                drjit_data_ptr_mut(power),
                drjit_data_ptr_mut(valid_pair_count),
                n_pairs,
                k,
                rx_pol_x,
                rx_pol_y,
                rx_pol_z,
                material
            );

        },

        "Launch the UTD scalar-power forward kernel into preallocated Dr.Jit output arrays."

    );



    // UTD backward raw launcher

    m.def(

        "utd_accumulate_backward_raw",

        [](nb::args, nb::kwargs) {
            throw std::runtime_error(
                "Legacy UTD raw backward launcher is unsupported. Finite-wedge UTD requires "
                "edge_line_min and edge_line_max, and native explicit backward is disabled."
            );
        },

        "Legacy UTD raw backward launcher. Always errors because finite-wedge bounds are required."

    );



    m.def(

        "utd_accumulate_backward_arrays",

        [](

            const Int32 &state_idx,

            const Int32 &rx_idx,

            const Int32 &ownership_code,

            const DiffFloat &edge_pos_x, const DiffFloat &edge_pos_y, const DiffFloat &edge_pos_z,

            const DiffFloat &edge_dir_x, const DiffFloat &edge_dir_y, const DiffFloat &edge_dir_z,

            const DiffFloat &n0_x, const DiffFloat &n0_y, const DiffFloat &n0_z,

            const DiffFloat &nn_x, const DiffFloat &nn_y, const DiffFloat &nn_z,

            const DiffFloat &wedge_n,

            const DiffFloat &source_pos_x, const DiffFloat &source_pos_y, const DiffFloat &source_pos_z,

            const DiffFloat &incident_field_re, const DiffFloat &incident_field_im,

            const DiffFloat &incident_nderiv_re, const DiffFloat &incident_nderiv_im,

            const DiffFloat &r0_re, const DiffFloat &r0_im,

            const DiffFloat &rn_re, const DiffFloat &rn_im,

            const DiffFloat &inc_vec_x_re, const DiffFloat &inc_vec_x_im,

            const DiffFloat &inc_vec_y_re, const DiffFloat &inc_vec_y_im,

            const DiffFloat &inc_vec_z_re, const DiffFloat &inc_vec_z_im,

            const DiffFloat &inc_dvec_x_re, const DiffFloat &inc_dvec_x_im,

            const DiffFloat &inc_dvec_y_re, const DiffFloat &inc_dvec_y_im,

            const DiffFloat &inc_dvec_z_re, const DiffFloat &inc_dvec_z_im,

            const DiffFloat &inc_jones_u_re, const DiffFloat &inc_jones_u_im,

            const DiffFloat &inc_jones_v_re, const DiffFloat &inc_jones_v_im,

            const DiffFloat &inc_djones_u_re, const DiffFloat &inc_djones_u_im,

            const DiffFloat &inc_djones_v_re, const DiffFloat &inc_djones_v_im,

            const DiffFloat &inc_basis_u_x, const DiffFloat &inc_basis_u_y, const DiffFloat &inc_basis_u_z,

            const DiffFloat &inc_basis_v_x, const DiffFloat &inc_basis_v_y, const DiffFloat &inc_basis_v_z,

            const DiffFloat &inc_basis_k_x, const DiffFloat &inc_basis_k_y, const DiffFloat &inc_basis_k_z,

            const DiffFloat &face0_op_m00_re, const DiffFloat &face0_op_m00_im,

            const DiffFloat &face0_op_m01_re, const DiffFloat &face0_op_m01_im,

            const DiffFloat &face0_op_m10_re, const DiffFloat &face0_op_m10_im,

            const DiffFloat &face0_op_m11_re, const DiffFloat &face0_op_m11_im,

            const DiffFloat &face1_op_m00_re, const DiffFloat &face1_op_m00_im,

            const DiffFloat &face1_op_m01_re, const DiffFloat &face1_op_m01_im,

            const DiffFloat &face1_op_m10_re, const DiffFloat &face1_op_m10_im,

            const DiffFloat &face1_op_m11_re, const DiffFloat &face1_op_m11_im,

            const DiffFloat &face0_eta_r, const DiffFloat &face0_sigma,

            const DiffFloat &face0_gain, const DiffFloat &face0_use_fresnel,

            const DiffFloat &face0_present,

            const DiffFloat &face1_eta_r, const DiffFloat &face1_sigma,

            const DiffFloat &face1_gain, const DiffFloat &face1_use_fresnel,

            const DiffFloat &face1_present,

            const DiffFloat &rx_x, const DiffFloat &rx_y, const DiffFloat &rx_z,

            const DiffFloat &grad_direct_re, const DiffFloat &grad_direct_im,

            const DiffFloat &grad_multi_re, const DiffFloat &grad_multi_im,

            const DiffFloat &grad_direct_vec_x_re, const DiffFloat &grad_direct_vec_x_im,

            const DiffFloat &grad_direct_vec_y_re, const DiffFloat &grad_direct_vec_y_im,

            const DiffFloat &grad_direct_vec_z_re, const DiffFloat &grad_direct_vec_z_im,

            const DiffFloat &grad_multi_vec_x_re, const DiffFloat &grad_multi_vec_x_im,

            const DiffFloat &grad_multi_vec_y_re, const DiffFloat &grad_multi_vec_y_im,

            const DiffFloat &grad_multi_vec_z_re, const DiffFloat &grad_multi_vec_z_im,

            int n_pairs,

            float k,

            witwin::channel::native_ext::MaterialParams material

        ) {

            size_t n_states = edge_pos_x.size();

            size_t n_rx = rx_x.size();



            DiffFloat grad_edge_pos_x = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_edge_pos_y = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_edge_pos_z = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_edge_dir_x = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_edge_dir_y = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_edge_dir_z = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_n0_x = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_n0_y = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_n0_z = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_nn_x = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_nn_y = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_nn_z = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_wedge_n = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_source_pos_x = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_source_pos_y = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_source_pos_z = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_field_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_field_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_nderiv_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_nderiv_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_r0_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_r0_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_rn_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_rn_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_vec_x_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_vec_x_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_vec_y_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_vec_y_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_vec_z_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_vec_z_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_dvec_x_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_dvec_x_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_dvec_y_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_dvec_y_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_dvec_z_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_inc_dvec_z_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_op_m00_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_op_m00_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_op_m01_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_op_m01_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_op_m10_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_op_m10_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_op_m11_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_op_m11_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_op_m00_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_op_m00_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_op_m01_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_op_m01_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_op_m10_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_op_m10_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_op_m11_re = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_op_m11_im = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_eta_r = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_sigma = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face0_gain = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_eta_r = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_sigma = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_face1_gain = drjit::zeros<DiffFloat>(n_states);

            DiffFloat grad_rx_x = drjit::zeros<DiffFloat>(n_rx);
            DiffFloat grad_rx_y = drjit::zeros<DiffFloat>(n_rx);

            DiffFloat grad_rx_z = drjit::zeros<DiffFloat>(n_rx);



            drjit::eval(

                state_idx, rx_idx, ownership_code,

                edge_pos_x, edge_pos_y, edge_pos_z,

                edge_dir_x, edge_dir_y, edge_dir_z,

                n0_x, n0_y, n0_z,

                nn_x, nn_y, nn_z,

                wedge_n,

                source_pos_x, source_pos_y, source_pos_z,

                incident_field_re, incident_field_im,

                incident_nderiv_re, incident_nderiv_im,

                r0_re, r0_im, rn_re, rn_im,

                inc_vec_x_re, inc_vec_x_im,

                inc_vec_y_re, inc_vec_y_im,

                inc_vec_z_re, inc_vec_z_im,

                inc_dvec_x_re, inc_dvec_x_im,

                inc_dvec_y_re, inc_dvec_y_im,

                inc_dvec_z_re, inc_dvec_z_im,

                inc_jones_u_re, inc_jones_u_im,

                inc_jones_v_re, inc_jones_v_im,

                inc_djones_u_re, inc_djones_u_im,

                inc_djones_v_re, inc_djones_v_im,

                inc_basis_u_x, inc_basis_u_y, inc_basis_u_z,

                inc_basis_v_x, inc_basis_v_y, inc_basis_v_z,

                inc_basis_k_x, inc_basis_k_y, inc_basis_k_z,

                face0_op_m00_re, face0_op_m00_im,

                face0_op_m01_re, face0_op_m01_im,

                face0_op_m10_re, face0_op_m10_im,

                face0_op_m11_re, face0_op_m11_im,

                face1_op_m00_re, face1_op_m00_im,

                face1_op_m01_re, face1_op_m01_im,

                face1_op_m10_re, face1_op_m10_im,

                face1_op_m11_re, face1_op_m11_im,

                face0_eta_r, face0_sigma, face0_gain, face0_use_fresnel, face0_present,

                face1_eta_r, face1_sigma, face1_gain, face1_use_fresnel, face1_present,

                rx_x, rx_y, rx_z,

                grad_direct_re, grad_direct_im,

                grad_multi_re, grad_multi_im,

                grad_direct_vec_x_re, grad_direct_vec_x_im,

                grad_direct_vec_y_re, grad_direct_vec_y_im,

                grad_direct_vec_z_re, grad_direct_vec_z_im,

                grad_multi_vec_x_re, grad_multi_vec_x_im,

                grad_multi_vec_y_re, grad_multi_vec_y_im,

                grad_multi_vec_z_re, grad_multi_vec_z_im,

                grad_edge_pos_x, grad_edge_pos_y, grad_edge_pos_z,

                grad_edge_dir_x, grad_edge_dir_y, grad_edge_dir_z,

                grad_n0_x, grad_n0_y, grad_n0_z,

                grad_nn_x, grad_nn_y, grad_nn_z,

                grad_wedge_n,

                grad_source_pos_x, grad_source_pos_y, grad_source_pos_z,

                grad_inc_field_re, grad_inc_field_im,

                grad_inc_nderiv_re, grad_inc_nderiv_im,

                grad_r0_re, grad_r0_im,

                grad_rn_re, grad_rn_im,

                grad_inc_vec_x_re, grad_inc_vec_x_im,

                grad_inc_vec_y_re, grad_inc_vec_y_im,

                grad_inc_vec_z_re, grad_inc_vec_z_im,

                grad_inc_dvec_x_re, grad_inc_dvec_x_im,

                grad_inc_dvec_y_re, grad_inc_dvec_y_im,

                grad_inc_dvec_z_re, grad_inc_dvec_z_im,

                grad_face0_op_m00_re, grad_face0_op_m00_im,

                grad_face0_op_m01_re, grad_face0_op_m01_im,

                grad_face0_op_m10_re, grad_face0_op_m10_im,

                grad_face0_op_m11_re, grad_face0_op_m11_im,

                grad_face1_op_m00_re, grad_face1_op_m00_im,

                grad_face1_op_m01_re, grad_face1_op_m01_im,

                grad_face1_op_m10_re, grad_face1_op_m10_im,

                grad_face1_op_m11_re, grad_face1_op_m11_im,

                grad_face0_eta_r, grad_face0_sigma, grad_face0_gain,

                grad_face1_eta_r, grad_face1_sigma, grad_face1_gain,

                grad_rx_x, grad_rx_y, grad_rx_z
            );



            witwin::channel::native_ext::utd_accumulate_backward(

                drjit_data_ptr(state_idx),

                drjit_data_ptr(rx_idx),

                drjit_data_ptr(ownership_code),

                drjit_data_ptr(edge_pos_x), drjit_data_ptr(edge_pos_y), drjit_data_ptr(edge_pos_z),

                drjit_data_ptr(edge_dir_x), drjit_data_ptr(edge_dir_y), drjit_data_ptr(edge_dir_z),

                drjit_data_ptr(n0_x), drjit_data_ptr(n0_y), drjit_data_ptr(n0_z),

                drjit_data_ptr(nn_x), drjit_data_ptr(nn_y), drjit_data_ptr(nn_z),

                drjit_data_ptr(wedge_n),

                drjit_data_ptr(source_pos_x), drjit_data_ptr(source_pos_y), drjit_data_ptr(source_pos_z),

                drjit_data_ptr(incident_field_re), drjit_data_ptr(incident_field_im),

                drjit_data_ptr(incident_nderiv_re), drjit_data_ptr(incident_nderiv_im),

                drjit_data_ptr(r0_re), drjit_data_ptr(r0_im),

                drjit_data_ptr(rn_re), drjit_data_ptr(rn_im),

                drjit_data_ptr(inc_vec_x_re), drjit_data_ptr(inc_vec_x_im),

                drjit_data_ptr(inc_vec_y_re), drjit_data_ptr(inc_vec_y_im),

                drjit_data_ptr(inc_vec_z_re), drjit_data_ptr(inc_vec_z_im),

                drjit_data_ptr(inc_dvec_x_re), drjit_data_ptr(inc_dvec_x_im),

                drjit_data_ptr(inc_dvec_y_re), drjit_data_ptr(inc_dvec_y_im),

                drjit_data_ptr(inc_dvec_z_re), drjit_data_ptr(inc_dvec_z_im),

                drjit_data_ptr(inc_jones_u_re), drjit_data_ptr(inc_jones_u_im),

                drjit_data_ptr(inc_jones_v_re), drjit_data_ptr(inc_jones_v_im),

                drjit_data_ptr(inc_djones_u_re), drjit_data_ptr(inc_djones_u_im),

                drjit_data_ptr(inc_djones_v_re), drjit_data_ptr(inc_djones_v_im),

                drjit_data_ptr(inc_basis_u_x), drjit_data_ptr(inc_basis_u_y), drjit_data_ptr(inc_basis_u_z),

                drjit_data_ptr(inc_basis_v_x), drjit_data_ptr(inc_basis_v_y), drjit_data_ptr(inc_basis_v_z),

                drjit_data_ptr(inc_basis_k_x), drjit_data_ptr(inc_basis_k_y), drjit_data_ptr(inc_basis_k_z),

                drjit_data_ptr(face0_op_m00_re), drjit_data_ptr(face0_op_m00_im),

                drjit_data_ptr(face0_op_m01_re), drjit_data_ptr(face0_op_m01_im),

                drjit_data_ptr(face0_op_m10_re), drjit_data_ptr(face0_op_m10_im),

                drjit_data_ptr(face0_op_m11_re), drjit_data_ptr(face0_op_m11_im),

                drjit_data_ptr(face1_op_m00_re), drjit_data_ptr(face1_op_m00_im),

                drjit_data_ptr(face1_op_m01_re), drjit_data_ptr(face1_op_m01_im),

                drjit_data_ptr(face1_op_m10_re), drjit_data_ptr(face1_op_m10_im),

                drjit_data_ptr(face1_op_m11_re), drjit_data_ptr(face1_op_m11_im),

                drjit_data_ptr(face0_eta_r), drjit_data_ptr(face0_sigma),

                drjit_data_ptr(face0_gain), drjit_data_ptr(face0_use_fresnel),

                drjit_data_ptr(face0_present),

                drjit_data_ptr(face1_eta_r), drjit_data_ptr(face1_sigma),

                drjit_data_ptr(face1_gain), drjit_data_ptr(face1_use_fresnel),

                drjit_data_ptr(face1_present),

                drjit_data_ptr(rx_x), drjit_data_ptr(rx_y), drjit_data_ptr(rx_z),

                drjit_data_ptr(grad_direct_re), drjit_data_ptr(grad_direct_im),

                drjit_data_ptr(grad_multi_re), drjit_data_ptr(grad_multi_im),

                drjit_data_ptr(grad_direct_vec_x_re), drjit_data_ptr(grad_direct_vec_x_im),

                drjit_data_ptr(grad_direct_vec_y_re), drjit_data_ptr(grad_direct_vec_y_im),

                drjit_data_ptr(grad_direct_vec_z_re), drjit_data_ptr(grad_direct_vec_z_im),

                drjit_data_ptr(grad_multi_vec_x_re), drjit_data_ptr(grad_multi_vec_x_im),

                drjit_data_ptr(grad_multi_vec_y_re), drjit_data_ptr(grad_multi_vec_y_im),

                drjit_data_ptr(grad_multi_vec_z_re), drjit_data_ptr(grad_multi_vec_z_im),

                drjit_data_ptr_mut(grad_edge_pos_x), drjit_data_ptr_mut(grad_edge_pos_y), drjit_data_ptr_mut(grad_edge_pos_z),

                drjit_data_ptr_mut(grad_edge_dir_x), drjit_data_ptr_mut(grad_edge_dir_y), drjit_data_ptr_mut(grad_edge_dir_z),

                drjit_data_ptr_mut(grad_n0_x), drjit_data_ptr_mut(grad_n0_y), drjit_data_ptr_mut(grad_n0_z),

                drjit_data_ptr_mut(grad_nn_x), drjit_data_ptr_mut(grad_nn_y), drjit_data_ptr_mut(grad_nn_z),

                drjit_data_ptr_mut(grad_wedge_n),

                drjit_data_ptr_mut(grad_source_pos_x), drjit_data_ptr_mut(grad_source_pos_y), drjit_data_ptr_mut(grad_source_pos_z),

                drjit_data_ptr_mut(grad_inc_field_re), drjit_data_ptr_mut(grad_inc_field_im),

                drjit_data_ptr_mut(grad_inc_nderiv_re), drjit_data_ptr_mut(grad_inc_nderiv_im),

                drjit_data_ptr_mut(grad_r0_re), drjit_data_ptr_mut(grad_r0_im),

                drjit_data_ptr_mut(grad_rn_re), drjit_data_ptr_mut(grad_rn_im),

                drjit_data_ptr_mut(grad_inc_vec_x_re), drjit_data_ptr_mut(grad_inc_vec_x_im),

                drjit_data_ptr_mut(grad_inc_vec_y_re), drjit_data_ptr_mut(grad_inc_vec_y_im),

                drjit_data_ptr_mut(grad_inc_vec_z_re), drjit_data_ptr_mut(grad_inc_vec_z_im),

                drjit_data_ptr_mut(grad_inc_dvec_x_re), drjit_data_ptr_mut(grad_inc_dvec_x_im),

                drjit_data_ptr_mut(grad_inc_dvec_y_re), drjit_data_ptr_mut(grad_inc_dvec_y_im),

                drjit_data_ptr_mut(grad_inc_dvec_z_re), drjit_data_ptr_mut(grad_inc_dvec_z_im),

                drjit_data_ptr_mut(grad_face0_op_m00_re), drjit_data_ptr_mut(grad_face0_op_m00_im),

                drjit_data_ptr_mut(grad_face0_op_m01_re), drjit_data_ptr_mut(grad_face0_op_m01_im),

                drjit_data_ptr_mut(grad_face0_op_m10_re), drjit_data_ptr_mut(grad_face0_op_m10_im),

                drjit_data_ptr_mut(grad_face0_op_m11_re), drjit_data_ptr_mut(grad_face0_op_m11_im),

                drjit_data_ptr_mut(grad_face1_op_m00_re), drjit_data_ptr_mut(grad_face1_op_m00_im),

                drjit_data_ptr_mut(grad_face1_op_m01_re), drjit_data_ptr_mut(grad_face1_op_m01_im),

                drjit_data_ptr_mut(grad_face1_op_m10_re), drjit_data_ptr_mut(grad_face1_op_m10_im),

                drjit_data_ptr_mut(grad_face1_op_m11_re), drjit_data_ptr_mut(grad_face1_op_m11_im),

                drjit_data_ptr_mut(grad_face0_eta_r), drjit_data_ptr_mut(grad_face0_sigma), drjit_data_ptr_mut(grad_face0_gain),

                drjit_data_ptr_mut(grad_face1_eta_r), drjit_data_ptr_mut(grad_face1_sigma), drjit_data_ptr_mut(grad_face1_gain),

                drjit_data_ptr_mut(grad_rx_x), drjit_data_ptr_mut(grad_rx_y), drjit_data_ptr_mut(grad_rx_z),
                n_pairs,

                k,

                material

            );



            return nb::make_tuple(

                grad_edge_pos_x, grad_edge_pos_y, grad_edge_pos_z,

                grad_edge_dir_x, grad_edge_dir_y, grad_edge_dir_z,

                grad_n0_x, grad_n0_y, grad_n0_z,

                grad_nn_x, grad_nn_y, grad_nn_z,

                grad_wedge_n,

                grad_source_pos_x, grad_source_pos_y, grad_source_pos_z,

                grad_inc_field_re, grad_inc_field_im,

                grad_inc_nderiv_re, grad_inc_nderiv_im,

                grad_r0_re, grad_r0_im,

                grad_rn_re, grad_rn_im,

                grad_inc_vec_x_re, grad_inc_vec_x_im,

                grad_inc_vec_y_re, grad_inc_vec_y_im,

                grad_inc_vec_z_re, grad_inc_vec_z_im,

                grad_inc_dvec_x_re, grad_inc_dvec_x_im,

                grad_inc_dvec_y_re, grad_inc_dvec_y_im,

                grad_inc_dvec_z_re, grad_inc_dvec_z_im,

                grad_face0_op_m00_re, grad_face0_op_m00_im,

                grad_face0_op_m01_re, grad_face0_op_m01_im,

                grad_face0_op_m10_re, grad_face0_op_m10_im,

                grad_face0_op_m11_re, grad_face0_op_m11_im,

                grad_face1_op_m00_re, grad_face1_op_m00_im,

                grad_face1_op_m01_re, grad_face1_op_m01_im,

                grad_face1_op_m10_re, grad_face1_op_m10_im,

                grad_face1_op_m11_re, grad_face1_op_m11_im,

                grad_face0_eta_r, grad_face0_sigma, grad_face0_gain,

                grad_face1_eta_r, grad_face1_sigma, grad_face1_gain,

                grad_rx_x, grad_rx_y, grad_rx_z

            );

        },

        "Launch the UTD backward mega-kernel with Dr.Jit arrays and return gradients."

    );



    // UTD JVP raw launcher

    m.def(

        "utd_accumulate_jvp_raw",

        [](nb::args, nb::kwargs) {
            throw std::runtime_error(
                "Legacy UTD raw JVP launcher is unsupported. Finite-wedge UTD requires "
                "edge_line_min and edge_line_max."
            );
        },

        "Legacy UTD raw JVP launcher. Always errors because finite-wedge bounds are required."

    );



    // UTD accumulation with C++-side Dr.Jit custom AD support

    m.def(

        "utd_accumulate",

        [](nb::args, nb::kwargs) {
            throw std::runtime_error(
                "Native UTD custom-op accumulation without finite-wedge edge_line_min and "
                "edge_line_max is unsupported. Use the finite-wedge tiled/native accumulation "
                "paths instead."
            );
        },

        "Legacy direct UTD custom-op entrypoint. Always errors because finite-wedge bounds are required."

    );

}

