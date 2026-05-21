#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <packed_state/packed_state.h>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

struct PackedStateCoreIn {
    const std::uint32_t* edge_idx;
    const float* edge_pos_x;
    const float* edge_pos_y;
    const float* edge_pos_z;
    const float* edge_dir_x;
    const float* edge_dir_y;
    const float* edge_dir_z;
    const float* n0_x;
    const float* n0_y;
    const float* n0_z;
    const float* nn_x;
    const float* nn_y;
    const float* nn_z;
    const float* wedge_n;
    const std::int32_t* adj_face0;
    const std::int32_t* adj_face1;
    const float* source_pos_x;
    const float* source_pos_y;
    const float* source_pos_z;
    const float* inc_field_re;
    const float* inc_field_im;
    const float* inc_nderiv_re;
    const float* inc_nderiv_im;
    const float* inc_jones_u_re;
    const float* inc_jones_u_im;
    const float* inc_jones_v_re;
    const float* inc_jones_v_im;
    const float* inc_djones_u_re;
    const float* inc_djones_u_im;
    const float* inc_djones_v_re;
    const float* inc_djones_v_im;
    const float* r0_re;
    const float* r0_im;
    const float* rn_re;
    const float* rn_im;
    const float* inc_basis_u_x;
    const float* inc_basis_u_y;
    const float* inc_basis_u_z;
    const float* inc_basis_v_x;
    const float* inc_basis_v_y;
    const float* inc_basis_v_z;
    const float* inc_basis_k_x;
    const float* inc_basis_k_y;
    const float* inc_basis_k_z;
    const float* f0_op_m00_re;
    const float* f0_op_m00_im;
    const float* f0_op_m01_re;
    const float* f0_op_m01_im;
    const float* f0_op_m10_re;
    const float* f0_op_m10_im;
    const float* f0_op_m11_re;
    const float* f0_op_m11_im;
    const float* f1_op_m00_re;
    const float* f1_op_m00_im;
    const float* f1_op_m01_re;
    const float* f1_op_m01_im;
    const float* f1_op_m10_re;
    const float* f1_op_m10_im;
    const float* f1_op_m11_re;
    const float* f1_op_m11_im;
    const float* f0_eta_r;
    const float* f0_mu_r;
    const float* f0_sigma;
    const float* f0_gain;
    const std::uint8_t* f0_use_fresnel;
    const float* f1_eta_r;
    const float* f1_mu_r;
    const float* f1_sigma;
    const float* f1_gain;
    const std::uint8_t* f1_use_fresnel;
    const std::uint32_t* prefix_refl_depth;
    const std::uint32_t* inter_refl_depth;
    const std::uint32_t* suffix_refl_depth;
    const std::uint32_t* order;
};

struct PackedStateCoreOut {
    std::uint32_t* edge_idx;
    float* edge_pos_x;
    float* edge_pos_y;
    float* edge_pos_z;
    float* edge_dir_x;
    float* edge_dir_y;
    float* edge_dir_z;
    float* n0_x;
    float* n0_y;
    float* n0_z;
    float* nn_x;
    float* nn_y;
    float* nn_z;
    float* wedge_n;
    std::int32_t* adj_face0;
    std::int32_t* adj_face1;
    float* source_pos_x;
    float* source_pos_y;
    float* source_pos_z;
    float* inc_field_re;
    float* inc_field_im;
    float* inc_nderiv_re;
    float* inc_nderiv_im;
    float* inc_jones_u_re;
    float* inc_jones_u_im;
    float* inc_jones_v_re;
    float* inc_jones_v_im;
    float* inc_djones_u_re;
    float* inc_djones_u_im;
    float* inc_djones_v_re;
    float* inc_djones_v_im;
    float* r0_re;
    float* r0_im;
    float* rn_re;
    float* rn_im;
    float* inc_basis_u_x;
    float* inc_basis_u_y;
    float* inc_basis_u_z;
    float* inc_basis_v_x;
    float* inc_basis_v_y;
    float* inc_basis_v_z;
    float* inc_basis_k_x;
    float* inc_basis_k_y;
    float* inc_basis_k_z;
    float* f0_op_m00_re;
    float* f0_op_m00_im;
    float* f0_op_m01_re;
    float* f0_op_m01_im;
    float* f0_op_m10_re;
    float* f0_op_m10_im;
    float* f0_op_m11_re;
    float* f0_op_m11_im;
    float* f1_op_m00_re;
    float* f1_op_m00_im;
    float* f1_op_m01_re;
    float* f1_op_m01_im;
    float* f1_op_m10_re;
    float* f1_op_m10_im;
    float* f1_op_m11_re;
    float* f1_op_m11_im;
    float* f0_eta_r;
    float* f0_mu_r;
    float* f0_sigma;
    float* f0_gain;
    float* f0_use_fresnel;
    float* f1_eta_r;
    float* f1_mu_r;
    float* f1_sigma;
    float* f1_gain;
    float* f1_use_fresnel;
    std::uint32_t* prefix_refl_depth;
    std::uint32_t* inter_refl_depth;
    std::uint32_t* suffix_refl_depth;
    std::uint32_t* order;
};

template <typename T>
__host__ __device__ T* ptr_mut(std::uintptr_t value) {
    return reinterpret_cast<T*>(value);
}

template <typename T>
__host__ __device__ const T* ptr(std::uintptr_t value) {
    return reinterpret_cast<const T*>(value);
}

__device__ __forceinline__ float bool_to_float(const std::uint8_t* value, int idx) {
    return value[idx] != 0 ? 1.0f : 0.0f;
}

struct DiffractionPathSlotKernelInputs {
    const std::int32_t* prefix_depth;
    const std::int32_t* order;
    const std::int32_t* const* path_edge_slots;
    const std::int32_t* const* inserted_depth_slots;
    const float* first_interaction_pos_x;
    const float* first_interaction_pos_y;
    const float* first_interaction_pos_z;
    const float* edge_pos_x;
    const float* edge_pos_y;
    const float* edge_pos_z;
    const float* edge_n0_x;
    const float* edge_n0_y;
    const float* edge_n0_z;
    const std::int32_t* edge_object_idx;
    int history_size;
    int n_edges;
};

struct DiffractionPathSlotKernelOutputs {
    std::int32_t** type_slots;
    float** vertex_x_slots;
    float** vertex_y_slots;
    float** vertex_z_slots;
    float** normal_x_slots;
    float** normal_y_slots;
    float** normal_z_slots;
    std::int32_t** object_slots;
};

PackedStateCoreIn make_core_in(const std::uintptr_t* p) {
    PackedStateCoreIn in{};
    in.edge_idx          = ptr<const std::uint32_t>(p[PSP_EDGE_IDX]);
    in.edge_pos_x        = ptr<const float>(p[PSP_EDGE_POS_X]);
    in.edge_pos_y        = ptr<const float>(p[PSP_EDGE_POS_Y]);
    in.edge_pos_z        = ptr<const float>(p[PSP_EDGE_POS_Z]);
    in.edge_dir_x        = ptr<const float>(p[PSP_EDGE_DIR_X]);
    in.edge_dir_y        = ptr<const float>(p[PSP_EDGE_DIR_Y]);
    in.edge_dir_z        = ptr<const float>(p[PSP_EDGE_DIR_Z]);
    in.n0_x              = ptr<const float>(p[PSP_N0_X]);
    in.n0_y              = ptr<const float>(p[PSP_N0_Y]);
    in.n0_z              = ptr<const float>(p[PSP_N0_Z]);
    in.nn_x              = ptr<const float>(p[PSP_NN_X]);
    in.nn_y              = ptr<const float>(p[PSP_NN_Y]);
    in.nn_z              = ptr<const float>(p[PSP_NN_Z]);
    in.wedge_n           = ptr<const float>(p[PSP_WEDGE_N]);
    in.adj_face0         = ptr<const std::int32_t>(p[PSP_ADJ_FACE0]);
    in.adj_face1         = ptr<const std::int32_t>(p[PSP_ADJ_FACE1]);
    in.source_pos_x      = ptr<const float>(p[PSP_SOURCE_POS_X]);
    in.source_pos_y      = ptr<const float>(p[PSP_SOURCE_POS_Y]);
    in.source_pos_z      = ptr<const float>(p[PSP_SOURCE_POS_Z]);
    in.inc_field_re      = ptr<const float>(p[PSP_INC_FIELD_RE]);
    in.inc_field_im      = ptr<const float>(p[PSP_INC_FIELD_IM]);
    in.inc_nderiv_re     = ptr<const float>(p[PSP_INC_NDERIV_RE]);
    in.inc_nderiv_im     = ptr<const float>(p[PSP_INC_NDERIV_IM]);
    in.inc_jones_u_re    = ptr<const float>(p[PSP_INC_JONES_U_RE]);
    in.inc_jones_u_im    = ptr<const float>(p[PSP_INC_JONES_U_IM]);
    in.inc_jones_v_re    = ptr<const float>(p[PSP_INC_JONES_V_RE]);
    in.inc_jones_v_im    = ptr<const float>(p[PSP_INC_JONES_V_IM]);
    in.inc_djones_u_re   = ptr<const float>(p[PSP_INC_DJONES_U_RE]);
    in.inc_djones_u_im   = ptr<const float>(p[PSP_INC_DJONES_U_IM]);
    in.inc_djones_v_re   = ptr<const float>(p[PSP_INC_DJONES_V_RE]);
    in.inc_djones_v_im   = ptr<const float>(p[PSP_INC_DJONES_V_IM]);
    in.r0_re             = ptr<const float>(p[PSP_R0_RE]);
    in.r0_im             = ptr<const float>(p[PSP_R0_IM]);
    in.rn_re             = ptr<const float>(p[PSP_RN_RE]);
    in.rn_im             = ptr<const float>(p[PSP_RN_IM]);
    in.inc_basis_u_x     = ptr<const float>(p[PSP_INC_BASIS_U_X]);
    in.inc_basis_u_y     = ptr<const float>(p[PSP_INC_BASIS_U_Y]);
    in.inc_basis_u_z     = ptr<const float>(p[PSP_INC_BASIS_U_Z]);
    in.inc_basis_v_x     = ptr<const float>(p[PSP_INC_BASIS_V_X]);
    in.inc_basis_v_y     = ptr<const float>(p[PSP_INC_BASIS_V_Y]);
    in.inc_basis_v_z     = ptr<const float>(p[PSP_INC_BASIS_V_Z]);
    in.inc_basis_k_x     = ptr<const float>(p[PSP_INC_BASIS_K_X]);
    in.inc_basis_k_y     = ptr<const float>(p[PSP_INC_BASIS_K_Y]);
    in.inc_basis_k_z     = ptr<const float>(p[PSP_INC_BASIS_K_Z]);
    in.f0_op_m00_re      = ptr<const float>(p[PSP_F0_OP_M00_RE]);
    in.f0_op_m00_im      = ptr<const float>(p[PSP_F0_OP_M00_IM]);
    in.f0_op_m01_re      = ptr<const float>(p[PSP_F0_OP_M01_RE]);
    in.f0_op_m01_im      = ptr<const float>(p[PSP_F0_OP_M01_IM]);
    in.f0_op_m10_re      = ptr<const float>(p[PSP_F0_OP_M10_RE]);
    in.f0_op_m10_im      = ptr<const float>(p[PSP_F0_OP_M10_IM]);
    in.f0_op_m11_re      = ptr<const float>(p[PSP_F0_OP_M11_RE]);
    in.f0_op_m11_im      = ptr<const float>(p[PSP_F0_OP_M11_IM]);
    in.f1_op_m00_re      = ptr<const float>(p[PSP_F1_OP_M00_RE]);
    in.f1_op_m00_im      = ptr<const float>(p[PSP_F1_OP_M00_IM]);
    in.f1_op_m01_re      = ptr<const float>(p[PSP_F1_OP_M01_RE]);
    in.f1_op_m01_im      = ptr<const float>(p[PSP_F1_OP_M01_IM]);
    in.f1_op_m10_re      = ptr<const float>(p[PSP_F1_OP_M10_RE]);
    in.f1_op_m10_im      = ptr<const float>(p[PSP_F1_OP_M10_IM]);
    in.f1_op_m11_re      = ptr<const float>(p[PSP_F1_OP_M11_RE]);
    in.f1_op_m11_im      = ptr<const float>(p[PSP_F1_OP_M11_IM]);
    in.f0_eta_r          = ptr<const float>(p[PSP_F0_ETA_R]);
    in.f0_mu_r           = ptr<const float>(p[PSP_F0_MU_R]);
    in.f0_sigma          = ptr<const float>(p[PSP_F0_SIGMA]);
    in.f0_gain           = ptr<const float>(p[PSP_F0_GAIN]);
    in.f0_use_fresnel    = ptr<const std::uint8_t>(p[PSP_F0_USE_FRESNEL]);
    in.f1_eta_r          = ptr<const float>(p[PSP_F1_ETA_R]);
    in.f1_mu_r           = ptr<const float>(p[PSP_F1_MU_R]);
    in.f1_sigma          = ptr<const float>(p[PSP_F1_SIGMA]);
    in.f1_gain           = ptr<const float>(p[PSP_F1_GAIN]);
    in.f1_use_fresnel    = ptr<const std::uint8_t>(p[PSP_F1_USE_FRESNEL]);
    in.prefix_refl_depth = ptr<const std::uint32_t>(p[PSP_PREFIX_REFL_DEPTH]);
    in.inter_refl_depth  = ptr<const std::uint32_t>(p[PSP_INTER_REFL_DEPTH]);
    in.suffix_refl_depth = ptr<const std::uint32_t>(p[PSP_SUFFIX_REFL_DEPTH]);
    in.order             = ptr<const std::uint32_t>(p[PSP_ORDER]);
    return in;
}

PackedStateCoreOut make_core_out(const std::uintptr_t* p) {
    PackedStateCoreOut out{};
    out.edge_idx          = ptr_mut<std::uint32_t>(p[PSP_EDGE_IDX]);
    out.edge_pos_x        = ptr_mut<float>(p[PSP_EDGE_POS_X]);
    out.edge_pos_y        = ptr_mut<float>(p[PSP_EDGE_POS_Y]);
    out.edge_pos_z        = ptr_mut<float>(p[PSP_EDGE_POS_Z]);
    out.edge_dir_x        = ptr_mut<float>(p[PSP_EDGE_DIR_X]);
    out.edge_dir_y        = ptr_mut<float>(p[PSP_EDGE_DIR_Y]);
    out.edge_dir_z        = ptr_mut<float>(p[PSP_EDGE_DIR_Z]);
    out.n0_x              = ptr_mut<float>(p[PSP_N0_X]);
    out.n0_y              = ptr_mut<float>(p[PSP_N0_Y]);
    out.n0_z              = ptr_mut<float>(p[PSP_N0_Z]);
    out.nn_x              = ptr_mut<float>(p[PSP_NN_X]);
    out.nn_y              = ptr_mut<float>(p[PSP_NN_Y]);
    out.nn_z              = ptr_mut<float>(p[PSP_NN_Z]);
    out.wedge_n           = ptr_mut<float>(p[PSP_WEDGE_N]);
    out.adj_face0         = ptr_mut<std::int32_t>(p[PSP_ADJ_FACE0]);
    out.adj_face1         = ptr_mut<std::int32_t>(p[PSP_ADJ_FACE1]);
    out.source_pos_x      = ptr_mut<float>(p[PSP_SOURCE_POS_X]);
    out.source_pos_y      = ptr_mut<float>(p[PSP_SOURCE_POS_Y]);
    out.source_pos_z      = ptr_mut<float>(p[PSP_SOURCE_POS_Z]);
    out.inc_field_re      = ptr_mut<float>(p[PSP_INC_FIELD_RE]);
    out.inc_field_im      = ptr_mut<float>(p[PSP_INC_FIELD_IM]);
    out.inc_nderiv_re     = ptr_mut<float>(p[PSP_INC_NDERIV_RE]);
    out.inc_nderiv_im     = ptr_mut<float>(p[PSP_INC_NDERIV_IM]);
    out.inc_jones_u_re    = ptr_mut<float>(p[PSP_INC_JONES_U_RE]);
    out.inc_jones_u_im    = ptr_mut<float>(p[PSP_INC_JONES_U_IM]);
    out.inc_jones_v_re    = ptr_mut<float>(p[PSP_INC_JONES_V_RE]);
    out.inc_jones_v_im    = ptr_mut<float>(p[PSP_INC_JONES_V_IM]);
    out.inc_djones_u_re   = ptr_mut<float>(p[PSP_INC_DJONES_U_RE]);
    out.inc_djones_u_im   = ptr_mut<float>(p[PSP_INC_DJONES_U_IM]);
    out.inc_djones_v_re   = ptr_mut<float>(p[PSP_INC_DJONES_V_RE]);
    out.inc_djones_v_im   = ptr_mut<float>(p[PSP_INC_DJONES_V_IM]);
    out.r0_re             = ptr_mut<float>(p[PSP_R0_RE]);
    out.r0_im             = ptr_mut<float>(p[PSP_R0_IM]);
    out.rn_re             = ptr_mut<float>(p[PSP_RN_RE]);
    out.rn_im             = ptr_mut<float>(p[PSP_RN_IM]);
    out.inc_basis_u_x     = ptr_mut<float>(p[PSP_INC_BASIS_U_X]);
    out.inc_basis_u_y     = ptr_mut<float>(p[PSP_INC_BASIS_U_Y]);
    out.inc_basis_u_z     = ptr_mut<float>(p[PSP_INC_BASIS_U_Z]);
    out.inc_basis_v_x     = ptr_mut<float>(p[PSP_INC_BASIS_V_X]);
    out.inc_basis_v_y     = ptr_mut<float>(p[PSP_INC_BASIS_V_Y]);
    out.inc_basis_v_z     = ptr_mut<float>(p[PSP_INC_BASIS_V_Z]);
    out.inc_basis_k_x     = ptr_mut<float>(p[PSP_INC_BASIS_K_X]);
    out.inc_basis_k_y     = ptr_mut<float>(p[PSP_INC_BASIS_K_Y]);
    out.inc_basis_k_z     = ptr_mut<float>(p[PSP_INC_BASIS_K_Z]);
    out.f0_op_m00_re      = ptr_mut<float>(p[PSP_F0_OP_M00_RE]);
    out.f0_op_m00_im      = ptr_mut<float>(p[PSP_F0_OP_M00_IM]);
    out.f0_op_m01_re      = ptr_mut<float>(p[PSP_F0_OP_M01_RE]);
    out.f0_op_m01_im      = ptr_mut<float>(p[PSP_F0_OP_M01_IM]);
    out.f0_op_m10_re      = ptr_mut<float>(p[PSP_F0_OP_M10_RE]);
    out.f0_op_m10_im      = ptr_mut<float>(p[PSP_F0_OP_M10_IM]);
    out.f0_op_m11_re      = ptr_mut<float>(p[PSP_F0_OP_M11_RE]);
    out.f0_op_m11_im      = ptr_mut<float>(p[PSP_F0_OP_M11_IM]);
    out.f1_op_m00_re      = ptr_mut<float>(p[PSP_F1_OP_M00_RE]);
    out.f1_op_m00_im      = ptr_mut<float>(p[PSP_F1_OP_M00_IM]);
    out.f1_op_m01_re      = ptr_mut<float>(p[PSP_F1_OP_M01_RE]);
    out.f1_op_m01_im      = ptr_mut<float>(p[PSP_F1_OP_M01_IM]);
    out.f1_op_m10_re      = ptr_mut<float>(p[PSP_F1_OP_M10_RE]);
    out.f1_op_m10_im      = ptr_mut<float>(p[PSP_F1_OP_M10_IM]);
    out.f1_op_m11_re      = ptr_mut<float>(p[PSP_F1_OP_M11_RE]);
    out.f1_op_m11_im      = ptr_mut<float>(p[PSP_F1_OP_M11_IM]);
    out.f0_eta_r          = ptr_mut<float>(p[PSP_F0_ETA_R]);
    out.f0_mu_r           = ptr_mut<float>(p[PSP_F0_MU_R]);
    out.f0_sigma          = ptr_mut<float>(p[PSP_F0_SIGMA]);
    out.f0_gain           = ptr_mut<float>(p[PSP_F0_GAIN]);
    out.f0_use_fresnel    = ptr_mut<float>(p[PSP_F0_USE_FRESNEL]);
    out.f1_eta_r          = ptr_mut<float>(p[PSP_F1_ETA_R]);
    out.f1_mu_r           = ptr_mut<float>(p[PSP_F1_MU_R]);
    out.f1_sigma          = ptr_mut<float>(p[PSP_F1_SIGMA]);
    out.f1_gain           = ptr_mut<float>(p[PSP_F1_GAIN]);
    out.f1_use_fresnel    = ptr_mut<float>(p[PSP_F1_USE_FRESNEL]);
    out.prefix_refl_depth = ptr_mut<std::uint32_t>(p[PSP_PREFIX_REFL_DEPTH]);
    out.inter_refl_depth  = ptr_mut<std::uint32_t>(p[PSP_INTER_REFL_DEPTH]);
    out.suffix_refl_depth = ptr_mut<std::uint32_t>(p[PSP_SUFFIX_REFL_DEPTH]);
    out.order             = ptr_mut<std::uint32_t>(p[PSP_ORDER]);
    return out;
}

__global__ void pack_state_arrays_kernel(
    PackedStateCoreIn in,
    float* dst,
    int n_states,
    int stride
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_states) {
        return;
    }

    float* row = dst + tid * stride;

    row[PF_EDGE_IDX]          = __uint_as_float(in.edge_idx[tid]);
    row[PF_EDGE_POS_X]        = in.edge_pos_x[tid];
    row[PF_EDGE_POS_Y]        = in.edge_pos_y[tid];
    row[PF_EDGE_POS_Z]        = in.edge_pos_z[tid];
    row[PF_EDGE_DIR_X]        = in.edge_dir_x[tid];
    row[PF_EDGE_DIR_Y]        = in.edge_dir_y[tid];
    row[PF_EDGE_DIR_Z]        = in.edge_dir_z[tid];
    row[PF_N0_X]              = in.n0_x[tid];
    row[PF_N0_Y]              = in.n0_y[tid];
    row[PF_N0_Z]              = in.n0_z[tid];
    row[PF_NN_X]              = in.nn_x[tid];
    row[PF_NN_Y]              = in.nn_y[tid];
    row[PF_NN_Z]              = in.nn_z[tid];
    row[PF_WEDGE_N]           = in.wedge_n[tid];
    row[PF_ADJ_FACE0]         = __int_as_float(in.adj_face0[tid]);
    row[PF_ADJ_FACE1]         = __int_as_float(in.adj_face1[tid]);
    row[PF_SOURCE_POS_X]      = in.source_pos_x[tid];
    row[PF_SOURCE_POS_Y]      = in.source_pos_y[tid];
    row[PF_SOURCE_POS_Z]      = in.source_pos_z[tid];
    row[PF_INC_FIELD_RE]      = in.inc_field_re[tid];
    row[PF_INC_FIELD_IM]      = in.inc_field_im[tid];
    row[PF_INC_NDERIV_RE]     = in.inc_nderiv_re[tid];
    row[PF_INC_NDERIV_IM]     = in.inc_nderiv_im[tid];
    row[PF_INC_JONES_U_RE]    = in.inc_jones_u_re[tid];
    row[PF_INC_JONES_U_IM]    = in.inc_jones_u_im[tid];
    row[PF_INC_JONES_V_RE]    = in.inc_jones_v_re[tid];
    row[PF_INC_JONES_V_IM]    = in.inc_jones_v_im[tid];
    row[PF_INC_DJONES_U_RE]   = in.inc_djones_u_re[tid];
    row[PF_INC_DJONES_U_IM]   = in.inc_djones_u_im[tid];
    row[PF_INC_DJONES_V_RE]   = in.inc_djones_v_re[tid];
    row[PF_INC_DJONES_V_IM]   = in.inc_djones_v_im[tid];
    row[PF_R0_RE]             = in.r0_re[tid];
    row[PF_R0_IM]             = in.r0_im[tid];
    row[PF_RN_RE]             = in.rn_re[tid];
    row[PF_RN_IM]             = in.rn_im[tid];
    row[PF_INC_BASIS_U_X]     = in.inc_basis_u_x[tid];
    row[PF_INC_BASIS_U_Y]     = in.inc_basis_u_y[tid];
    row[PF_INC_BASIS_U_Z]     = in.inc_basis_u_z[tid];
    row[PF_INC_BASIS_V_X]     = in.inc_basis_v_x[tid];
    row[PF_INC_BASIS_V_Y]     = in.inc_basis_v_y[tid];
    row[PF_INC_BASIS_V_Z]     = in.inc_basis_v_z[tid];
    row[PF_INC_BASIS_K_X]     = in.inc_basis_k_x[tid];
    row[PF_INC_BASIS_K_Y]     = in.inc_basis_k_y[tid];
    row[PF_INC_BASIS_K_Z]     = in.inc_basis_k_z[tid];
    row[PF_F0_OP_M00_RE]      = in.f0_op_m00_re[tid];
    row[PF_F0_OP_M00_IM]      = in.f0_op_m00_im[tid];
    row[PF_F0_OP_M01_RE]      = in.f0_op_m01_re[tid];
    row[PF_F0_OP_M01_IM]      = in.f0_op_m01_im[tid];
    row[PF_F0_OP_M10_RE]      = in.f0_op_m10_re[tid];
    row[PF_F0_OP_M10_IM]      = in.f0_op_m10_im[tid];
    row[PF_F0_OP_M11_RE]      = in.f0_op_m11_re[tid];
    row[PF_F0_OP_M11_IM]      = in.f0_op_m11_im[tid];
    row[PF_F1_OP_M00_RE]      = in.f1_op_m00_re[tid];
    row[PF_F1_OP_M00_IM]      = in.f1_op_m00_im[tid];
    row[PF_F1_OP_M01_RE]      = in.f1_op_m01_re[tid];
    row[PF_F1_OP_M01_IM]      = in.f1_op_m01_im[tid];
    row[PF_F1_OP_M10_RE]      = in.f1_op_m10_re[tid];
    row[PF_F1_OP_M10_IM]      = in.f1_op_m10_im[tid];
    row[PF_F1_OP_M11_RE]      = in.f1_op_m11_re[tid];
    row[PF_F1_OP_M11_IM]      = in.f1_op_m11_im[tid];
    row[PF_F0_ETA_R]          = in.f0_eta_r[tid];
    row[PF_F0_MU_R]           = in.f0_mu_r[tid];
    row[PF_F0_SIGMA]          = in.f0_sigma[tid];
    row[PF_F0_GAIN]           = in.f0_gain[tid];
    row[PF_F0_USE_FRESNEL]    = bool_to_float(in.f0_use_fresnel, tid);
    row[PF_F1_ETA_R]          = in.f1_eta_r[tid];
    row[PF_F1_MU_R]           = in.f1_mu_r[tid];
    row[PF_F1_SIGMA]          = in.f1_sigma[tid];
    row[PF_F1_GAIN]           = in.f1_gain[tid];
    row[PF_F1_USE_FRESNEL]    = bool_to_float(in.f1_use_fresnel, tid);
    row[PF_PREFIX_REFL_DEPTH] = __uint_as_float(in.prefix_refl_depth[tid]);
    row[PF_INTER_REFL_DEPTH]  = __uint_as_float(in.inter_refl_depth[tid]);
    row[PF_SUFFIX_REFL_DEPTH] = __uint_as_float(in.suffix_refl_depth[tid]);
    row[PF_ORDER]             = __uint_as_float(in.order[tid]);

    for (int i = PACKED_CORE_FLOATS; i < stride; ++i) {
        row[i] = 0.0f;
    }
}

__global__ void unpack_state_arrays_kernel(
    const float* src,
    PackedStateCoreOut out,
    int n_states,
    int stride
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_states) {
        return;
    }

    const float* row = src + tid * stride;

    out.edge_idx[tid]          = __float_as_uint(row[PF_EDGE_IDX]);
    out.edge_pos_x[tid]        = row[PF_EDGE_POS_X];
    out.edge_pos_y[tid]        = row[PF_EDGE_POS_Y];
    out.edge_pos_z[tid]        = row[PF_EDGE_POS_Z];
    out.edge_dir_x[tid]        = row[PF_EDGE_DIR_X];
    out.edge_dir_y[tid]        = row[PF_EDGE_DIR_Y];
    out.edge_dir_z[tid]        = row[PF_EDGE_DIR_Z];
    out.n0_x[tid]              = row[PF_N0_X];
    out.n0_y[tid]              = row[PF_N0_Y];
    out.n0_z[tid]              = row[PF_N0_Z];
    out.nn_x[tid]              = row[PF_NN_X];
    out.nn_y[tid]              = row[PF_NN_Y];
    out.nn_z[tid]              = row[PF_NN_Z];
    out.wedge_n[tid]           = row[PF_WEDGE_N];
    out.adj_face0[tid]         = __float_as_int(row[PF_ADJ_FACE0]);
    out.adj_face1[tid]         = __float_as_int(row[PF_ADJ_FACE1]);
    out.source_pos_x[tid]      = row[PF_SOURCE_POS_X];
    out.source_pos_y[tid]      = row[PF_SOURCE_POS_Y];
    out.source_pos_z[tid]      = row[PF_SOURCE_POS_Z];
    out.inc_field_re[tid]      = row[PF_INC_FIELD_RE];
    out.inc_field_im[tid]      = row[PF_INC_FIELD_IM];
    out.inc_nderiv_re[tid]     = row[PF_INC_NDERIV_RE];
    out.inc_nderiv_im[tid]     = row[PF_INC_NDERIV_IM];
    out.inc_jones_u_re[tid]    = row[PF_INC_JONES_U_RE];
    out.inc_jones_u_im[tid]    = row[PF_INC_JONES_U_IM];
    out.inc_jones_v_re[tid]    = row[PF_INC_JONES_V_RE];
    out.inc_jones_v_im[tid]    = row[PF_INC_JONES_V_IM];
    out.inc_djones_u_re[tid]   = row[PF_INC_DJONES_U_RE];
    out.inc_djones_u_im[tid]   = row[PF_INC_DJONES_U_IM];
    out.inc_djones_v_re[tid]   = row[PF_INC_DJONES_V_RE];
    out.inc_djones_v_im[tid]   = row[PF_INC_DJONES_V_IM];
    out.r0_re[tid]             = row[PF_R0_RE];
    out.r0_im[tid]             = row[PF_R0_IM];
    out.rn_re[tid]             = row[PF_RN_RE];
    out.rn_im[tid]             = row[PF_RN_IM];
    out.inc_basis_u_x[tid]     = row[PF_INC_BASIS_U_X];
    out.inc_basis_u_y[tid]     = row[PF_INC_BASIS_U_Y];
    out.inc_basis_u_z[tid]     = row[PF_INC_BASIS_U_Z];
    out.inc_basis_v_x[tid]     = row[PF_INC_BASIS_V_X];
    out.inc_basis_v_y[tid]     = row[PF_INC_BASIS_V_Y];
    out.inc_basis_v_z[tid]     = row[PF_INC_BASIS_V_Z];
    out.inc_basis_k_x[tid]     = row[PF_INC_BASIS_K_X];
    out.inc_basis_k_y[tid]     = row[PF_INC_BASIS_K_Y];
    out.inc_basis_k_z[tid]     = row[PF_INC_BASIS_K_Z];
    out.f0_op_m00_re[tid]      = row[PF_F0_OP_M00_RE];
    out.f0_op_m00_im[tid]      = row[PF_F0_OP_M00_IM];
    out.f0_op_m01_re[tid]      = row[PF_F0_OP_M01_RE];
    out.f0_op_m01_im[tid]      = row[PF_F0_OP_M01_IM];
    out.f0_op_m10_re[tid]      = row[PF_F0_OP_M10_RE];
    out.f0_op_m10_im[tid]      = row[PF_F0_OP_M10_IM];
    out.f0_op_m11_re[tid]      = row[PF_F0_OP_M11_RE];
    out.f0_op_m11_im[tid]      = row[PF_F0_OP_M11_IM];
    out.f1_op_m00_re[tid]      = row[PF_F1_OP_M00_RE];
    out.f1_op_m00_im[tid]      = row[PF_F1_OP_M00_IM];
    out.f1_op_m01_re[tid]      = row[PF_F1_OP_M01_RE];
    out.f1_op_m01_im[tid]      = row[PF_F1_OP_M01_IM];
    out.f1_op_m10_re[tid]      = row[PF_F1_OP_M10_RE];
    out.f1_op_m10_im[tid]      = row[PF_F1_OP_M10_IM];
    out.f1_op_m11_re[tid]      = row[PF_F1_OP_M11_RE];
    out.f1_op_m11_im[tid]      = row[PF_F1_OP_M11_IM];
    out.f0_eta_r[tid]          = row[PF_F0_ETA_R];
    out.f0_mu_r[tid]           = row[PF_F0_MU_R];
    out.f0_sigma[tid]          = row[PF_F0_SIGMA];
    out.f0_gain[tid]           = row[PF_F0_GAIN];
    out.f0_use_fresnel[tid]    = row[PF_F0_USE_FRESNEL];
    out.f1_eta_r[tid]          = row[PF_F1_ETA_R];
    out.f1_mu_r[tid]           = row[PF_F1_MU_R];
    out.f1_sigma[tid]          = row[PF_F1_SIGMA];
    out.f1_gain[tid]           = row[PF_F1_GAIN];
    out.f1_use_fresnel[tid]    = row[PF_F1_USE_FRESNEL];
    out.prefix_refl_depth[tid] = __float_as_uint(row[PF_PREFIX_REFL_DEPTH]);
    out.inter_refl_depth[tid]  = __float_as_uint(row[PF_INTER_REFL_DEPTH]);
    out.suffix_refl_depth[tid] = __float_as_uint(row[PF_SUFFIX_REFL_DEPTH]);
    out.order[tid]             = __float_as_uint(row[PF_ORDER]);
}

__global__ void gather_packed_states_kernel(
    const float* __restrict__ src,
    const int* __restrict__ indices,
    float* __restrict__ dst,
    int n_out,
    int stride
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_out) {
        return;
    }
    const float* src_row = src + indices[tid] * stride;
    float* dst_row = dst + tid * stride;
    for (int i = 0; i < stride; ++i) {
        dst_row[i] = src_row[i];
    }
}

__global__ void gather_inserted_reflection_state_fields_kernel(
    const float* __restrict__ src,
    const int* __restrict__ indices,
    float* __restrict__ edge_pos_x,
    float* __restrict__ edge_pos_y,
    float* __restrict__ edge_pos_z,
    std::uint32_t* __restrict__ prefix_refl_depth,
    std::uint32_t* __restrict__ inter_refl_depth,
    std::uint32_t* __restrict__ suffix_refl_depth,
    std::uint32_t* __restrict__ order,
    int n_out,
    int stride
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_out) {
        return;
    }

    const float* row = src + indices[tid] * stride;
    edge_pos_x[tid] = row[PF_EDGE_POS_X];
    edge_pos_y[tid] = row[PF_EDGE_POS_Y];
    edge_pos_z[tid] = row[PF_EDGE_POS_Z];
    prefix_refl_depth[tid] = __float_as_uint(row[PF_PREFIX_REFL_DEPTH]);
    inter_refl_depth[tid] = __float_as_uint(row[PF_INTER_REFL_DEPTH]);
    suffix_refl_depth[tid] = __float_as_uint(row[PF_SUFFIX_REFL_DEPTH]);
    order[tid] = __float_as_uint(row[PF_ORDER]);
}

__global__ void build_diffraction_path_slots_kernel(
    DiffractionPathSlotKernelInputs in,
    DiffractionPathSlotKernelOutputs out,
    int n_states,
    int max_depth,
    bool return_geometry,
    int reflection_code,
    int diffraction_code
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_states) {
        return;
    }

    for (int depth = 0; depth < max_depth; ++depth) {
        out.type_slots[depth][tid] = 0;
        if (!return_geometry) {
            continue;
        }
        out.vertex_x_slots[depth][tid] = 0.0f;
        out.vertex_y_slots[depth][tid] = 0.0f;
        out.vertex_z_slots[depth][tid] = 0.0f;
        out.normal_x_slots[depth][tid] = 0.0f;
        out.normal_y_slots[depth][tid] = 0.0f;
        out.normal_z_slots[depth][tid] = 0.0f;
        out.object_slots[depth][tid] = -1;
    }

    int prefix = in.prefix_depth != nullptr ? in.prefix_depth[tid] : 0;
    int ord = in.order != nullptr ? in.order[tid] : 0;
    if (prefix < 0) {
        prefix = 0;
    }
    if (ord < 0) {
        ord = 0;
    }

    for (int depth = 0; depth < max_depth && depth < prefix; ++depth) {
        out.type_slots[depth][tid] = reflection_code;
    }

    if (return_geometry && prefix > 0
        && in.first_interaction_pos_x != nullptr
        && in.first_interaction_pos_y != nullptr
        && in.first_interaction_pos_z != nullptr) {
        out.vertex_x_slots[0][tid] = in.first_interaction_pos_x[tid];
        out.vertex_y_slots[0][tid] = in.first_interaction_pos_y[tid];
        out.vertex_z_slots[0][tid] = in.first_interaction_pos_z[tid];
    }

    int inserted_before = 0;
    for (int diff_slot = 0; diff_slot < in.history_size; ++diff_slot) {
        bool active = ord > diff_slot;
        int diff_position = prefix + diff_slot + inserted_before;
        int edge_idx = in.path_edge_slots[diff_slot] != nullptr ? in.path_edge_slots[diff_slot][tid] : -1;

        if (active && diff_position >= 0 && diff_position < max_depth) {
            out.type_slots[diff_position][tid] = diffraction_code;
            if (return_geometry
                && in.n_edges > 0
                && edge_idx >= 0
                && edge_idx < in.n_edges
                && in.edge_pos_x != nullptr
                && in.edge_pos_y != nullptr
                && in.edge_pos_z != nullptr
                && in.edge_n0_x != nullptr
                && in.edge_n0_y != nullptr
                && in.edge_n0_z != nullptr
                && in.edge_object_idx != nullptr) {
                out.vertex_x_slots[diff_position][tid] = in.edge_pos_x[edge_idx];
                out.vertex_y_slots[diff_position][tid] = in.edge_pos_y[edge_idx];
                out.vertex_z_slots[diff_position][tid] = in.edge_pos_z[edge_idx];
                out.normal_x_slots[diff_position][tid] = in.edge_n0_x[edge_idx];
                out.normal_y_slots[diff_position][tid] = in.edge_n0_y[edge_idx];
                out.normal_z_slots[diff_position][tid] = in.edge_n0_z[edge_idx];
                out.object_slots[diff_position][tid] = in.edge_object_idx[edge_idx];
            }
        }

        if (diff_slot < in.history_size - 1 && in.inserted_depth_slots[diff_slot] != nullptr) {
            int inserted_depth = in.inserted_depth_slots[diff_slot][tid];
            if (inserted_depth < 0) {
                inserted_depth = 0;
            }
            if (ord > diff_slot + 1) {
                int inserted_start = diff_position + 1;
                for (int offset = 0; offset < inserted_depth; ++offset) {
                    int depth_position = inserted_start + offset;
                    if (depth_position >= 0 && depth_position < max_depth) {
                        out.type_slots[depth_position][tid] = reflection_code;
                    }
                }
                inserted_before += inserted_depth;
            }
        }
    }
}

__global__ void concat_packed_states_kernel(
    const float* const* __restrict__ srcs,
    const int* __restrict__ cum_offsets,
    float* __restrict__ dst,
    int total_states,
    int n_sources,
    int stride
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_states) {
        return;
    }

    int lo = 0;
    int hi = n_sources;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cum_offsets[mid + 1] <= tid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    int src_buf = lo;
    int local_idx = tid - cum_offsets[src_buf];
    const float* src_row = srcs[src_buf] + local_idx * stride;
    float* dst_row = dst + tid * stride;
    for (int i = 0; i < stride; ++i) {
        dst_row[i] = src_row[i];
    }
}

} // anonymous namespace

void gather_packed_states(
    const float* src,
    const int* indices,
    float* dst,
    int n_out,
    int stride
) {
    if (n_out <= 0) {
        return;
    }

    constexpr int BLOCK = 256;
    int grid = (n_out + BLOCK - 1) / BLOCK;
    gather_packed_states_kernel<<<grid, BLOCK>>>(src, indices, dst, n_out, stride);
    throw_cuda(cudaGetLastError(), "gather_packed_states_kernel launch");
}

void gather_inserted_reflection_state_fields(
    const float* src,
    const int* indices,
    float* edge_pos_x,
    float* edge_pos_y,
    float* edge_pos_z,
    std::uint32_t* prefix_refl_depth,
    std::uint32_t* inter_refl_depth,
    std::uint32_t* suffix_refl_depth,
    std::uint32_t* order,
    int n_out,
    int stride
) {
    if (n_out <= 0) {
        return;
    }

    constexpr int BLOCK = 256;
    int grid = (n_out + BLOCK - 1) / BLOCK;
    gather_inserted_reflection_state_fields_kernel<<<grid, BLOCK>>>(
        src,
        indices,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        prefix_refl_depth,
        inter_refl_depth,
        suffix_refl_depth,
        order,
        n_out,
        stride
    );
    throw_cuda(cudaGetLastError(), "gather_inserted_reflection_state_fields_kernel launch");
}

void build_diffraction_path_slots(
    const DiffractionPathSlotInputs& in,
    const DiffractionPathSlotOutputs& out,
    int n_states,
    int max_depth,
    bool return_geometry,
    int reflection_code,
    int diffraction_code
) {
    if (n_states <= 0 || max_depth <= 0) {
        return;
    }

    const std::int32_t** d_path_edge_slots = nullptr;
    const std::int32_t** d_inserted_depth_slots = nullptr;
    std::int32_t** d_type_slots = nullptr;
    float** d_vertex_x_slots = nullptr;
    float** d_vertex_y_slots = nullptr;
    float** d_vertex_z_slots = nullptr;
    float** d_normal_x_slots = nullptr;
    float** d_normal_y_slots = nullptr;
    float** d_normal_z_slots = nullptr;
    std::int32_t** d_object_slots = nullptr;

    auto free_if = [](const void* ptr) {
        if (ptr != nullptr) {
            cudaFree(const_cast<void*>(ptr));
        }
    };
    auto copy_pointer_table = [&](auto*& dst, const auto* src, size_t count, const char* alloc_name, const char* copy_name) {
        if (count == 0) {
            dst = nullptr;
            return;
        }
        throw_cuda(cudaMalloc(reinterpret_cast<void**>(&dst), count * sizeof(*src)), alloc_name);
        throw_cuda(cudaMemcpy(dst, src, count * sizeof(*src), cudaMemcpyHostToDevice), copy_name);
    };

    constexpr int BLOCK = 256;
    int grid = (n_states + BLOCK - 1) / BLOCK;
    try {
        if (in.history_size > 0) {
            void* raw_path_edge_slots = nullptr;
            throw_cuda(
                cudaMalloc(
                    &raw_path_edge_slots,
                    static_cast<size_t>(in.history_size) * sizeof(*in.path_edge_slots)
                ),
                "malloc path_edge_slots"
            );
            d_path_edge_slots = reinterpret_cast<const std::int32_t**>(raw_path_edge_slots);
            throw_cuda(
                cudaMemcpy(
                    raw_path_edge_slots,
                    in.path_edge_slots,
                    static_cast<size_t>(in.history_size) * sizeof(*in.path_edge_slots),
                    cudaMemcpyHostToDevice
                ),
                "memcpy path_edge_slots"
            );
        }
        if (in.history_size > 1) {
            void* raw_inserted_depth_slots = nullptr;
            throw_cuda(
                cudaMalloc(
                    &raw_inserted_depth_slots,
                    static_cast<size_t>(in.history_size - 1) * sizeof(*in.inserted_depth_slots)
                ),
                "malloc inserted_depth_slots"
            );
            d_inserted_depth_slots = reinterpret_cast<const std::int32_t**>(raw_inserted_depth_slots);
            throw_cuda(
                cudaMemcpy(
                    raw_inserted_depth_slots,
                    in.inserted_depth_slots,
                    static_cast<size_t>(in.history_size - 1) * sizeof(*in.inserted_depth_slots),
                    cudaMemcpyHostToDevice
                ),
                "memcpy inserted_depth_slots"
            );
        }
        copy_pointer_table(
            d_type_slots,
            out.type_slots,
            static_cast<size_t>(max_depth),
            "malloc type_slots",
            "memcpy type_slots"
        );
        if (return_geometry) {
            copy_pointer_table(
                d_vertex_x_slots,
                out.vertex_x_slots,
                static_cast<size_t>(max_depth),
                "malloc vertex_x_slots",
                "memcpy vertex_x_slots"
            );
            copy_pointer_table(
                d_vertex_y_slots,
                out.vertex_y_slots,
                static_cast<size_t>(max_depth),
                "malloc vertex_y_slots",
                "memcpy vertex_y_slots"
            );
            copy_pointer_table(
                d_vertex_z_slots,
                out.vertex_z_slots,
                static_cast<size_t>(max_depth),
                "malloc vertex_z_slots",
                "memcpy vertex_z_slots"
            );
            copy_pointer_table(
                d_normal_x_slots,
                out.normal_x_slots,
                static_cast<size_t>(max_depth),
                "malloc normal_x_slots",
                "memcpy normal_x_slots"
            );
            copy_pointer_table(
                d_normal_y_slots,
                out.normal_y_slots,
                static_cast<size_t>(max_depth),
                "malloc normal_y_slots",
                "memcpy normal_y_slots"
            );
            copy_pointer_table(
                d_normal_z_slots,
                out.normal_z_slots,
                static_cast<size_t>(max_depth),
                "malloc normal_z_slots",
                "memcpy normal_z_slots"
            );
            copy_pointer_table(
                d_object_slots,
                out.object_slots,
                static_cast<size_t>(max_depth),
                "malloc object_slots",
                "memcpy object_slots"
            );
        }

        DiffractionPathSlotKernelInputs kernel_in{};
        kernel_in.prefix_depth = in.prefix_depth;
        kernel_in.order = in.order;
        kernel_in.path_edge_slots = d_path_edge_slots;
        kernel_in.inserted_depth_slots = d_inserted_depth_slots;
        kernel_in.first_interaction_pos_x = in.first_interaction_pos_x;
        kernel_in.first_interaction_pos_y = in.first_interaction_pos_y;
        kernel_in.first_interaction_pos_z = in.first_interaction_pos_z;
        kernel_in.edge_pos_x = in.edge_pos_x;
        kernel_in.edge_pos_y = in.edge_pos_y;
        kernel_in.edge_pos_z = in.edge_pos_z;
        kernel_in.edge_n0_x = in.edge_n0_x;
        kernel_in.edge_n0_y = in.edge_n0_y;
        kernel_in.edge_n0_z = in.edge_n0_z;
        kernel_in.edge_object_idx = in.edge_object_idx;
        kernel_in.history_size = in.history_size;
        kernel_in.n_edges = in.n_edges;

        DiffractionPathSlotKernelOutputs kernel_out{};
        kernel_out.type_slots = d_type_slots;
        kernel_out.vertex_x_slots = d_vertex_x_slots;
        kernel_out.vertex_y_slots = d_vertex_y_slots;
        kernel_out.vertex_z_slots = d_vertex_z_slots;
        kernel_out.normal_x_slots = d_normal_x_slots;
        kernel_out.normal_y_slots = d_normal_y_slots;
        kernel_out.normal_z_slots = d_normal_z_slots;
        kernel_out.object_slots = d_object_slots;

        build_diffraction_path_slots_kernel<<<grid, BLOCK>>>(
            kernel_in,
            kernel_out,
            n_states,
            max_depth,
            return_geometry,
            reflection_code,
            diffraction_code
        );
        throw_cuda(cudaGetLastError(), "build_diffraction_path_slots_kernel launch");
    } catch (...) {
        free_if(d_path_edge_slots);
        free_if(d_inserted_depth_slots);
        free_if(d_type_slots);
        free_if(d_vertex_x_slots);
        free_if(d_vertex_y_slots);
        free_if(d_vertex_z_slots);
        free_if(d_normal_x_slots);
        free_if(d_normal_y_slots);
        free_if(d_normal_z_slots);
        free_if(d_object_slots);
        throw;
    }

    free_if(d_path_edge_slots);
    free_if(d_inserted_depth_slots);
    free_if(d_type_slots);
    free_if(d_vertex_x_slots);
    free_if(d_vertex_y_slots);
    free_if(d_vertex_z_slots);
    free_if(d_normal_x_slots);
    free_if(d_normal_y_slots);
    free_if(d_normal_z_slots);
    free_if(d_object_slots);
}

void concat_packed_states(
    const float* const* srcs,
    const int* sizes,
    float* dst,
    int n_sources,
    int stride
) {
    if (n_sources <= 0) {
        return;
    }

    int* h_cum = new int[n_sources + 1];
    h_cum[0] = 0;
    for (int i = 0; i < n_sources; ++i) {
        h_cum[i + 1] = h_cum[i] + sizes[i];
    }

    int total = h_cum[n_sources];
    if (total <= 0) {
        delete[] h_cum;
        return;
    }

    int* d_cum = nullptr;
    throw_cuda(cudaMalloc(&d_cum, (n_sources + 1) * sizeof(int)), "malloc concat offsets");
    throw_cuda(
        cudaMemcpy(d_cum, h_cum, (n_sources + 1) * sizeof(int), cudaMemcpyHostToDevice),
        "memcpy concat offsets"
    );

    const float** d_srcs = nullptr;
    throw_cuda(cudaMalloc(&d_srcs, n_sources * sizeof(float*)), "malloc concat srcs");
    throw_cuda(
        cudaMemcpy(d_srcs, srcs, n_sources * sizeof(float*), cudaMemcpyHostToDevice),
        "memcpy concat srcs"
    );

    constexpr int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    concat_packed_states_kernel<<<grid, BLOCK>>>(d_srcs, d_cum, dst, total, n_sources, stride);
    throw_cuda(cudaGetLastError(), "concat_packed_states_kernel launch");

    cudaFree(d_cum);
    cudaFree(d_srcs);
    delete[] h_cum;
}

void pack_state_arrays(
    const std::uintptr_t* core_ptrs,
    float* dst,
    int n_states,
    int stride
) {
    if (n_states <= 0) {
        return;
    }

    PackedStateCoreIn in = make_core_in(core_ptrs);

    constexpr int BLOCK = 256;
    int grid = (n_states + BLOCK - 1) / BLOCK;
    pack_state_arrays_kernel<<<grid, BLOCK>>>(in, dst, n_states, stride);
    throw_cuda(cudaGetLastError(), "pack_state_arrays_kernel launch");
}

void unpack_state_arrays(
    const float* src,
    const std::uintptr_t* core_ptrs,
    int n_states,
    int stride
) {
    if (n_states <= 0) {
        return;
    }

    PackedStateCoreOut out = make_core_out(core_ptrs);

    constexpr int BLOCK = 256;
    int grid = (n_states + BLOCK - 1) / BLOCK;
    unpack_state_arrays_kernel<<<grid, BLOCK>>>(src, out, n_states, stride);
    throw_cuda(cudaGetLastError(), "unpack_state_arrays_kernel launch");
}

} // namespace witwin::channel::native_ext
