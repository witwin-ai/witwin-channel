#pragma once

#include <cstdint>

namespace witwin::channel::native_ext {

// =========================================================================
// Packed state buffer layout
//
// The native packed-state buffer stores only propagation-hot fields. Cold
// metadata and lineage replay data are handled in Python alongside the packed
// buffer so the native gather/concat/subset path can stay compact.
// =========================================================================

constexpr int PACKED_MAX_HISTORY_SLOTS = 8;

constexpr int PF_EDGE_IDX          =  0;
constexpr int PF_EDGE_POS_X        =  1;
constexpr int PF_EDGE_POS_Y        =  2;
constexpr int PF_EDGE_POS_Z        =  3;
constexpr int PF_EDGE_DIR_X        =  4;
constexpr int PF_EDGE_DIR_Y        =  5;
constexpr int PF_EDGE_DIR_Z        =  6;
constexpr int PF_N0_X              =  7;
constexpr int PF_N0_Y              =  8;
constexpr int PF_N0_Z              =  9;
constexpr int PF_NN_X              = 10;
constexpr int PF_NN_Y              = 11;
constexpr int PF_NN_Z              = 12;
constexpr int PF_WEDGE_N           = 13;
constexpr int PF_ADJ_FACE0         = 14;
constexpr int PF_ADJ_FACE1         = 15;
constexpr int PF_SOURCE_POS_X      = 16;
constexpr int PF_SOURCE_POS_Y      = 17;
constexpr int PF_SOURCE_POS_Z      = 18;
constexpr int PF_INC_FIELD_RE      = 19;
constexpr int PF_INC_FIELD_IM      = 20;
constexpr int PF_INC_NDERIV_RE     = 21;
constexpr int PF_INC_NDERIV_IM     = 22;
constexpr int PF_INC_JONES_U_RE    = 23;
constexpr int PF_INC_JONES_U_IM    = 24;
constexpr int PF_INC_JONES_V_RE    = 25;
constexpr int PF_INC_JONES_V_IM    = 26;
constexpr int PF_INC_DJONES_U_RE   = 27;
constexpr int PF_INC_DJONES_U_IM   = 28;
constexpr int PF_INC_DJONES_V_RE   = 29;
constexpr int PF_INC_DJONES_V_IM   = 30;
constexpr int PF_R0_RE             = 31;
constexpr int PF_R0_IM             = 32;
constexpr int PF_RN_RE             = 33;
constexpr int PF_RN_IM             = 34;
constexpr int PF_INC_BASIS_U_X     = 35;
constexpr int PF_INC_BASIS_U_Y     = 36;
constexpr int PF_INC_BASIS_U_Z     = 37;
constexpr int PF_INC_BASIS_V_X     = 38;
constexpr int PF_INC_BASIS_V_Y     = 39;
constexpr int PF_INC_BASIS_V_Z     = 40;
constexpr int PF_INC_BASIS_K_X     = 41;
constexpr int PF_INC_BASIS_K_Y     = 42;
constexpr int PF_INC_BASIS_K_Z     = 43;
constexpr int PF_F0_OP_M00_RE      = 44;
constexpr int PF_F0_OP_M00_IM      = 45;
constexpr int PF_F0_OP_M01_RE      = 46;
constexpr int PF_F0_OP_M01_IM      = 47;
constexpr int PF_F0_OP_M10_RE      = 48;
constexpr int PF_F0_OP_M10_IM      = 49;
constexpr int PF_F0_OP_M11_RE      = 50;
constexpr int PF_F0_OP_M11_IM      = 51;
constexpr int PF_F1_OP_M00_RE      = 52;
constexpr int PF_F1_OP_M00_IM      = 53;
constexpr int PF_F1_OP_M01_RE      = 54;
constexpr int PF_F1_OP_M01_IM      = 55;
constexpr int PF_F1_OP_M10_RE      = 56;
constexpr int PF_F1_OP_M10_IM      = 57;
constexpr int PF_F1_OP_M11_RE      = 58;
constexpr int PF_F1_OP_M11_IM      = 59;
constexpr int PF_F0_ETA_R          = 60;
constexpr int PF_F0_SIGMA          = 61;
constexpr int PF_F0_GAIN           = 62;
constexpr int PF_F0_USE_FRESNEL    = 63;
constexpr int PF_F1_ETA_R          = 64;
constexpr int PF_F1_SIGMA          = 65;
constexpr int PF_F1_GAIN           = 66;
constexpr int PF_F1_USE_FRESNEL    = 67;
constexpr int PF_PREFIX_REFL_DEPTH = 68;
constexpr int PF_INTER_REFL_DEPTH  = 69;
constexpr int PF_SUFFIX_REFL_DEPTH = 70;
constexpr int PF_ORDER             = 71;

constexpr int PACKED_CORE_FLOATS   = 72;

constexpr int packed_state_stride(int /*history_size*/) {
    return (PACKED_CORE_FLOATS + 3) & ~3;
}

constexpr int PACKED_STATE_STRIDE_MAX = packed_state_stride(PACKED_MAX_HISTORY_SLOTS);
constexpr int DIFFRACTION_PATH_SLOT_MAX_DEPTH = 64;

struct DiffractionPathSlotInputs {
    const std::int32_t* prefix_depth = nullptr;
    const std::int32_t* order = nullptr;
    const std::int32_t* path_edge_slots[PACKED_MAX_HISTORY_SLOTS] = {};
    const std::int32_t* inserted_depth_slots[PACKED_MAX_HISTORY_SLOTS] = {};
    const float* first_interaction_pos_x = nullptr;
    const float* first_interaction_pos_y = nullptr;
    const float* first_interaction_pos_z = nullptr;
    const float* edge_pos_x = nullptr;
    const float* edge_pos_y = nullptr;
    const float* edge_pos_z = nullptr;
    const float* edge_n0_x = nullptr;
    const float* edge_n0_y = nullptr;
    const float* edge_n0_z = nullptr;
    const std::int32_t* edge_object_idx = nullptr;
    int history_size = 0;
    int n_edges = 0;
};

struct DiffractionPathSlotOutputs {
    std::int32_t* type_slots[DIFFRACTION_PATH_SLOT_MAX_DEPTH] = {};
    float* vertex_x_slots[DIFFRACTION_PATH_SLOT_MAX_DEPTH] = {};
    float* vertex_y_slots[DIFFRACTION_PATH_SLOT_MAX_DEPTH] = {};
    float* vertex_z_slots[DIFFRACTION_PATH_SLOT_MAX_DEPTH] = {};
    float* normal_x_slots[DIFFRACTION_PATH_SLOT_MAX_DEPTH] = {};
    float* normal_y_slots[DIFFRACTION_PATH_SLOT_MAX_DEPTH] = {};
    float* normal_z_slots[DIFFRACTION_PATH_SLOT_MAX_DEPTH] = {};
    std::int32_t* object_slots[DIFFRACTION_PATH_SLOT_MAX_DEPTH] = {};
};

// Pointer order shared by Python native_impl.py and the pack/unpack launchers.
constexpr int PSP_EDGE_IDX                =  0;
constexpr int PSP_EDGE_POS_X              =  1;
constexpr int PSP_EDGE_POS_Y              =  2;
constexpr int PSP_EDGE_POS_Z              =  3;
constexpr int PSP_EDGE_DIR_X              =  4;
constexpr int PSP_EDGE_DIR_Y              =  5;
constexpr int PSP_EDGE_DIR_Z              =  6;
constexpr int PSP_N0_X                    =  7;
constexpr int PSP_N0_Y                    =  8;
constexpr int PSP_N0_Z                    =  9;
constexpr int PSP_NN_X                    = 10;
constexpr int PSP_NN_Y                    = 11;
constexpr int PSP_NN_Z                    = 12;
constexpr int PSP_WEDGE_N                 = 13;
constexpr int PSP_ADJ_FACE0               = 14;
constexpr int PSP_ADJ_FACE1               = 15;
constexpr int PSP_SOURCE_POS_X            = 16;
constexpr int PSP_SOURCE_POS_Y            = 17;
constexpr int PSP_SOURCE_POS_Z            = 18;
constexpr int PSP_INC_FIELD_RE            = 19;
constexpr int PSP_INC_FIELD_IM            = 20;
constexpr int PSP_INC_NDERIV_RE           = 21;
constexpr int PSP_INC_NDERIV_IM           = 22;
constexpr int PSP_INC_JONES_U_RE          = 23;
constexpr int PSP_INC_JONES_U_IM          = 24;
constexpr int PSP_INC_JONES_V_RE          = 25;
constexpr int PSP_INC_JONES_V_IM          = 26;
constexpr int PSP_INC_DJONES_U_RE         = 27;
constexpr int PSP_INC_DJONES_U_IM         = 28;
constexpr int PSP_INC_DJONES_V_RE         = 29;
constexpr int PSP_INC_DJONES_V_IM         = 30;
constexpr int PSP_R0_RE                   = 31;
constexpr int PSP_R0_IM                   = 32;
constexpr int PSP_RN_RE                   = 33;
constexpr int PSP_RN_IM                   = 34;
constexpr int PSP_INC_BASIS_U_X           = 35;
constexpr int PSP_INC_BASIS_U_Y           = 36;
constexpr int PSP_INC_BASIS_U_Z           = 37;
constexpr int PSP_INC_BASIS_V_X           = 38;
constexpr int PSP_INC_BASIS_V_Y           = 39;
constexpr int PSP_INC_BASIS_V_Z           = 40;
constexpr int PSP_INC_BASIS_K_X           = 41;
constexpr int PSP_INC_BASIS_K_Y           = 42;
constexpr int PSP_INC_BASIS_K_Z           = 43;
constexpr int PSP_F0_OP_M00_RE            = 44;
constexpr int PSP_F0_OP_M00_IM            = 45;
constexpr int PSP_F0_OP_M01_RE            = 46;
constexpr int PSP_F0_OP_M01_IM            = 47;
constexpr int PSP_F0_OP_M10_RE            = 48;
constexpr int PSP_F0_OP_M10_IM            = 49;
constexpr int PSP_F0_OP_M11_RE            = 50;
constexpr int PSP_F0_OP_M11_IM            = 51;
constexpr int PSP_F1_OP_M00_RE            = 52;
constexpr int PSP_F1_OP_M00_IM            = 53;
constexpr int PSP_F1_OP_M01_RE            = 54;
constexpr int PSP_F1_OP_M01_IM            = 55;
constexpr int PSP_F1_OP_M10_RE            = 56;
constexpr int PSP_F1_OP_M10_IM            = 57;
constexpr int PSP_F1_OP_M11_RE            = 58;
constexpr int PSP_F1_OP_M11_IM            = 59;
constexpr int PSP_F0_ETA_R                = 60;
constexpr int PSP_F0_SIGMA                = 61;
constexpr int PSP_F0_GAIN                 = 62;
constexpr int PSP_F0_USE_FRESNEL          = 63;
constexpr int PSP_F1_ETA_R                = 64;
constexpr int PSP_F1_SIGMA                = 65;
constexpr int PSP_F1_GAIN                 = 66;
constexpr int PSP_F1_USE_FRESNEL          = 67;
constexpr int PSP_PREFIX_REFL_DEPTH       = 68;
constexpr int PSP_INTER_REFL_DEPTH        = 69;
constexpr int PSP_SUFFIX_REFL_DEPTH       = 70;
constexpr int PSP_ORDER                   = 71;
constexpr int PACKED_CORE_POINTER_COUNT   = 72;

void gather_packed_states(
    const float* src,
    const int* indices,
    float* dst,
    int n_out,
    int stride
);

void concat_packed_states(
    const float* const* srcs,
    const int* sizes,
    float* dst,
    int n_sources,
    int stride
);

void pack_state_arrays(
    const std::uintptr_t* core_ptrs,
    float* dst,
    int n_states,
    int stride
);

void unpack_state_arrays(
    const float* src,
    const std::uintptr_t* core_ptrs,
    int n_states,
    int stride
);

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
);

void build_diffraction_path_slots(
    const DiffractionPathSlotInputs& in,
    const DiffractionPathSlotOutputs& out,
    int n_states,
    int max_depth,
    bool return_geometry,
    int reflection_code,
    int diffraction_code
);

} // namespace witwin::channel::native_ext
