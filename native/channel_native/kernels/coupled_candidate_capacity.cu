#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "../tensor_checks.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kBlockSize = 256;
constexpr int64_t kRdChunkSize = 65'536;
constexpr int64_t kMaxCoupledCandidates = 1'000'000;

struct CandidateOutput {
    bool *valid;
    int *candidate_count;
    bool *overflow;
    int *tx_id;
    int *rx_id;
    int *component_id;
    int *face_id;
    int *edge1_id;
    int *edge2_id;
};

__global__ void coupled_candidate_capacity_kernel(
    const int *__restrict__ representative_faces,
    const int *__restrict__ selected_edges,
    CandidateOutput output,
    int64_t tx_count,
    int64_t rx_count,
    int64_t rx_id_offset,
    int64_t group_count,
    int64_t edge_count,
    int64_t base_candidate_count,
    int64_t theoretical_candidate_count,
    int64_t candidate_capacity,
    bool capacity_overflow) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t slot = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         slot < candidate_capacity;
         slot += stride) {
        output.valid[slot] = false;
        output.tx_id[slot] = -1;
        output.rx_id[slot] = -1;
        output.component_id[slot] = -1;
        output.face_id[slot] = -1;
        output.edge1_id[slot] = -1;
        output.edge2_id[slot] = -1;
        if (capacity_overflow || slot >= theoretical_candidate_count) {
            continue;
        }

        output.valid[slot] = true;
        if (slot < 2 * base_candidate_count) {
            const int64_t full_candidate_count =
                (base_candidate_count / kRdChunkSize) * kRdChunkSize;
            const int64_t full_slot_count = 2 * full_candidate_count;
            int64_t linear = 0;
            bool reverse = false;
            if (slot < full_slot_count) {
                const int64_t chunk_slot = slot % (2 * kRdChunkSize);
                reverse = chunk_slot >= kRdChunkSize;
                linear = (slot / (2 * kRdChunkSize)) * kRdChunkSize +
                         (chunk_slot % kRdChunkSize);
            } else {
                const int64_t tail_count = base_candidate_count - full_candidate_count;
                const int64_t tail_slot = slot - full_slot_count;
                reverse = tail_slot >= tail_count;
                linear = full_candidate_count +
                         (reverse ? tail_slot - tail_count : tail_slot);
            }

            const int64_t candidates_per_pair = group_count * edge_count;
            const int64_t pair_slot = linear / candidates_per_pair;
            const int64_t local_slot = linear % candidates_per_pair;
            const int64_t tx_slot = pair_slot / rx_count;
            const int64_t rx_slot = pair_slot % rx_count;
            const int64_t face_slot = local_slot / edge_count;
            const int64_t edge_slot = local_slot % edge_count;
            output.tx_id[slot] = static_cast<int>(tx_slot);
            output.rx_id[slot] = static_cast<int>(rx_id_offset + rx_slot);
            output.component_id[slot] = reverse ? 4 : 3;
            output.face_id[slot] = representative_faces[face_slot];
            output.edge1_id[slot] = selected_edges[edge_slot];
            continue;
        }

        const int64_t linear = slot - 2 * base_candidate_count;
        const int64_t candidates_per_pair = edge_count * (edge_count - 1);
        const int64_t pair_slot = linear / candidates_per_pair;
        const int64_t local_slot = linear % candidates_per_pair;
        const int64_t tx_slot = pair_slot / rx_count;
        const int64_t rx_slot = pair_slot % rx_count;
        const int64_t first_slot = local_slot / (edge_count - 1);
        const int64_t remainder_slot = local_slot % (edge_count - 1);
        const int64_t second_slot =
            remainder_slot < first_slot ? remainder_slot : remainder_slot + 1;
        output.tx_id[slot] = static_cast<int>(tx_slot);
        output.rx_id[slot] = static_cast<int>(rx_id_offset + rx_slot);
        output.component_id[slot] = 7;
        output.edge1_id[slot] = selected_edges[first_slot];
        output.edge2_id[slot] = selected_edges[second_slot];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output.candidate_count[0] =
            capacity_overflow ? 0 : static_cast<int>(theoretical_candidate_count);
        output.overflow[0] = capacity_overflow;
    }
}

__global__ void coupled_candidate_capacity_trap_kernel(
    const bool *__restrict__ overflow) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && overflow[0]) {
        asm volatile("trap;");
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

int64_t checked_product(int64_t lhs, int64_t rhs, const char *message) {
    TORCH_CHECK(
        lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs,
        message);
    return lhs * rhs;
}

}  // namespace

pybind11::dict cn_coupled_candidate_capacity_block(
    at::Tensor representative_faces,
    at::Tensor selected_edges,
    int64_t tx_count,
    int64_t rx_count,
    int64_t rx_id_offset,
    int64_t candidate_capacity,
    int64_t candidate_limit) {
    channel_native::check_tensor(
        representative_faces, "representative_faces", at::kInt, 1);
    channel_native::check_tensor(selected_edges, "selected_edges", at::kInt, 1);
    TORCH_CHECK(
        selected_edges.get_device() == representative_faces.get_device(),
        "selected_edges must share representative_faces device");
    TORCH_CHECK(tx_count >= 0, "tx_count must be non-negative");
    TORCH_CHECK(rx_count >= 0, "rx_count must be non-negative");
    TORCH_CHECK(rx_id_offset >= 0, "rx_id_offset must be non-negative");
    TORCH_CHECK(candidate_capacity >= 0, "candidate_capacity must be non-negative");
    TORCH_CHECK(candidate_limit > 0, "candidate_limit must be positive");
    const int64_t effective_limit =
        std::min(candidate_limit, kMaxCoupledCandidates);
    TORCH_CHECK(
        candidate_capacity <= effective_limit,
        "candidate_capacity exceeds coupled candidate guardrail");
    TORCH_CHECK(
        tx_count <= std::numeric_limits<int>::max(),
        "tx_count exceeds int32 identifier capacity");
    TORCH_CHECK(
        rx_count == 0 ||
            rx_id_offset <= std::numeric_limits<int>::max() - (rx_count - 1),
        "receiver identifiers exceed int32 capacity");

    const int64_t group_count = representative_faces.size(0);
    const int64_t edge_count = selected_edges.size(0);
    const int64_t candidates_per_pair = checked_product(
        group_count, edge_count, "coupled R-D candidate axes overflow int64");
    const int64_t dd_candidates_per_pair = edge_count > 1
        ? checked_product(
              edge_count,
              edge_count - 1,
              "coupled D-D candidate axes overflow int64")
        : 0;
    TORCH_CHECK(
        candidates_per_pair <=
            (std::numeric_limits<int64_t>::max() - dd_candidates_per_pair) / 2,
        "coupled candidate union overflows int64");
    const int64_t union_per_pair =
        2 * candidates_per_pair + dd_candidates_per_pair;
    const int64_t pair_count = checked_product(
        tx_count, rx_count, "coupled endpoint pair count overflows int64");
    const int64_t theoretical_candidate_count = checked_product(
        pair_count,
        union_per_pair,
        "coupled theoretical candidate count overflows int64");
    TORCH_CHECK(
        theoretical_candidate_count <= effective_limit,
        "coupled reflection-diffraction topology requires ",
        theoretical_candidate_count,
        " candidates, exceeding coupled_candidate_limit=",
        effective_limit);
    const int64_t base_candidate_count = checked_product(
        pair_count,
        candidates_per_pair,
        "coupled R-D candidate count overflows int64");
    const bool capacity_overflow =
        theoretical_candidate_count > candidate_capacity;

    auto bool_options = representative_faces.options().dtype(at::kBool);
    auto int_options = representative_faces.options().dtype(at::kInt);
    auto valid = at::empty({candidate_capacity}, bool_options);
    auto candidate_count = at::empty({1}, int_options);
    auto overflow = at::empty({1}, bool_options);
    auto tx_id = at::empty({candidate_capacity}, int_options);
    auto rx_id = at::empty({candidate_capacity}, int_options);
    auto component_id = at::empty({candidate_capacity}, int_options);
    auto face_id = at::empty({candidate_capacity}, int_options);
    auto edge1_id = at::empty({candidate_capacity}, int_options);
    auto edge2_id = at::empty({candidate_capacity}, int_options);
    const CandidateOutput output{
        valid.data_ptr<bool>(),
        candidate_count.data_ptr<int>(),
        overflow.data_ptr<bool>(),
        tx_id.data_ptr<int>(),
        rx_id.data_ptr<int>(),
        component_id.data_ptr<int>(),
        face_id.data_ptr<int>(),
        edge1_id.data_ptr<int>(),
        edge2_id.data_ptr<int>()};
    const int device = representative_faces.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const int64_t init_count = std::max<int64_t>(candidate_capacity, 1);
    coupled_candidate_capacity_kernel<<<
        launch_blocks(init_count), kBlockSize, 0, stream>>>(
        representative_faces.data_ptr<int>(),
        selected_edges.data_ptr<int>(),
        output,
        tx_count,
        rx_count,
        rx_id_offset,
        group_count,
        edge_count,
        base_candidate_count,
        theoretical_candidate_count,
        candidate_capacity,
        capacity_overflow);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    coupled_candidate_capacity_trap_kernel<<<1, 1, 0, stream>>>(
        overflow.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    pybind11::dict result;
    result["valid"] = valid;
    result["candidate_count"] = candidate_count;
    result["overflow"] = overflow;
    result["tx_id"] = tx_id;
    result["rx_id"] = rx_id;
    result["component_id"] = component_id;
    result["face_id"] = face_id;
    result["edge1_id"] = edge1_id;
    result["edge2_id"] = edge2_id;
    return result;
}
