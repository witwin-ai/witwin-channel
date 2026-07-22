#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

namespace channel::capacity {

struct FinalizeState {
    at::Tensor failure_state;
    at::Tensor selected_row_index;
    at::Tensor valid;
    at::Tensor num_paths;
    at::Tensor overflow_flag;
    at::Tensor contract_error;
};

int64_t deterministic_capacity_validate(
    const at::Tensor& candidate_valid,
    const at::Tensor& tx_id,
    const at::Tensor& rx_id,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair);

FinalizeState deterministic_capacity_finalize_no_trap(
    at::Tensor failure_state,
    at::Tensor candidate_valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair,
    at::Tensor selected_row_index,
    at::Tensor output_valid,
    at::Tensor num_paths,
    bool initialize_public_outputs);

void deterministic_capacity_publish_status(
    const FinalizeState& state,
    at::Tensor overflow,
    cudaStream_t stream);

}  // namespace channel::capacity
