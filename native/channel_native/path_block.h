#pragma once

#include <torch/extension.h>

#include <tuple>
#include <vector>

using PathBlockTuple = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

#define CN_PATH_BLOCK_FIELDS(X)        \
    X(0, valid, "valid")              \
    X(1, tx_id, "tx_id")              \
    X(2, rx_id, "rx_id")              \
    X(3, depth, "depth")              \
    X(4, component_id, "component_id") \
    X(5, primitive_id, "primitive_id") \
    X(6, edge_id, "edge_id")          \
    X(7, path_length, "path_length_m") \
    X(8, delay, "delay_s")            \
    X(9, path_gain, "path_gain")

inline pybind11::dict path_block_dict(const PathBlockTuple& block) {
    pybind11::dict out;
#define CN_PATH_BLOCK_TO_DICT(index, member, key) out[key] = std::get<index>(block);
    CN_PATH_BLOCK_FIELDS(CN_PATH_BLOCK_TO_DICT)
#undef CN_PATH_BLOCK_TO_DICT
    return out;
}

struct PathBlockLists {
#define CN_PATH_BLOCK_LIST(index, member, key) std::vector<at::Tensor> member;
    CN_PATH_BLOCK_FIELDS(CN_PATH_BLOCK_LIST)
#undef CN_PATH_BLOCK_LIST

    void append(const PathBlockTuple& block) {
#define CN_PATH_BLOCK_APPEND(index, member, key) member.push_back(std::get<index>(block));
        CN_PATH_BLOCK_FIELDS(CN_PATH_BLOCK_APPEND)
#undef CN_PATH_BLOCK_APPEND
    }

    void reserve(size_t count) {
#define CN_PATH_BLOCK_RESERVE(index, member, key) member.reserve(count);
        CN_PATH_BLOCK_FIELDS(CN_PATH_BLOCK_RESERVE)
#undef CN_PATH_BLOCK_RESERVE
    }
};

#undef CN_PATH_BLOCK_FIELDS
