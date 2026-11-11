// Copyright Xingyu Chen.
// Receiver block arguments shared by the MC reflection bindings and kernels.

#pragma once

#include <ATen/ATen.h>

#include <cstdint>

// One receiver block of an MC reflection launch: an axis-aligned grid, or, when `points` holds
// P > 0 rows, P point receivers whose capture radius is point_radius (> 0) or -point_radius
// times the straight transmitter distance (< 0).
struct ReflectionReceiverArgs {
    int64_t axis = 2;
    double plane_position = 0.0;
    double coord0_min = 0.0;
    double coord0_max = 0.0;
    double coord1_min = 0.0;
    double coord1_max = 0.0;
    int64_t resolution0 = 0;
    int64_t resolution1 = 0;
    double cell_area = 0.0;
    at::Tensor points;
    double point_radius = 0.0;
};
