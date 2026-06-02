#include "drjit_common.h"
#include <monitors/common/receiver_tiles/bind.h>

#include <monitors/common/receiver_tiles/receiver_tiles.h>

void register_receiver_tiles_bindings(nb::module_ &m) {
    m.def(
        "build_receiver_tiles_arrays",
        [](
            int plane_axis,
            float plane_position,
            Float coord_0,
            Float coord_1,
            int n_coord_0,
            int n_coord_1,
            int tile_size_0,
            int tile_size_1
        ) {
            if (n_coord_0 <= 0 || n_coord_1 <= 0 || tile_size_0 <= 0 || tile_size_1 <= 0) {
                return nb::make_tuple(
                    0,
                    0,
                    drjit::zeros<Int32>(0),
                    drjit::zeros<Int32>(0),
                    drjit::zeros<Int32>(0),
                    drjit::zeros<Int32>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0)
                );
            }

            int n_tiles_0 = (n_coord_0 + tile_size_0 - 1) / tile_size_0;
            int n_tiles_1 = (n_coord_1 + tile_size_1 - 1) / tile_size_1;
            int n_tiles = n_tiles_0 * n_tiles_1;

            Int32 tile_i0 = drjit::zeros<Int32>(static_cast<size_t>(n_tiles));
            Int32 tile_i1 = drjit::zeros<Int32>(static_cast<size_t>(n_tiles));
            Int32 tile_extent_0 = drjit::zeros<Int32>(static_cast<size_t>(n_tiles));
            Int32 tile_extent_1 = drjit::zeros<Int32>(static_cast<size_t>(n_tiles));
            Float coord_0_min = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float coord_0_max = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float coord_1_min = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float coord_1_max = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float aabb_min_x = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float aabb_min_y = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float aabb_min_z = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float aabb_max_x = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float aabb_max_y = drjit::zeros<Float>(static_cast<size_t>(n_tiles));
            Float aabb_max_z = drjit::zeros<Float>(static_cast<size_t>(n_tiles));

            drjit::eval(
                coord_0,
                coord_1,
                tile_i0,
                tile_i1,
                tile_extent_0,
                tile_extent_1,
                coord_0_min,
                coord_0_max,
                coord_1_min,
                coord_1_max,
                aabb_min_x,
                aabb_min_y,
                aabb_min_z,
                aabb_max_x,
                aabb_max_y,
                aabb_max_z
            );

            witwin::channel::native_ext::build_receiver_tiles(
                plane_axis,
                plane_position,
                drjit_data_ptr(coord_0),
                drjit_data_ptr(coord_1),
                n_coord_0,
                n_coord_1,
                tile_size_0,
                tile_size_1,
                drjit_data_ptr_mut(tile_i0),
                drjit_data_ptr_mut(tile_i1),
                drjit_data_ptr_mut(tile_extent_0),
                drjit_data_ptr_mut(tile_extent_1),
                drjit_data_ptr_mut(coord_0_min),
                drjit_data_ptr_mut(coord_0_max),
                drjit_data_ptr_mut(coord_1_min),
                drjit_data_ptr_mut(coord_1_max),
                drjit_data_ptr_mut(aabb_min_x),
                drjit_data_ptr_mut(aabb_min_y),
                drjit_data_ptr_mut(aabb_min_z),
                drjit_data_ptr_mut(aabb_max_x),
                drjit_data_ptr_mut(aabb_max_y),
                drjit_data_ptr_mut(aabb_max_z)
            );

            return nb::make_tuple(
                n_tiles_0,
                n_tiles_1,
                tile_i0,
                tile_i1,
                tile_extent_0,
                tile_extent_1,
                coord_0_min,
                coord_0_max,
                coord_1_min,
                coord_1_max,
                aabb_min_x,
                aabb_min_y,
                aabb_min_z,
                aabb_max_x,
                aabb_max_y,
                aabb_max_z
            );
        },
        "Build receiver-tile index and world-space AABB buffers from monitor-plane metadata."
    );
}
