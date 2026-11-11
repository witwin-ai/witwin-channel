// Copyright Xingyu Chen.
// Generates reflection and interaction-chain candidates: plane groups, image trees, tests.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <rayd/utd.h>

#include "math.cuh"
#include "path_compaction.cuh"

namespace {

namespace cmath = channel::math;

// Deepest specular chain the candidate kernels unroll; matches the EPC bounce limit.
constexpr int kDiscoveryMaxDepth = 8;
constexpr int kDiscoveryBlock = 256;
// Each plane group carries a kOccupancyCells x kOccupancyCells grid over its bounds, projected
// along the dominant normal axis; a cell is set when a member triangle's bounds overlap it.
constexpr int kOccupancyCells = 32;

struct GroupTables {
    const float* plane_point;   // [G, 3]
    const float* plane_normal;  // [G, 3]
    const float* bounds;        // [G, 6]: min xyz, max xyz
    const int* occupancy;       // [G, kOccupancyCells]: one bit row per cell row
    int64_t group_count;
};

// Mirror image of a point across a plane (unit normal), as the EPC solve forms it.
__device__ __forceinline__ float3 mirror(float3 point, float3 plane_point, float3 unit_normal) {
    const float distance = cmath::dot(cmath::sub(point, plane_point), unit_normal);
    return cmath::sub(point, cmath::scale(unit_normal, 2.0f * distance));
}

__device__ __forceinline__ bool inside_bounds(const GroupTables& tables, int group, float3 p, float tolerance) {
    const float* b = tables.bounds + static_cast<int64_t>(group) * 6;
    return p.x >= b[0] - tolerance && p.y >= b[1] - tolerance && p.z >= b[2] - tolerance &&
           p.x <= b[3] + tolerance && p.y <= b[4] + tolerance && p.z <= b[5] + tolerance;
}

// The two in-plane axes of a group: every axis but the dominant one of its normal.
__device__ __forceinline__ void plane_axes(const float* plane_normal, int64_t group, int& u, int& v) {
    const float nx = fabsf(plane_normal[group * 3]);
    const float ny = fabsf(plane_normal[group * 3 + 1]);
    const float nz = fabsf(plane_normal[group * 3 + 2]);
    const int drop = nx >= ny && nx >= nz ? 0 : (ny >= nz ? 1 : 2);
    u = drop == 0 ? 1 : 0;
    v = drop == 2 ? 1 : 2;
}

__device__ __forceinline__ float axis_of(float3 p, int axis) { return axis == 0 ? p.x : (axis == 1 ? p.y : p.z); }

// Cell range [first, last] covering [lo, hi] on one axis of a group whose bounds span
// [axis_lo, axis_hi]; an empty range (first > last) when the interval misses the bounds.
__device__ __forceinline__ void cell_range(float lo, float hi, float axis_lo, float axis_hi, int& first, int& last) {
    const float extent = axis_hi - axis_lo;
    if (!(extent > 0.0f)) {
        first = 0;
        last = 0;
        return;
    }
    const float scale = static_cast<float>(kOccupancyCells) / extent;
    first = max(0, static_cast<int>(floorf((lo - axis_lo) * scale)));
    last = min(kOccupancyCells - 1, static_cast<int>(floorf((hi - axis_lo) * scale)));
}

// True when the box [p - tolerance, p + tolerance] overlaps an occupied cell of the group.
__device__ bool occupied(const GroupTables& tables, int group, float3 p, float tolerance) {
    int u, v;
    plane_axes(tables.plane_normal, group, u, v);
    const float* b = tables.bounds + static_cast<int64_t>(group) * 6;
    int u0, u1, v0, v1;
    cell_range(axis_of(p, u) - tolerance, axis_of(p, u) + tolerance, b[u], b[3 + u], u0, u1);
    cell_range(axis_of(p, v) - tolerance, axis_of(p, v) + tolerance, b[v], b[3 + v], v0, v1);
    const int* rows = tables.occupancy + static_cast<int64_t>(group) * kOccupancyCells;
    for (int row = v0; row <= v1; ++row) {
        const unsigned int bits = static_cast<unsigned int>(rows[row]);
        for (int column = u0; column <= u1; ++column) {
            if (bits & (1u << column)) {
                return true;
            }
        }
    }
    return false;
}

// Order-preserving float <-> int mapping for atomic min/max on float bounds.
__device__ __forceinline__ int float_to_ordered(float value) {
    const int bits = __float_as_int(value);
    return bits >= 0 ? bits : bits ^ 0x7fffffff;
}

__device__ __forceinline__ float ordered_to_float(int value) {
    return __int_as_float(value >= 0 ? value : value ^ 0x7fffffff);
}

__global__ void group_bounds_init_kernel(int64_t group_count, int* __restrict__ ordered) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= group_count * 6) {
        return;
    }
    const int axis = static_cast<int>(index % 6);
    ordered[index] = axis < 3 ? float_to_ordered(INFINITY) : float_to_ordered(-INFINITY);
}

__global__ void group_bounds_accumulate_kernel(int64_t face_count, const float* __restrict__ vertices,
                                               const int* __restrict__ faces, const int* __restrict__ face_group_id,
                                               int64_t group_count, int* __restrict__ ordered) {
    const int64_t face = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (face >= face_count) {
        return;
    }
    const int group = face_group_id[face];
    if (group < 0 || group >= group_count) {
        return;
    }
    int* slot = ordered + static_cast<int64_t>(group) * 6;
    for (int corner = 0; corner < 3; ++corner) {
        const int vertex = faces[face * 3 + corner];
        for (int axis = 0; axis < 3; ++axis) {
            const int value = float_to_ordered(vertices[static_cast<int64_t>(vertex) * 3 + axis]);
            atomicMin(slot + axis, value);
            atomicMax(slot + 3 + axis, value);
        }
    }
}

__global__ void group_planes_kernel(int64_t group_count, const int* __restrict__ ordered,
                                    const int64_t* __restrict__ representative_faces, const float* __restrict__ tri_a,
                                    const float* __restrict__ normals, float* __restrict__ bounds,
                                    float* __restrict__ plane_point, float* __restrict__ plane_normal) {
    const int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= group_count) {
        return;
    }
    for (int axis = 0; axis < 6; ++axis) {
        bounds[group * 6 + axis] = ordered_to_float(ordered[group * 6 + axis]);
    }
    const int64_t face = representative_faces[group];
    for (int axis = 0; axis < 3; ++axis) {
        plane_point[group * 3 + axis] = tri_a[face * 3 + axis];
        plane_normal[group * 3 + axis] = normals[face * 3 + axis];
    }
}

__global__ void group_occupancy_kernel(int64_t face_count, const float* __restrict__ vertices,
                                       const int* __restrict__ faces, const int* __restrict__ face_group_id,
                                       int64_t group_count, const float* __restrict__ bounds,
                                       const float* __restrict__ plane_normal, float tolerance,
                                       int* __restrict__ occupancy) {
    const int64_t face = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (face >= face_count) {
        return;
    }
    const int group = face_group_id[face];
    if (group < 0 || group >= group_count) {
        return;
    }
    float lo[3] = {INFINITY, INFINITY, INFINITY};
    float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int corner = 0; corner < 3; ++corner) {
        const int vertex = faces[face * 3 + corner];
        for (int axis = 0; axis < 3; ++axis) {
            const float value = vertices[static_cast<int64_t>(vertex) * 3 + axis];
            lo[axis] = fminf(lo[axis], value);
            hi[axis] = fmaxf(hi[axis], value);
        }
    }
    int u, v;
    plane_axes(plane_normal, group, u, v);
    const float* b = bounds + static_cast<int64_t>(group) * 6;
    int u0, u1, v0, v1;
    cell_range(lo[u] - tolerance, hi[u] + tolerance, b[u], b[3 + u], u0, u1);
    cell_range(lo[v] - tolerance, hi[v] + tolerance, b[v], b[3 + v], v0, v1);
    if (u0 > u1) {
        return;
    }
    const unsigned int mask = (u1 - u0 + 1 >= 32 ? 0xffffffffu : ((1u << (u1 - u0 + 1)) - 1u)) << u0;
    int* rows = occupancy + static_cast<int64_t>(group) * kOccupancyCells;
    for (int row = v0; row <= v1; ++row) {
        atomicOr(rows + row, static_cast<int>(mask));
    }
}

// The occupancy grid read as kMacroCells x kMacroCells macro cells; a macro cell is occupied when
// any of its cells is. Its box spans the macro cell in-plane and the group's full extent along
// the dropped axis.
constexpr int kMacroCells = 4;
constexpr int kCellsPerMacro = kOccupancyCells / kMacroCells;

__device__ bool macro_box(const GroupTables& tables, int group, int index, float3& lo, float3& hi) {
    const int mi = index % kMacroCells;
    const int mj = index / kMacroCells;
    const int* rows = tables.occupancy + static_cast<int64_t>(group) * kOccupancyCells;
    const unsigned int mask = ((1u << kCellsPerMacro) - 1u) << (mi * kCellsPerMacro);
    bool any = false;
    for (int row = mj * kCellsPerMacro; row < (mj + 1) * kCellsPerMacro && !any; ++row) {
        any = (static_cast<unsigned int>(rows[row]) & mask) != 0u;
    }
    if (!any) {
        return false;
    }
    int u, v;
    plane_axes(tables.plane_normal, group, u, v);
    const float* b = tables.bounds + static_cast<int64_t>(group) * 6;
    float l[3] = {b[0], b[1], b[2]};
    float h[3] = {b[3], b[4], b[5]};
    const float du = (b[3 + u] - b[u]) / kMacroCells;
    const float dv = (b[3 + v] - b[v]) / kMacroCells;
    l[u] = b[u] + du * mi;
    h[u] = mi == kMacroCells - 1 ? b[3 + u] : b[u] + du * (mi + 1);
    l[v] = b[v] + dv * mj;
    h[v] = mj == kMacroCells - 1 ? b[3 + v] : b[v] + dv * (mj + 1);
    lo = make_float3(l[0], l[1], l[2]);
    hi = make_float3(h[0], h[1], h[2]);
    return true;
}

// True unless no specular path can continue from the node (image `image`, last plane group
// `last`) onto plane group `group`. Every rejection is geometric and conservative: the rays
// leaving the image through the last polygon form a beam that lies beyond the last plane and
// inside the union of the cones from the image around the bounding spheres of the last group's
// occupied macro cells, so a group none of whose occupied macro cells reaches beyond the last
// plane and into one of those cones cannot host the next reflection point.
__device__ bool image_tree_keep(const GroupTables& tables, int last, float3 image, int group, float tolerance) {
    if (group == last) {
        return false;
    }
    if (last < 0) {
        return true;
    }
    const float3 last_point = cmath::load_vec3(tables.plane_point, last);
    const float3 last_normal = cmath::load_vec3(tables.plane_normal, last);
    const float image_side = cmath::dot(cmath::sub(image, last_point), last_normal);
    if (fabsf(image_side) <= tolerance) {
        return true;
    }
    const float sign = image_side > 0.0f ? 1.0f : -1.0f;
    for (int a = 0; a < kMacroCells * kMacroCells; ++a) {
        float3 alo, ahi;
        if (!macro_box(tables, last, a, alo, ahi)) {
            continue;
        }
        const float3 axis = cmath::sub(cmath::scale(cmath::add(alo, ahi), 0.5f), image);
        const float axis_length = cmath::length(axis);
        const float cone_radius = 0.5f * cmath::length(cmath::sub(ahi, alo)) + tolerance;
        const bool everywhere = axis_length <= cone_radius;
        const float cone = everywhere ? 0.0f : asinf(fminf(1.0f, cone_radius / axis_length));
        for (int c = 0; c < kMacroCells * kMacroCells; ++c) {
            float3 lo, hi;
            if (!macro_box(tables, group, c, lo, hi)) {
                continue;
            }
            bool beyond = false;
            for (int corner = 0; corner < 8 && !beyond; ++corner) {
                const float3 p = make_float3((corner & 1) ? hi.x : lo.x, (corner & 2) ? hi.y : lo.y,
                                             (corner & 4) ? hi.z : lo.z);
                beyond = sign * cmath::dot(cmath::sub(p, last_point), last_normal) < tolerance;
            }
            if (!beyond) {
                continue;
            }
            if (everywhere) {
                return true;
            }
            const float3 offset = cmath::sub(cmath::scale(cmath::add(lo, hi), 0.5f), image);
            const float offset_length = cmath::length(offset);
            const float radius = 0.5f * cmath::length(cmath::sub(hi, lo)) + tolerance;
            if (offset_length <= radius) {
                return true;
            }
            const float spread = asinf(fminf(1.0f, radius / offset_length));
            const float cosine = fmaxf(-1.0f, fminf(1.0f, cmath::dot(axis, offset) / (axis_length * offset_length)));
            if (acosf(cosine) <= cone + spread + 1.0e-3f) {
                return true;
            }
        }
    }
    return false;
}

__global__ void image_tree_flags_kernel(int64_t pair_count, int64_t node_start, int64_t width,
                                        const int* __restrict__ node_sequences, const float* __restrict__ node_images,
                                        GroupTables tables, float tolerance, bool prune, int* __restrict__ flags) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair < pair_count;
         pair += stride) {
        const int64_t node = node_start + pair / tables.group_count;
        const int group = static_cast<int>(pair % tables.group_count);
        const int last = width > 0 ? node_sequences[node * width + width - 1] : -1;
        bool keep;
        if (prune) {
            keep = image_tree_keep(tables, last, cmath::load_vec3(node_images, node), group, tolerance);
        } else {
            keep = group != last;
        }
        flags[pair] = keep ? 1 : 0;
    }
}

__global__ void image_tree_write_kernel(int64_t pair_count, int64_t node_start, int64_t width,
                                        const int* __restrict__ node_tx, const int* __restrict__ node_sequences,
                                        const float* __restrict__ node_images, GroupTables tables,
                                        const int* __restrict__ flags, const int* __restrict__ offsets,
                                        int* __restrict__ child_tx, int* __restrict__ child_sequences,
                                        float* __restrict__ child_images) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair < pair_count;
         pair += stride) {
        if (flags[pair] == 0) {
            continue;
        }
        const int64_t node = node_start + pair / tables.group_count;
        const int group = static_cast<int>(pair % tables.group_count);
        const int64_t dst = offsets[pair];
        child_tx[dst] = node_tx[node];
        for (int64_t column = 0; column < width; ++column) {
            child_sequences[dst * (width + 1) + column] = node_sequences[node * width + column];
        }
        child_sequences[dst * (width + 1) + width] = group;
        const float3 image =
            mirror(cmath::load_vec3(node_images, node), cmath::load_vec3(tables.plane_point, group), cmath::load_vec3(tables.plane_normal, group));
        child_images[dst * 3] = image.x;
        child_images[dst * 3 + 1] = image.y;
        child_images[dst * 3 + 2] = image.z;
    }
}

struct PairTolerances {
    float box;          // bounding-box inflation, metres
    float segment_t;    // slack on the back-trace intersection parameter
    float min_segment;  // segments shorter than this cannot be valid, metres
};

// One sequence's mirrored chain, shared by the threads that test it against a receiver tile.
struct SequenceChain {
    float3 images[kDiscoveryMaxDepth + 1];
    float3 plane_point[kDiscoveryMaxDepth];
    float3 plane_normal[kDiscoveryMaxDepth];
    int groups[kDiscoveryMaxDepth];
};

// Thread 0 mirrors the transmitter through the sequence's planes; the caller synchronizes.
__device__ void load_chain(const GroupTables& tables, const int* sequence, int depth, float3 tx,
                           SequenceChain& chain) {
    if (threadIdx.x != 0) {
        return;
    }
    chain.images[0] = tx;
    for (int bounce = 0; bounce < depth; ++bounce) {
        const int group = sequence[bounce];
        chain.groups[bounce] = group;
        chain.plane_point[bounce] = cmath::load_vec3(tables.plane_point, group);
        chain.plane_normal[bounce] = cmath::load_vec3(tables.plane_normal, group);
        chain.images[bounce + 1] = mirror(chain.images[bounce], chain.plane_point[bounce], chain.plane_normal[bounce]);
    }
}

// True unless the image method proves the chain cannot connect its transmitter to `rx`: some
// back-traced reflection point misses its segment, falls in no occupied cell of its group, or
// leaves a segment shorter than the validity threshold. Conservative relative to the EPC solve.
__device__ bool receiver_candidate(const GroupTables& tables, const SequenceChain& chain, int depth, float3 rx,
                                   const PairTolerances& tol) {
    float3 endpoint = rx;
    for (int bounce = depth - 1; bounce >= 0; --bounce) {
        const int group = chain.groups[bounce];
        const float3 start = chain.images[bounce + 1];
        const float3 normal = chain.plane_normal[bounce];
        const float3 span = cmath::sub(endpoint, start);
        const float denominator = cmath::dot(span, normal);
        float3 point = endpoint;
        if (fabsf(denominator) > 1.0e-12f) {
            const float t = cmath::dot(cmath::sub(chain.plane_point[bounce], start), normal) / denominator;
            if (t < -tol.segment_t || t > 1.0f + tol.segment_t) {
                return false;
            }
            point = cmath::add(start, cmath::scale(span, t));
        }
        if (!inside_bounds(tables, group, point, tol.box) || !occupied(tables, group, point, tol.box)) {
            return false;
        }
        if (cmath::length(cmath::sub(endpoint, point)) < tol.min_segment) {
            return false;
        }
        endpoint = point;
    }
    return cmath::length(cmath::sub(endpoint, chain.images[0])) >= tol.min_segment;
}

// One block covers one (sequence, receiver tile); tile t of sequence s is block s * tiles + t.
__global__ void receiver_candidate_count_kernel(int64_t tile_total, int64_t tiles, int64_t sequence_start,
                                                int64_t depth, int64_t rx_count, const int* __restrict__ sequences,
                                                const int* __restrict__ sequence_tx,
                                                const float* __restrict__ tx_positions,
                                                const float* __restrict__ rx_positions, GroupTables tables,
                                                PairTolerances tol, int* __restrict__ tile_counts) {
    __shared__ SequenceChain chain;
    for (int64_t tile = blockIdx.x; tile < tile_total; tile += gridDim.x) {
        const int64_t sequence = sequence_start + tile / tiles;
        const int64_t rx = (tile % tiles) * kDiscoveryBlock + threadIdx.x;
        __syncthreads();
        load_chain(tables, sequences + sequence * depth, static_cast<int>(depth),
                   cmath::load_vec3(tx_positions, sequence_tx[sequence]), chain);
        __syncthreads();
        const bool keep = rx < rx_count &&
                          receiver_candidate(tables, chain, static_cast<int>(depth), cmath::load_vec3(rx_positions, rx), tol);
        const int count = __syncthreads_count(keep ? 1 : 0);
        if (threadIdx.x == 0) {
            tile_counts[tile] = count;
        }
    }
}

__global__ void receiver_candidate_write_kernel(int64_t tile_total, int64_t tiles, int64_t sequence_start,
                                                int64_t depth, int64_t rx_count, const int* __restrict__ sequences,
                                                const int* __restrict__ sequence_tx,
                                                const float* __restrict__ tx_positions,
                                                const float* __restrict__ rx_positions, GroupTables tables,
                                                PairTolerances tol, const int* __restrict__ tile_offsets,
                                                int* __restrict__ candidate_sequence,
                                                int* __restrict__ candidate_receiver) {
    using BlockScan = cub::BlockScan<int, kDiscoveryBlock>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ SequenceChain chain;
    for (int64_t tile = blockIdx.x; tile < tile_total; tile += gridDim.x) {
        const int64_t sequence = sequence_start + tile / tiles;
        const int64_t rx = (tile % tiles) * kDiscoveryBlock + threadIdx.x;
        __syncthreads();
        load_chain(tables, sequences + sequence * depth, static_cast<int>(depth),
                   cmath::load_vec3(tx_positions, sequence_tx[sequence]), chain);
        __syncthreads();
        const bool keep = rx < rx_count &&
                          receiver_candidate(tables, chain, static_cast<int>(depth), cmath::load_vec3(rx_positions, rx), tol);
        int rank = 0;
        BlockScan(scan_storage).ExclusiveSum(keep ? 1 : 0, rank);
        if (keep) {
            const int64_t dst = static_cast<int64_t>(tile_offsets[tile]) + rank;
            candidate_sequence[dst] = static_cast<int>(sequence);
            candidate_receiver[dst] = static_cast<int>(rx);
        }
    }
}

// ---------------------------------------------------------------------------
// Interaction chains (reflections, one diffraction, reflections). A chain is
// the join of a transmitter-side half (an image-tree node or the transmitter,
// with an edge in its beam) and a receiver-side half (the same from the
// receiver) on a shared edge. The diffraction point is the Keller point
// between the two half images; both halves are then back-traced through their
// planes exactly as the reflection receiver test does.
// ---------------------------------------------------------------------------

// Edge pieces tested against a beam; a long edge is split so each piece's bounding sphere stays tight.
constexpr int kChainEdgePieces = 8;

struct ChainEdges {
    const float* position;   // [E, 3]
    const float* direction;  // [E, 3]
    const float* t_min;      // [E]
    const float* t_max;      // [E]
    int64_t edge_count;
    const float* n0 = nullptr;  // [E, 3] outward face normals of the wedge; null when unused
    const float* n1 = nullptr;  // [E, 3]
};

// Segment start, unit direction and length of one edge, as the coupled prepare kernels form them.
struct EdgeFrame {
    float3 origin;
    float3 direction;
    float length;
};

__device__ __forceinline__ EdgeFrame edge_frame(const ChainEdges& edges, int edge) {
    EdgeFrame frame;
    frame.direction = cmath::normalize_or_zero(cmath::load_vec3(edges.direction, edge), 1.0e-6f);
    frame.origin = cmath::add(cmath::load_vec3(edges.position, edge), cmath::scale(frame.direction, edges.t_min[edge]));
    frame.length = edges.t_max[edge] - edges.t_min[edge];
    return frame;
}

__device__ __forceinline__ float3 frame_point(const EdgeFrame& frame, float t) {
    return cmath::add(frame.origin, cmath::scale(frame.direction, t));
}

// True unless no point of the edge can lie in the beam of the node (image `image`, last plane
// group `last`): the beam lies beyond the last plane and inside the cones from the image around
// the bounding spheres of the last group's occupied macro cells. Conservative, like image_tree_keep.
__device__ bool chain_edge_keep(const GroupTables& tables, int last, float3 image, const ChainEdges& edges, int edge,
                                float tolerance) {
    const EdgeFrame frame = edge_frame(edges, edge);
    const float length = frame.length;
    if (!(isfinite(length) && length > 0.0f)) {
        return false;
    }
    const float3 a = frame_point(frame, 0.0f);
    const float3 b = frame_point(frame, length);
    const float3 last_point = cmath::load_vec3(tables.plane_point, last);
    const float3 last_normal = cmath::load_vec3(tables.plane_normal, last);
    const float image_side = cmath::dot(cmath::sub(image, last_point), last_normal);
    if (fabsf(image_side) <= tolerance) {
        return true;
    }
    const float sign = image_side > 0.0f ? 1.0f : -1.0f;
    const bool beyond = sign * cmath::dot(cmath::sub(a, last_point), last_normal) < tolerance ||
                        sign * cmath::dot(cmath::sub(b, last_point), last_normal) < tolerance;
    if (!beyond) {
        return false;
    }
    const float piece = length / kChainEdgePieces;
    for (int c = 0; c < kMacroCells * kMacroCells; ++c) {
        float3 alo, ahi;
        if (!macro_box(tables, last, c, alo, ahi)) {
            continue;
        }
        const float3 axis = cmath::sub(cmath::scale(cmath::add(alo, ahi), 0.5f), image);
        const float axis_length = cmath::length(axis);
        const float cone_radius = 0.5f * cmath::length(cmath::sub(ahi, alo)) + tolerance;
        if (axis_length <= cone_radius) {
            return true;
        }
        const float cone = asinf(fminf(1.0f, cone_radius / axis_length));
        for (int p = 0; p < kChainEdgePieces; ++p) {
            const float3 center = frame_point(frame, piece * (p + 0.5f));
            const float3 offset = cmath::sub(center, image);
            const float offset_length = cmath::length(offset);
            const float radius = 0.5f * piece + tolerance;
            if (offset_length <= radius) {
                return true;
            }
            const float spread = asinf(fminf(1.0f, radius / offset_length));
            const float cosine =
                fmaxf(-1.0f, fminf(1.0f, cmath::dot(axis, offset) / (axis_length * offset_length)));
            if (acosf(cosine) <= cone + spread + 1.0e-3f) {
                return true;
            }
        }
    }
    return false;
}

__global__ void chain_node_edge_flags_kernel(int64_t pair_count, int64_t node_start, int64_t width,
                                             const int* __restrict__ node_sequences,
                                             const float* __restrict__ node_images, GroupTables tables,
                                             ChainEdges edges, float tolerance, int* __restrict__ flags) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair < pair_count;
         pair += stride) {
        const int64_t node = node_start + pair / edges.edge_count;
        const int edge = static_cast<int>(pair % edges.edge_count);
        const int last = node_sequences[node * width + width - 1];
        flags[pair] = chain_edge_keep(tables, last, cmath::load_vec3(node_images, node), edges, edge, tolerance) ? 1 : 0;
    }
}

__global__ void chain_node_edge_write_kernel(int64_t pair_count, int64_t node_start, int64_t edge_count,
                                             const int* __restrict__ flags, const int* __restrict__ offsets,
                                             int* __restrict__ out_node, int* __restrict__ out_edge) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair < pair_count;
         pair += stride) {
        if (flags[pair] == 0) {
            continue;
        }
        const int64_t dst = offsets[pair];
        out_node[dst] = static_cast<int>(node_start + pair / edge_count);
        out_edge[dst] = static_cast<int>(pair % edge_count);
    }
}

// One half of a chain: per-edge lists (CSR over edges) of entries. An entry is an endpoint
// (transmitter or receiver index) and, for a half with reflections, its image-tree node.
struct ChainHalf {
    const int64_t* edge_offsets;  // [E + 1]
    const int* endpoint;          // [K]
    const int* node;              // [K]; null when the half has no reflection
    const float* endpoints;       // [endpoint count, 3]
    const int* node_sequences;    // [nodes, depth]
    const float* node_images;     // [nodes, 3]
    int depth;
};

__device__ __forceinline__ void half_entry(const ChainHalf& half, int64_t entry, int& endpoint, int& node,
                                           float3& image) {
    endpoint = half.endpoint[entry];
    node = half.node != nullptr ? half.node[entry] : -1;
    image = node >= 0 ? cmath::load_vec3(half.node_images, node) : cmath::load_vec3(half.endpoints, endpoint);
}

// Per-thread mirrored chain of one half (the shared-memory loader serves whole blocks only).
__device__ void half_chain(const GroupTables& tables, const ChainHalf& half, int node, int endpoint,
                           SequenceChain& chain) {
    chain.images[0] = cmath::load_vec3(half.endpoints, endpoint);
    for (int bounce = 0; bounce < half.depth; ++bounce) {
        const int group = half.node_sequences[static_cast<int64_t>(node) * half.depth + bounce];
        chain.groups[bounce] = group;
        chain.plane_point[bounce] = cmath::load_vec3(tables.plane_point, group);
        chain.plane_normal[bounce] = cmath::load_vec3(tables.plane_normal, group);
        chain.images[bounce + 1] = mirror(chain.images[bounce], chain.plane_point[bounce], chain.plane_normal[bounce]);
    }
}

struct ChainLane {
    int edge;
    int64_t a_entry;
    int64_t b_entry;
};

// Lane -> (edge, transmitter-side entry, receiver-side entry) over the per-edge products.
__device__ __forceinline__ ChainLane chain_lane(const int64_t* lane_offsets, int64_t edge_count, const ChainHalf& b,
                                                const ChainHalf& a, int64_t lane) {
    int64_t lo = 0;
    int64_t hi = edge_count;
    while (hi - lo > 1) {
        const int64_t mid = (lo + hi) / 2;
        if (lane_offsets[mid] <= lane) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int edge = static_cast<int>(lo);
    const int64_t local = lane - lane_offsets[edge];
    const int64_t b_count = b.edge_offsets[edge + 1] - b.edge_offsets[edge];
    return {edge, a.edge_offsets[edge] + local / b_count, b.edge_offsets[edge] + local % b_count};
}

struct ChainCandidate {
    bool keep;
    int tx;
    int rx;
    int a_node;
    int b_node;
    float3 point;
};

__device__ ChainCandidate chain_candidate(const GroupTables& tables, const ChainEdges& edges, const ChainHalf& a,
                                          const ChainHalf& b, const ChainLane& lane, const PairTolerances& tol,
                                          float edge_epsilon) {
    ChainCandidate out{};
    float3 a_image, b_image;
    half_entry(a, lane.a_entry, out.tx, out.a_node, a_image);
    half_entry(b, lane.b_entry, out.rx, out.b_node, b_image);
    const EdgeFrame frame = edge_frame(edges, lane.edge);
    const float3 origin = frame.origin;
    const float3 direction = frame.direction;
    const float length = frame.length;
    const float parameter = rayd::shared::diffraction::first_order_diffraction_parameter(
        rayd::shared::diffraction::make_f3(a_image.x, a_image.y, a_image.z),
        rayd::shared::diffraction::make_f3(b_image.x, b_image.y, b_image.z),
        rayd::shared::diffraction::make_f3(origin.x, origin.y, origin.z),
        rayd::shared::diffraction::make_f3(direction.x, direction.y, direction.z));
    if (!(isfinite(length) && length > edge_epsilon && isfinite(parameter) && parameter > edge_epsilon &&
          parameter < length - edge_epsilon)) {
        return out;
    }
    out.point = cmath::add(origin, cmath::scale(direction, parameter));
    // Both legs leave the diffraction point into the open region around the wedge. The leg towards
    // a half points at that half's image, so a chain whose leg would cross the solid behind the edge
    // (a reflection reached from inside a closed body) is rejected here.
    {
        namespace utd = rayd::shared::diffraction;
        const utd::float3a axis = utd::make_f3(direction.x, direction.y, direction.z);
        const float3 n0 = cmath::load_vec3(edges.n0, lane.edge);
        const float3 n1 = cmath::load_vec3(edges.n1, lane.edge);
        const utd::float3a wn0 = utd::make_f3(n0.x, n0.y, n0.z);
        const utd::float3a wn1 = utd::make_f3(n1.x, n1.y, n1.z);
        const float3 to_a = cmath::sub(a_image, out.point);
        const float3 to_b = cmath::sub(b_image, out.point);
        if (!utd::wedge_exterior_mask(utd::make_f3(to_a.x, to_a.y, to_a.z), axis, wn0, wn1) ||
            !utd::wedge_exterior_mask(utd::make_f3(to_b.x, to_b.y, to_b.z), axis, wn0, wn1)) {
            return out;
        }
    }
    SequenceChain chain;
    if (a.depth > 0) {
        half_chain(tables, a, out.a_node, out.tx, chain);
        if (!receiver_candidate(tables, chain, a.depth, out.point, tol)) {
            return out;
        }
    } else if (cmath::length(cmath::sub(out.point, a_image)) < tol.min_segment) {
        return out;
    }
    if (b.depth > 0) {
        half_chain(tables, b, out.b_node, out.rx, chain);
        if (!receiver_candidate(tables, chain, b.depth, out.point, tol)) {
            return out;
        }
    } else if (cmath::length(cmath::sub(out.point, b_image)) < tol.min_segment) {
        return out;
    }
    out.keep = true;
    return out;
}

__global__ void chain_candidate_flags_kernel(int64_t lane_start, int64_t lane_count, const int64_t* lane_offsets,
                                             GroupTables tables, ChainEdges edges, ChainHalf a, ChainHalf b,
                                             PairTolerances tol, float edge_epsilon, int* __restrict__ flags) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < lane_count;
         index += stride) {
        const ChainLane lane = chain_lane(lane_offsets, edges.edge_count, b, a, lane_start + index);
        flags[index] = chain_candidate(tables, edges, a, b, lane, tol, edge_epsilon).keep ? 1 : 0;
    }
}

__global__ void chain_candidate_write_kernel(int64_t lane_start, int64_t lane_count, const int64_t* lane_offsets,
                                             GroupTables tables, ChainEdges edges, ChainHalf a, ChainHalf b,
                                             PairTolerances tol, float edge_epsilon, const int* __restrict__ flags,
                                             const int* __restrict__ offsets, int* __restrict__ out_tx,
                                             int* __restrict__ out_rx, int* __restrict__ out_edge,
                                             int* __restrict__ out_a_node, int* __restrict__ out_b_node,
                                             float* __restrict__ out_point) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < lane_count;
         index += stride) {
        if (flags[index] == 0) {
            continue;
        }
        const ChainLane lane = chain_lane(lane_offsets, edges.edge_count, b, a, lane_start + index);
        const ChainCandidate candidate = chain_candidate(tables, edges, a, b, lane, tol, edge_epsilon);
        const int64_t dst = offsets[index];
        out_tx[dst] = candidate.tx;
        out_rx[dst] = candidate.rx;
        out_edge[dst] = lane.edge;
        out_a_node[dst] = candidate.a_node;
        out_b_node[dst] = candidate.b_node;
        out_point[dst * 3] = candidate.point.x;
        out_point[dst * 3 + 1] = candidate.point.y;
        out_point[dst * 3 + 2] = candidate.point.z;
    }
}

// Double diffraction: lane -> (transmitter, receiver, first edge, second edge) over the products
// of each transmitter's visible edges with each receiver's visible edges.
__global__ void chain_double_expand_kernel(int64_t lane_start, int64_t lane_count, int64_t rx_count,
                                           const int64_t* __restrict__ pair_offsets,
                                           const int64_t* __restrict__ tx_offsets, const int* __restrict__ tx_edges,
                                           const int64_t* __restrict__ rx_offsets, const int* __restrict__ rx_edges,
                                           int64_t pair_count, int* __restrict__ out_tx, int* __restrict__ out_rx,
                                           int* __restrict__ out_edge1, int* __restrict__ out_edge2) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < lane_count;
         index += stride) {
        const int64_t lane = lane_start + index;
        int64_t lo = 0;
        int64_t hi = pair_count;
        while (hi - lo > 1) {
            const int64_t mid = (lo + hi) / 2;
            if (pair_offsets[mid] <= lane) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const int64_t tx = lo / rx_count;
        const int64_t rx = lo % rx_count;
        const int64_t local = lane - pair_offsets[lo];
        const int64_t rx_visible = rx_offsets[rx + 1] - rx_offsets[rx];
        out_tx[index] = static_cast<int>(tx);
        out_rx[index] = static_cast<int>(rx);
        out_edge1[index] = tx_edges[tx_offsets[tx] + local / rx_visible];
        out_edge2[index] = rx_edges[rx_offsets[rx] + local % rx_visible];
    }
}

int discovery_grid(int64_t work) {
    const int64_t blocks = (work + kDiscoveryBlock - 1) / kDiscoveryBlock;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(blocks, 65535)));
}

GroupTables group_tables(const at::Tensor& plane_point, const at::Tensor& plane_normal, const at::Tensor& bounds,
                         const at::Tensor& occupancy) {
    check_vec3_table(plane_point, "plane_point");
    check_vec3_table(plane_normal, "plane_normal");
    check_tensor(bounds, "bounds", at::kFloat, 2);
    check_tensor(occupancy, "occupancy", at::kInt, 2);
    TORCH_CHECK(bounds.size(1) == 6, "bounds must have shape (G, 6)");
    TORCH_CHECK(occupancy.size(1) == kOccupancyCells, "occupancy must have one bit row per cell row");
    TORCH_CHECK(plane_normal.size(0) == plane_point.size(0) && bounds.size(0) == plane_point.size(0) &&
                    occupancy.size(0) == plane_point.size(0),
                "plane group tables must have one row per group");
    TORCH_CHECK(plane_point.size(0) > 0, "plane group tables must hold at least one group");
    return GroupTables{plane_point.data_ptr<float>(), plane_normal.data_ptr<float>(), bounds.data_ptr<float>(),
                       occupancy.data_ptr<int>(), plane_point.size(0)};
}

}  // namespace

pybind11::dict channel_reflection_group_tables(at::Tensor vertices, at::Tensor faces, at::Tensor face_group_id,
                                               at::Tensor representative_faces, at::Tensor tri_a, at::Tensor normals,
                                               double tolerance) {
    TORCH_CHECK(std::isfinite(tolerance) && tolerance >= 0.0, "tolerance must be finite and non-negative");
    check_vec3_table(vertices, "vertices");
    check_tensor(faces, "faces", at::kInt, 2);
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    check_tensor(face_group_id, "face_group_id", at::kInt, 1);
    check_tensor(representative_faces, "representative_faces", at::kLong, 1);
    check_vec3_table(tri_a, "tri_a");
    check_vec3_table(normals, "normals");
    TORCH_CHECK(face_group_id.size(0) == faces.size(0), "face_group_id must match faces");
    TORCH_CHECK(tri_a.size(0) == faces.size(0) && normals.size(0) == faces.size(0),
                "tri_a and normals must match faces");
    const int device = vertices.get_device();
    for (const at::Tensor* tensor : {&faces, &face_group_id, &representative_faces, &tri_a, &normals}) {
        TORCH_CHECK(tensor->get_device() == device, "plane group inputs must share one device");
    }
    const int64_t group_count = representative_faces.size(0);
    const int64_t face_count = faces.size(0);
    auto float_options = vertices.options().dtype(at::kFloat);
    pybind11::dict out;
    out["plane_point"] = at::empty({group_count, 3}, float_options);
    out["plane_normal"] = at::empty({group_count, 3}, float_options);
    out["bounds"] = at::empty({group_count, 6}, float_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    out["occupancy"] = channel::empty_zero_cuda({group_count, kOccupancyCells}, vertices.options().dtype(at::kInt), stream);
    if (group_count == 0) {
        return out;
    }
    auto ordered = at::empty({group_count, 6}, vertices.options().dtype(at::kInt));
    group_bounds_init_kernel<<<::channel::launch_blocks(group_count * 6, kPathBlockSize),
                               kPathBlockSize, 0, stream>>>(
        group_count, ordered.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (face_count > 0) {
        group_bounds_accumulate_kernel<<<::channel::launch_blocks(face_count, kPathBlockSize),
                                         kPathBlockSize, 0, stream>>>(
            face_count, vertices.data_ptr<float>(), faces.data_ptr<int>(), face_group_id.data_ptr<int>(), group_count,
            ordered.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    group_planes_kernel<<<::channel::launch_blocks(group_count, kPathBlockSize),
                          kPathBlockSize, 0, stream>>>(
        group_count, ordered.data_ptr<int>(), representative_faces.data_ptr<int64_t>(), tri_a.data_ptr<float>(),
        normals.data_ptr<float>(), pybind11::cast<at::Tensor>(out["bounds"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["plane_point"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["plane_normal"]).data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (face_count > 0) {
        group_occupancy_kernel<<<::channel::launch_blocks(face_count, kPathBlockSize),
                                 kPathBlockSize, 0, stream>>>(
            face_count, vertices.data_ptr<float>(), faces.data_ptr<int>(), face_group_id.data_ptr<int>(), group_count,
            pybind11::cast<at::Tensor>(out["bounds"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["plane_normal"]).data_ptr<float>(), static_cast<float>(tolerance),
            pybind11::cast<at::Tensor>(out["occupancy"]).data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

pybind11::dict channel_reflection_image_tree_expand(at::Tensor node_tx, at::Tensor node_sequences,
                                                    at::Tensor node_images, at::Tensor plane_point,
                                                    at::Tensor plane_normal, at::Tensor bounds,
                                                    at::Tensor occupancy, int64_t node_start,
                                                    int64_t node_end, double tolerance, bool prune) {
    check_tensor(node_tx, "node_tx", at::kInt, 1);
    check_tensor(node_sequences, "node_sequences", at::kInt, 2);
    check_vec3_table(node_images, "node_images");
    const GroupTables tables = group_tables(plane_point, plane_normal, bounds, occupancy);
    const int64_t node_count = node_tx.size(0);
    const int64_t width = node_sequences.size(1);
    TORCH_CHECK(node_sequences.size(0) == node_count && node_images.size(0) == node_count,
                "node tables must have one row per node");
    TORCH_CHECK(width < kDiscoveryMaxDepth, "image tree depth exceeds the specular chain limit");
    TORCH_CHECK(0 <= node_start && node_start <= node_end && node_end <= node_count, "node range is out of range");
    TORCH_CHECK(std::isfinite(tolerance) && tolerance >= 0.0, "tolerance must be finite and non-negative");
    const int device = node_tx.get_device();
    for (const at::Tensor* tensor : {&node_sequences, &node_images, &plane_point, &plane_normal, &bounds, &occupancy}) {
        TORCH_CHECK(tensor->get_device() == device, "image tree inputs must share one device");
    }
    const int64_t pair_count = (node_end - node_start) * tables.group_count;
    TORCH_CHECK(pair_count < std::numeric_limits<int>::max(), "image tree expansion chunk exceeds the int32 range");
    auto int_options = node_tx.options();
    auto float_options = node_images.options();
    auto allocate = [&](int64_t rows) {
        pybind11::dict out;
        out["node_tx"] = at::empty({rows}, int_options);
        out["node_sequences"] = at::empty({rows, width + 1}, int_options);
        out["node_images"] = at::empty({rows, 3}, float_options);
        return out;
    };
    if (pair_count == 0) {
        return allocate(0);
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    auto flags = at::empty({pair_count}, int_options);
    auto offsets = at::empty({pair_count + 1}, int_options);
    image_tree_flags_kernel<<<discovery_grid(pair_count), kDiscoveryBlock, 0, stream>>>(
        pair_count, node_start, width, node_sequences.data_ptr<int>(), node_images.data_ptr<float>(), tables,
        static_cast<float>(tolerance), prune, flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(flags, offsets, pair_count, stream);
    const int64_t out_count = observe_compact_count(offsets, pair_count, stream).count;
    pybind11::dict out = allocate(out_count);
    if (out_count == 0) {
        return out;
    }
    image_tree_write_kernel<<<discovery_grid(pair_count), kDiscoveryBlock, 0, stream>>>(
        pair_count, node_start, width, node_tx.data_ptr<int>(), node_sequences.data_ptr<int>(),
        node_images.data_ptr<float>(), tables, flags.data_ptr<int>(), offsets.data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["node_tx"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["node_sequences"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["node_images"]).data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

pybind11::dict channel_reflection_receiver_candidates(at::Tensor tx_positions, at::Tensor rx_positions,
                                                      at::Tensor sequences, at::Tensor sequence_tx,
                                                      at::Tensor plane_point, at::Tensor plane_normal,
                                                      at::Tensor bounds, at::Tensor occupancy,
                                                      int64_t sequence_start,
                                                      int64_t sequence_end, double box_tolerance,
                                                      double segment_tolerance, double min_segment_length) {
    check_vec3_table(tx_positions, "tx_positions");
    check_vec3_table(rx_positions, "rx_positions");
    check_tensor(sequences, "sequences", at::kInt, 2);
    check_tensor(sequence_tx, "sequence_tx", at::kInt, 1);
    const GroupTables tables = group_tables(plane_point, plane_normal, bounds, occupancy);
    const int64_t sequence_count = sequences.size(0);
    const int64_t depth = sequences.size(1);
    const int64_t rx_count = rx_positions.size(0);
    TORCH_CHECK(depth >= 1 && depth <= kDiscoveryMaxDepth, "sequence depth exceeds the specular chain limit");
    TORCH_CHECK(sequence_tx.size(0) == sequence_count, "sequence_tx must match sequences");
    TORCH_CHECK(0 <= sequence_start && sequence_start <= sequence_end && sequence_end <= sequence_count,
                "sequence range is out of range");
    TORCH_CHECK(std::isfinite(box_tolerance) && box_tolerance >= 0.0 && std::isfinite(segment_tolerance) &&
                    segment_tolerance >= 0.0 && std::isfinite(min_segment_length) && min_segment_length >= 0.0,
                "receiver candidate tolerances must be finite and non-negative");
    const int device = tx_positions.get_device();
    for (const at::Tensor* tensor :
         {&rx_positions, &sequences, &sequence_tx, &plane_point, &plane_normal, &bounds, &occupancy}) {
        TORCH_CHECK(tensor->get_device() == device, "receiver candidate inputs must share one device");
    }
    const int64_t tiles = (rx_count + kDiscoveryBlock - 1) / kDiscoveryBlock;
    const int64_t tile_total = (sequence_end - sequence_start) * tiles;
    TORCH_CHECK(tile_total < std::numeric_limits<int>::max() &&
                    (sequence_end - sequence_start) * rx_count < std::numeric_limits<int>::max(),
                "receiver candidate chunk exceeds the int32 range");
    auto int_options = sequence_tx.options();
    pybind11::dict out;
    if (tile_total == 0) {
        out["candidate_sequence"] = at::empty({0}, int_options);
        out["candidate_receiver"] = at::empty({0}, int_options);
        return out;
    }
    const PairTolerances tol{static_cast<float>(box_tolerance), static_cast<float>(segment_tolerance),
                             static_cast<float>(min_segment_length)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    auto tile_counts = at::empty({tile_total}, int_options);
    auto tile_offsets = at::empty({tile_total + 1}, int_options);
    const int grid = static_cast<int>(std::min<int64_t>(tile_total, 65535));
    receiver_candidate_count_kernel<<<grid, kDiscoveryBlock, 0, stream>>>(
        tile_total, tiles, sequence_start, depth, rx_count, sequences.data_ptr<int>(), sequence_tx.data_ptr<int>(),
        tx_positions.data_ptr<float>(), rx_positions.data_ptr<float>(), tables, tol, tile_counts.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(tile_counts, tile_offsets, tile_total, stream);
    const int64_t count = observe_compact_count(tile_offsets, tile_total, stream).count;
    out["candidate_sequence"] = at::empty({count}, int_options);
    out["candidate_receiver"] = at::empty({count}, int_options);
    if (count == 0) {
        return out;
    }
    receiver_candidate_write_kernel<<<grid, kDiscoveryBlock, 0, stream>>>(
        tile_total, tiles, sequence_start, depth, rx_count, sequences.data_ptr<int>(), sequence_tx.data_ptr<int>(),
        tx_positions.data_ptr<float>(), rx_positions.data_ptr<float>(), tables, tol, tile_offsets.data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["candidate_sequence"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["candidate_receiver"]).data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

namespace {

ChainEdges chain_edges(const at::Tensor& position, const at::Tensor& direction, const at::Tensor& t_min,
                       const at::Tensor& t_max) {
    check_vec3_table(position, "edge_position");
    check_vec3_table(direction, "edge_direction");
    check_tensor(t_min, "edge_t_min", at::kFloat, 1);
    check_tensor(t_max, "edge_t_max", at::kFloat, 1);
    TORCH_CHECK(direction.size(0) == position.size(0) && t_min.size(0) == position.size(0) &&
                    t_max.size(0) == position.size(0),
                "edge tables must have one row per edge");
    return ChainEdges{position.data_ptr<float>(), direction.data_ptr<float>(), t_min.data_ptr<float>(),
                      t_max.data_ptr<float>(), position.size(0)};
}

ChainHalf chain_half(const at::Tensor& edge_offsets, const at::Tensor& endpoint, const at::Tensor& node,
                     const at::Tensor& endpoints, const at::Tensor& node_sequences, const at::Tensor& node_images,
                     int64_t edge_count, const char* name) {
    check_tensor(edge_offsets, "edge_offsets", at::kLong, 1);
    check_tensor(endpoint, "endpoint", at::kInt, 1);
    check_tensor(node, "node", at::kInt, 1);
    check_vec3_table(endpoints, "endpoints");
    check_tensor(node_sequences, "node_sequences", at::kInt, 2);
    check_vec3_table(node_images, "node_images");
    TORCH_CHECK(edge_offsets.size(0) == edge_count + 1, name, " edge offsets must have one row per edge plus one");
    TORCH_CHECK(node.size(0) == endpoint.size(0), name, " node must match endpoint");
    TORCH_CHECK(node_sequences.size(0) == node_images.size(0), name, " node tables must have one row per node");
    TORCH_CHECK(node_sequences.size(1) < kDiscoveryMaxDepth, name, " depth exceeds the specular chain limit");
    return ChainHalf{edge_offsets.data_ptr<int64_t>(), endpoint.data_ptr<int>(), node.data_ptr<int>(),
                     endpoints.data_ptr<float>(), node_sequences.data_ptr<int>(), node_images.data_ptr<float>(),
                     static_cast<int>(node_sequences.size(1))};
}

}  // namespace

pybind11::dict channel_chain_node_edges(at::Tensor node_sequences, at::Tensor node_images, at::Tensor plane_point,
                                        at::Tensor plane_normal, at::Tensor bounds, at::Tensor occupancy,
                                        at::Tensor edge_position, at::Tensor edge_direction, at::Tensor edge_t_min, at::Tensor edge_t_max,
                                        int64_t node_start, int64_t node_end, double tolerance) {
    check_tensor(node_sequences, "node_sequences", at::kInt, 2);
    check_vec3_table(node_images, "node_images");
    const GroupTables tables = group_tables(plane_point, plane_normal, bounds, occupancy);
    const ChainEdges edges = chain_edges(edge_position, edge_direction, edge_t_min, edge_t_max);
    const int64_t node_count = node_sequences.size(0);
    const int64_t width = node_sequences.size(1);
    TORCH_CHECK(width >= 1 && width < kDiscoveryMaxDepth, "chain node depth must be in [1, 7]");
    TORCH_CHECK(node_images.size(0) == node_count, "node tables must have one row per node");
    TORCH_CHECK(0 <= node_start && node_start <= node_end && node_end <= node_count, "node range is out of range");
    TORCH_CHECK(std::isfinite(tolerance) && tolerance >= 0.0, "tolerance must be finite and non-negative");
    const int64_t pair_count = (node_end - node_start) * edges.edge_count;
    TORCH_CHECK(pair_count < std::numeric_limits<int>::max(), "chain node-edge chunk exceeds the int32 range");
    auto int_options = node_sequences.options();
    pybind11::dict out;
    if (pair_count == 0) {
        out["node"] = at::empty({0}, int_options);
        out["edge"] = at::empty({0}, int_options);
        return out;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(node_sequences.get_device()).stream();
    auto flags = at::empty({pair_count}, int_options);
    auto offsets = at::empty({pair_count + 1}, int_options);
    chain_node_edge_flags_kernel<<<discovery_grid(pair_count), kDiscoveryBlock, 0, stream>>>(
        pair_count, node_start, width, node_sequences.data_ptr<int>(), node_images.data_ptr<float>(), tables, edges,
        static_cast<float>(tolerance), flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(flags, offsets, pair_count, stream);
    const int64_t count = observe_compact_count(offsets, pair_count, stream).count;
    out["node"] = at::empty({count}, int_options);
    out["edge"] = at::empty({count}, int_options);
    if (count == 0) {
        return out;
    }
    chain_node_edge_write_kernel<<<discovery_grid(pair_count), kDiscoveryBlock, 0, stream>>>(
        pair_count, node_start, edges.edge_count, flags.data_ptr<int>(), offsets.data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["node"]).data_ptr<int>(), pybind11::cast<at::Tensor>(out["edge"]).data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

pybind11::dict channel_chain_candidates(
    at::Tensor lane_offsets, int64_t lane_start, int64_t lane_end, at::Tensor plane_point, at::Tensor plane_normal,
    at::Tensor bounds, at::Tensor occupancy, at::Tensor edge_position, at::Tensor edge_direction, at::Tensor edge_t_min, at::Tensor edge_t_max,
    at::Tensor edge_n0, at::Tensor edge_n1,
    at::Tensor a_edge_offsets, at::Tensor a_endpoint, at::Tensor a_node, at::Tensor a_endpoints,
    at::Tensor a_node_sequences, at::Tensor a_node_images, at::Tensor b_edge_offsets, at::Tensor b_endpoint,
    at::Tensor b_node, at::Tensor b_endpoints, at::Tensor b_node_sequences, at::Tensor b_node_images,
    double box_tolerance, double segment_tolerance, double min_segment_length, double edge_epsilon) {
    const GroupTables tables = group_tables(plane_point, plane_normal, bounds, occupancy);
    ChainEdges edges = chain_edges(edge_position, edge_direction, edge_t_min, edge_t_max);
    check_vec3_table(edge_n0, "edge_n0");
    check_vec3_table(edge_n1, "edge_n1");
    TORCH_CHECK(edge_n0.size(0) == edges.edge_count && edge_n1.size(0) == edges.edge_count,
                "edge normal tables must have one row per edge");
    edges.n0 = edge_n0.data_ptr<float>();
    edges.n1 = edge_n1.data_ptr<float>();
    check_tensor(lane_offsets, "lane_offsets", at::kLong, 1);
    TORCH_CHECK(lane_offsets.size(0) == edges.edge_count + 1, "lane_offsets must have one row per edge plus one");
    TORCH_CHECK(edges.edge_count > 0, "chain candidates need at least one edge");
    const ChainHalf a = chain_half(a_edge_offsets, a_endpoint, a_node, a_endpoints, a_node_sequences, a_node_images,
                                   edges.edge_count, "transmitter half");
    const ChainHalf b = chain_half(b_edge_offsets, b_endpoint, b_node, b_endpoints, b_node_sequences, b_node_images,
                                   edges.edge_count, "receiver half");
    TORCH_CHECK(a.depth + b.depth + 1 <= kDiscoveryMaxDepth, "chain depth exceeds the specular chain limit");
    TORCH_CHECK(0 <= lane_start && lane_start <= lane_end, "lane range is out of range");
    const int64_t lane_count = lane_end - lane_start;
    TORCH_CHECK(lane_count < std::numeric_limits<int>::max(), "chain candidate chunk exceeds the int32 range");
    TORCH_CHECK(std::isfinite(box_tolerance) && box_tolerance >= 0.0 && std::isfinite(segment_tolerance) &&
                    segment_tolerance >= 0.0 && std::isfinite(min_segment_length) && min_segment_length >= 0.0 &&
                    std::isfinite(edge_epsilon) && edge_epsilon >= 0.0,
                "chain candidate tolerances must be finite and non-negative");
    auto int_options = a_endpoint.options();
    auto allocate = [&](int64_t rows) {
        pybind11::dict out;
        out["tx"] = at::empty({rows}, int_options);
        out["rx"] = at::empty({rows}, int_options);
        out["edge"] = at::empty({rows}, int_options);
        out["a_node"] = at::empty({rows}, int_options);
        out["b_node"] = at::empty({rows}, int_options);
        out["point"] = at::empty({rows, 3}, edge_position.options());
        return out;
    };
    if (lane_count == 0) {
        return allocate(0);
    }
    const PairTolerances tol{static_cast<float>(box_tolerance), static_cast<float>(segment_tolerance),
                             static_cast<float>(min_segment_length)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a_endpoint.get_device()).stream();
    auto flags = at::empty({lane_count}, int_options);
    auto offsets = at::empty({lane_count + 1}, int_options);
    chain_candidate_flags_kernel<<<discovery_grid(lane_count), kDiscoveryBlock, 0, stream>>>(
        lane_start, lane_count, lane_offsets.data_ptr<int64_t>(), tables, edges, a, b, tol,
        static_cast<float>(edge_epsilon), flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(flags, offsets, lane_count, stream);
    const int64_t count = observe_compact_count(offsets, lane_count, stream).count;
    pybind11::dict out = allocate(count);
    if (count == 0) {
        return out;
    }
    chain_candidate_write_kernel<<<discovery_grid(lane_count), kDiscoveryBlock, 0, stream>>>(
        lane_start, lane_count, lane_offsets.data_ptr<int64_t>(), tables, edges, a, b, tol,
        static_cast<float>(edge_epsilon), flags.data_ptr<int>(), offsets.data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["tx"]).data_ptr<int>(), pybind11::cast<at::Tensor>(out["rx"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["edge"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["a_node"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["b_node"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["point"]).data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

pybind11::dict channel_chain_double_candidates(at::Tensor pair_offsets, int64_t lane_start, int64_t lane_end,
                                               int64_t rx_count, at::Tensor tx_offsets, at::Tensor tx_edges,
                                               at::Tensor rx_offsets, at::Tensor rx_edges) {
    check_tensor(pair_offsets, "pair_offsets", at::kLong, 1);
    check_tensor(tx_offsets, "tx_offsets", at::kLong, 1);
    check_tensor(rx_offsets, "rx_offsets", at::kLong, 1);
    check_tensor(tx_edges, "tx_edges", at::kInt, 1);
    check_tensor(rx_edges, "rx_edges", at::kInt, 1);
    TORCH_CHECK(rx_count > 0 && rx_offsets.size(0) == rx_count + 1, "rx_offsets must have one row per receiver plus one");
    const int64_t tx_count = tx_offsets.size(0) - 1;
    TORCH_CHECK(tx_count > 0, "tx_offsets must have one row per transmitter plus one");
    const int64_t pair_count = tx_count * rx_count;
    TORCH_CHECK(pair_offsets.size(0) == pair_count + 1, "pair_offsets must have one row per endpoint pair plus one");
    TORCH_CHECK(0 <= lane_start && lane_start <= lane_end, "lane range is out of range");
    const int64_t lane_count = lane_end - lane_start;
    TORCH_CHECK(lane_count < std::numeric_limits<int>::max(), "double-diffraction chunk exceeds the int32 range");
    auto int_options = tx_edges.options();
    pybind11::dict out;
    out["tx"] = at::empty({lane_count}, int_options);
    out["rx"] = at::empty({lane_count}, int_options);
    out["edge1"] = at::empty({lane_count}, int_options);
    out["edge2"] = at::empty({lane_count}, int_options);
    if (lane_count == 0) {
        return out;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_edges.get_device()).stream();
    chain_double_expand_kernel<<<discovery_grid(lane_count), kDiscoveryBlock, 0, stream>>>(
        lane_start, lane_count, rx_count, pair_offsets.data_ptr<int64_t>(), tx_offsets.data_ptr<int64_t>(),
        tx_edges.data_ptr<int>(), rx_offsets.data_ptr<int64_t>(), rx_edges.data_ptr<int>(), pair_count,
        pybind11::cast<at::Tensor>(out["tx"]).data_ptr<int>(), pybind11::cast<at::Tensor>(out["rx"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["edge1"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["edge2"]).data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
