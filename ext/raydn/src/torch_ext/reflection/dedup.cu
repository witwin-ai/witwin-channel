#include <raydn/reflection/dedup.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <string>

#include <raydn/common/native_compat.h>


namespace raydn {

namespace {

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) {
        allocate(count);
    }

    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    CudaBuffer(CudaBuffer &&other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer &operator=(CudaBuffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }

        count_ = count;
        if (count_ == 0) {
            return;
        }

        const cudaError_t error =
            cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_);
        require(error == cudaSuccess,
                std::string("reflection_dedup_gpu(): cudaMalloc failed: ") +
                    cudaGetErrorString(error));
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

// Deduplication runs as a sort-and-segment pipeline; each kernel maps one thread to
// one ray (blockIdx.x * blockDim.x + threadIdx.x, bounds-checked).

/// One ray per thread; hashes its reflector sequence (via canonical ids) into a 64-bit grouping key.
__global__ void reflection_dedup_build_keys_kernel(
    int n_rays,
    int max_bounces,
    const int *bounce_count,
    const int *shape_ids,
    const int *prim_ids,
    const int *face_offsets,
    int n_meshes,
    const int *canonical_table,
    int canonical_table_size,
    uint64_t *out_keys,
    int *out_ray_indices) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_rays) {
        return;
    }

    const int bc = bounce_count[i];
    out_ray_indices[i] = i;
    if (bc <= 0) {
        out_keys[i] = UINT64_MAX;
        return;
    }

    uint64_t hash = 14695981039346656037ull;
    const int base = i * max_bounces;
    for (int b = 0; b < bc; ++b) {
        const int shape_id = shape_ids[base + b];
        const int local_prim = prim_ids[base + b];
        const int face_offset =
            (shape_id >= 0 && shape_id < n_meshes) ? face_offsets[shape_id] : 0;
        int global_prim = face_offset + local_prim;

        if (canonical_table != nullptr &&
            global_prim >= 0 &&
            global_prim < canonical_table_size) {
            const int mapped = canonical_table[global_prim];
            if (mapped >= 0) {
                global_prim = mapped;
            }
        }

        hash ^= static_cast<uint64_t>(static_cast<uint32_t>(global_prim));
        hash *= 1099511628211ull;
    }

    hash ^= static_cast<uint64_t>(bc) << 56;
    out_keys[i] = hash;
}

/// One ray per thread; flags the first entry of each run of equal sorted keys (segment boundary).
__global__ void reflection_dedup_mark_boundaries_kernel(
    int n_rays,
    const uint64_t *sorted_keys,
    int *out_boundary_flags) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_rays) {
        return;
    }

    if (sorted_keys[i] == UINT64_MAX) {
        out_boundary_flags[i] = 0;
        return;
    }

    out_boundary_flags[i] =
        (i == 0 || sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0;
}

/// One ray per thread; converts the inclusive-scan boundary counts into 0-based group ids.
__global__ void reflection_dedup_zero_base_ids_kernel(
    int n_rays,
    const uint64_t *sorted_keys,
    int *inout_ids) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_rays || sorted_keys[i] == UINT64_MAX) {
        return;
    }

    inout_ids[i] -= 1;
}

/// One ray per thread; refines each hash group by the spatially-quantized final image source,
/// so paths that merely hash-collide are split into distinct clusters.
__global__ void reflection_dedup_sub_cluster_kernel(
    int n_rays,
    int max_bounces,
    const uint64_t *sorted_keys,
    const int *sorted_ray_indices,
    const int *hash_group_ids,
    const int *bounce_count,
    const float *img_x,
    const float *img_y,
    const float *img_z,
    float tolerance,
    uint64_t *out_cluster_keys,
    int *out_cluster_ray_indices) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_rays) {
        return;
    }

    const int ray_index = sorted_ray_indices[i];
    out_cluster_ray_indices[i] = ray_index;

    if (sorted_keys[i] == UINT64_MAX) {
        out_cluster_keys[i] = UINT64_MAX;
        return;
    }

    const int bc = bounce_count[ray_index];
    const int last_slot = ray_index * max_bounces + (bc > 0 ? bc - 1 : 0);

    const float inv_tol = 1.0f / fmaxf(tolerance, 1e-12f);
    const int qx = __float2int_rn(img_x[last_slot] * inv_tol);
    const int qy = __float2int_rn(img_y[last_slot] * inv_tol);
    const int qz = __float2int_rn(img_z[last_slot] * inv_tol);
    const uint32_t spatial =
        static_cast<uint32_t>(qx * 73856093u ^
                              qy * 19349663u ^
                              qz * 83492791u);

    out_cluster_keys[i] =
        (static_cast<uint64_t>(static_cast<uint32_t>(hash_group_ids[i])) << 32) |
        static_cast<uint64_t>(spatial);
}

/// One ray per thread; writes each cluster's representative path to its compacted output slot
/// and tallies how many paths it represents.
__global__ void reflection_dedup_compact_kernel(
    int n_rays,
    int max_bounces,
    const uint64_t *final_sorted_keys,
    const int *final_sorted_ray_indices,
    const int *unique_path_ids,
    const int *raw_bounce_count,
    const int *raw_shape_ids,
    const int *raw_prim_ids,
    const float *raw_t,
    const float *raw_bary_u,
    const float *raw_bary_v,
    const float *raw_hit_x,
    const float *raw_hit_y,
    const float *raw_hit_z,
    const float *raw_norm_x,
    const float *raw_norm_y,
    const float *raw_norm_z,
    const float *raw_img_x,
    const float *raw_img_y,
    const float *raw_img_z,
    int *out_n_unique,
    int *out_bounce_count,
    int *out_shape_ids,
    int *out_prim_ids,
    float *out_t,
    float *out_bary_u,
    float *out_bary_v,
    float *out_hit_x,
    float *out_hit_y,
    float *out_hit_z,
    float *out_norm_x,
    float *out_norm_y,
    float *out_norm_z,
    float *out_img_x,
    float *out_img_y,
    float *out_img_z,
    int *out_discovery_count,
    int *out_representative_ray_index) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_rays || final_sorted_keys[i] == UINT64_MAX) {
        return;
    }

    const int uid = unique_path_ids[i];
    atomicAdd(out_discovery_count + uid, 1);

    const bool is_representative =
        i == 0 || unique_path_ids[i] != unique_path_ids[i - 1];
    if (!is_representative) {
        return;
    }

    atomicMax(out_n_unique, uid + 1);

    const int ray_index = final_sorted_ray_indices[i];
    out_representative_ray_index[uid] = ray_index;

    const int bc = raw_bounce_count[ray_index];
    out_bounce_count[uid] = bc;

    const int src_base = ray_index * max_bounces;
    const int dst_base = uid * max_bounces;
    for (int b = 0; b < bc; ++b) {
        const int src = src_base + b;
        const int dst = dst_base + b;
        out_shape_ids[dst] = raw_shape_ids[src];
        out_prim_ids[dst] = raw_prim_ids[src];
        out_t[dst] = raw_t[src];
        out_bary_u[dst] = raw_bary_u[src];
        out_bary_v[dst] = raw_bary_v[src];
        out_hit_x[dst] = raw_hit_x[src];
        out_hit_y[dst] = raw_hit_y[src];
        out_hit_z[dst] = raw_hit_z[src];
        out_norm_x[dst] = raw_norm_x[src];
        out_norm_y[dst] = raw_norm_y[src];
        out_norm_z[dst] = raw_norm_z[src];
        out_img_x[dst] = raw_img_x[src];
        out_img_y[dst] = raw_img_y[src];
        out_img_z[dst] = raw_img_z[src];
    }
}

} // namespace

int reflection_dedup_gpu(
    int n_rays,
    int max_bounces,
    const int *bounce_count,
    const int *shape_ids,
    const int *prim_ids,
    const float *t,
    const float *bary_u,
    const float *bary_v,
    const float *hit_x,
    const float *hit_y,
    const float *hit_z,
    const float *norm_x,
    const float *norm_y,
    const float *norm_z,
    const float *img_x,
    const float *img_y,
    const float *img_z,
    const int *face_offsets,
    int n_meshes,
    const int *canonical_prim_table,
    int canonical_table_size,
    float image_source_tolerance,
    int *out_bounce_count,
    int *out_shape_ids,
    int *out_prim_ids,
    float *out_t,
    float *out_bary_u,
    float *out_bary_v,
    float *out_hit_x,
    float *out_hit_y,
    float *out_hit_z,
    float *out_norm_x,
    float *out_norm_y,
    float *out_norm_z,
    float *out_img_x,
    float *out_img_y,
    float *out_img_z,
    int *out_discovery_count,
    int *out_representative_ray_index) {
    require(n_rays >= 0, "reflection_dedup_gpu(): n_rays must be non-negative.");
    require(max_bounces > 0,
            "reflection_dedup_gpu(): max_bounces must be positive.");

    if (n_rays == 0) {
        return 0;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());

    constexpr int block_size = 256;
    const int block_count = (n_rays + block_size - 1) / block_size;

    CudaBuffer<uint64_t> keys_in(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> keys_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> ray_indices_in(static_cast<size_t>(n_rays));
    CudaBuffer<int> ray_indices_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> boundary_flags(static_cast<size_t>(n_rays));
    CudaBuffer<int> hash_group_ids(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> cluster_keys_in(static_cast<size_t>(n_rays));
    CudaBuffer<uint64_t> cluster_keys_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> cluster_ray_indices_in(static_cast<size_t>(n_rays));
    CudaBuffer<int> cluster_ray_indices_out(static_cast<size_t>(n_rays));
    CudaBuffer<int> unique_path_ids(static_cast<size_t>(n_rays));
    CudaBuffer<int> unique_count_device(1);

    check_cuda_call(cudaMemsetAsync(out_discovery_count,
                                    0,
                                    sizeof(int) * static_cast<size_t>(n_rays),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear discovery counts");
    audit_cuda_memset_async();
    check_cuda_call(cudaMemsetAsync(out_representative_ray_index,
                                    0xFF,
                                    sizeof(int) * static_cast<size_t>(n_rays),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear representative indices");
    audit_cuda_memset_async();
    check_cuda_call(cudaMemsetAsync(unique_count_device.get(),
                                    0,
                                    sizeof(int),
                                    stream),
                    "reflection_dedup_gpu(): failed to clear unique counter");
    audit_cuda_memset_async();

    audit_cuda_kernel_launch("reflection_dedup_build_keys_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    reflection_dedup_build_keys_kernel<<<block_count, block_size, 0, stream>>>(
        n_rays,
        max_bounces,
        bounce_count,
        shape_ids,
        prim_ids,
        face_offsets,
        n_meshes,
        canonical_prim_table,
        canonical_table_size,
        keys_in.get(),
        ray_indices_in.get());
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch build-keys kernel");

    size_t sort_temp_size = 0;
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr,
                                                    sort_temp_size,
                                                    keys_in.get(),
                                                    keys_out.get(),
                                                    ray_indices_in.get(),
                                                    ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to size first radix sort");
    CudaBuffer<char> sort_temp(std::max<size_t>(sort_temp_size, 1));
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(sort_temp.get(),
                                                    sort_temp_size,
                                                    keys_in.get(),
                                                    keys_out.get(),
                                                    ray_indices_in.get(),
                                                    ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to run first radix sort");

    audit_cuda_kernel_launch("reflection_dedup_mark_boundaries_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    reflection_dedup_mark_boundaries_kernel<<<block_count, block_size, 0, stream>>>(
        n_rays, keys_out.get(), boundary_flags.get());
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch first boundary kernel");

    size_t scan_temp_size = 0;
    audit_cub_scan();
    check_cuda_call(cub::DeviceScan::InclusiveSum(nullptr,
                                                  scan_temp_size,
                                                  boundary_flags.get(),
                                                  hash_group_ids.get(),
                                                  n_rays,
                                                  stream),
                    "reflection_dedup_gpu(): failed to size first scan");
    CudaBuffer<char> scan_temp(std::max<size_t>(scan_temp_size, 1));
    audit_cub_scan();
    check_cuda_call(cub::DeviceScan::InclusiveSum(scan_temp.get(),
                                                  scan_temp_size,
                                                  boundary_flags.get(),
                                                  hash_group_ids.get(),
                                                  n_rays,
                                                  stream),
                    "reflection_dedup_gpu(): failed to run first scan");
    audit_cuda_kernel_launch("reflection_dedup_zero_base_ids_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    reflection_dedup_zero_base_ids_kernel<<<block_count, block_size, 0, stream>>>(
        n_rays, keys_out.get(), hash_group_ids.get());
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch first id-fix kernel");

    audit_cuda_kernel_launch("reflection_dedup_sub_cluster_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    reflection_dedup_sub_cluster_kernel<<<block_count, block_size, 0, stream>>>(
        n_rays,
        max_bounces,
        keys_out.get(),
        ray_indices_out.get(),
        hash_group_ids.get(),
        bounce_count,
        img_x,
        img_y,
        img_z,
        image_source_tolerance,
        cluster_keys_in.get(),
        cluster_ray_indices_in.get());
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch sub-cluster kernel");

    size_t cluster_sort_temp_size = 0;
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(nullptr,
                                                    cluster_sort_temp_size,
                                                    cluster_keys_in.get(),
                                                    cluster_keys_out.get(),
                                                    cluster_ray_indices_in.get(),
                                                    cluster_ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to size second radix sort");
    CudaBuffer<char> cluster_sort_temp(std::max<size_t>(cluster_sort_temp_size, 1));
    audit_cub_sort();
    check_cuda_call(cub::DeviceRadixSort::SortPairs(cluster_sort_temp.get(),
                                                    cluster_sort_temp_size,
                                                    cluster_keys_in.get(),
                                                    cluster_keys_out.get(),
                                                    cluster_ray_indices_in.get(),
                                                    cluster_ray_indices_out.get(),
                                                    n_rays,
                                                    0,
                                                    64,
                                                    stream),
                    "reflection_dedup_gpu(): failed to run second radix sort");

    audit_cuda_kernel_launch("reflection_dedup_mark_boundaries_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    reflection_dedup_mark_boundaries_kernel<<<block_count, block_size, 0, stream>>>(
        n_rays, cluster_keys_out.get(), boundary_flags.get());
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch second boundary kernel");

    audit_cub_scan();
    check_cuda_call(cub::DeviceScan::InclusiveSum(scan_temp.get(),
                                                  scan_temp_size,
                                                  boundary_flags.get(),
                                                  unique_path_ids.get(),
                                                  n_rays,
                                                  stream),
                    "reflection_dedup_gpu(): failed to run second scan");
    audit_cuda_kernel_launch("reflection_dedup_zero_base_ids_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    reflection_dedup_zero_base_ids_kernel<<<block_count, block_size, 0, stream>>>(
        n_rays, cluster_keys_out.get(), unique_path_ids.get());
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch second id-fix kernel");

    audit_cuda_kernel_launch("reflection_dedup_compact_kernel",
                             static_cast<uint32_t>(block_count), 1, 1,
                             block_size, 1, 1,
                             static_cast<uint64_t>(n_rays));
    reflection_dedup_compact_kernel<<<block_count, block_size, 0, stream>>>(
        n_rays,
        max_bounces,
        cluster_keys_out.get(),
        cluster_ray_indices_out.get(),
        unique_path_ids.get(),
        bounce_count,
        shape_ids,
        prim_ids,
        t,
        bary_u,
        bary_v,
        hit_x,
        hit_y,
        hit_z,
        norm_x,
        norm_y,
        norm_z,
        img_x,
        img_y,
        img_z,
        unique_count_device.get(),
        out_bounce_count,
        out_shape_ids,
        out_prim_ids,
        out_t,
        out_bary_u,
        out_bary_v,
        out_hit_x,
        out_hit_y,
        out_hit_z,
        out_norm_x,
        out_norm_y,
        out_norm_z,
        out_img_x,
        out_img_y,
        out_img_z,
        out_discovery_count,
        out_representative_ray_index);
    check_cuda_last_error("reflection_dedup_gpu(): failed to launch compact kernel");

    int unique_count = 0;
    audit_cuda_memcpy_async();
    check_cuda_call(cudaMemcpyAsync(&unique_count,
                                    unique_count_device.get(),
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    stream),
                    "reflection_dedup_gpu(): failed to copy unique count");
    audit_cuda_stream_synchronize();
    check_cuda_call(cudaStreamSynchronize(stream),
                    "reflection_dedup_gpu(): failed to finish dedup stream");
    return unique_count;
}

} // namespace raydn
