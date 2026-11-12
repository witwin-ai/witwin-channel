// Copyright Xingyu Chen.
// Order-independent scatter sums: fixed-point or exact binned integer accumulation per target.

#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace channel::fixed_sum {

// A scatter sum adds values into targets in two passes over the same contributions.
// observe() raises the target's scale word to the binary exponent of the largest magnitude
// seen; deposit() then rounds each value to an integer multiple of the target's quantum
// 2^(exponent - headroom) and adds it with a 64-bit integer atomic. Integer addition is
// associative, so the total does not depend on thread order, launch chunking or the order of
// the contributions. resolve() converts the integer back to the floating-point sum.
//
// Range and error: every rounded term is below 2^headroom in magnitude and headroom is
// 62 - ceil(log2(N)) for at most N terms per target, so the integer sum cannot overflow. Each
// term is off by at most half a quantum, so a target that receives n terms is off by at most
// n * 2^(exponent - headroom - 1) before the final rounding to the output type.
//
// Non-finite values set flag bits above the exponent field instead (atomicOr); a flagged target
// resolves to NaN, +inf or -inf exactly as an IEEE sum of the same values would.
struct Targets {
    int *scale = nullptr;  // biased exponent (0: no nonzero term) or non-finite flag bits
    unsigned long long *sum = nullptr;
    int headroom = 62;
};

constexpr int kExponentBias = 1100;  // above the magnitude of any float or double ilogb result
constexpr int kPositiveInf = 1 << 28;
constexpr int kNegativeInf = 1 << 29;
constexpr int kNaN = 1 << 30;
constexpr int kNonFiniteMask = kPositiveInf | kNegativeInf | kNaN;

// Headroom bits for at most max_terms contributions per target.
inline int headroom_for(int64_t max_terms) {
    int bits = 0;
    while (bits < 62 && (int64_t{1} << bits) < max_terms) ++bits;
    TORCH_CHECK(bits <= 40, "fixed-point scatter sum supports at most 2^40 terms per target");
    return 62 - bits;
}

__device__ __forceinline__ int binary_exponent(float value) { return ilogbf(value) + 1; }
__device__ __forceinline__ int binary_exponent(double value) { return ilogb(value) + 1; }

__device__ __forceinline__ long long to_fixed(float value, int shift) {
    return __float2ll_rn(scalbnf(value, shift));
}
__device__ __forceinline__ long long to_fixed(double value, int shift) {
    return __double2ll_rn(scalbn(value, shift));
}

template <typename T>
__device__ __forceinline__ void observe(const Targets &targets, int64_t target, T value) {
    if (value == T(0)) return;
    if (!isfinite(value)) {
        const int flag = isnan(value) ? kNaN : (value > T(0) ? kPositiveInf : kNegativeInf);
        atomicOr(targets.scale + target, flag);
        return;
    }
    atomicMax(targets.scale + target, binary_exponent(value) + kExponentBias);
}

template <typename T>
__device__ __forceinline__ void deposit(const Targets &targets, int64_t target, T value) {
    if (value == T(0) || !isfinite(value)) return;
    const int scale = targets.scale[target];
    if (scale & kNonFiniteMask) return;
    const long long fixed = to_fixed(value, targets.headroom - (scale - kExponentBias));
    if (fixed != 0)
        atomicAdd(targets.sum + target, static_cast<unsigned long long>(fixed));
}

template <typename T>
__device__ __forceinline__ T resolve(const Targets &targets, int64_t target);

template <>
__device__ __forceinline__ float resolve<float>(const Targets &targets, int64_t target) {
    const int scale = targets.scale[target];
    if (scale & kNonFiniteMask) {
        if ((scale & kNaN) || (scale & kPositiveInf && scale & kNegativeInf))
            return __int_as_float(0x7fc00000);
        return (scale & kPositiveInf) ? __int_as_float(0x7f800000) : __int_as_float(0xff800000);
    }
    if (scale == 0) return 0.0f;
    const long long fixed = static_cast<long long>(targets.sum[target]);
    return scalbnf(__ll2float_rn(fixed), (scale - kExponentBias) - targets.headroom);
}

template <>
__device__ __forceinline__ double resolve<double>(const Targets &targets, int64_t target) {
    const int scale = targets.scale[target];
    if (scale & kNonFiniteMask) {
        if ((scale & kNaN) || (scale & kPositiveInf && scale & kNegativeInf))
            return __longlong_as_double(0x7ff8000000000000LL);
        return (scale & kPositiveInf) ? __longlong_as_double(0x7ff0000000000000LL)
                                      : __longlong_as_double(static_cast<long long>(0xfff0000000000000ULL));
    }
    if (scale == 0) return 0.0;
    const long long fixed = static_cast<long long>(targets.sum[target]);
    return scalbn(__ll2double_rn(fixed), (scale - kExponentBias) - targets.headroom);
}

// Single-pass exact variant for float32 terms, used where a second pass over the
// contributions would cost a launch: every finite float is an integer M < 2^24 times
// 2^(p - 149) with 0 <= p <= 253, so M * 2^(p mod 32) is split into two 32-bit halves added to
// bins p / 32 and p / 32 + 1 of nine signed 64-bit bins. The bins hold the exact sum for up to
// 2^31 terms per target; slot 9 holds the non-finite flags.
constexpr int kBinSlots = 10;

struct BinnedTargets {
    unsigned long long *bins = nullptr;  // kBinSlots words per target
};

__device__ __forceinline__ void binned_add(const BinnedTargets &targets, int64_t target, float value) {
    if (value == 0.0f) return;
    unsigned long long *bins = targets.bins + target * kBinSlots;
    if (!isfinite(value)) {
        const int flag = isnan(value) ? kNaN : (value > 0.0f ? kPositiveInf : kNegativeInf);
        atomicOr(bins + 9, static_cast<unsigned long long>(flag));
        return;
    }
    const unsigned int bits = __float_as_uint(value);
    const int biased = static_cast<int>((bits >> 23) & 0xffu);
    unsigned int mantissa = bits & 0x7fffffu;
    int position = 0;
    if (biased != 0) {
        mantissa |= 0x800000u;
        position = biased - 1;
    }
    const unsigned long long shifted = static_cast<unsigned long long>(mantissa) << (position & 31);
    long long low = static_cast<long long>(shifted & 0xffffffffull);
    long long high = static_cast<long long>(shifted >> 32);
    if (bits >> 31) {
        low = -low;
        high = -high;
    }
    const int bin = position >> 5;
    if (low != 0) atomicAdd(bins + bin, static_cast<unsigned long long>(low));
    if (high != 0) atomicAdd(bins + bin + 1, static_cast<unsigned long long>(high));
}

// Reads the bins through L2 (they were written by atomics of other blocks) and adds them from
// the top bin down in double precision, then rounds once to float.
__device__ __forceinline__ float binned_resolve(const BinnedTargets &targets, int64_t target) {
    const unsigned long long *bins = targets.bins + target * kBinSlots;
    const long long flags = static_cast<long long>(__ldcg(bins + 9));
    if (flags != 0) {
        if ((flags & kNaN) || ((flags & kPositiveInf) && (flags & kNegativeInf)))
            return __int_as_float(0x7fc00000);
        return (flags & kPositiveInf) ? __int_as_float(0x7f800000) : __int_as_float(0xff800000);
    }
    double total = 0.0;
    for (int bin = kBinSlots - 2; bin >= 0; --bin) {
        const long long word = static_cast<long long>(__ldcg(bins + bin));
        if (word != 0) total += scalbn(__ll2double_rn(word), 32 * bin - 149);
    }
    return __double2float_rn(total);
}

// True in exactly one block of the grid, after every other block has passed this point and
// made its global writes visible; that block resets the counter for the next launch.
__device__ __forceinline__ bool last_block(unsigned int *counter) {
    __shared__ bool is_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        is_last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (is_last) __threadfence();
    return is_last;
}

// A kernel's receiver output: add() deposits a term, and finish(), reached by every thread at
// the end of the kernel, lets the last block to finish write the resolved float sums.
struct BinnedOutput {
    BinnedTargets bins;
    float *output = nullptr;
    int64_t count = 0;
    unsigned int *counter = nullptr;

    __device__ __forceinline__ void add(int64_t target, float value) const {
        binned_add(bins, target, value);
    }

    __device__ __forceinline__ void finish() const {
        if (!last_block(counter)) return;
        for (int64_t index = threadIdx.x; index < count; index += blockDim.x)
            output[index] = binned_resolve(bins, index);
        if (threadIdx.x == 0) *counter = 0;
    }
};

// Zeroed bins for `count` targets plus one launch counter, one memset on the caller's stream.
struct BinnedStorage {
    at::Tensor words;
    int64_t count;

    BinnedStorage(int64_t count_, const at::TensorOptions &options, cudaStream_t stream) : count(count_) {
        words = at::empty({count * kBinSlots + 1}, options.dtype(at::kLong));
        C10_CUDA_CHECK(cudaMemsetAsync(
            words.data_ptr<int64_t>(), 0, (count * kBinSlots + 1) * sizeof(int64_t), stream));
    }

    BinnedTargets targets(int64_t offset) {
        BinnedTargets view;
        view.bins = reinterpret_cast<unsigned long long *>(words.data_ptr<int64_t>()) + offset * kBinSlots;
        return view;
    }

    unsigned int *counter() {
        return reinterpret_cast<unsigned int *>(words.data_ptr<int64_t>() + count * kBinSlots);
    }

    // Output view for targets [offset, offset + targets) written to output.
    BinnedOutput output(int64_t offset, int64_t targets, float *output_values) {
        BinnedOutput view;
        view.bins = this->targets(offset);
        view.output = output_values + offset;
        view.count = targets;
        view.counter = counter();
        return view;
    }
};

// Zeroed scale and sum storage for `count` targets, two memsets on the caller's stream.
struct Storage {
    at::Tensor scale;
    at::Tensor sum;

    Storage(int64_t count, const at::TensorOptions &options, cudaStream_t stream) {
        scale = at::empty({count}, options.dtype(at::kInt));
        sum = at::empty({count}, options.dtype(at::kLong));
        if (count > 0) {
            C10_CUDA_CHECK(cudaMemsetAsync(scale.data_ptr<int>(), 0, count * sizeof(int), stream));
            C10_CUDA_CHECK(
                cudaMemsetAsync(sum.data_ptr<int64_t>(), 0, count * sizeof(int64_t), stream));
        }
    }

    Targets targets(int64_t offset, int headroom) {
        Targets view;
        view.scale = scale.data_ptr<int>() + offset;
        view.sum = reinterpret_cast<unsigned long long *>(sum.data_ptr<int64_t>()) + offset;
        view.headroom = headroom;
        return view;
    }
};

namespace {

constexpr int kScatterThreads = 256;

// One pass over buffered contributions: values[i] goes to target targets[i] (target 0 when
// targets is null); negative targets and zero values are skipped.
template <typename T, bool kDeposit>
__global__ void scatter_pass_kernel(const T *values, const int *targets, int64_t count, Targets cells) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const int target = targets != nullptr ? targets[index] : 0;
        if (target < 0) continue;
        if constexpr (kDeposit) {
            deposit(cells, target, values[index]);
        } else {
            observe(cells, target, values[index]);
        }
    }
}

template <typename T>
__global__ void resolve_kernel(Targets cells, int64_t count, T *output) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride)
        output[index] = resolve<T>(cells, index);
}

inline unsigned int scatter_blocks(int64_t count) {
    return static_cast<unsigned int>(std::min<int64_t>((count + kScatterThreads - 1) / kScatterThreads, 65535));
}

// Writes output[t] = sum of values[i] over targets[i] == t for t < output_count, in an order-
// independent fixed-point sum. Three launches plus two memsets.
template <typename T>
[[maybe_unused]] void scatter_sum(const T *values, const int *targets, int64_t count, T *output,
                                  int64_t output_count, const at::TensorOptions &options,
                                  cudaStream_t stream) {
    if (output_count <= 0) return;
    Storage storage(output_count, options, stream);
    const Targets cells = storage.targets(0, headroom_for(count));
    if (count > 0) {
        scatter_pass_kernel<T, false><<<scatter_blocks(count), kScatterThreads, 0, stream>>>(
            values, targets, count, cells);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scatter_pass_kernel<T, true><<<scatter_blocks(count), kScatterThreads, 0, stream>>>(
            values, targets, count, cells);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    resolve_kernel<T><<<scatter_blocks(output_count), kScatterThreads, 0, stream>>>(
        cells, output_count, output);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

}  // namespace channel::fixed_sum
