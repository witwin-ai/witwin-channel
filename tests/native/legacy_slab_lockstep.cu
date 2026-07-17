#include "field_transport_ad.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

namespace ad = channel_native::field_transport_ad;
namespace transport = channel_native::field_transport;
namespace utd = witwin::channel::native_ext;

struct SlabCase {
    const char* name;
    float cos_theta;
    float eps_r;
    float sigma_e;
    float gain;
    float thickness;
    float wavelength;
};

struct TestResult {
    std::uint32_t value_mismatch_mask;
    std::uint32_t zero_seed_mismatch_mask;
    std::uint32_t nonfinite_derivative_mask;
    std::uint32_t finite_difference_mismatch_mask;
    std::uint32_t exp_clamp_mismatch_mask;
};

__device__ bool same_bits(float lhs, float rhs) {
    return __float_as_uint(lhs) == __float_as_uint(rhs);
}

__device__ bool finite_complex(utd::Complex value) {
    return isfinite(value.re) && isfinite(value.im);
}

__device__ bool close_fd(float actual, float expected) {
    const float tolerance = 2.0e-3f * fmaxf(1.0f, fabsf(expected));
    return fabsf(actual - expected) <= tolerance;
}

__global__ void run_lockstep_cases(const SlabCase* cases, TestResult* results, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }

    const SlabCase input = cases[index];
    TestResult result{};
    utd::Complex primal_te;
    utd::Complex primal_tm;
    transport::legacy_sionna_slab_fresnel(
        input.cos_theta,
        input.eps_r,
        input.sigma_e,
        input.gain,
        input.thickness,
        input.wavelength,
        primal_te,
        primal_tm);

    ad::DualC zero_te;
    ad::DualC zero_tm;
    ad::legacy_sionna_slab_fresnel_dual(
        input.cos_theta,
        input.eps_r,
        input.sigma_e,
        input.gain,
        input.thickness,
        input.wavelength,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        zero_te,
        zero_tm);

    ad::DualC seeded_te;
    ad::DualC seeded_tm;
    ad::legacy_sionna_slab_fresnel_dual(
        input.cos_theta,
        input.eps_r,
        input.sigma_e,
        input.gain,
        input.thickness,
        input.wavelength,
        0.125f,
        1.0e-4f,
        0.25f,
        1.0e-3f,
        1.0e-3f,
        seeded_te,
        seeded_tm);

    const float primal_values[4] = {
        primal_te.re, primal_te.im, primal_tm.re, primal_tm.im};
    const float zero_values[4] = {
        zero_te.v.re, zero_te.v.im, zero_tm.v.re, zero_tm.v.im};
    const float seeded_values[4] = {
        seeded_te.v.re, seeded_te.v.im, seeded_tm.v.re, seeded_tm.v.im};
    const float zero_derivatives[4] = {
        zero_te.d.re, zero_te.d.im, zero_tm.d.re, zero_tm.d.im};
    for (int component = 0; component < 4; ++component) {
        if (!same_bits(primal_values[component], zero_values[component])) {
            result.value_mismatch_mask |= 1u << component;
        }
        if (!same_bits(primal_values[component], seeded_values[component])) {
            result.value_mismatch_mask |= 1u << (component + 4);
        }
        if (zero_derivatives[component] != 0.0f) {
            result.zero_seed_mismatch_mask |= 1u << component;
        }
    }
    if (!finite_complex(seeded_te.d)) {
        result.nonfinite_derivative_mask |= 1u;
    }
    if (!finite_complex(seeded_tm.d)) {
        result.nonfinite_derivative_mask |= 2u;
    }

    // A stable, unclamped gain direction supplies an independent derivative
    // oracle. The gain enters linearly, so central finite differences are
    // well-conditioned for the first two ordinary-material cases.
    if (index < 2) {
        constexpr float kGainStep = 1.0e-3f;
        utd::Complex plus_te;
        utd::Complex plus_tm;
        utd::Complex minus_te;
        utd::Complex minus_tm;
        transport::legacy_sionna_slab_fresnel(
            input.cos_theta,
            input.eps_r,
            input.sigma_e,
            input.gain + kGainStep,
            input.thickness,
            input.wavelength,
            plus_te,
            plus_tm);
        transport::legacy_sionna_slab_fresnel(
            input.cos_theta,
            input.eps_r,
            input.sigma_e,
            input.gain - kGainStep,
            input.thickness,
            input.wavelength,
            minus_te,
            minus_tm);
        ad::DualC gain_te;
        ad::DualC gain_tm;
        ad::legacy_sionna_slab_fresnel_dual(
            input.cos_theta,
            input.eps_r,
            input.sigma_e,
            input.gain,
            input.thickness,
            input.wavelength,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            gain_te,
            gain_tm);
        const float inverse_span = 0.5f / kGainStep;
        const float finite_differences[4] = {
            (plus_te.re - minus_te.re) * inverse_span,
            (plus_te.im - minus_te.im) * inverse_span,
            (plus_tm.re - minus_tm.re) * inverse_span,
            (plus_tm.im - minus_tm.im) * inverse_span};
        const float gain_derivatives[4] = {
            gain_te.d.re, gain_te.d.im, gain_tm.d.re, gain_tm.d.im};
        for (int component = 0; component < 4; ++component) {
            if (!close_fd(gain_derivatives[component], finite_differences[component])) {
                result.finite_difference_mismatch_mask |= 1u << component;
            }
        }
    }

    // Passive slab inputs cannot make q.im positive, so exercise the frozen
    // exp(80) branch directly at the shared helper boundary.
    if (index == 0) {
        const transport::LegacySlabComplex q = {0.25f, 45.0f};
        const transport::LegacySlabComplex primal_phase =
            transport::legacy_exp_neg_2i(q);
        const ad::DualLC dual_phase = ad::dlc_exp_neg_2i({q, {0.125f, 1.0f}});
        if (!same_bits(primal_phase.re, dual_phase.v.re)) {
            result.exp_clamp_mismatch_mask |= 1u;
        }
        if (!same_bits(primal_phase.im, dual_phase.v.im)) {
            result.exp_clamp_mismatch_mask |= 2u;
        }
        if (!isfinite(dual_phase.d.re) || !isfinite(dual_phase.d.im)) {
            result.exp_clamp_mismatch_mask |= 4u;
        }
    }

    results[index] = result;
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "No CUDA device available; skipping lockstep test.\n");
        return 77;
    }

    const SlabCase host_cases[] = {
        {"lossless", 0.65f, 4.0f, 0.0f, 0.8f, 0.12f, 0.125f},
        {"lossy", 0.72f, 5.2f, 0.03f, 0.9f, 0.08f, 0.0857f},
        {"high-conductivity", 0.45f, 2.5f, 1.0e7f, 0.7f, 0.01f, 0.05f},
        {"zero-thickness", 0.8f, 3.0f, 0.01f, 1.0f, 0.0f, 0.1f},
        {"negative-thickness-clamp", 0.8f, 3.0f, 0.01f, 1.0f, -0.2f, 0.1f},
        {"small-wavelength-clamp", 0.6f, 4.5f, 0.02f, 0.6f, 1.0e-4f, 1.0e-12f},
        {"grazing-incidence", 0.0f, 6.0f, 0.04f, 0.75f, 0.03f, 0.06f},
        {"material-input-clamps", -0.3f, 0.0f, -1.0f, 0.5f, 0.04f, 0.09f},
    };
    constexpr int kCaseCount = static_cast<int>(sizeof(host_cases) / sizeof(host_cases[0]));

    SlabCase* device_cases = nullptr;
    TestResult* results = nullptr;
    if (!check_cuda(cudaMalloc(&device_cases, sizeof(host_cases)), "cudaMalloc(cases)") ||
        !check_cuda(cudaMallocManaged(&results, sizeof(TestResult) * kCaseCount),
                    "cudaMallocManaged(results)")) {
        cudaFree(device_cases);
        cudaFree(results);
        return 1;
    }
    if (!check_cuda(cudaMemcpy(
                        device_cases, host_cases, sizeof(host_cases), cudaMemcpyHostToDevice),
                    "cudaMemcpy(cases)")) {
        cudaFree(device_cases);
        cudaFree(results);
        return 1;
    }

    run_lockstep_cases<<<1, 32>>>(device_cases, results, kCaseCount);
    bool cuda_ok = check_cuda(cudaGetLastError(), "run_lockstep_cases launch") &&
                   check_cuda(cudaDeviceSynchronize(), "run_lockstep_cases sync");
    bool passed = cuda_ok;
    if (cuda_ok) {
        for (int index = 0; index < kCaseCount; ++index) {
            const TestResult& result = results[index];
            if (result.value_mismatch_mask != 0 ||
                result.zero_seed_mismatch_mask != 0 ||
                result.nonfinite_derivative_mask != 0 ||
                result.finite_difference_mismatch_mask != 0 ||
                result.exp_clamp_mismatch_mask != 0) {
                std::fprintf(
                    stderr,
                    "%s failed: value=0x%x zero_seed=0x%x nonfinite=0x%x "
                    "finite_difference=0x%x exp_clamp=0x%x\n",
                    host_cases[index].name,
                    result.value_mismatch_mask,
                    result.zero_seed_mismatch_mask,
                    result.nonfinite_derivative_mask,
                    result.finite_difference_mismatch_mask,
                    result.exp_clamp_mismatch_mask);
                passed = false;
            }
        }
    }

    cudaFree(device_cases);
    cudaFree(results);
    if (!passed) {
        return 1;
    }
    std::printf("LegacySlabComplex primal/dual lockstep passed for %d cases.\n", kCaseCount);
    return 0;
}
