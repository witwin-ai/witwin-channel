#pragma once

#include <ATen/ATen.h>

namespace channel_native::capacity {

enum FailureBit : int {
    kDiffractionStateOverflow = 1 << 0,
    kDiffractionPathOverflow = 1 << 1,
    kDiffractionPathContractError = 1 << 2,
    kPairCapacityOverflow = 1 << 3,
    kPairContractError = 1 << 4,
    kCoupledCandidateOverflow = 1 << 5,
    kReflectionCandidateOverflow = 1 << 6,
    kSegmentPenetrationFailure = 1 << 7,
};

void validate_failure_state(
    const at::Tensor& failure_state,
    const at::Tensor& reference);

}  // namespace channel_native::capacity
