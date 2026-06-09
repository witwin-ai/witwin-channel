#pragma once

namespace raydtorch {

enum DfrStrategyMask {
    RAYDTORCH_DFR_DIRECT = 1 << 0,
    RAYDTORCH_DFR_KELLER = 1 << 1,
    RAYDTORCH_DFR_SUFFIX_REFL = 1 << 2
};

enum DfrSampleSequence {
    RAYDTORCH_DFR_HASH = 0,
    RAYDTORCH_DFR_SOBOL = 1
};

enum DfrReceiverModel {
    RAYDTORCH_DFR_MATCHED_ISO = 0
};

} // namespace raydtorch
