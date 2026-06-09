#pragma once

namespace raydn {

enum DfrStrategyMask {
    RAYDN_DFR_DIRECT = 1 << 0,
    RAYDN_DFR_KELLER = 1 << 1,
    RAYDN_DFR_SUFFIX_REFL = 1 << 2
};

enum DfrSampleSequence {
    RAYDN_DFR_HASH = 0,
    RAYDN_DFR_SOBOL = 1
};

enum DfrReceiverModel {
    RAYDN_DFR_MATCHED_ISO = 0
};

} // namespace raydn
