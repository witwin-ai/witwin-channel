// Copyright Xingyu Chen.
// Owns the physical constants and path component ids shared by the native kernels.

#pragma once

namespace channel {

// Each constant has a double and a float spelling. The float spelling is the double value
// rounded to float, so a float site and a double site agree with the literals they replace.

// Speed of light in vacuum, m/s.
constexpr double kSpeedOfLight = 299792458.0;
constexpr float kSpeedOfLightF = 299792458.0f;

constexpr double kPi = 3.14159265358979323846;
constexpr float kPiF = 3.14159265358979323846f;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr float kTwoPiF = 6.28318530717958647692f;

// Vacuum permittivity, F/m (CODATA 2018).
constexpr float kVacuumPermittivityF = 8.8541878128e-12f;

static_assert(kSpeedOfLightF == static_cast<float>(kSpeedOfLight), "c float spelling");
static_assert(kPiF == static_cast<float>(kPi), "pi float spelling");
static_assert(kTwoPi == 2.0 * kPi, "two pi is exactly twice pi");
static_assert(kTwoPiF == 2.0f * kPiF, "two pi float is exactly twice pi float");

// Path component ids of the enumerated row contract (``witwin.channel.path`` publishes the
// same ids). An interaction chain diffracts last (3), first (4) or at interior slot s
// (kComponentChainInteriorBase + s).
enum ComponentId : int {
    kComponentLineOfSight = 0,
    kComponentReflection = 1,
    kComponentDiffraction = 2,
    kComponentReflectionDiffraction = 3,
    kComponentDiffractionReflection = 4,
    kComponentTransmission = 5,
    kComponentScattering = 6,
    kComponentDoubleDiffraction = 7,
    kComponentChainInteriorBase = 8,
};

}  // namespace channel
