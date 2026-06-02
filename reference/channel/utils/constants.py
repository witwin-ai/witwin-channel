"""Shared numeric constants for RFDT."""

# Shared CUDA AD runtime label used by legacy script imports.
DEFAULT_VARIANT = "cuda_ad_rgb"

# Small epsilon for denominators and normalization.
EPS = 1e-10

# Geometry and numeric thresholds.
SMALL_EPS = 1e-6
EDGE_2D_EPS = 1e-4
RAY_EPS = 1e-4
BARY_EPS = 1e-12

# Ray tracing offsets and cutoffs.
RAY_ORIGIN_BIAS = 1e-3
DIFFRACTION_MIN_DISTANCE = 0.05
UTD_SLOPE_DERIVATIVE_STEP = 1e-4

# Power conversion floor to avoid log(0).
POWER_DB_FLOOR = 1e-20

# Vacuum permittivity (F/m) to avoid SciPy dependency.
EPSILON_0 = 8.854187817e-12

# Speed of light in vacuum (m/s).
SPEED_OF_LIGHT = 299792458.0
