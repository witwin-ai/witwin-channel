# Copyright Xingyu Chen.
# Shared AD gradient-test tolerances and finite-difference steps.

"""Shared AD gradient-test tolerances and finite-difference steps."""

# Central finite-difference steps, calibrated per parameter magnitude.
FD_STEP_POSITION = 1.0e-2
FD_STEP_GEOMETRY = 1.0e-3

# Geometry steps for the complex-coefficient solvers (geometry AD). The
# carrier phase runs at k ~ 63 rad/m at 3 GHz, so the position step has to keep
# k*h well below 1 for a two-point difference to stay in the linear regime;
# 1e-2 m (the incoherent-power step above) would carry a 0.6 rad phase swing and
# a percent-level truncation error.
FD_STEP_POSITION_PHASE = 1.0e-3
FD_STEP_VERTEX = 1.0e-3

# Material FD steps (material and frequency derivatives), calibrated so the two-point difference
# clears the float32 forward noise floor for eps_r ~ 2..5, sigma_e ~ 0.01..0.1
# and layer thickness ~ 0.05..0.2 m.
FD_STEP_EPS_R = 1.0e-2
FD_STEP_SIGMA_E = 1.0e-3
FD_STEP_GAIN = 1.0e-3
FD_STEP_THICKNESS = 1.0e-4

# Frequency uses a relative step: h = FD_REL_STEP_FREQUENCY * frequency, small
# enough that the carrier phase change over a scene-sized path stays in the
# linear regime while remaining representable in the kernels' float32 k.
FD_REL_STEP_FREQUENCY = 1.0e-4

# Relative tolerances.
REL_TOL_PATH = 5.0e-3
REL_TOL_GENERAL = 5.0e-2

# Absolute tolerance floor.
ABS_TOL = 1.0e-12