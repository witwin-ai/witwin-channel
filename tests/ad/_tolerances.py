"""Shared AD gradient-test tolerances and finite-difference steps.

Single source of truth for the AD test suite (plan 07 section 9.2). Do not
copy these constants into individual test files.
"""

# Central finite-difference steps, calibrated per parameter magnitude.
FD_STEP_POSITION = 1.0e-2
FD_STEP_GEOMETRY = 1.0e-3

# Relative tolerances.
REL_TOL_PATH = 5.0e-3
REL_TOL_GENERAL = 5.0e-2

# Absolute tolerance floor.
ABS_TOL = 1.0e-12
