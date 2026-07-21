"""Runtime ownership for the compiled Channel Native extension and symbols."""

from .capacity import (
    CapacityFailureBit,
    CapacityFailureState,
    capacity_failure_terminal_check,
    create_capacity_failure_state,
)
from .extension import build_info
from .symbols import (
    NativeSymbolError,
    has_symbol,
    native_extension,
    optional_symbol,
    required_symbol,
)

__all__ = [
    "NativeSymbolError",
    "CapacityFailureBit",
    "CapacityFailureState",
    "build_info",
    "capacity_failure_terminal_check",
    "create_capacity_failure_state",
    "has_symbol",
    "native_extension",
    "optional_symbol",
    "required_symbol",
]
