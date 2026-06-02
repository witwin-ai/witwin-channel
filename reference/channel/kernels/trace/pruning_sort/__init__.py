"""GPU-native pruning sort for state budget enforcement."""

from .drjit_impl import _state_pruning_metric

def _use_native():
    try:
        from witwin.channel._native import extension_available

        return extension_available()
    except ImportError:
        return False

if _use_native():
    from .native_impl import prune_state_arrays_by_budget, prune_state_arrays_by_budget_pair
else:
    from .drjit_impl import prune_state_arrays_by_budget, prune_state_arrays_by_budget_pair


__all__ = [
    "_state_pruning_metric",
    "prune_state_arrays_by_budget",
    "prune_state_arrays_by_budget_pair",
]

