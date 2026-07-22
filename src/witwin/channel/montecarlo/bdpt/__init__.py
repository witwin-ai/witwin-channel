"""Native CUDA/OptiX BDPT Monte Carlo solver with Torch tensor storage."""

from typing import TYPE_CHECKING

from .config import Config
from .result import BDPTPathSamples, Result

if TYPE_CHECKING:
    from .solver import solve

__all__ = ["BDPTPathSamples", "Config", "Result", "solve"]


def __getattr__(name: str):
    if name != "solve":
        raise AttributeError(name)
    from .solver import solve

    globals()[name] = solve
    return solve
