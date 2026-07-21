"""Stable CUDA profiler annotations for architecture evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from functools import wraps
from typing import ParamSpec, TypeVar

import torch


class CudaProfileRange(StrEnum):
    """Closed set of semantic ranges consumed by performance evidence."""

    ENUMERATED_PENETRATION_DISCOVERY = (
        "witwin.channel_native:enumerated_penetration_discovery"
    )
    MONTECARLO_BASIC_PENETRATION_DISCOVERY = (
        "witwin.channel_native:montecarlo_basic_penetration_discovery"
    )
    DIFFRACTION_EXPORTER = "witwin.channel_native:diffraction_exporter"
    CAPACITY_STATUS = "witwin.channel_native:capacity_status"
    DIFFRACTION_PAIR_REDUCER = "witwin.channel_native:diffraction_pair_reducer"
    DIFFRACTION_TOPOLOGY_PACKING = (
        "witwin.channel_native:diffraction_topology_packing"
    )
    DIFFRACTION_TOTAL_STAGE = "witwin.channel_native:diffraction_total_stage"


class CudaProfileMark(StrEnum):
    """Closed set of semantic point annotations consumed by the runner."""

    OPTIX_TRAVERSAL = "witwin.channel_native:optix_traversal"
    DIFFRACTION_EXPORTER_REQUEST = (
        "witwin.channel_native:diffraction_exporter_request"
    )


@contextmanager
def cuda_profile_range(name: CudaProfileRange) -> Iterator[None]:
    """Emit one balanced NVTX range without CUDA work or synchronization."""

    torch.cuda.nvtx.range_push(name.value)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


_P = ParamSpec("_P")
_R = TypeVar("_R")


def profiled_cuda_range(
    name: CudaProfileRange,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap an operation owner in one balanced semantic NVTX range."""

    def decorate(operation: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(operation)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with cuda_profile_range(name):
                return operation(*args, **kwargs)

        return wrapped

    return decorate


def cuda_profile_mark(name: CudaProfileMark) -> None:
    """Emit one semantic NVTX point annotation without CUDA work."""

    torch.cuda.nvtx.mark(name.value)


__all__ = [
    "CudaProfileMark",
    "CudaProfileRange",
    "cuda_profile_mark",
    "cuda_profile_range",
    "profiled_cuda_range",
]
