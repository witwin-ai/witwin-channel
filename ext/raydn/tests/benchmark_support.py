from __future__ import annotations

import time
from typing import Callable

import torch


def synchronize() -> None:
    torch.cuda.synchronize()


def time_ms(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat
