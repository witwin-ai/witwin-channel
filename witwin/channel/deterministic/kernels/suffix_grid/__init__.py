"""Reflection-suffix grid accumulation backends."""

from .drjit_impl import (
    accumulate_reflected_segment_fields_batched as drjit_accumulate_reflected_segment_fields_batched,
    accumulate_reflected_segment_fields_chunk as drjit_accumulate_reflected_segment_fields_chunk,
)
from .native_impl import (
    accumulate_reflected_segment_fields_batched as native_accumulate_reflected_segment_fields_batched,
    accumulate_reflected_segment_fields_chunk as native_accumulate_reflected_segment_fields_chunk,
)

__all__ = [
    "drjit_accumulate_reflected_segment_fields_batched",
    "drjit_accumulate_reflected_segment_fields_chunk",
    "native_accumulate_reflected_segment_fields_batched",
    "native_accumulate_reflected_segment_fields_chunk",
]
