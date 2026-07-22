"""Deterministic-versus-full-wave validation harness."""

from .metrics import (
    analyze_boundaries,
    compare_fields,
    compare_magnitudes,
    comparison_report,
)
from .models import CaseSpec, FieldMap, MaterialSpec
from .scenarios import load_case

__all__ = [
    "CaseSpec",
    "FieldMap",
    "MaterialSpec",
    "analyze_boundaries",
    "compare_fields",
    "compare_magnitudes",
    "comparison_report",
    "load_case",
]
