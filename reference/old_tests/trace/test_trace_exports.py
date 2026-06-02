"""Regression tests for the trace package exports."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin.channel.trace as trace_pkg
from witwin.channel.trace.los import compute_los_field
from witwin.channel.trace.reflection import compute_reflection_field
from witwin.channel.trace.tracer import Tracer


def test_trace_package_re_exports_solver_symbols():
    assert trace_pkg.Tracer is Tracer
    assert trace_pkg.compute_los_field is compute_los_field
    assert trace_pkg.compute_reflection_field is compute_reflection_field


def test_legacy_trace_diffraction_module_is_removed():
    assert importlib.util.find_spec("witwin.channel.trace_diffraction") is None


def test_diffraction_package_moved_under_trace():
    assert importlib.util.find_spec("witwin.channel.trace.diffraction") is not None
    assert importlib.util.find_spec("witwin.channel.diffraction") is None
