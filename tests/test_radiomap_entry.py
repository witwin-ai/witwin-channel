# Copyright Xingyu Chen.
# Tests the radiomap entry point: config-type dispatch and the removed 0.5 import paths.

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from witwin.channel import path, radiomap
from witwin.channel.experimental import bdpt

ROOT = Path(__file__).resolve().parents[1]


def test_radiomap_exports_the_estimator_and_result_types():
    assert radiomap.__all__ == [
        "Coherent", "CoherentRadioMap", "CoherentRadioMapMetadata", "PathTable", "RadioMap",
        "RadioMapMetadata", "Stochastic", "solve",
    ]
    assert radiomap.Coherent is radiomap._coherent.Coherent
    assert radiomap.Stochastic is radiomap._stochastic.Stochastic
    assert radiomap.CoherentRadioMap is radiomap._coherent.CoherentRadioMap
    assert radiomap.RadioMap is radiomap._stochastic.RadioMap


def test_estimator_configs_pickle_through_the_public_module():
    for config in (radiomap.Coherent(combine="power"), radiomap.Stochastic(samples=64)):
        assert pickle.loads(pickle.dumps(config)) == config
    assert repr(radiomap.Coherent()).startswith("Coherent(")


@pytest.mark.parametrize("config", [path.Config(), bdpt.Config(), object(), None])
def test_solve_refuses_a_config_of_another_solver_before_any_work(config):
    with pytest.raises(TypeError, match=r"radiomap\.Coherent or witwin\.channel\.radiomap\.Stochastic"):
        radiomap.solve(None, config, reference_frequency_hz=3.0e9)  # type: ignore[arg-type]


def test_solve_dispatches_on_the_config_type(monkeypatch):
    calls = []
    monkeypatch.setattr(radiomap._coherent, "solve", lambda *a, **k: calls.append("coherent"))
    monkeypatch.setattr(radiomap._stochastic, "solve", lambda *a, **k: calls.append("stochastic"))

    radiomap.solve(None, radiomap.Coherent(), reference_frequency_hz=1.0)  # type: ignore[arg-type]
    radiomap.solve(None, radiomap.Stochastic(), reference_frequency_hz=1.0)  # type: ignore[arg-type]

    assert calls == ["coherent", "stochastic"]


@pytest.mark.parametrize("module,names", [
    ("witwin.channel.deterministic", ("witwin.channel.radiomap", "radiomap.Coherent")),
    ("witwin.channel.montecarlo", ("witwin.channel.radiomap", "experimental.bdpt")),
    ("witwin.channel.montecarlo.basic", ("radiomap.Stochastic", "radiomap.RadioMap")),
    ("witwin.channel.montecarlo.bdpt", ("witwin.channel.experimental.bdpt",)),
])
def test_removed_import_paths_raise_import_error_naming_the_new_path(module, names):
    # A fresh interpreter, so a failed import cannot leave a partial module behind.
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"], cwd=ROOT, capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert completed.returncode != 0
    assert "ImportError" in completed.stderr
    assert "Channel 0.6.0" in completed.stderr
    for name in names:
        assert name in completed.stderr
    assert "docs/migration-0.6.md" in completed.stderr
