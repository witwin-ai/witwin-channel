# Copyright Xingyu Chen.
# Tests the generated native binding manifest and its ownership and liveness checks.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ci import check_native_bindings as check


ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_SPLIT = check.KNOWN_SPLIT_OWNER_SYMBOLS

_BINDING = """\
#include "registry.h"
torch::Tensor channel_alpha(torch::Tensor values, int64_t count);
torch::Tensor channel_beta(torch::Tensor values);
void register_demo(pybind11::module_& module) {
    module.def("alpha", &channel_alpha, "Alpha op.", pybind11::arg("values"), pybind11::arg("count") = 1);
    module.def("beta", &channel_beta, "Beta op.", pybind11::arg("values"));
}
"""


def _repo(tmp_path: Path, kernels: str, solver: str) -> Path:
    (tmp_path / "native/channel/binding").mkdir(parents=True)
    (tmp_path / "native/channel/binding/demo.cpp").write_text(_BINDING, encoding="utf-8")
    package = tmp_path / "witwin/channel"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(kernels, encoding="utf-8")
    (package / "solver.py").write_text(solver, encoding="utf-8")
    (tmp_path / "ci").mkdir()
    snapshot = {
        "modules": [
            {"module": "witwin.channel.solver", "exports": [{"name": "solve", "target": "witwin.channel.solver.solve"}]}
        ]
    }
    (tmp_path / "ci/public-api-snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    check._parsed.cache_clear()
    return tmp_path


_KERNELS = """\
from witwin.channel.runtime import required_symbol as _required_native_op


def alpha(values):
    return _required_native_op("alpha")(values, 1)


def beta(values):
    return _required_native_op("beta")(values)
"""

_SOLVER_USES_BOTH = """\
from witwin.channel import kernels
from witwin.channel.kernels import beta as _beta


def solve(values):
    return _helper(values)


def _helper(values):
    return kernels.alpha(values) + _beta(values)
"""


@pytest.fixture(autouse=True)
def _no_known_debt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "KNOWN_SPLIT_OWNER_SYMBOLS", frozenset())


def test_live_symbols_with_one_owner_pass(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _KERNELS, _SOLVER_USES_BOTH)

    assert check.check_invariants(repo) == []
    manifest = check.generate_manifest(repo)
    rows = {row["name"]: row for row in manifest["symbols"]}
    assert rows["alpha"]["python_owners"] == ["witwin.channel.kernels.alpha"]
    assert rows["alpha"]["parameters"] == [["values", "torch::Tensor", None], ["count", "int64_t", None]]
    assert rows["alpha"]["pybind_arguments"] == [["values", None], ["count", "1"]]
    assert "line" not in rows["alpha"] and "path" not in rows["alpha"]


def test_a_facade_reached_only_from_dead_code_is_reported(tmp_path: Path) -> None:
    solver = """\
from witwin.channel import kernels


def solve(values):
    return kernels.alpha(values)


def _unused(values):
    return kernels.beta(values)
"""
    repo = _repo(tmp_path, _KERNELS, solver)

    assert check.check_invariants(repo) == ["native symbol has no production caller: beta"]


def test_tests_do_not_count_as_production_callers(tmp_path: Path) -> None:
    solver = "from witwin.channel import kernels\n\n\ndef solve(values):\n    return kernels.alpha(values)\n"
    repo = _repo(tmp_path, _KERNELS, solver)
    (repo / "tests").mkdir()
    (repo / "tests/test_beta.py").write_text("from witwin.channel.kernels import beta\nbeta(1)\n", encoding="utf-8")

    assert "native symbol has no production caller: beta" in check.check_invariants(repo)


def test_a_symbol_dispatched_from_two_modules_is_reported(tmp_path: Path) -> None:
    solver = _SOLVER_USES_BOTH + (
        "\n\ndef direct(values):\n"
        "    from witwin.channel.runtime import required_symbol\n"
        "    return required_symbol('beta')(values)\n"
        "\n\nDIRECT = direct\n"
    )
    repo = _repo(tmp_path, _KERNELS, solver)

    assert check.check_invariants(repo) == ["native symbol is dispatched from more than one module: beta"]


def test_a_symbol_without_a_dispatch_site_is_reported(tmp_path: Path) -> None:
    kernels = _KERNELS.replace('_required_native_op("beta")', "_required_native_op(name)")
    repo = _repo(tmp_path, kernels, _SOLVER_USES_BOTH.replace("+ _beta(values)", ""))

    assert check.check_invariants(repo) == ["native symbol has no Python dispatch site: beta"]


_SPEC_KERNELS = """\
from witwin.channel.kernels.spec import NativeOps, integer, tensor
from witwin.channel.runtime import required_symbol as _required_native_op

_native = NativeOps(lambda symbol: _required_native_op(symbol))
_ALPHA = _native.op("alpha", tensor("values"), integer("count"))
_BETA = _native.op("beta", tensor("values"))


def alpha(values):
    return _ALPHA(values, 1)


def beta(values):
    return _BETA(values)
"""


def test_declared_ops_are_owned_by_the_definitions_that_dispatch_them(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _SPEC_KERNELS, _SOLVER_USES_BOTH)

    assert check.check_invariants(repo) == []
    rows = {row["name"]: row for row in check.generate_manifest(repo)["symbols"]}
    assert rows["alpha"]["python_owners"] == ["witwin.channel.kernels.alpha"]
    assert rows["beta"]["python_owners"] == ["witwin.channel.kernels.beta"]


def test_a_declared_op_reached_only_from_dead_code_is_reported(tmp_path: Path) -> None:
    solver = "from witwin.channel import kernels\n\n\ndef solve(values):\n    return kernels.alpha(values)\n"
    repo = _repo(tmp_path, _SPEC_KERNELS, solver)

    assert check.check_invariants(repo) == ["native symbol has no production caller: beta"]


def test_the_declaration_resolver_does_not_make_symbol_literals_dispatch_sites(
    tmp_path: Path,
) -> None:
    kernels = _SPEC_KERNELS.replace(
        '_BETA = _native.op("beta", tensor("values"))\n', ""
    ).replace("    return _BETA(values)", '    return "beta"')
    repo = _repo(tmp_path, kernels, _SOLVER_USES_BOTH)

    assert check.check_invariants(repo) == ["native symbol has no Python dispatch site: beta"]


@pytest.mark.parametrize(
    ("declaration", "problem"),
    [
        ('tensor("count"), integer("values")', "arguments ['count', 'values'] differ"),
        ('tensor("values")', "arguments ['values'] differ"),
        ('tensor("values"), integer("count"), integer("extra")', "arguments ['values', 'count', 'extra'] differ"),
    ],
)
def test_a_declaration_must_match_the_cpp_parameters(tmp_path: Path, declaration: str, problem: str) -> None:
    kernels = _SPEC_KERNELS.replace('tensor("values"), integer("count")', declaration)
    repo = _repo(tmp_path, kernels, _SOLVER_USES_BOTH)

    problems = check.check_invariants(repo)
    assert len(problems) == 1
    assert problems[0].startswith("witwin.channel.kernels._ALPHA ") and problem in problems[0]


def test_known_debt_entries_must_shrink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, _KERNELS, _SOLVER_USES_BOTH)
    monkeypatch.setattr(check, "KNOWN_SPLIT_OWNER_SYMBOLS", frozenset({"beta"}))

    assert check.check_invariants(repo) == [
        "KNOWN_SPLIT_OWNER_SYMBOLS entry is stale (now single-owner or deleted): beta",
    ]


def test_there_is_no_uncalled_symbol_allowance() -> None:
    assert not hasattr(check, "KNOWN_UNCALLED_SYMBOLS")


def test_a_generation_suffixed_binding_name_is_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _KERNELS, _SOLVER_USES_BOTH)
    binding = repo / "native/channel/binding/demo.cpp"
    binding.write_text(_BINDING.replace('"beta"', '"beta_v2"'), encoding="utf-8")
    (repo / "witwin/channel/kernels.py").write_text(_KERNELS.replace('"beta"', '"beta_v2"'), encoding="utf-8")
    check._parsed.cache_clear()

    assert "binding name carries a generation or work-in-progress marker: beta_v2" in check.check_invariants(repo)


def test_moving_a_binding_between_files_leaves_the_manifest_unchanged(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _KERNELS, _SOLVER_USES_BOTH)
    before = check.render_manifest(check.generate_manifest(repo))
    binding = repo / "native/channel/binding/demo.cpp"
    binding.write_text("// A comment shifts every line.\n\n" + _BINDING, encoding="utf-8")
    binding.rename(repo / "native/channel/binding/other.cpp")
    check._parsed.cache_clear()

    assert check.render_manifest(check.generate_manifest(repo)) == before


def test_the_cli_reports_a_stale_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path, _KERNELS, _SOLVER_USES_BOTH)

    assert check.main(["--repo", str(repo), "--write"]) == 0
    manifest = repo / "ci/native-binding-manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"Alpha op."', '"Old doc."'), encoding="utf-8")
    capsys.readouterr()
    assert check.main(["--repo", str(repo)]) == 1
    captured = capsys.readouterr()
    assert '+      "doc": "Alpha op.",' in captured.out
    assert "is out of date" in captured.err


def test_the_repository_manifest_is_current(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "KNOWN_SPLIT_OWNER_SYMBOLS", _REPOSITORY_SPLIT)
    check._parsed.cache_clear()
    assert check.main(["--repo", str(ROOT)]) == 0


@pytest.mark.parametrize(("name", "rejected"), [("beta_v2", True), ("v3_beta", True), ("beta_" + "w" + "ip", True), ("vec3_pack", False)])
def test_generation_marker_pattern(name: str, rejected: bool) -> None:
    assert bool(check._FORBIDDEN_NAME.search(name)) is rejected
