# Copyright Xingyu Chen.
# The orphan-module gate passes here, and FAILS on a planted violation.

"""The orphan-module gate passes here, and FAILS on a planted violation."""

from __future__ import annotations

from pathlib import Path
import shutil

from ci import check_orphan_modules as orphan


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"


def _synthetic_package(tmp_path: Path, files: dict[str, str]) -> Path:
    package_root = tmp_path / "witwin" / "channel"
    for relative, source in files.items():
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package_root


def _mirror(tmp_path: Path) -> Path:
    package_root = tmp_path / "witwin" / "channel"
    shutil.copytree(
        PACKAGE_ROOT,
        package_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyd", "*.so"),
    )
    return package_root


def test_the_real_package_has_no_orphan_module():
    assert orphan.find_orphans(PACKAGE_ROOT) == []


def test_every_entry_point_exists():
    assert orphan.stale_entry_points(PACKAGE_ROOT) == []


def test_the_entry_points_are_the_public_surface():
    """The root plus the four solver entries, and nothing else."""

    assert set(orphan.ENTRY_POINTS) == {
        "witwin.channel",
        "witwin.channel.path",
        "witwin.channel.deterministic",
        "witwin.channel.montecarlo.basic",
        "witwin.channel.montecarlo.bdpt",
    }


def test_a_resurrected_dead_module_is_unreachable(tmp_path: Path):
    """A recreated ``scattering/energy.py`` module remains unreachable from every public entry point."""

    package_root = _mirror(tmp_path)
    (package_root / "scattering").mkdir()
    (package_root / "scattering" / "energy.py").write_text(
        "from __future__ import annotations\n\n"
        "import torch\n\n\n"
        "def scattered_energy(field: torch.Tensor) -> torch.Tensor:\n"
        "    return field.abs().square().sum(-1)\n",
        encoding="utf-8",
    )

    assert orphan.find_orphans(package_root) == ["witwin.channel.scattering.energy"]


def test_a_whole_dead_subpackage_is_unreachable(tmp_path: Path):
    """Two modules that import each other are still unreachable."""

    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "path/__init__.py": "",
            "capacity/__init__.py": "from .results import CapacityResult\n",
            "capacity/results.py": (
                "from witwin.channel.capacity import __name__ as _package\n\n\n"
                "class CapacityResult:\n    pass\n"
            ),
        },
    )

    assert orphan.find_orphans(package_root) == [
        "witwin.channel.capacity",
        "witwin.channel.capacity.results",
    ]


def test_a_module_reached_only_through_a_lazy_string_is_live(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": (
                "from importlib import import_module\n\n"
                "_LAZY = {'build_info': ('deployment', 'build_info')}\n\n\n"
                "def __getattr__(name):\n"
                "    module, attribute = _LAZY[name]\n"
                "    return getattr(import_module(f'.{module}', __name__), attribute)\n"
            ),
            "deployment.py": "",
        },
    )

    assert orphan.find_orphans(package_root) == []


def test_a_module_reached_only_by_a_test_is_still_an_orphan(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path, {"__init__.py": "", "oracle.py": "VALUE = 1\n"}
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_oracle.py").write_text(
        "from witwin.channel.oracle import VALUE\n", encoding="utf-8"
    )

    assert orphan.find_orphans(package_root) == ["witwin.channel.oracle"]


def test_a_stale_entry_point_is_refused(tmp_path: Path):
    package_root = _synthetic_package(tmp_path, {"__init__.py": ""})

    assert orphan.stale_entry_points(package_root) == [
        "witwin.channel.deterministic",
        "witwin.channel.montecarlo.basic",
        "witwin.channel.montecarlo.bdpt",
        "witwin.channel.path",
    ]


def test_cli_passes_with_repository_defaults(capsys):
    assert orphan.main(["--repository-root", str(REPOSITORY_ROOT)]) == 0
    assert "orphan module check passed" in capsys.readouterr().out


def test_cli_fails_and_names_the_orphan(tmp_path: Path, capsys):
    package_root = _mirror(tmp_path)
    (package_root / "scattering").mkdir()
    (package_root / "scattering" / "energy.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert (
        orphan.main(
            [
                "--repository-root",
                str(tmp_path),
                "--package-root",
                str(package_root),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "unreachable production module" in output
    assert "witwin.channel.scattering.energy" in output
    assert "witwin/channel/scattering/energy.py" in output