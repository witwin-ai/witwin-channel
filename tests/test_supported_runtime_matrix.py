from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def _load_toml(relative_path: str):
    with (ROOT / relative_path).open("rb") as stream:
        return tomllib.load(stream)


def _constraints():
    entries = {}
    path = ROOT / "constraints" / "ci-py311-cu128.txt"
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        entries[name.lower().replace("_", "-")] = version
    return entries


def test_supported_runtime_matrix_matches_project_metadata_and_constraints():
    project = _load_toml("pyproject.toml")
    matrix = _load_toml("ci/support-matrix.toml")["runtime"]

    assert len(matrix) == 1
    runtime = matrix[0]
    assert runtime["python_spec"] == project["project"]["requires-python"]
    assert f"torch{runtime['torch_spec']}" in project["project"]["dependencies"]
    assert runtime["constraints"] == "constraints/ci-py311-cu128.txt"
    assert runtime["verified_sm"] == [120]
    assert runtime["declared_unverified_sm"] == [70, 75, 80, 86, 87, 89, 90, 100, 101]

    constraints = _constraints()
    assert constraints["torch"] == "2.10.0"
    assert constraints["scikit-build-core"] == "0.12.2"
    assert constraints["cmake"] == "4.3.0"
    assert constraints["ninja"] == "1.13.0"
    assert constraints["pytest"] == "8.4.2"
    assert constraints["ruff"] == "0.15.10"


def test_build_requirements_are_bounded_and_have_exact_ci_constraints():
    project = _load_toml("pyproject.toml")
    constraints = _constraints()

    requirements = project["build-system"]["requires"]
    assert requirements == [
        "cmake>=4.3,<4.4",
        "ninja>=1.13,<1.14",
        "scikit-build-core>=0.12,<0.13",
    ]
    for package in ("cmake", "ninja", "scikit-build-core"):
        assert package in constraints
