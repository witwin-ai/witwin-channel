# Copyright Xingyu Chen.
# Tests supported runtime matrix.

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
    project = _load_toml("pyproject.toml")["project"]
    matrix = _load_toml("ci/support-matrix.toml")

    assert matrix["schema_version"] == 2
    pythons = [tuple(int(part) for part in python.split(".")) for python in matrix["pythons"]]
    assert pythons == sorted(pythons)
    lowest, highest = pythons[0], pythons[-1]
    assert project["requires-python"] == (
        f">={lowest[0]}.{lowest[1]},<{highest[0]}.{highest[1] + 1}"
    )
    classifiers = set(project["classifiers"])
    assert {f"Programming Language :: Python :: {python}" for python in matrix["pythons"]} <= (
        classifiers
    )
    pypi = [torch for torch in matrix["torch"] if torch["channel"] == "pypi"]
    assert len(pypi) == 1
    assert f"torch=={pypi[0]['version']}" in project["dependencies"]
    assert "witwin>=0.4,<0.5" in project["dependencies"]
    assert {torch["channel"] for torch in matrix["torch"]} <= {"pypi", "release-asset"}
    assert matrix["constraints"] == "constraints/ci-py311-cu128.txt"
    assert matrix["cuda_toolkit"] == "12.8.1"
    assert set(matrix["platforms"]) == {"manylinux_2_28_x86_64", "win_amd64"}
    assert matrix["ptx"] and max(matrix["ptx"]) >= max(matrix["sass"])
    assert 0 < matrix["max_wheel_mib"] < 100
    for verified in matrix["verified"]:
        assert verified["sm"] in matrix["sass"]
        assert (ROOT / verified["evidence"]).is_file()

    constraints = _constraints()
    assert constraints["torch"] == pypi[0]["version"]
    assert constraints["scikit-build-core"] == "0.12.2"
    assert constraints["cmake"] == "4.3.0"
    assert constraints["ninja"] == "1.13.0"
    assert constraints["pytest"] == "8.4.2"
    assert constraints["ruff"] == "0.15.10"


def test_project_metadata_names_the_license_and_ships_the_notice():
    project = _load_toml("pyproject.toml")["project"]

    assert project["license"].startswith("LicenseRef-")
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert (ROOT / "NOTICE").is_file()
    assert {"Homepage", "Repository", "Issues", "License"} <= set(project["urls"])
    assert project["scripts"] == {"witwin-channel-doctor": "witwin.channel.deployment:main"}


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

def test_dev_extra_pins_every_tool_to_its_ci_constraint():
    project = _load_toml("pyproject.toml")
    constraints = _constraints()

    dev = project["project"]["optional-dependencies"]["dev"]
    pinned = dict(requirement.split("==", maxsplit=1) for requirement in dev)
    assert set(pinned) == {"coverage", "mypy", "pytest", "ruff", "scipy"}
    assert all(constraints[name] == version for name, version in pinned.items())
