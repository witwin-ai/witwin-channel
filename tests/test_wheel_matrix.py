# Copyright Xingyu Chen.
# Tests the release wheel matrix derived from ci/support-matrix.toml.

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from ci import wheel_matrix


def test_gencode_flags_build_every_cubin_once_and_the_requested_ptx():
    flags = wheel_matrix.gencode_flags([70, 75, 80, 86, 89, 90, 100, 120], [120])

    assert flags == (
        "--generate-code=arch=compute_70,code=[sm_70,sm_75] "
        "--generate-code=arch=compute_80,code=[sm_80,sm_86,sm_89] "
        "--generate-code=arch=compute_90,code=sm_90 "
        "--generate-code=arch=compute_100,code=sm_100 "
        "--generate-code=arch=compute_120,code=[sm_120,compute_120]"
    )


def test_release_profile_matches_the_matrix():
    matrix = wheel_matrix.load()
    profile = wheel_matrix.profile(matrix)

    cubins = sorted(int(sm) for sm in re.findall(r"sm_(\d+)", profile["gencode"]))
    assert cubins == sorted(matrix["sass"])
    assert profile["expected_sass"] == ",".join(map(str, matrix["sass"]))
    assert profile["cuda_architectures"].split(";") == [
        *(f"{sm}-real" for sm in matrix["sass"]),
        *(f"{sm}-virtual" for sm in matrix["ptx"]),
    ]


def test_build_matrix_has_one_row_per_python_and_torch():
    matrix = wheel_matrix.load()
    rows = wheel_matrix.build_matrix(matrix)

    assert len(rows) == len(matrix["pythons"]) * len(matrix["torch"])
    assert {row["local"] for row in rows if row["channel"] == "pypi"} == {""}
    assert all(row["local"].startswith("torch") for row in rows if row["channel"] != "pypi")
    smoke = wheel_matrix.build_matrix(matrix, smoke=True)
    assert [(row["python"], row["channel"]) for row in smoke] == [("3.11", "pypi")]


def test_expected_wheel_names_cover_every_platform():
    matrix = {
        "pythons": ["3.10", "3.13"], "platforms": ["manylinux_2_28_x86_64", "win_amd64"],
        "torch": [{"version": "2.10.0", "channel": "pypi"},
                  {"version": "2.11.0", "channel": "release-asset"}],
    }

    assert wheel_matrix.expected_wheels(matrix, "pypi", "0.6.0") == [
        "witwin_channel-0.6.0-cp310-cp310-manylinux_2_28_x86_64.whl",
        "witwin_channel-0.6.0-cp310-cp310-win_amd64.whl",
        "witwin_channel-0.6.0-cp313-cp313-manylinux_2_28_x86_64.whl",
        "witwin_channel-0.6.0-cp313-cp313-win_amd64.whl",
    ]
    assert wheel_matrix.expected_wheels(matrix, "release-asset", "0.6.0")[0] == (
        "witwin_channel-0.6.0+torch211-cp310-cp310-manylinux_2_28_x86_64.whl"
    )


def test_size_gate_rejects_oversized_wheels(tmp_path: Path):
    wheel = tmp_path / "witwin_channel-0.6.0-cp311-cp311-win_amd64.whl"
    wheel.write_bytes(b"\0" * (1024 * 1024 + 1))

    wheel_matrix.check_sizes([wheel], 2)
    with pytest.raises(SystemExit, match="over the 1 MiB limit"):
        wheel_matrix.check_sizes([wheel], 1)


def test_select_torch_rewrites_the_pin_and_version_and_keeps_the_tree_clean(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nversion = "0.6.0"\ndependencies = ["torch==2.10.0", "witwin>=0.4,<0.5"]\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "pyproject.toml"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "init"],
        check=True,
    )

    wheel_matrix.select_torch(pyproject, "2.11.0", "torch211")

    text = pyproject.read_text(encoding="utf-8")
    assert 'version = "0.6.0+torch211"' in text
    assert '"torch==2.11.0"' in text
    status = subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, check=True,
    )
    assert status.stdout == ""
