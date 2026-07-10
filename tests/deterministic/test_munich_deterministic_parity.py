import json
from pathlib import Path
import subprocess
import uuid

import pytest
import torch

import tests.support.bin.benchmark_munich_deterministic_native_vs_original as munich_bench
from tests.support.bin.benchmark_munich_deterministic_native_vs_original import (
    DEFAULT_MUNICH_XML,
    _run_original,
    _load_scene,
    _parser,
    run,
)
from witwin.channel_native.deterministic import Config, solve


def test_reduced_munich_deterministic_parity_emits_artifacts():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Munich deterministic parity")
    if not DEFAULT_MUNICH_XML.exists():
        pytest.skip("Munich reference scene is not available")

    artifact_dir = Path(__file__).resolve().parents[2] / "artifacts" / f"test_munich_{uuid.uuid4().hex}"
    args = _parser().parse_args(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--grid-size",
            "32",
            "--max-depth",
            "2",
            "--original-timeout-seconds",
            "240",
        ]
    )
    payload = run(args)

    metadata_path = Path(payload["artifacts"]["metadata"])
    assert metadata_path.exists()
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved["native"]["path_gain_shape"] == [1, 32, 32]
    assert saved["warmup_runs"] == 1
    assert saved["original"]["metadata"]["max_bounces"] == saved["max_depth"]
    assert saved["original"]["metadata"]["enable_rd_diffraction"] is False
    assert saved["original"]["component_finite_counts"]["los"] > 0
    assert saved["original"]["component_finite_counts"]["reflection"] > 0
    assert saved["original"]["component_finite_counts"]["diffraction"] > 0
    assert saved["max_depth"] == 2
    assert saved["original"]["available"] is True
    assert saved["delta"]["finite_count"] > 0
    assert saved["delta"]["max_abs_delta_db"] == pytest.approx(saved["delta"]["max_abs_delta_db"])
    assert saved["delta"]["median_abs_delta_db"] == pytest.approx(saved["delta"]["median_abs_delta_db"])
    assert saved["delta"]["max_abs_delta_db"] < 250.0
    assert saved["delta"]["median_abs_delta_db"] < 20.0
    assert saved["component_delta"]["los"]["max_abs_delta_db"] < 1.0e-3
    assert saved["component_delta"]["reflection"]["median_abs_delta_db"] < 1.0
    assert saved["component_delta"]["diffraction"]["median_abs_delta_db"] < 25.0
    assert saved["native"]["metadata"]["kernel"]["launch_count"] <= 10
    assert saved["performance"]["native_solve_time_ms"] < saved["performance"]["original_solve_time_ms"]
    assert saved["performance"]["original_solve_time_ms"] > 0.0
    assert saved["native"]["path_count_histogram"]["los"] > 0
    assert saved["native"]["path_count_histogram"]["reflection"] > 0
    assert saved["native"]["path_count_histogram"]["diffraction"] > 0
    assert saved["native"]["component_shapes"] == saved["original"]["component_shapes"]
    for key in (
        "native_total",
        "original_total",
        "native_los",
        "native_reflection",
        "native_diffraction",
        "delta_db",
        "delta_los_db",
        "delta_reflection_db",
        "delta_diffraction_db",
    ):
        artifact = Path(payload["artifacts"][key])
        assert artifact.exists()
        assert artifact.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_munich_benchmark_defaults_to_stage8_depth_two():
    from benchmarks import bench_deterministic_munich

    captured = {}

    def fake_parity_main(argv):
        captured["argv"] = argv

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(bench_deterministic_munich, "parity_main", fake_parity_main)
    monkeypatch.setattr("sys.argv", ["bench_deterministic_munich.py"])
    try:
        bench_deterministic_munich.main()
    finally:
        monkeypatch.undo()

    depth_index = captured["argv"].index("--max-depth") + 1
    assert captured["argv"][depth_index] == "2"
    warmup_index = captured["argv"].index("--warmup-runs") + 1
    assert captured["argv"][warmup_index] == "1"


def test_reduced_munich_native_depth_two_exports_reflection_paths():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Munich deterministic native depth-two coverage")
    if not DEFAULT_MUNICH_XML.exists():
        pytest.skip("Munich reference scene is not available")

    args = _parser().parse_args(["--grid-size", "32", "--max-depth", "2"])
    scene = _load_scene(args)

    result = solve(
        scene,
        Config(
            max_depth=2,
            max_diffraction_order=1,
            components={"los", "reflection", "diffraction"},
            coherent=False,
            return_field=False,
            export_paths=True,
            diagnostics=True,
        ),
    )

    assert result.paths is not None
    assert result.path_gain.shape == (1, 32, 32)
    assert bool(((result.paths.component_id == 1) & (result.paths.depth == 2)).any())
    assert result.diagnostics is not None
    assert result.diagnostics["path_planning"]["guardrail_count"] == 0
    assert result.diagnostics["path_planning"]["candidate_count"] < 200_000
    # Diffraction chunks receivers to bound the rx x edge-state workspace
    # (audit P-2), trading a handful of extra launches for city-scale memory
    # safety; the bound still guards against per-pair launch storms (P-4).
    assert result.diagnostics["native_launch_count"] <= 24


def test_original_munich_worker_timeout_returns_unavailable(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0] if args else []), timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    channel_root = Path(__file__).resolve().parents[2]
    artifact_dir = channel_root / "artifacts" / f"test_original_timeout_{uuid.uuid4().hex}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args = _parser().parse_args(
        [
            "--channel-root",
            str(channel_root),
            "--original-timeout-seconds",
            "0.01",
        ]
    )

    original, tensors = _run_original(args, artifact_dir)

    assert tensors == {}
    assert original["available"] is False
    assert original["reason"] == "original subprocess timed out"
    assert original["timeout_seconds"] == pytest.approx(0.01)


def test_original_worker_defaults_to_direct_diffraction_without_rd_mixed_path():
    args = _parser().parse_args([])

    assert args.original_enable_rd_diffraction is False
    worker_code = munich_bench._original_worker_code()
    assert "parser.add_argument(\"--enable-rd-diffraction\", action=\"store_true\")" in worker_code
    assert "enable_rd_diffraction=bool(args.enable_rd_diffraction)" in worker_code
    assert "max_diffraction_order=1" in worker_code


def test_run_original_sanitizes_pythonpath_and_disables_rd_by_default(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        output = Path(command[command.index("--output") + 1])
        torch.save(
            {
                "path_gain": torch.ones((1, 1), dtype=torch.float32),
                "components": {
                    "los": torch.ones((1, 1), dtype=torch.float32),
                    "reflection": torch.ones((1, 1), dtype=torch.float32),
                    "diffraction": torch.ones((1, 1), dtype=torch.float32),
                },
                "metadata": {"max_bounces": 2},
            },
            output,
        )

        class Completed:
            returncode = 0
            stdout = "{\"available\": true}\n"
            stderr = ""

        return Completed()

    monkeypatch.setenv("PYTHONPATH", "polluted")
    monkeypatch.setattr(subprocess, "run", fake_run)
    channel_root = Path(__file__).resolve().parents[2]
    artifact_dir = channel_root / "artifacts" / f"test_original_env_{uuid.uuid4().hex}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args = _parser().parse_args(["--channel-root", str(channel_root), "--max-depth", "2"])

    original, tensors = _run_original(args, artifact_dir)

    assert original["available"] is True
    assert tensors["path_gain"].shape == (1, 1)
    assert "--enable-rd-diffraction" not in captured["command"]
    assert captured["command"][captured["command"].index("--warmup-runs") + 1] == "1"
    assert captured["env"].get("PYTHONPATH") is None
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"


def test_run_executes_original_before_native(monkeypatch):
    order = []

    class FakePaths:
        component_id = torch.tensor([0, 1, 2], dtype=torch.int32)
        valid = torch.ones((3,), dtype=torch.bool)

    class FakeResult:
        path_gain = torch.ones((1, 1, 1), dtype=torch.float32)
        component_power = {
            "los": torch.ones((1, 1, 1), dtype=torch.float32),
            "reflection": torch.ones((1, 1, 1), dtype=torch.float32),
            "diffraction": torch.ones((1, 1, 1), dtype=torch.float32),
        }
        metadata = {"fake": True}
        paths = FakePaths()

    def fake_run_original(args, artifact_dir):
        order.append("original")
        return (
            {"available": True, "metadata": {"max_bounces": int(args.max_depth)}},
            {
                "path_gain": torch.ones((1, 1), dtype=torch.float32),
                "los": torch.ones((1, 1), dtype=torch.float32),
                "reflection": torch.ones((1, 1), dtype=torch.float32),
                "diffraction": torch.ones((1, 1), dtype=torch.float32),
            },
        )

    def fake_solve(scene, config):
        order.append("native")
        return FakeResult()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(munich_bench, "_run_original", fake_run_original)
    monkeypatch.setattr(munich_bench, "_load_scene", lambda args: object())
    monkeypatch.setattr(munich_bench, "solve", fake_solve)
    artifact_dir = Path(__file__).resolve().parents[2] / "artifacts" / f"test_order_{uuid.uuid4().hex}"
    args = _parser().parse_args(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--scene-xml",
            __file__,
            "--grid-size",
            "1",
            "--max-depth",
            "2",
        ]
    )

    payload = munich_bench.run(args)

    assert order == ["original", "native", "native"]
    assert payload["max_depth"] == 2
