from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_and_doc_files_under(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root_name in roots:
        root = ROOT / root_name
        for pattern in ("*.py", "*.md", "*.ipynb"):
            files.extend(
                path
                for path in root.rglob(pattern)
                if "__pycache__" not in path.parts
            )
    files.extend(ROOT.glob("*.md"))
    return files


def test_phase5_channel_owns_solvers_config_and_native_helpers() -> None:
    channel_root = ROOT / "witwin" / "channel"

    assert (channel_root / "_config" / "base.py").is_file()
    assert (channel_root / "_native" / "loader.py").is_file()
    assert (channel_root / "_native" / "deterministic.py").is_file()
    assert (channel_root / "_native" / "montecarlo.py").is_file()
    assert (channel_root / "types.py").is_file()
    assert (channel_root / "deterministic" / "solver.py").is_file()
    assert (channel_root / "montecarlo" / "solver.py").is_file()
    assert (channel_root / "path" / "solver.py").is_file()

    assert not (ROOT / "witwin" / "_config").exists()
    assert not (ROOT / "witwin" / "_native").exists()
    assert not (ROOT / "witwin" / "deterministic").exists()
    assert not (ROOT / "witwin" / "montecarlo").exists()
    assert not (ROOT / "witwin" / "path").exists()
    assert not (ROOT / "witwin" / "types.py").exists()
    assert not (channel_root / "paths").exists()


def test_phase5_public_channel_exports_solver_namespaces() -> None:
    import witwin.channel as wc
    import witwin.channel.deterministic as deterministic
    import witwin.channel.montecarlo as montecarlo
    import witwin.channel.path as path

    assert wc.deterministic is deterministic
    assert wc.montecarlo is montecarlo
    assert wc.path is path
    assert not hasattr(wc, "paths")


def test_phase5_imports_do_not_reference_removed_top_level_packages() -> None:
    forbidden = (
        b"witwin.deterministic",
        b"witwin.montecarlo",
        b"witwin.path",
        b"witwin._config",
        b"witwin._native",
        b"witwin.types",
        b"witwin.channel.paths",
        b"wc.paths",
        b"import witwin as wt",
        b"from witwin import Float",
        b"from witwin import Bool",
        b"from witwin import Point",
        b"from witwin import Vector",
        b"from witwin import Complex",
        b"from witwin import Matrix",
        b"from witwin import deterministic",
        b"from witwin import montecarlo",
        b"from witwin import path",
    )

    offenders: list[str] = []
    for path in _python_and_doc_files_under("witwin", "tests", "examples", "docs"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_bytes()
        for needle in forbidden:
            if needle in text:
                offenders.append(
                    f"{path.relative_to(ROOT)} contains {needle.decode('ascii')}"
                )

    assert offenders == []
