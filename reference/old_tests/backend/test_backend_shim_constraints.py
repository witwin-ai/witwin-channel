from __future__ import annotations

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = (
    REPO_ROOT / "witwin" / "channel",
    REPO_ROOT / "tests",
    REPO_ROOT / "basic",
    REPO_ROOT / "samples",
)


def test_repo_contains_no_direct_mitsuba_imports():
    direct_import_markers = ("import " + "mitsuba", "from " + "mitsuba")
    offenders = []
    for root in SCANNED_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in text for marker in direct_import_markers):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_runtime_module_exposes_no_mitsuba_runtime_helpers():
    import witwin as wt

    removed = (
        "ensure_variant",
        "mitsuba_module",
        "set_variant",
        "variant",
        "to_mitsuba_ray",
        "ScalarPoint3f",
        "ScalarVector3f",
        "scalar_transform4f",
    )

    for name in removed:
        assert not hasattr(wt.runtime, name)


def test_witwin_root_exports_runtime_aliases():
    import rayd
    import witwin as wt
    import witwin.channel.types as channel_types
    import witwin.channel.utils.transform as channel_transform

    assert wt.runtime is wt.channel.runtime
    assert wt.Vector2f is channel_types.Vector2f
    assert wt.Transform4f is channel_transform.Transform4f
    assert wt.Ray is rayd.Ray
    assert not hasattr(wt, "Ray3f")
    assert not hasattr(wt, "rayd_module")
    assert not hasattr(wt, "to_rayd_ray")
    assert wt.core is not None
    assert wt.channel is not None
