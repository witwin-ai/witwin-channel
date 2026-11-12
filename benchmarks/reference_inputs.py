# Copyright Xingyu Chen.
# Resolves the external scene assets and reference checkouts that comparison benchmarks need.

"""External scene assets and reference checkouts of the comparison benchmarks."""

from __future__ import annotations

import os
from pathlib import Path


# City scenes resolve from an explicit path or $WITWIN_CHANNEL_SCENE_ROOT and an archived Channel
# reference checkout from an explicit path or $WITWIN_CHANNEL_ORIGINAL_ROOT. Nothing is guessed from
# sibling directories, and a missing input raises instead of substituting a synthetic scene. The
# module imports no witwin package, so a reference worker can still import an archived checkout.
SCENE_ROOT_ENV = "WITWIN_CHANNEL_SCENE_ROOT"
ORIGINAL_CHANNEL_ROOT_ENV = "WITWIN_CHANNEL_ORIGINAL_ROOT"


class FullScenarioAssetError(RuntimeError):
    """Raised when a required full city asset cannot be resolved."""


def scene_asset_root(explicit: str | Path | None) -> Path:
    """``explicit`` or ``$WITWIN_CHANNEL_SCENE_ROOT`` as an existing directory."""

    value = explicit if explicit is not None else os.environ.get(SCENE_ROOT_ENV)
    if value is None or not str(value).strip():
        raise FullScenarioAssetError(
            "full Munich/SF scenarios require an explicit asset_root or "
            f"the {SCENE_ROOT_ENV} environment variable; no reduced fallback is used"
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FullScenarioAssetError(f"Phase E scene asset root is not a directory: {root}")
    return root


def city_xml_under(root: Path, city: str) -> Path:
    """The ``city`` XML in the Sionna source, ``<city>/`` or flat layout under ``root``."""

    candidates = (
        root / "sionna" / "rt" / "scenes" / city / f"{city}.xml",
        root / city / f"{city}.xml",
        root / f"{city}.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FullScenarioAssetError(
        f"full {city} scene is unavailable under {root}; searched: {searched}. "
        "No reduced or synthetic city scene is substituted."
    )


def env_path(name: str) -> Path | None:
    """The path named by environment variable ``name``, or ``None`` when it is unset or empty."""

    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def default_city_xml(city: str) -> Path | None:
    """The ``city`` XML under ``$WITWIN_CHANNEL_SCENE_ROOT``, or ``None`` when it is not available."""

    try:
        return city_xml_under(scene_asset_root(None), city)
    except FullScenarioAssetError:
        return None


def require_city_xml(city: str, scene_xml: str | Path | None) -> Path:
    """An explicit scene XML, else the one under ``$WITWIN_CHANNEL_SCENE_ROOT``; raises when absent."""

    if scene_xml is None or not str(scene_xml).strip():
        return city_xml_under(scene_asset_root(None), city)
    path = Path(scene_xml).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{city} scene XML is not a file: {path}")
    return path.resolve()


def sionna_source_root_of(scene_xml: Path) -> Path | None:
    """The Sionna source root that holds ``scene_xml`` in its ``sionna/rt/scenes/<city>/`` layout."""

    parents = scene_xml.resolve().parents
    if len(parents) > 4 and [parents[3].name, parents[2].name, parents[1].name] == [
        "sionna", "rt", "scenes",
    ]:
        return parents[4]
    return None


__all__ = [
    "ORIGINAL_CHANNEL_ROOT_ENV",
    "SCENE_ROOT_ENV",
    "FullScenarioAssetError",
    "city_xml_under",
    "default_city_xml",
    "env_path",
    "require_city_xml",
    "scene_asset_root",
    "sionna_source_root_of",
]
