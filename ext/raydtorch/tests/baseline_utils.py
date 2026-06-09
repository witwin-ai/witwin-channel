from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASELINE_ROOT = Path(__file__).resolve().parent / "baselines"


def load_baseline(*parts: str) -> Any:
    path = BASELINE_ROOT.joinpath(*parts)
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
