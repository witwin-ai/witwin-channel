"""Execute one canonical evidence script against a runner-built installation."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    site_packages = args.site_packages.resolve(strict=True)
    script = args.script.resolve(strict=True)
    if not site_packages.is_dir() or not script.is_file():
        raise RuntimeError("evidence bootstrap inputs are not regular installation/script paths")
    sys.path.insert(0, str(site_packages))
    sys.argv = [str(script), *args.arguments]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
