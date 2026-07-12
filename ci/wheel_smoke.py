from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install a built wheel into an isolated target and smoke its native ABI."
    )
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if wheel.suffix != ".whl" or not wheel.is_file():
        parser.error(f"wheel does not exist: {wheel}")

    with tempfile.TemporaryDirectory(prefix="channel-native-wheel-smoke-") as raw:
        target = Path(raw) / "site-packages"
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if install.returncode != 0:
            print(install.stderr, file=sys.stderr)
            return install.returncode

        code = (
            "import json,sys;"
            f"sys.path.insert(0,{str(target)!r});"
            "import witwin.channel_native as c;"
            "info=c.build_info();"
            "print(json.dumps({'wheel_smoke':True,'build_info':info},sort_keys=True))"
        )
        smoke = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=target,
            capture_output=True,
            text=True,
            check=False,
        )
        if smoke.stdout:
            print(smoke.stdout.strip())
        if smoke.stderr:
            print(smoke.stderr.strip(), file=sys.stderr)
        return smoke.returncode


if __name__ == "__main__":
    raise SystemExit(main())
