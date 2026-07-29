# Copyright Xingyu Chen.
# Checks that native scalar, vector, and complex helpers have one shared owner.

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = ROOT / "native" / "channel"
MATH_OWNER = NATIVE_ROOT / "kernels" / "math.cuh"
NATIVE_SUFFIXES = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".cu", ".cuh"}

SIMPLE_TYPE = re.compile(
    r"\b(?:struct|class)\s+"
    r"(?:Float3|Vec3|Complex|Complex3|SubpathComplex|SubC)\b"
)
LOCAL_VECTOR_FUNCTION = re.compile(
    r"__device__[^\n{;]*\b"
    r"(?:make_f3|load_f3|load3|store3|add3|sub3|mul3|scale3|dot3|"
    r"cross3|norm3|length3|normalize3|normalized3)\s*\("
)
LOCAL_COMPLEX_FUNCTION = re.compile(
    r"__device__[^\n{;]*\b"
    r"(?:c_make|c_add|c_sub|c_mul|c_scale|c_div|c_abs2|"
    r"sp_c_\w+|subc_\w+)\s*\("
)


def main() -> int:
    if not MATH_OWNER.is_file():
        print(f"missing shared native math owner: {MATH_OWNER.relative_to(ROOT)}")
        return 1

    errors: list[str] = []
    for path in NATIVE_ROOT.rglob("*"):
        if path == MATH_OWNER or path.suffix.lower() not in NATIVE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig")
        relative = path.relative_to(ROOT).as_posix()
        for label, pattern in (
            ("simple math type", SIMPLE_TYPE),
            ("local vector helper", LOCAL_VECTOR_FUNCTION),
            ("local complex helper", LOCAL_COMPLEX_FUNCTION),
        ):
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{relative}:{line}: {label} belongs in "
                    "native/channel/kernels/math.cuh"
                )

    if (ROOT / "witwin" / "channel" / "tensor_math.py").exists():
        errors.append(
            "witwin/channel/tensor_math.py: generic Python vector math belongs to "
            "witwin.core.math"
        )

    if errors:
        print("\n".join(errors))
        return 1
    print("shared math: one native owner and no Channel tensor_math module")
    return 0


if __name__ == "__main__":
    sys.exit(main())