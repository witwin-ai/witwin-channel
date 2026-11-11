# Copyright Xingyu Chen.
# Checks that native math helpers, physical constants and launch helpers have one shared owner.

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = ROOT / "native" / "channel"
MATH_OWNER = NATIVE_ROOT / "kernels" / "math.cuh"
CONSTANTS_OWNER = NATIVE_ROOT / "kernels" / "constants.cuh"
LAUNCH_OWNER = NATIVE_ROOT / "kernels" / "torch_cuda.h"
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


# Speed of light, pi, two pi and vacuum permittivity literals (any precision spelling).
CONSTANT_LITERAL = re.compile(r"(?<![\w.])(?:299792458|3\.14159|6\.28318|8\.854187)")
# A local definition of a constant or a component id the constants owner publishes.
CONSTANT_DEFINITION = re.compile(
    r"\b(?:constexpr|const|#define)\b[^\n;=]*\b(?:"
    r"kPiF?|kTwoPiF?|kC0|k\w*LightSpeed\w*|kSpeedOfLight\w*|k\w*Epsilon0\w*|"
    r"kVacuumPermittivity\w*|kComponent(?:Los|LineOfSight|Reflection|Diffraction|"
    r"Transmission|Scattering)|k(?:LineOfSight|Reflection|Diffraction)Component"
    r")\b\s*(?:=|[A-Za-z_])"
)
# A local launch_blocks helper or a macro that renames one per section.
LOCAL_LAUNCH_HELPER = re.compile(
    r"^\s*(?:inline\s+|static\s+)?(?:int|int64_t)\s+\w*launch_blocks\s*\(|"
    r"^#define\s+(?:launch_blocks|kBlockSize)\b",
    re.MULTILINE,
)


def _owned_errors(path: Path, text: str, relative: str) -> list[str]:
    errors = []
    for owner, label, pattern in (
        (CONSTANTS_OWNER, "physical constant literal", CONSTANT_LITERAL),
        (CONSTANTS_OWNER, "local constant or component id", CONSTANT_DEFINITION),
        (LAUNCH_OWNER, "local launch_blocks helper or rename macro", LOCAL_LAUNCH_HELPER),
    ):
        if path == owner:
            continue
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(f"{relative}:{line}: {label} belongs in {owner.relative_to(ROOT).as_posix()}")
    return errors


def main() -> int:
    for owner in (MATH_OWNER, CONSTANTS_OWNER, LAUNCH_OWNER):
        if not owner.is_file():
            print(f"missing shared native owner: {owner.relative_to(ROOT)}")
            return 1

    errors: list[str] = []
    for path in NATIVE_ROOT.rglob("*"):
        if path.suffix.lower() not in NATIVE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig")
        relative = path.relative_to(ROOT).as_posix()
        errors.extend(_owned_errors(path, text, relative))
        if path == MATH_OWNER:
            continue
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
    print(
        "shared math: one native owner each for math, physical constants and launch helpers, "
        "and no Channel tensor_math module"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())