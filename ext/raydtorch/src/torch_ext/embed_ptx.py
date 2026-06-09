from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: embed_ptx.py <input.ptx> <output.h> <symbol>")
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    symbol = sys.argv[3]
    ptx = input_path.read_bytes() + b"\0"
    rows = []
    for index in range(0, len(ptx), 16):
        chunk = ptx[index : index + 16]
        rows.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    payload = "\n".join(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "#pragma once\n\n"
        "namespace raydtorch {\n"
        f"inline constexpr char {symbol}[] = {{\n"
        f"{payload}\n"
        "};\n"
        "} // namespace raydtorch\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
