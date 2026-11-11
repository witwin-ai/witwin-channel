# Copyright Xingyu Chen.
# Writes the small street-canyon OBJ and Mitsuba XML scene shared by the examples.

"""Write the small street-canyon OBJ and Mitsuba XML scene shared by the examples."""

from __future__ import annotations

from pathlib import Path

# (x_min, x_max, y_min, y_max, height) of each building, metres; the street runs along x.
BUILDINGS = (
    (-60.0, -25.0, 8.0, 30.0, 18.0),
    (-20.0, 10.0, 8.0, 26.0, 24.0),
    (15.0, 60.0, 8.0, 34.0, 12.0),
    (-60.0, -10.0, -30.0, -8.0, 15.0),
    (-5.0, 25.0, -28.0, -8.0, 30.0),
    (30.0, 60.0, -32.0, -8.0, 20.0),
)
GROUND_HALF_SIZE = 70.0


def _box_obj(lines: list[str], offset: int, box: tuple[float, ...]) -> int:
    x0, x1, y0, y1, height = box
    # Footprints start 1 m below the ground plane so no face is coplanar with it.
    for z in (-1.0, height):
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            lines.append(f"v {x} {y} {z}")
    quads = ((1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8))
    for quad in quads:
        lines.append("f " + " ".join(str(offset + index) for index in quad))
    return offset + 8


def write_obj(path: Path) -> Path:
    """Write the canyon as one OBJ with ``usemtl concrete`` buildings and ``usemtl ground``."""

    lines = ["# Street canyon written by examples/street_canyon.py", "usemtl ground"]
    h = GROUND_HALF_SIZE
    lines += [f"v {-h} {-h} 0", f"v {h} {-h} 0", f"v {h} {h} 0", f"v {-h} {h} 0", "f 1 2 3 4"]
    lines.append("usemtl concrete")
    offset = 4
    for box in BUILDINGS:
        offset = _box_obj(lines, offset, box)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_mitsuba(directory: Path) -> Path:
    """Write ``canyon.xml`` (Sionna RT style ITU materials) plus its OBJ next to it."""

    directory.mkdir(parents=True, exist_ok=True)
    write_obj(directory / "canyon.obj")
    xml = directory / "canyon.xml"
    xml.write_text(
        """<scene version="3.0.0">
    <bsdf type="itu-radio-material" id="concrete">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.3"/>
    </bsdf>
    <shape type="obj" id="canyon">
        <string name="filename" value="canyon.obj"/>
        <ref id="concrete" name="bsdf"/>
    </shape>
</scene>
""",
        encoding="utf-8",
    )
    return xml
