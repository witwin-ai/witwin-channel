from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def notebook_source(path: str) -> str:
    notebook = json.loads(read_text(path))
    chunks: list[str] = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


def test_deterministic_munich_notebook_keeps_refactor_workload() -> None:
    source = notebook_source("examples/deterministic_radiomap_munich.ipynb")

    assert "GRID_SHAPE = (512, 512)" in source
    assert "MAX_DIFFRACTIONS = 1" in source
    assert "SIONNA_SAMPLE_BUDGET = 1_000_000" in source
    assert "max_diffractions=2" in source
    assert "grid_shape=(256, 256)" in source


def test_deterministic_three_cubes_examples_keep_refactor_workload() -> None:
    notebook = notebook_source("examples/deterministic_radiomap_three_cubes.ipynb")
    module = read_text("examples/deterministic_radiomap_three_cubes.py")

    assert "grid_shape=(256, 256)" in notebook
    assert "forward_num_samples=384" in notebook
    assert "gradient_num_samples=128" in notebook
    assert "max_diffraction_order=1" in notebook
    assert "DEFAULT_MAX_DIFFRACTION_ORDER = 1" in module


def test_field_solver_examples_keep_refactor_workload() -> None:
    multipath_notebook = notebook_source("examples/field_solver_multipath_main.ipynb")
    multipath_module = read_text("examples/field_solver_multipath_main.py")
    field3d_notebook = notebook_source("examples/field_solver_three_cubes_3d.ipynb")
    field3d_module = read_text("examples/field_solver_three_cubes_3d.py")

    assert "GRID_SHAPE = field_example.DEFAULT_GRID_SHAPE" in multipath_notebook
    assert "GRADIENT_GRID_SHAPE = (64, 64)" in multipath_notebook
    assert "GRADIENT_MAX_DIFFRACTIONS = 1" in multipath_notebook
    assert "DEFAULT_MAX_DIFFRACTIONS = 2" in multipath_module

    assert "grid_shape=(256, 256)" in field3d_notebook
    assert "forward_num_samples=384" in field3d_notebook
    assert "gradient_num_samples=128" in field3d_notebook
    assert "max_diffraction_order=1" in field3d_notebook
    assert "DEFAULT_MAX_DIFFRACTION_ORDER = 1" in field3d_module


def test_monte_carlo_notebooks_keep_refactor_workload() -> None:
    basic_munich = notebook_source("examples/monte_carlo_radiomap_basic_munich.ipynb")
    bdpt = notebook_source("examples/monte_carlo_radiomap_bdpt.ipynb")
    bdpt_munich = notebook_source("examples/monte_carlo_radiomap_bdpt_munich.ipynb")

    assert "GRID_SHAPE = (512, 512)" in basic_munich
    assert "DIFFRACTION_SAMPLE_BUDGET = 1_000_000" in basic_munich
    assert "REFLECTION_RAY_BUDGET = 1_000_000" in basic_munich
    assert "SIONNA_SAMPLE_BUDGET = 1_000_000" in basic_munich
    assert "enable_rd_diffraction=True" in basic_munich
    assert "max_diffraction_order=1" in basic_munich

    assert "GRID_SHAPE = (256, 256)" in bdpt
    assert "SAMPLE_BUDGET = 1_000_000" in bdpt
    assert "GRADIENT_GRID_SHAPE = (256, 256)" in bdpt
    assert "GRADIENT_MAX_DIFFRACTIONS = 1" in bdpt
    assert "enable_rd_diffraction=True" in bdpt
    assert "max_diffraction_order=2" in bdpt

    assert "GRID_SHAPE = (512, 512)" in bdpt_munich
    assert "DIFFRACTION_SAMPLE_BUDGET = 10_000_000" in bdpt_munich
    assert "REFLECTION_RAY_BUDGET = 10_000_000" in bdpt_munich
    assert "SIONNA_SAMPLE_BUDGET = 10_000_000" in bdpt_munich
    assert "MAX_DIFFRACTIONS = 2" in bdpt_munich
    assert "enable_rd_diffraction=True" in bdpt_munich
