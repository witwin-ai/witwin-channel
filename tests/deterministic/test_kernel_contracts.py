from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.deterministic.kernels import fields
from witwin.channel_native.runtime import symbols, tensor_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_OWNER_NAMES = (
    "deterministic_delay_to_path_length",
    "deterministic_diffraction_vector_field",
    "deterministic_field_from_power_phase",
    "deterministic_los_field",
    "deterministic_pack_complex",
    "deterministic_phase_from_field",
    "deterministic_phase_from_length",
    "deterministic_reflection_field",
    "deterministic_reflection_sequence_field",
    "deterministic_zero_field_phase",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_deterministic_fields_is_the_single_object_owner(name: str):
    owner = getattr(fields, name)

    assert owner.__module__ == fields.__name__
    assert getattr(ops, name) is owner


def test_deterministic_fields_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{fields.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert len(definitions) == 10
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_deterministic_fields_uses_canonical_runtime_dependencies():
    assert fields.native_extension is symbols.native_extension
    assert fields.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.core.kernels import ops; "
            "from witwin.channel_native.deterministic.kernels import fields"
        ),
        (
            "from witwin.channel_native.deterministic.kernels import fields; "
            "from witwin.channel_native.core.kernels import ops"
        ),
    ),
)
def test_deterministic_fields_import_order_preserves_facade_identity(imports: str):
    names = repr(_OWNER_NAMES)
    code = (
        f"{imports}; "
        f"names={names}; "
        "assert all(getattr(ops, name) is getattr(fields, name) for name in names)"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH"))
        if value
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
