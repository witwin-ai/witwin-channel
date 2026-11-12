# Copyright Xingyu Chen.
# Tests public api snapshot.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.refactor_baseline import api_manifest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "ci" / "public-api-snapshot.json"


def _contract_sha256(contract: dict[str, object]) -> str:
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _current_snapshot(public_modules: list[str]) -> dict[str, object]:
    current = api_manifest(ROOT, public_modules)
    modules = []
    for module in current["modules"]:
        exports = []
        for contract in module["objects"]:
            export = {
                "name": contract["name"],
                "kind": contract["kind"],
                "target": contract["target"],
                "contract_sha256": _contract_sha256(contract),
            }
            if "signature" in contract:
                export["signature"] = contract["signature"]
            exports.append(export)
        modules.append({"module": module["module"], "exports": exports})
    return {
        "schema_version": 1,
        "generator": "tools.refactor_baseline.api_manifest/v1",
        "modules": modules,
    }


def test_curated_public_api_matches_frozen_snapshot():
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    public_modules = [module["module"] for module in expected["modules"]]

    assert expected["schema_version"] == 1
    assert len(public_modules) == len(set(public_modules)) == 6
    assert _current_snapshot(public_modules) == expected


def test_the_experimental_tier_is_outside_the_stable_snapshot():
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    modules = [module["module"] for module in expected["modules"]]
    targets = [export["target"] for module in expected["modules"] for export in module["exports"]]

    assert not [name for name in modules if name.startswith("witwin.channel.experimental")]
    assert not [target for target in targets if ".experimental." in target]

def test_dataclass_contracts_include_inherited_fields():
    """A field added to a shared base class must change every derived Config contract."""

    manifest = api_manifest(ROOT, ["witwin.channel.path", "witwin.channel.radiomap"])
    fields = {
        module["module"]: [field["name"] for field in contract["fields"]]
        for module in manifest["modules"]
        for contract in module["objects"]
        if contract["name"] in {"Config", "Coherent"}
    }
    assert fields["witwin.channel.path"][:2] == ["max_depth", "components"]
    assert "discovery" in fields["witwin.channel.path"]
    assert fields["witwin.channel.radiomap"][: len(fields["witwin.channel.path"])] == (
        fields["witwin.channel.path"]
    )
