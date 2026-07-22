from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
RAYD_COMMIT = "3988f0934fec7b521ee5190b0defc0883c84b9e6"
INTEGRATION_V2_SHA256 = (
    "6cb18f682e08cb0bb0853507e3b4b82a68e681bb1dad89dc8c36518705f74989"
)
INTEGRATION_V2_IDENTITY = (
    "rayd.torch.integration.v2.20260719.rf-transmission-sequence"
)
BINDING_MANIFEST_SHA256 = (
    "6c9ac270c8f243ff63f1c6f4466e5410e78683e10d906a0c7226fd84a208c252"
)
TRANSMISSION_SYMBOLS = {
    "field_transmission_sequence",
    "field_transmission_sequence_backward",
    "field_transmission_sequence_jvp",
}
REMOVED_CHANNEL_SOURCE = (
    ROOT / "native/channel/kernels/field_transport_transmission.cu"
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase6b_pin_manifest_and_owner_counts_are_consistent() -> None:
    inventory = _json(AUDIT / "phase13-current-native-owner-inventory.json")
    migration = _json(AUDIT / "phase13-migration-delta.json")
    contracts = _json(AUDIT / "phase13-transmission-contracts.json")
    decision = _json(
        AUDIT / "phase13-shared-rf-helper-ownership-decision.json"
    )
    ledger = _json(AUDIT / "phase13-shared-rf-helper-ledger.json")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")

    assert {
        inventory["phase6b_transmission_sequence"]["rayd_commit"],  # type: ignore[index]
        migration["phase6b_current"]["rayd_commit"],  # type: ignore[index]
        contracts["phase6b_activation"]["rayd_commit"],  # type: ignore[index]
        decision["phase6b_activation"]["rayd_commit"],  # type: ignore[index]
        ledger["phase6b_activation"]["rayd_commit"],  # type: ignore[index]
        graph["phase6b_activation"]["rayd_commit"],  # type: ignore[index]
    } == {RAYD_COMMIT}
    assert {
        inventory["phase6b_transmission_sequence"][  # type: ignore[index]
            "integration_header_sha256"
        ],
        migration["phase6b_current"]["integration_header_sha256"],  # type: ignore[index]
        contracts["phase6b_activation"][  # type: ignore[index]
            "integration_header_sha256"
        ],
        decision["phase6b_activation"][  # type: ignore[index]
            "integration_header_sha256"
        ],
        ledger["phase6b_activation"][  # type: ignore[index]
            "integration_header_sha256"
        ],
        graph["phase6b_activation"][  # type: ignore[index]
            "integration_header_sha256"
        ],
    } == {INTEGRATION_V2_SHA256}
    assert {
        inventory["phase6b_transmission_sequence"][  # type: ignore[index]
            "integration_header_identity"
        ],
        migration["phase6b_current"]["integration_header_identity"],  # type: ignore[index]
        contracts["phase6b_activation"][  # type: ignore[index]
            "integration_header_identity"
        ],
        decision["phase6b_activation"][  # type: ignore[index]
            "integration_header_identity"
        ],
        ledger["phase6b_activation"][  # type: ignore[index]
            "integration_header_identity"
        ],
        graph["phase6b_activation"][  # type: ignore[index]
            "integration_header_identity"
        ],
    } == {INTEGRATION_V2_IDENTITY}
    assert (
        inventory["phase6b_transmission_sequence"][  # type: ignore[index]
            "binding_manifest_sha256"
        ]
        == BINDING_MANIFEST_SHA256
    )
    assert migration["phase6b_current"][  # type: ignore[index]
        "binding_manifest_sha256"
    ] == BINDING_MANIFEST_SHA256

    owners = {
        record["symbol"]: record["numerical_owner"]
        for record in inventory["symbols"]  # type: ignore[index]
    }
    assert all(owners[symbol] == "RayD" for symbol in TRANSMISSION_SYMBOLS)


def test_phase6b_is_a_complete_source_owner_move_without_a_fallback() -> None:
    fields = (
        ROOT / "native/channel/binding/fields.cpp"
    ).read_text(encoding="utf-8-sig")
    retained = (
        ROOT / "native/channel/kernels/field_transport.cu"
    ).read_text(encoding="utf-8-sig")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")

    assert not REMOVED_CHANNEL_SOURCE.exists()
    assert "field_transport_transmission.cu" not in cmake
    assert "transmission_sequence" not in retained
    assert "<<<" not in fields
    for symbol in TRANSMISSION_SYMBOLS:
        assert fields.count(f"rayd::torch::{symbol}(") == 1

    nodes = {record["id"] for record in graph["nodes"]}  # type: ignore[index]
    assert {
        "RayD:backends/torch/src/torch_ext/rf/transmission_sequence.cu",
        "RayD:backends/torch/src/torch_ext/rf/transmission_sequence_ad.cu",
    } <= nodes
    assert all(
        not (
            edge["from"].startswith("RayD:")
            and edge["to"].startswith("native/channel/")
        )
        for edge in graph["edges"]  # type: ignore[index]
    )


def test_phase6b_keeps_the_helper_partition_and_frozen_budget() -> None:
    decision = _json(
        AUDIT / "phase13-shared-rf-helper-ownership-decision.json"
    )["phase6b_activation"]
    ledger = _json(AUDIT / "phase13-shared-rf-helper-ledger.json")
    duplication = _json(AUDIT / "duplication-classification.json")

    expected = {
        "RayD ADR-024 active": 112,
        "Channel boundary retained": 10,
        "Channel pending ADR-026": 7,
    }
    assert decision["helper_count_delta"] == 0  # type: ignore[index]
    assert decision["helper_ownership"] == expected  # type: ignore[index]
    assert ledger["helper_count"] == 129
    assert ledger["phase6b_activation"]["ownership_projection"] == expected  # type: ignore[index]

    refresh = duplication["phase6b_refresh"]  # type: ignore[index]
    assert refresh["region_count"] == 178
    assert refresh["coverage_percent"] == 12.560455
    assert refresh["frozen_coverage_percent"] == 10.211512
    assert duplication["baseline"]["coverage_percent"] == 10.211512  # type: ignore[index]


def test_phase6b_transmission_contract_and_symbol_ledgers_are_closed() -> None:
    contracts = _json(AUDIT / "phase13-transmission-contracts.json")
    symbols = _json(AUDIT / "phase13-symbol-delta-ledger.json")
    by_contract = {
        record["symbol"]: record
        for record in contracts["contracts"]  # type: ignore[index]
    }
    by_action = {
        record["symbol"]: record
        for record in symbols["actions"]  # type: ignore[index]
    }

    assert contracts["phase6b_activation"]["pending_contract_count"] == 0  # type: ignore[index]
    assert all(
        by_contract[symbol]["current_numerical_owner"] == "RayD"
        and by_contract[symbol]["activation_phase"] == "6B"
        and by_action[symbol]["status"] == "applied in Phase 6B"
        for symbol in TRANSMISSION_SYMBOLS
    )
