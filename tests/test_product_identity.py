from __future__ import annotations

from pathlib import Path

from ci import check_product_identity as identity


ROOT = Path(__file__).resolve().parents[1]


def test_live_repository_uses_only_channel_identity() -> None:
    assert identity.scan_repository(ROOT) == []


def test_predecessor_identity_is_rejected_in_live_source() -> None:
    predecessor = "channel" + "_native"
    findings = identity.scan_text("src/witwin/channel/example.py", predecessor)
    assert [(finding.path, finding.line) for finding in findings] == [
        ("src/witwin/channel/example.py", 1)
    ]


def test_checkout_name_and_frozen_history_are_explicit_exceptions() -> None:
    checkout_line = "`" + "channel" + "_native/` and"
    predecessor = "Channel" + " Native"
    assert identity.scan_text("AGENTS.md", checkout_line) == []
    assert identity.scan_text("docs/dev/audit/evidence.json", predecessor) == []
