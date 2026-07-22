from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.phase13_phase12 import workers
from benchmarks.phase13_phase12.contracts import EvidenceError


def _install_route_replay(
    monkeypatch: pytest.MonkeyPatch, *, keep_stable_candidate: bool
) -> None:
    matches = {
        ("base", "incident_te_tm_fractions"): [
            "src/witwin/channel/montecarlo/events/transmission.py:129:def incident_te_tm_fractions("
        ],
        ("candidate", "incident_te_tm_fractions"): [],
        ("base", "straight_transmission_chains"): [
            "src/witwin/channel/montecarlo/events/transmission.py:185:def straight_transmission_chains("
        ],
        ("candidate", "straight_transmission_chains"): (
            [
                "src/witwin/channel/montecarlo/events/transmission.py:150:def straight_transmission_chains("
            ]
            if keep_stable_candidate
            else []
        ),
        ("base", "MonteCarloTargetInset"): [],
        ("candidate", "MonteCarloTargetInset"): [
            "src/witwin/channel/montecarlo/events/transmission.py:212:    policy=MonteCarloTargetInset,"
        ],
    }

    monkeypatch.setattr(
        workers,
        "_tracked_matches_revision",
        lambda _git, _repo, revision, pattern: matches[(revision, pattern)],
    )
    monkeypatch.setattr(
        workers,
        "_git",
        lambda _git, _repo, *args, **_kwargs: (
            "src/witwin/channel/montecarlo/events/transmission.py"
            if args[:2] == ("diff", "--name-only")
            else ""
        ),
    )


def test_montecarlo_route_keeps_stable_owner_and_deletes_old_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_route_replay(monkeypatch, keep_stable_candidate=True)

    audit = workers._route_transition_at_revisions(
        "montecarlo_penetration",
        "base",
        "candidate",
        repository=Path("repository"),
        git_executable=Path("git"),
    )

    assert audit["passed"] is True
    assert audit["checks"]["candidate_deleted:incident_te_tm_fractions"] is True
    assert audit["checks"]["candidate_stable:straight_transmission_chains"] is True


def test_montecarlo_route_rejects_deleting_stable_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_route_replay(monkeypatch, keep_stable_candidate=False)

    with pytest.raises(EvidenceError, match="candidate_stable:straight_transmission_chains"):
        workers._route_transition_at_revisions(
            "montecarlo_penetration",
            "base",
            "candidate",
            repository=Path("repository"),
            git_executable=Path("git"),
        )
