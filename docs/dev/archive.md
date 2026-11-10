# Archived governance records

The files below were removed from the working tree because no check, test or tool reads them any
more. They remain in Git history. The last commit that contains all of them is
`63c63e6` ("test: never auto-select a local extension build, add the quick-tier workflow"); restore
any of them with

```bash
git show 63c63e6:<path> > <path>
git checkout 63c63e6 -- docs/dev/audit docs/dev/baselines   # everything at once
```

Design records (ADRs, plans, audit narratives in `docs/dev/audit/*.md`) still link to some of these
paths. Those links refer to the archived copies at that commit.

## Audit ledgers (`docs/dev/audit/*.json`)

All 43 JSON ledgers were removed. They recorded per-phase migration evidence: hand-maintained
symbol counts, owner inventories with line and body-hash pins, launch-count ledgers, the duplication
classification, resource ledgers and release-acceptance records. The invariants that were still
worth keeping are now checked directly from the source, without a ledger:

| Former record | Replacement |
|---|---|
| `phase13-current-native-owner-inventory.json`, `phase9-native-owner-inventory.json`, `phase13-symbol-delta-ledger.json`, `phase13-migration-delta.json`, `phase10-legacy-dead-binding.json`, `phase12-ops-migration-ledger.json` | `ci/check_native_bindings.py` derives each pybind symbol's Python owner and production callers from the AST |
| `adr-044-native-tu-consolidation.json` (translation-unit set, launch/sync/trap counts) | `ci/check_native_inventory.py` generates `ci/native-inventory.json` (kernel, launch and host-sync names per file) and checks the translation-unit layout |
| `duplication-classification.json` | `ci/check_duplication.py` gates total duplicate coverage against a tighten-only ceiling |
| everything else (`phase*`, `stage1-*`, `adr-0xx-*` evidence and resource ledgers) | historical only; the behaviour they describe is covered by the test suite |

## Refactor baselines (`docs/dev/baselines/<sha>/`)

The snapshot directories `4c9f3ec…`, `812e195…`, `a741f8d…` and `d2ba697…` were removed, and from
`0892d85…` everything except the two files that live code still reads:

- `0892d855b27ee851521a181f5158b0bf41091eda/coverage.json` - floor for `ci/check_coverage.py`
  (via `ci/coverage-policy.json`);
- `0892d855b27ee851521a181f5158b0bf41091eda/static/environment.json` - the SM 120 evidence path
  published by `witwin.channel.deployment` and `ci/support-matrix.toml`.

`tools/refactor_baseline.py` and `tools/refactor_runtime_baseline.py` can still regenerate such
snapshots into a scratch directory when an exactness comparison needs one.

## Removed governance code

- `ci/check_contract_coverage.py` and `ci/contract-coverage-manifest.json` (hand-labelled owner,
  contract-test and end-to-end-scenario columns) - replaced by `ci/check_native_bindings.py`.
- `ci/run_phase9_reduced_gates.ps1` - superseded by `ci/run_ci_tier.py`.
- `docs/dev/audit/generate_phase13_phase0_audit.py` - generator of the removed phase-0 ledgers.
- Tests that pinned ledger contents, symbol counts, `<<<` counts, source line order, or SHA-256
  digests of function bodies or ASTs (the `tests/test_phase13_*` and `tests/test_*_native_boundaries`
  files, `tests/test_native_owner_inventory.py`, and others listed in the change that removed them).
