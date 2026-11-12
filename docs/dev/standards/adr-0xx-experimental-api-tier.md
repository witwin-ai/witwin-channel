# ADR-0xx: The experimental API tier

- **Status:** Accepted (owner decision D2/D3 for Channel 0.6.0, report section 10.13).
- **Date:** 2026-11-11.
- **Decision ID:** `experimental-api-tier`.
- **Scope:** `witwin.channel.experimental` and the modules under it. First member:
  `witwin.channel.experimental.bdpt` (formerly `witwin.channel.montecarlo.bdpt`).
  No numerical change.

## Decision

Channel has three API tiers:

| Tier | Modules | Promise |
| --- | --- | --- |
| Stable | the package root, `path`, `radiomap`, `propagation.consumer`, `importers`, `presets` | Frozen in `ci/public-api-snapshot.json`; a change needs a snapshot update and a migration note. |
| Experimental | `witwin.channel.experimental.*` | Importable, built, tested and held to every engineering rule; its names, signatures, result schema and numbers may change or the module may be removed in any release without a migration path. |
| Internal | everything else | No promise; callers must not import it. |

Rules for the experimental tier:

1. **Not in the snapshot.** No `witwin.channel.experimental` module or target appears in
   `ci/public-api-snapshot.json` (`tests/test_public_api_snapshot.py` checks this). The
   stable modules never import an experimental module, and the package root does not
   re-export one.
2. **Explicit import.** Users opt in by naming the path, for example
   `from witwin.channel.experimental import bdpt`. `witwin.channel.experimental` itself
   exports nothing. Importing emits no warning: the path is the marker, and a warning would
   change behaviour for existing callers.
3. **Same engineering rules.** An experimental module is production code: one native
   backend, no fallbacks, launch/sync budgets, the CUDA-context rule, native-binding
   ownership, source headers, the import-graph layer rules (it is a solver for
   `ci/check_import_graph.py`, so it may not import another solver), and reachability in
   `ci/check_orphan_modules.py`, where each experimental module is a declared entry point.
4. **Moving into or out of the tier** is a rename, done as a `git mv` commit of its own with
   no behaviour change. Promotion to stable adds the module to the snapshot and to the
   stable tier in `CLAUDE.md`; demotion removes it and records the move in the release's
   migration note.

## Why BDPT is experimental

BDPT is a second stochastic estimator beside `radiomap.Stochastic`. Its wavefront AD and
several of its component paths are still being reworked (ADR-049 wavefront), and the owner
chose not to freeze its configuration and result schema in 0.6.0. Its functionality,
numbers and tests are unchanged by the move. Its configuration keeps the `coherent` flag
(the stable `radiomap.Coherent` uses `combine="field" | "power"`), because the experimental
tier is not held to the stable vocabulary.

## Consequences

- Earlier ADRs, plans and audits name the pre-0.6.0 module paths
  (`witwin.channel.deterministic`, `witwin.channel.montecarlo.basic`,
  `witwin.channel.montecarlo.bdpt`). They are records of their time and are not rewritten;
  `docs/migration-0.6.md` maps the old paths to the new ones.
- `capabilities()["solvers"]` keeps its keys (`deterministic`, `montecarlo_basic`,
  `montecarlo_bdpt`); they name capability records, not import paths.

## Evidence

- `tests/test_public_api_snapshot.py::test_the_experimental_tier_is_outside_the_stable_snapshot`.
- `tests/test_orphan_modules_contract.py` (entry points), `tests/test_import_graph_contract.py`
  (the BDPT enumerated-oracle debt is the same single edge under its new source name).
- Bit-identical BDPT results before and after the move on the rename corpus (see the
  `rename` task report).
