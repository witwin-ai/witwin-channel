# Repository Documentation Crosswalk

Status: Active
Category: Standard
Last reviewed: 2026-05-16

## Purpose

This document explains which repository rules belong at the root, which belong under `docs/dev/`, and which active standards already define the authoritative workflow.

## Root vs `docs/dev/`

The repository root should stay limited to repository-wide entrypoints and stable user-facing inventories:

- `README.md`
- `FEATURE_LIST.md`
- `AGENTS.md`
- `CLAUDE.md`

Development-cycle Markdown lives under `docs/dev/`. This follows the current repository rule in `AGENTS.md` / `CLAUDE.md` and the active placement rule in `docs/dev/standards/00-documentation-naming-standard.md`:

> Put active workflow, invariant, and validation-reference docs in `docs/dev/standards/`.

It also matches the active development index in `docs/dev/README.md`, which declares itself as:

> the authoritative map for development-facing documentation under `docs/dev/`.

## Active Standards And Their Authority

Use the following documents as the canonical sources before adding new rule text elsewhere. Files in `docs/dev/standards/` use a two-digit numeric prefix; see `00-documentation-naming-standard.md` for the grouping convention.

| Topic | Canonical document |
|-------|--------------------|
| Documentation placement, naming, and metadata | `standards/00-documentation-naming-standard.md` |
| Root-vs-dev crosswalk (this file) | `standards/01-repository-documentation-crosswalk.md` |
| Detailed agent repository reference | `standards/10-agent-reference.md` |
| Codex CLI operating notes (Windows, PowerShell, env quirks) | `standards/11-codex-operating-guide.md` |
| Claude Code operating notes (permissions, hooks, skills, plan-file flow) | `standards/12-claude-code-operating-guide.md` |
| Superpowers plan-file conventions and execution flow | `standards/13-superpowers-operating-guide.md` |
| Python package architecture and naming discipline | `standards/20-python-package-architecture-standard.md` |
| Python runtime typing and boundary normalization | `standards/21-python-runtime-typing-standard.md` |
| Architecture lessons from gsplat | `standards/22-gsplat-architecture-lessons.md` |
| Native CUDA implementation pitfalls | `standards/30-cuda-kernel-development-guide.md` |
| CUDA migration process | `standards/31-cuda-kernel-migration-workflow.md` |
| Diffraction path taxonomy and solver metadata | `standards/40-diffraction-path-taxonomy.md` |
| Monte Carlo radiomap package overview | `standards/42-monte-carlo-radiomap-package-overview.md` |
| DrJit-native runtime transport boundary | `standards/30-cuda-kernel-development-guide.md` and `standards/31-cuda-kernel-migration-workflow.md` |
| Test order and acceptance gates | `standards/50-test-and-acceptance-workflow.md` |

## Maintenance Rule

When a contributor wants to add a new workflow rule, use this order:

1. Check whether an active standard already owns the topic.
2. If yes, update that standard and only add a short pointer in root docs if needed.
3. If no, add a properly named, prefix-numbered document under `docs/dev/standards/` and register it in `docs/dev/README.md`.

When a contributor wants to add a new agent runtime guide (for example a new CLI agent), add it in the `1x` range alongside the existing Codex / Claude Code / Superpowers guides and update this crosswalk.

## Cleanup Decisions

The active set under `docs/dev/standards/` no longer contains:

- `agent-reference.md` (pre-numeric, referenced the deleted `witwin/channel/scene/`, `trace/`, `monitors/`, and `tracer.py` layout). Archived as `archive/superseded/agent-reference-2026-04-09.md`; replaced by `standards/10-agent-reference.md`.
- `grid-monitor-sampling-standard.md` (described `FieldMonitor` sampling semantics; the monitor abstraction has been removed). Archived as `archive/superseded/grid-monitor-sampling-standard.md`.
- `41-closed-form-multi-diffraction-validation.md` (the validation harness in `witwin/channel/validation.py`, `samples/save_wedge_validation_suite.py`, and `tests/test_validation_references.py` was removed during the package rewrite and has not been re-ported to the new layout). Archived as `archive/superseded/41-closed-form-multi-diffraction-validation.md`; a fresh validation standard will be written once the harness is rebuilt under the new packages.

Both files are kept only as historical traceability and must not be cited as the source of truth for current behavior.
