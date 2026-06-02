# Documentation Naming Standard

Status: Active
Category: Standard
Last reviewed: 2026-05-16

## Purpose

This document defines how development-facing Markdown files in `docs/dev/` should be named, placed, and maintained.

## Placement Rules

- Put active workflow, invariant, and validation-reference docs in `docs/dev/standards/`.
- Put active defect inventories in `docs/dev/bugs/`.
- Put future-facing design and feature proposals in `docs/dev/plans/`.
- Put profiling, scaling, and performance-rollout docs in `docs/dev/optimization/`.
- Move completed or superseded documents to `docs/dev/archive/` instead of leaving them mixed into the active set.

## Numeric Prefix Convention

Documents in `docs/dev/standards/` and `docs/dev/plans/` use a two-digit numeric prefix so the alphabetic order on disk matches the recommended reading order. Group ranges have ten-unit gaps so new entries can be inserted without renumbering.

### standards/

| Range | Group | Purpose |
| --- | --- | --- |
| `00`-`09` | Meta | Naming rules, root-vs-dev crosswalk, documentation governance |
| `10`-`19` | Agents | Agent-facing repository reference and per-agent operating guides (Codex, Claude Code, Superpowers) |
| `20`-`29` | Python | Package architecture, runtime typing, language-level discipline |
| `30`-`39` | CUDA | Native kernel development guide and migration workflow |
| `40`-`49` | Physics / packages | Solver taxonomies, validation references, and per-package overviews |
| `50`-`59` | Workflow | Tests, acceptance, release, and other process docs |

### plans/

| Range | Group | Purpose |
| --- | --- | --- |
| `00`-`09` | Roadmap | Strategic and long-range feature roadmaps |
| `10`-`19` | Architecture | Package architecture, refactor, and ownership-migration plans |
| `20`-`29` | Features | Feature, physics, and algorithm implementation plans |
| `30`-`39` | Reserved | Reserved for future operational rollouts (CI, packaging, release) |

Plan numbers are assigned when the plan first lands as `Status: Draft` or `Status: Active` and stay with the file when it moves to `archive/completed/` or `archive/superseded/` so cross-references remain stable.

`bugs/` and `optimization/` do not use numeric prefixes. Bug inventories and dated reports do not need a fixed reading order.

## File Naming Rules

- Use lowercase kebab-case only. Inside `standards/`, prepend the two-digit group prefix described above.
- Use concrete topic names. Prefer `material-diffraction-rebuild.md` over vague names such as `features.md`.
- Do not use mixed case, underscores, spaces, or typo-prone names such as `BUGs.md` or `multiplath.md`.
- Use a date suffix only when the document is a point-in-time report, investigation note, or dataset snapshot.
- Date format is `YYYY-MM-DD`, appended as `topic-YYYY-MM-DD.md`.
- Long-lived standards and living plans should usually be undated.

## Status Rules

- Every new or heavily revised document should declare:
  - `Status: Active | Draft | In Progress | Completed | Superseded | Archived`
  - `Category: Standard | Bug | Plan | Optimization | Archive`
  - `Last reviewed: YYYY-MM-DD`
- When a document stops being authoritative, move it to `archive/` and update the replacement path in `docs/dev/README.md` or `docs/dev/archive/README.md`.

### Plan-Specific Status Rule

`docs/dev/plans/` is the **active backlog only**. A plan must never sit in `plans/` with `Status: Completed` or `Status: Superseded`.

- When a plan's last task is checked off and the corresponding code change has landed:
  1. Flip the plan header to `Status: Completed` and update `Last reviewed`.
  2. In the same change, `git mv` the plan into `archive/completed/`, keeping its numeric prefix.
  3. Update `docs/dev/README.md` (remove from the Active Plans table) and `docs/dev/archive/README.md` (add the new row).
- When a plan is replaced by a newer plan before completing, move it into `archive/superseded/` in the same change, again keeping its numeric prefix.

Agents executing plans through `superpowers:executing-plans` are responsible for performing the move at completion; reviewers should reject PRs that leave a `Status: Completed` plan inside `plans/`.

## Scope Rules

- One document should own one topic.
- If a topic accumulates many short phase logs, keep one stable summary document in the active tree and archive the raw phase notes.
- Do not keep multiple active documents that describe the same plan at different maturity levels without clearly marking one as superseded.

## Language Rules

- Use English for document titles, headings, comments, and maintenance notes.
- If historical content is preserved only for traceability and contains mixed language or encoding damage, keep it under `archive/superseded/` and do not treat it as authoritative.
