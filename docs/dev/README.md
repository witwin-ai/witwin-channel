# Development Documentation Index

This index is the authoritative map for development-facing documentation under `docs/dev/`.

## Directory Rules

- `standards/`: active workflow, convention, and validation-reference documents. Filenames use a two-digit numeric prefix; see `standards/00-documentation-naming-standard.md`.
- `bugs/`: active known defects with repro, impact, and current handling.
- `plans/`: **ongoing plans only.** Filenames use a two-digit numeric prefix. A plan must never sit here with `Status: Completed` or `Status: Superseded`; the agent that lands the final task moves the file into `archive/completed/` (or `archive/superseded/`) in the same change, keeping its numeric prefix. Details in `standards/00-documentation-naming-standard.md`.
- `optimization/`: performance analysis, scaling reports, and optimization rollout notes.
- `archive/completed/`: finished plans and optimization rollouts kept as historical record. Plans keep the numeric prefix they had in `plans/`.
- `archive/superseded/`: obsolete or replaced documents that are no longer authoritative.

## Standards

Standards filenames use a two-digit numeric prefix grouped by topic:

| Range | Group |
| --- | --- |
| `00`-`09` | Meta (documentation governance) |
| `10`-`19` | Agents (repository reference and per-agent operating guides) |
| `20`-`29` | Python (architecture, typing, language discipline) |
| `30`-`39` | CUDA (kernel development, migration) |
| `40`-`49` | Physics and per-package overviews |
| `50`-`59` | Workflow (tests, acceptance, release process) |

| Path | Purpose |
| --- | --- |
| `standards/00-documentation-naming-standard.md` | Naming, status, placement, and numeric-prefix rules for `docs/dev/` |
| `standards/01-repository-documentation-crosswalk.md` | Root-vs-`docs/dev` authority map and standards crosswalk |
| `standards/10-agent-reference.md` | Detailed repository layout, public API surface, shared utility ownership, command examples |
| `standards/11-codex-operating-guide.md` | Codex CLI operating notes (Windows, PowerShell, environment quirks) |
| `standards/12-claude-code-operating-guide.md` | Claude Code operating notes (permissions, hooks, skills, plan-file flow) |
| `standards/13-superpowers-operating-guide.md` | Plan-file conventions and execution flow for `superpowers:executing-plans` and `superpowers:subagent-driven-development` |
| `standards/20-python-package-architecture-standard.md` | Python package structure, naming discipline, utility ownership, thin-wrapper bans, OOP placement |
| `standards/21-python-runtime-typing-standard.md` | Python runtime typing, boundary normalization, anti-duck-typing, and DrJit-first internal contract rules |
| `standards/22-gsplat-architecture-lessons.md` | Architecture lessons from gsplat, including narrow public APIs, explicit state lifetimes, and avoided config/registry patterns |
| `standards/30-cuda-kernel-development-guide.md` | Practical CUDA development rules and pitfalls |
| `standards/31-cuda-kernel-migration-workflow.md` | Standard migration workflow for DrJit-to-CUDA work |
| `standards/40-diffraction-path-taxonomy.md` | Current diffraction-path taxonomy and solver metadata semantics |
| `standards/42-monte-carlo-radiomap-package-overview.md` | Current ownership map, call graph, and public API overview for `witwin.channel.montecarlo` |
| `standards/50-test-and-acceptance-workflow.md` | Canonical test and acceptance workflow |

## Bugs

| Path | Purpose |
| --- | --- |
| `bugs/known-bugs.md` | Current known defects only |
| `bugs/rayd-epc-field-optix-pipeline-repro-2026-05-25.md` | Reproduction flow for RayD batched `trace_refl_epc_field(...)` `optixPipelineCreate(multipath)` failures after rebuilding into `witwin2` |

## Active Plans

Plan filenames use a two-digit numeric prefix. Group ranges:

| Range | Group |
| --- | --- |
| `00`-`09` | Roadmap (strategic, long-range) |
| `10`-`19` | Architecture refactors and ownership migrations |
| `20`-`29` | Feature, physics, and algorithm implementation |
| `30`-`39` | Reserved for future operational rollouts |

| Path | Status | Purpose |
| --- | --- | --- |
| `plans/00-research-feature-roadmap.md` | Active | Long-range feature roadmap |
| `plans/01-platform-strategy-and-research-directions.md` | Active | Strategy layer above plan 00: moat thesis (differentiability + full-wave grounding), big-feature layering, the three bets, and an exploratory brainstorm of dynamics / real-time radiomap / scattering-transmission-Doppler / novel scene representations |
| `plans/10-scene-owned-endpoints-and-material-config-pruning.md` | In Progress | Migration plan for moving transmitter, receiver, grid, power, polarization, and material ownership into `Scene` objects while pruning solver config classes down to algorithm controls |
| `plans/11-deterministic-radiomap-package-architecture-plan.md` | Draft | Architecture plan for bringing `witwin.channel.deterministic/` to the structural bar set by `witwin.channel.montecarlo/` |
| `plans/12-monte-carlo-radiomap-package-architecture-plan.md` | Draft | Maintenance architecture plan for the standalone `witwin.channel.montecarlo` package |
| `plans/13-channel-readability-conciseness-followup-plan.md` | Active | Post–MidMay-Refactor follow-up: native loader, endpoints, shadow-boundary dedupe, megamodule splits |
| `plans/14-compact-call-formatting-plan.md` | Active | Ruff line-length policy, compact call layout rules, and context-dataclass rollout to cut vertical kwargs bloat (`witwin/` only) |
| `plans/15-source-tree-restructure-plan.md` | Complete | Directory-level package restructure plan for channel-owned core placement, path/trace naming, native loader consolidation, solver config sharing, and channel-owned solver packages |
| `plans/20-material-diffraction-rebuild.md` | Active | Target rebuild for material-aware diffraction |
| `plans/21-diffraction-shared-extraction-plan.md` | In Progress | Shared diffraction scaffolding extraction; Phases B1/B2/A3 complete |
| `plans/22-deterministic-discontinuity-plan.md` | Active | Deterministic radiomap discontinuity diagnostics, starting with single-cube y = -4 / y = 4 power-line component checks |
| `plans/25-rfdt-reflection-f-weight-plan.md` | Active | RFDT (MobiCom '26 §3.3) reflection differentiability rollout. Primary visibility F(x_a)E_a + F(x_b)E_b landed in reference / native CUDA forward / native CUDA JVP; next step is secondary visibility (Eq. 9) w(γ)E_spec + E_diffract reusing RayD triangle and edge BVHs |
| `plans/26-mc-sionna-parity-acceleration-plan.md` | Active | Tier-ordered work list to push `witwin.channel.montecarlo` (basic + BDPT) measurably past Sionna RT 2.0 on the same scene. Companion to plan 24; uses the in-tree `sionna-rt-reference-2.0.0/` as ground-truth baseline |
| `plans/27-path-reflection-scheduling-optimization-plan.md` | Active | Path reflection scheduling optimization and native prefix-compaction rollout |
| `plans/28-rayd-optix-diffraction-kernel-plan.md` | Active | RayD OptiX diffraction-kernel rollout for Monte Carlo basic, BDPT, and path solver diffraction integration |
| `plans/29-radiomap-differentiability-parity-plan.md` | Active | Make radiomap gradients FD-trustworthy: parity harness/gate first, then reverse-mode reflection, geometry-motion gradients, and native AD through RayD diffraction kernels |
| `plans/30-coherent-scattering-model-plan.md` | Active | Coherent, complex, full-polarimetric (2×2) surface scattering shared by MC + deterministic solvers, staged so the parametric model and the future full-wave T-matrix share one `scattering_jones_operator` interface |
| `plans/31-rayd-deterministic-exact-coherent-accumulator-plan.md` | Active | RayD exact coherent diffraction accumulator for deterministic radiomap acceleration, default non-AD first/second/third-order auto routing, Munich stress benchmarks, and AD routing discipline |

When a plan completes, the agent that lands the final task moves it into `archive/completed/` (keeping its numeric prefix) in the same change and removes the row above.

## Active Optimization

| Path | Purpose |
| --- | --- |
| `optimization/channel-package-architecture-audit-2026-05-15.md` | Architecture audit for `channel_scene`, `channel_utils`, `deterministic`, and `montecarlo`, including code-quality hotspots and refactor priorities |
| `optimization/channel-package-deep-audit-2026-05-15.md` | Evidence-and-ordering companion to the architecture audit |
| `optimization/channel-defensive-code-reduction-2026-05-15.md` | Defensive-code line-savings ceiling across `witwin/` |
| `optimization/channel-readability-conciseness-audit-2026-05-16.md` | Readability and conciseness audit, including trivial delegates, deterministic/Monte Carlo duplication, megamodules, and a prioritized line/file reduction roadmap |
| `optimization/rayd-diffraction-task9-benchmark-record-2026-05-22.md` | Smoke benchmark record for RayD OptiX diffraction Tasks 8-9, including strict grid/path gates and three-cube supplemental timing |
| `optimization/rayd-ad-routing-and-epc-performance-2026-05-23.md` | Active routing decision for RayD handwritten AD, DrJit reflection tracing, and EPC performance/correctness gates |
| `optimization/munich-performance-regression-gates-2026-05-22.md` | Unified opt-in Munich performance regression workflow for path, Monte Carlo basic, and Monte Carlo BDPT solver workloads |
| `optimization/sionna-2.0.1-radiomap-grad-probe-2026-05-20.md` | V1.1 / V1.2 closure for plan 26 W1 — empirical probe of Sionna 2.0.1 RadioMapSolver gradient behaviour (symbolic vs evaluated × material vs geometry), with FD-vs-AD cross-check on the 3-cube reference scene |

## Archive

See `archive/README.md` for the full inventory of completed and superseded documents.

## Archive Policy

- If a document describes a still-relevant invariant, workflow, or supported behavior, keep it in `standards/`.
- If a document is an active defect tracker, keep it in `bugs/`.
- If a document proposes unfinished work, keep it in `plans/` or `optimization/` depending on whether the focus is feature design or performance work.
- If a document is complete, replaced, duplicated, or no longer authoritative, move it to `archive/` instead of leaving it in the active set.
