# Archive Inventory

This directory holds documentation that is intentionally no longer part of the active working set.

## Completed

| Path | Reason |
| --- | --- |
| `completed/channel-core-scene-migration-plan.md` | Public scene migration is already the repository baseline |
| `completed/multiorder-diffraction-completeness-checklist.md` | Completion checklist kept for historical traceability only |
| `completed/multiorder-diffraction-post-completion-audit.md` | Post-completion audit of the finished rollout |
| `completed/trace-architecture-boundary-refactor.md` | Boundary-stabilization refactor landed; retained only as the historical implementation plan |
| `completed/cell-state-memory-rollout.md` | Consolidated summary of the completed cell-state memory rollout |
| `completed/cell-state-memory/cell-state-memory-optimization-plan-2026-04-01.md` | Raw optimization plan, superseded by the consolidated rollout summary |
| `completed/cell-state-memory/cell-state-memory-phase-0-1-implementation-2026-04-01.md` | Raw implementation log |
| `completed/cell-state-memory/cell-state-memory-phase-2-3-implementation-2026-04-01.md` | Raw implementation log |
| `completed/cell-state-memory/cell-state-memory-phase-4-native-packed-state-implementation-2026-04-01.md` | Raw implementation log |
| `completed/cell-state-memory/cell-state-memory-phase-5-6-implementation-2026-04-02.md` | Raw implementation log |
| `completed/cell-state-memory/cell-state-memory-phase-7-implementation-2026-04-02.md` | Raw implementation log |
| `completed/cell-state-memory/cell-state-memory-native-first-followup-2026-04-02.md` | Raw follow-up note |
| `completed/cell-state-memory/cell-state-memory-native-first-followup-round-2-2026-04-02.md` | Raw follow-up note |
| `completed/deterministic-munich-scaling-optimization-plan.md` | All five phases self-marked completed; superseded by the deterministic-package architecture plan still in `plans/` |
| `completed/monte-carlo-power-filtering-plan.md` | Filtering shipped in `witwin/channel/montecarlo/filtering.py` |
| `completed/monte-carlo-power-filtering-implementation-plan.md` | Implementation checklist for the shipped filtering work |
| `completed/multipath-native-symbolic-improvement-plan.md` | Native/symbolic improvement phases landed before the package rename |
| `completed/multipath-performance-scaling-report.md` | Historical multipath scaling report; targets the pre-rename `witwin.channel.trace` layout |
| `completed/multipath-scaling-stress-report-2026-04-01.md` | Historical forward stress dataset snapshot |
| `completed/multipath-scaling-stress-fb-report-2026-04-01.md` | Historical forward/backward stress dataset snapshot |
| `completed/radio-map-monte-carlo-diffraction-kernel-launch-analysis-2026-04-08.md` | Historical kernel-launch analysis whose findings folded into the current `witwin.channel.montecarlo` package |
| `completed/unified-radiomap-grid-result-contract.md` | Contract shipped as `witwin.channel.core.radiomap_result.RadioMapResult` |
| `completed/unified-radiomap-grid-result-implementation-plan.md` | Implementation checklist for the shipped unified-result contract |

## Superseded

| Path | Replacement or reason |
| --- | --- |
| `superseded/agent-reference-2026-04-09.md` | Replaced by `../standards/10-agent-reference.md` after the channel package layout was rewritten away from `witwin/channel/scene/`, `trace/`, `monitors/`, and `tracer.py` |
| `superseded/grid-monitor-sampling-standard.md` | `FieldMonitor` and the monitor abstraction were removed; receiver-grid sampling lives in `witwin.channel.core.grid` and is owned by the scene-owned `ReceiverGrid` |
| `superseded/41-closed-form-multi-diffraction-validation.md` | Documented the `witwin/channel/validation.py` + `tests/test_validation_references.py` harness, which was removed during the package rewrite and has not been re-ported under `witwin/channel/deterministic/` or `witwin/channel/montecarlo/`; a fresh standard will be written once the harness is rebuilt |
| `superseded/jones-vector-material-model-plan.md` | Replaced by `../plans/material-diffraction-rebuild.md` |
| `superseded/los-shadow-boundary-jump-analysis-2026-03-29.md` | Historical investigation only; target direction is in `../plans/material-diffraction-rebuild.md` |
| `superseded/transmission-refraction-analysis.md` | Replaced by `superseded/transmission-and-scattering-plan.md` and the broader rebuild plan; transmission and scattering are not yet re-scoped under the new package layout |
| `superseded/legacy-bugs-note.md` | Folded into `../bugs/known-bugs.md` |
| `superseded/cuda-migration-analysis.md` | Older migration analysis replaced by current standards and optimization docs |
| `superseded/discontinuity-analysis.md` | Historical analysis that is not an active implementation direction |
| `superseded/gpu-performance-analysis-2024-12.md` | Older benchmark snapshot replaced by the current channel-package audits in `../optimization/` |
| `superseded/native-utd-receiver-tiling-reference-2026-04-09.md` | Historical code snapshot for the removed UTD receiver-tiling rollout |
| `superseded/perf-audit-legacy.md` | Older audit replaced by current optimization and channel-package audit docs |
| `superseded/rd-shadow-boundary-smoothing-plan.md` | Historical smoothing proposal that is not the active solver direction |
| `superseded/multipath-legacy-options-note.md` | Legacy brainstorming note with naming and encoding issues; archive only |
| `superseded/field-monitor-3d-extension.md` | `FieldMonitor` and the monitor abstraction have been removed |
| `superseded/matched-isb-completion-gradient-noise-plan.md` | Targeted the removed `RadioMapMonitor` shadow-boundary modes; replaced by current deterministic shadow-boundary correction work |
| `superseded/monitor-solver-separation-refactor.md` | Targeted the deleted `witwin.channel.tracer`; replaced by the standalone `witwin.channel.deterministic` and `witwin.channel.montecarlo` packages |
| `superseded/path-monitor-design.md` | `PathMonitor` removed; replaced by the standalone `witwin.channel.path` package |
| `superseded/per-object-material-analysis.md` | Targeted the deleted `Tracer` and `reflection/materials.py` layout |
| `superseded/radio-map-architecture-refactor-plan.md` | Targeted the deleted `witwin/channel/monitors/radio_map/` tree |
| `superseded/radio-map-deterministic-decoupling-plan.md` | Landed via the deterministic-package extraction |
| `superseded/radio-map-monitor-plan.md` | `RadioMapMonitor` was replaced by the standalone deterministic and Monte Carlo packages |
| `superseded/radio-map-monte-carlo-gradient-roadmap.md` | Targeted the deleted Monte Carlo radiomap monitor tree; replaced by `witwin/channel/montecarlo/integrators/` |
| `superseded/radio-map-monte-carlo-mode-plan.md` | Already shipped as the standalone `witwin.channel.montecarlo` package |
| `superseded/radio-map-native-cell-accumulation-plan.md` | `RadioMapMonitor` was removed; native accumulation lives under `witwin/channel/_native/` plus solver-owned `witwin/channel/{deterministic,montecarlo}/kernels/` |
| `superseded/transmission-and-scattering-plan.md` | Targets obsolete `witwin.channel.monitors` API; transmission and diffuse scattering require a fresh plan under the new package layout |
| `superseded/wedge-backend-neutral-migration.md` | Targeted the deleted `witwin/channel/scene/runtime.py` and `trace/diffraction/` tree; wedge runtime now lives in `witwin/channel/core/scene/wedge_runtime.py` and `wedge_types.py` |
| `superseded/k1-diffraction-accumulation-refactor.md` | Targeted the deleted `witwin.channel.trace.diffraction.field`; current diffraction kernels live in `witwin/channel/deterministic/kernels/` and `witwin/channel/montecarlo/kernels/` |
| `superseded/kernel-symbolicization-analysis.md` | Targeted `RadioMapMonitor`; symbolic and JIT analysis now belongs in the Monte Carlo package overview |
| `superseded/memory-optimization-and-kernel-candidates.md` | Targeted the deleted `witwin.channel.trace` tree |
| `superseded/path-monitor-acceleration-plan.md` | `PathMonitor` removed; replaced by `witwin.channel.path` |
| `superseded/path-monitor-phase6-rollout-gates.md` | Benchmark gates frozen against the removed `PathMonitor` entrypoint |
| `superseded/radio-map-monitor-baseline-benchmark-2026-04-03.md` | `RadioMapMonitor` benchmark snapshot; the monitor was removed |
| `superseded/radio-map-monitor-sionna-alignment-analysis-2026-04-04.md` | Sionna alignment analysis targeted `RadioMapMonitor`; alignment work continues under the standalone Monte Carlo package |
| `superseded/receiver-tiled-cuda-path-family-refactor-plan.md` | Targeted `FieldMonitor` workloads that no longer exist |
| `superseded/trace-module-audit.md` | Audit of the deleted `witwin/channel/trace/` tree; replaced by the `channel-package-*-2026-05-15` audits |

## Rule

Archived documents may still be useful for archaeology, but they must not be treated as the source of truth for current behavior or new implementation work.
