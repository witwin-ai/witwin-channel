# Cell-State Memory Rollout Summary

Status: Active
Category: Optimization
Last reviewed: 2026-04-02

## Purpose

This document is the consolidated entry point for the completed cell-state memory optimization rollout. It replaces the scattered `cell_state_memory_*` plan and phase logs as the active reference.

## Problem The Rollout Solved

The original tracing pipeline stored too much repeated state per propagation branch and per exported path. The main cost came from:

- wide packed state with hot and cold fields mixed together,
- duplicated lineage/history payloads,
- dense path-export payload materialization before final selection,
- late pruning that allowed weak states to survive into expensive Cartesian expansion,
- Python-side fallback paths remaining on native-first hot paths.

The goal of the rollout was to reduce memory pressure without changing the public `Scene + Tracer + Result` architecture or introducing CPU fallback behavior.

## Final Status

The rollout is complete through Phase 7, plus two native-first follow-up passes.

### Delivered outcomes

- benchmark entrypoints and memory-profile reporting were stabilized,
- state layout was split into hot propagation data and optional cold metadata,
- repeated fixed history arrays were replaced by parent-link lineage,
- native packed-state support was re-enabled for the new lineage layout,
- diffraction path export moved to sparse references with lazy replay,
- bounded-mode pruning now happens earlier for weak states,
- packed-state width was reduced after the layout cleanup,
- top-K path export no longer fully materializes sparse diffraction paths before selection,
- follow-up work moved remaining AD-sensitive gather/replay slices back onto the native hot path.

## Phase Timeline

| Phase | Status | Main result |
| --- | --- | --- |
| Phase 0 | Completed | Repeatable benchmark entrypoint for cell-state memory cases |
| Phase 1 | Completed | Memory-safe profile and measurement guardrails |
| Phase 2 | Completed | Hot-state vs cold-metadata split |
| Phase 3 | Completed | Parent-link lineage replaces repeated history arrays |
| Phase 4 | Completed | Native packed-state migration plus sparse diffraction path references |
| Phase 5 | Completed | Early pruning before expensive Cartesian expansion |
| Phase 6 | Completed | Packed-state width reduction |
| Phase 7 | Completed | Lazy replay for path export and chunked selected-only materialization |
| Follow-up 1 | Completed | Native-first gather/subset cleanup and replay cost reduction |
| Follow-up 2 | Completed | Native path slot/depth assembly, native inserted-reflection gather, paired pruning-sort path |

## What Became The New Baseline

- Keep propagation-hot fields compact and isolated from optional metadata.
- Keep lineage reconstructable by links rather than duplicating full history arrays.
- Keep path export sparse until a caller explicitly requests the kept subset or geometry.
- Keep AD-sensitive gather/subset/replay logic on native hot paths whenever a native representation already exists.
- Use benchmarks to decide the next optimization stage instead of extending the rollout by assumption.

## What Was Archived

The raw planning and implementation notes were moved to `docs/dev/archive/completed/cell-state-memory/`. Keep those files only for historical traceability or benchmark archaeology.

## Next Related Work

The cell-state rollout fixed state-width and replay duplication, but it did not remove the dense many-to-many accumulation cliff. The next optimization documents to read are:

- `receiver-tiled-cuda-path-family-refactor-plan.md`
- `memory-optimization-and-kernel-candidates.md`
- `multipath-performance-scaling-report.md`
