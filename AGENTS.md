# Repository Agent Guide

This document defines the always-on working rules for coding agents in this repository. Keep `AGENTS.md` and `CLAUDE.md` identical.

## Always-On Rules

- Use the `witwin2` conda environment for all Python commands, tests, and scripts.
- Keep the codebase clean. Do not preserve legacy code, compatibility shims, or backward-support paths unless explicitly requested.
- Write all code comments and commit messages in English.
- Update `FEATURE_LIST.md` in the same change for every new user-visible feature, public API capability, or meaningful workflow addition.
- Keep development-cycle Markdown under `docs/dev/`, and keep repository-root Markdown limited to repository-wide entrypoints and operating rules.
- Prioritize correctness and efficiency. Core computation is GPU-first, so minimize GPU-CPU transfers and do not add new CPU fallback paths unless explicitly requested.
- The stable public architecture is `Scene + solver.solve(scene, config) + Result`. Solver entrypoints live in `witwin.channel.deterministic`, `witwin.channel.montecarlo`, and `witwin.channel.path`; there is no shared `Tracer` object.
- New public scene APIs must use `witwin.core` structures, materials, and geometry objects. Do not add raw `vertices/faces` scene constructors or `Scene.from_meshes(...)` compatibility helpers.
- Keep runtime internals DrJit-native. Do not introduce NumPy, Torch, or DLPack bridges in solver internals, native-kernel paths, or other hot paths. Torch is allowed only at explicit public API or result-adapter boundaries.
- Do not introduce heuristic smoothing, soft approximations, or ad hoc gradient hacks for diffraction or intersection derivatives unless explicitly requested.
- Shared core geometry constructors default to `device=None`; `Scene(...)` owns device placement and defaults to CUDA.
- Avoid duplicate implementations in parallel files. Update the primary module instead.
- Check `witwin/channel/core/` before adding local helpers. Do not duplicate generic geometry or DrJit utility helpers.
- Respect package layering: `channel/core/ -> channel/core/scene/ -> {channel/deterministic, channel/montecarlo, channel/path}/`. Solver packages must not import from each other.
- Keep exploratory or script-style gradient workflows under `tests/support/bin/`, not under `witwin/`.
- Do not introduce new `rfdt` import paths. User-facing code should import from the `witwin.*` namespace.
- On Windows, do not write large files in one CLI command. Use small chunks or incremental patches.
- When running `pip install .`, include `--no-deps`.
- For native C++/CUDA-only iteration, prefer CMake incremental builds over repeated editable reinstalls: `cmake --build build\cp311-cp311-win_amd64 --config Release --target <native_target>`, then `cmake --install build\cp311-cp311-win_amd64 --config Release --prefix <witwin2>\Lib\site-packages` to sync `.pyd` files. Close Python/Jupyter processes that have loaded the extension before installing, because Windows locks loaded `.pyd` files.
- When writing, reviewing, refactoring, or debugging code, use the karpathy-guidelines skill.

## Canonical References

- Use `docs/dev/README.md` as the index for active development documentation.
- Use `docs/dev/standards/10-agent-reference.md` for repository layout, public API surface, shared utility ownership, command examples, and dependencies.
- Use `docs/dev/standards/11-codex-operating-guide.md`, `12-claude-code-operating-guide.md`, and `13-superpowers-operating-guide.md` for runtime-specific operating notes (Codex CLI quirks, Claude Code permission and hook discipline, Superpowers plan-file conventions).
- Before changing documentation or workflows, check the canonical standards in `docs/dev/standards/`: `00-documentation-naming-standard.md`, `01-repository-documentation-crosswalk.md`, `50-test-and-acceptance-workflow.md`, `30-cuda-kernel-development-guide.md`, and `31-cuda-kernel-migration-workflow.md`.
- If a workflow rule already exists in an active standard, update that document instead of duplicating the rule here.
