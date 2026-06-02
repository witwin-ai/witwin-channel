# Claude Code Operating Guide

Status: Active
Category: Standard
Last reviewed: 2026-05-16

## Purpose

This document records the operating rules and conventions for running Anthropic Claude Code (the CLI and IDE-extension agent) against this repository. The intent is to keep Claude Code sessions deterministic, to make permission, hook, and skill use predictable, and to record the conventions that already exist in `.claude/settings.json`.

The general repository rules in `10-agent-reference.md` still apply. This document only adds Claude Code-specific guidance.

## Environment

- Use the `witwin2` conda environment for all Python commands, tests, and scripts.
- Claude Code runs in PowerShell on this machine. Bash is available through the `Bash` tool, but prefer PowerShell-native syntax for anything that has a clean PowerShell form.
- Direct interpreter path when needed: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe`.

## Tooling Preferences

- Prefer dedicated tools over `Bash`: `Read`, `Edit`, `Write`, `Glob`, `Grep`. Reserve `Bash` for shell-only operations.
- Use the `Agent` tool with the `Explore` subagent for broad codebase research instead of running long manual grep sweeps in the main context.
- Use `TodoWrite` for multi-step tasks. Mark each task complete as soon as it lands; do not batch completions.

## Permissions Discipline

Permissions live in `.claude/settings.json` (project-wide) and `.claude/settings.local.json` (per-checkout overrides). The allow list has accumulated many one-off bash forms over time. When adding new permissions:

- Prefer narrow command patterns over wildcards.
- Do not add `Bash(rm:*)`, `Bash(git push:*)`, or other destructive wildcards.
- Group related entries logically. Do not duplicate an entry that already exists in a slightly different form; consolidate instead.
- Remove permissions for paths that no longer exist (for example, anything that targeted the deleted `witwin/channel/trace/*` tree).

When a permission prompt fires frequently for a safe, read-only command, prefer adding it to the allow list over teaching the agent to avoid the command.

## Hooks

Hooks live in `.claude/settings.json` and execute outside the model. Treat hook output as coming from the user.

- Do not add hooks that run destructive commands (`git push`, `rm`, `Remove-Item -Force`) automatically.
- Hooks that run formatters or linters on file save should be idempotent and fast.
- If a hook ever rewrites Python in ways the agent did not author, log that fact in the hook block and document it here.

## Skills

Project-relevant Claude Code skills currently in use:

- `superpowers:executing-plans` and `superpowers:subagent-driven-development` for working through checkbox-tracked plan files. See `13-superpowers-operating-guide.md`.
- `python-code-standard` for auditing new and existing Python against the standards in this directory (especially `20-python-package-architecture-standard.md` and `21-python-runtime-typing-standard.md`).
- `update-config` for permission and hook edits, so changes go through one consistent flow.
- `simplify` for post-implementation review passes that look for reuse and dead code.

Do not invent or invoke skills that are not present in the active skill list for the session.

## CLAUDE.md and AGENTS.md Discipline

- `CLAUDE.md` and `AGENTS.md` at the repository root and at `channel/` must stay byte-identical. Both files are checked into the repository.
- When comparing the two on Windows, call `fc.exe` explicitly. Bare `fc` resolves to `Format-Custom` in PowerShell.
- Root `CLAUDE.md` covers monorepo-wide rules. Subproject `CLAUDE.md` files cover project-specific rules and take precedence inside their subdirectory.
- Long workflow rules belong in active standards under `docs/dev/standards/`, not duplicated into `CLAUDE.md`.

## Agent Subprocesses

When delegating to subagents (`general-purpose`, `Explore`, `Plan`):

- Brief the subagent with file paths, line numbers, and the specific question. Generic prompts produce shallow work.
- Use `Explore` for read-only research that would otherwise eat the main context window.
- Run independent subagents in parallel within a single tool-call block.
- Verify subagent claims about code changes; the summary is what the agent intended, not necessarily what it did.

## Plan Files

Plan files in `docs/dev/plans/` use the checkbox convention described in `13-superpowers-operating-guide.md`. Claude Code should:

- Read the `Status:` header before treating a plan as a backlog. `Completed` plans live in `docs/dev/archive/completed/` and are reference-only.
- Update checkbox state in the same edit that lands the corresponding code change.
- When a plan finishes, mark it `Status: Completed`, update `docs/dev/README.md`, and move it to `docs/dev/archive/completed/`.
