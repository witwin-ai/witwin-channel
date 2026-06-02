# Codex Operating Guide

Status: Active
Category: Standard
Last reviewed: 2026-05-16

## Purpose

This document records the operating rules and known pitfalls for running OpenAI Codex (the CLI agent) against this repository on the maintainer's Windows + PowerShell setup. The intent is to keep agent runs deterministic and to prevent Codex sessions from being blocked by environment quirks that have already been diagnosed.

The general repository rules in `10-agent-reference.md` still apply. This document only adds Codex-specific guidance.

## Environment

- Use the `witwin2` conda environment for all Python commands, tests, and scripts.
- Default shell on this machine is PowerShell (Windows 11 Pro). All shell snippets below assume PowerShell unless they explicitly call `python -c ...`.

Direct interpreter path when conda activation is unreliable:

```text
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe
```

## Search

- `rg` is available in PATH but frequently fails on this machine with `Access is denied`. When that happens, switch immediately to PowerShell-native search:
  - `Get-ChildItem -Recurse -File -Include *.py | Select-String -Pattern '...'`
- Do not retry `rg` repeatedly hoping it will succeed.

## Python Invocation

- `conda run -n witwin2` may fail to pass stdin through to `python -` in this Codex setup. Prefer:
  - `conda run -n witwin2 python -c "..."` for short snippets.
  - Invoking the environment interpreter directly with `python.exe -c "..."` or `python.exe path/to/script.py`.
- Windows command-line length limits are easy to hit with `python -c` and inline scripts. For longer code, push the script into an environment variable and read it back:

```powershell
$env:PYCODE = @'
import os
# ... long script ...
'@
& C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -c "import os; exec(os.environ['PYCODE'])"
```

## File Editing

- Do not attempt to rewrite a large file in one shot from the CLI. Split large rewrites into chunked patches.
- Large single-shot `apply_patch` updates can hit Windows path and command-length limits. If a full-file replacement fails for size reasons, rewrite it in smaller patch chunks instead of retrying the same oversized patch.
- When comparing `AGENTS.md` and `CLAUDE.md`, do not use bare `fc` in PowerShell because it resolves to `Format-Custom`. Call `fc.exe` explicitly.

## Plan Files

When Codex picks up a plan file from `docs/dev/plans/`, read the `Status:` header first. Files marked `Status: Completed` or located under `docs/dev/archive/` are kept only for traceability and must not be treated as implementation backlogs.

Plan-file checkbox conventions are described in `13-superpowers-operating-guide.md`. The same checkbox semantics apply when Codex executes the plan instead of Claude Code.

## Sanity Checks Before Long Runs

1. Confirm `conda activate witwin2` succeeds in the current shell.
2. Confirm `python -c "import drjit, torch; print(torch.cuda.is_available())"` returns `True`.
3. Confirm `python -m pytest tests --collect-only -q` finishes without import errors.
4. Confirm the working tree matches the expected branch with `git status` before mutating files.

If any of those checks fails, fix the environment first instead of letting the agent paper over the problem with retries.
