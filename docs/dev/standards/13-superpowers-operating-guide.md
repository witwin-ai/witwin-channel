# Superpowers Operating Guide

Status: Active
Category: Standard
Last reviewed: 2026-05-16

## Purpose

This document defines the plan-file conventions and workflow rules used when implementing work in this repository through the Superpowers skill pack (typically `superpowers:executing-plans` and `superpowers:subagent-driven-development`).

Superpowers itself is a generic Claude Code skill pack. This guide is the project-side contract that plan files in `docs/dev/plans/` should follow so they are executable end-to-end without per-plan ad-hoc instructions.

## Plan File Header

Every plan that is meant to be executed by a Superpowers skill must begin with:

```markdown
Status: Active | Draft | In Progress
Category: Plan
Last reviewed: YYYY-MM-DD

# Plan Title

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One sentence describing the user-visible or repository-visible outcome.

**Architecture:** Two to four sentences describing where the change lands in the package tree and which public surfaces are affected.

**Tech Stack:** Bullet list of the languages, libraries, and environments touched (for this repository, almost always Python, DrJit CUDA AD, pytest, and the `witwin2` conda environment).
```

The `> **For agentic workers:** ...` block is the contract between the plan author and any agent runtime. It is required.

Plans intended only as design notes (not executable backlogs) should not carry the agentic-workers header and should be filed as `Status: Draft` so they are not picked up by `executing-plans`.

## Task Structure

Tasks use this shape, in order:

```markdown
### Task N: Short Name

**Files:**
- Modify: `path/to/file.py`
- Add: `path/to/new_file.py`
- Test: `tests/<area>/test_<topic>.py`

- [ ] **Step 1: Write failing tests**

Brief description of the failing test surface.

- [ ] **Step 2: Run tests to verify red**

Run: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests/<area>/test_<topic>.py -q`

Expected: <named failure>.

- [ ] **Step 3: Implement <feature>**

Code-level description with the public surface and key invariants.

- [ ] **Step 4: Run tests to verify green**

Run the same pytest command. Expected: <named tests> pass.
```

Rules:

- Each task is independently testable. Do not merge two test surfaces into one task.
- Each step is a single checkbox. Do not nest checkboxes.
- Step ordering must always be: write red tests, observe red, implement, observe green. Refactors and follow-ups go in later tasks, not extra steps under the first task.
- Test commands must use the direct interpreter path (`C:\Users\Asixa\miniconda3\envs\witwin2\python.exe`) so they work regardless of whether `conda activate` has run in the agent's shell.

## File Targeting

- `Modify` lists existing files that the task changes.
- `Add` lists new files the task creates.
- `Test` lists the pytest files that gate the task.

Do not name a file under both `Modify` and `Add`. Do not list utility files that are only read but not edited.

## Status Transitions

- Plans land as `Status: Draft` while they are being refined.
- When the plan is approved and ready for execution, flip to `Status: Active` (or `Status: In Progress` once the first task has started).
- When all tasks have green checkboxes and the corresponding code changes have landed, the agent that lands the final task must, in the same change:
  1. Flip the header to `Status: Completed` and update `Last reviewed`.
  2. `git mv` the plan into `docs/dev/archive/completed/`, keeping the numeric prefix that was assigned when the plan landed in `plans/`.
  3. Remove the entry from the Active Plans table in `docs/dev/README.md` and add it to the Completed section of `docs/dev/archive/README.md`.

Leaving a `Status: Completed` plan inside `plans/` is a documentation bug. `plans/` is reserved for ongoing work.

A plan that becomes obsolete before completion (for example, because the underlying API was rewritten) goes to `docs/dev/archive/superseded/` instead, with a one-line replacement pointer in `docs/dev/archive/README.md`. The numeric prefix is preserved across the move so cross-references stay stable.

## Subagent-Driven Development

When using `superpowers:subagent-driven-development`:

- The driver agent owns plan-file edits and never modifies code directly.
- Each Task is delegated to a fresh subagent with the plan path, the task heading, and the required file list pasted into the subagent prompt.
- The driver verifies the subagent's reported changes by reading the listed files before flipping checkboxes.

## Anti-Patterns

- Do not include checkbox items that are not real implementation steps (for example, "remind the user to update FEATURE_LIST.md"). Real steps land code or run tests.
- Do not leave a single trailing unchecked meta-step at the end of a plan to keep it "open." If all real work is done, the plan is `Status: Completed`.
- Do not invent file paths. Every `Modify` and `Add` path must correspond to a real package location in the current layout (`witwin/channel/core/scene/`, `witwin/channel/core/`, `witwin/channel/deterministic/`, `witwin/channel/montecarlo/`, `witwin/channel/path/`).
- Do not write multi-paragraph rationale inside step bodies. Long rationale belongs in the plan introduction.
