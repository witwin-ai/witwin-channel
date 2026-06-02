---
name: xy_py
description: Minimal-code Python coding rules for this project. Disable staticmethod forwarding, set/get attribute shims, and unnecessary fallbacks. Avoid over-abstraction, unrequested "extensibility", excessive defensive validation, and coerce functions. Minimize isinstance; declare explicit input/output types. Invoke when writing, reviewing, refactoring, or debugging Python code in this repo, or when the user runs /xy_py.
---

# xy_py — Minimal Python Rules

Apply these rules to any Python you write, review, refactor, or debug in this project.

## Hard bans

- **No staticmethod forwarding.** Do not wrap an existing function in a `@staticmethod` just to expose it on a class.
- **No `__setattr__` / `__getattr__` / `__getattribute__` shims** for attribute proxying, lazy forwarding, or "convenience" access.
- **No unnecessary fallbacks.** Do not add `try/except` that swallows errors to return a default, alternate code path "in case", or backup implementations. Let it fail.
- **No coerce / normalize helpers** that accept "anything" and convert it. Callers pass the right type.
- **No defensive validation** of internal inputs. Validate only at true system boundaries (user input, external APIs, file I/O).

## Style

- **Minimum code.** Solve exactly what was asked, nothing more. No extra features, no unrequested "extensibility", no future-proofing.
- **No over-abstraction.** Three similar lines beat a premature abstraction. No base classes, registries, or factories unless the concrete need already exists.
- **Minimize `isinstance`.** If you need it often, the types are wrong — fix the types instead.
- **Explicit types on inputs and outputs.** Every public function has typed parameters and a typed return. No `Any`, no untyped `**kwargs` unless truly opaque.
- **No oversized dict returns.** Avoid returning large untyped `dict` blobs. Return a typed dataclass / `NamedTuple` with the exact fields needed, or split into focused functions. Modules stay high-cohesion, low-coupling, and concise.
- **No one-line forwarding wrappers.** Do not write a function whose body is a single call to another function. Inline the call at the use site.
- **Compact call sites.** For short function calls and constructors, keep arguments on one line. Do not split every argument onto its own line unless the call genuinely exceeds the line budget or has many keyword args worth aligning. Keep each function body visually compact.
- **No stringly-typed state.** Do not encode state, kind, or type with string literals (`kind="reflect"`) or untyped dicts holding heterogeneous fields. Use `Enum` / `Literal` for discrete tags and `dataclass` / `NamedTuple` for structured payloads. Every stored field has a declared type — no "object bag" containers.

## Surgical changes

When editing existing code, touch only what the task requires.

- Do not "improve" adjacent code, comments, formatting, or imports that the task did not call for.
- Do not refactor things that aren't broken; match the existing style even if you'd write it differently.
- Only remove imports / variables / functions that *your own change* made unused. Do not delete pre-existing dead code unless asked — mention it instead.
- Every changed line should trace directly to the request.

## Decision rule

When tempted to add a helper, wrapper, fallback, or check: ask "did the task require this?" If no, delete it.
