# Source code style

Every tracked Python, C++, CUDA, and native header file begins with two comment
lines: the exact copyright `Copyright Xingyu Chen.` and one plain sentence that
says what the file does. The purpose sentence is at most 100 characters.
All source comments and docstrings describe current behavior, a current
constraint, or the reason for nearby code in plain language. They do not use
ADR numbers, numbered plans, phases, waves, migration history, audit labels, or
changelog prose. A Python module docstring, when present, is one sentence no
longer than 120 characters and does not repeat an architecture narrative.
Detailed rationale belongs in owner documentation; algorithm comments belong
beside the code they explain. Frozen identifiers and test data may keep required
historical spellings, but prose does not use them as explanations. Living
documentation names canonical files and symbols instead of brittle source line
numbers; generated line-sensitive evidence is refreshed with the code that
moves.

Native kernels share ordinary vector and complex math through
`native/channel/kernels/math.cuh`. That header owns `Vec3`, `Complex`,
`Complex3`, memory load/store helpers, vector arithmetic, and basic complex
arithmetic. Translation units may select a clearly named numerical policy from
the shared header, but may not restate the type or helper. Policy names preserve
meaningful differences such as epsilon floors, zero or explicit fallback
vectors, explicit rounding order, and reciprocal-square-root normalization.
Specialized derivative and tape carriers are algorithm state, not substitutes
for the shared ordinary math types.

Python uses the vector and quaternion utilities exported by `witwin.core`.
Channel does not add a second generic tensor-math module. Domain-specific
offline material math remains with its owner, while independent reference
implementations remain under `tests/` and are never production dependencies.

The quick CI tier runs `ci/check_source_headers.py` and
`ci/check_shared_math.py` so new files and new duplicate math owners fail before
CUDA tests start.
