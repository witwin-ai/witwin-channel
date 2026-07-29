# Source code style

## File ownership and layout

File boundaries follow domain ownership, ABI and fusion boundaries, compile
mode, and tape lifetime, not a line-count target. There is no Python or native
file-length ceiling. A move carries the complete owner and updates imports,
manifests, tests, and owner documentation together; deleted internal paths do
not get aliases or compatibility re-exports.

The current production layout is package-root contracts, `scene`, `materials`,
one `interactions` module per RF concept, one `kernels` facade module per native
domain, shared `propagation` stages, four solver entry points, and the native
operation-family translation units recorded by the native source inventory.
The structure gates are the executable source of truth.

## Function signatures

Python and native code use a 100-column target and pack as many complete
parameters as fit on each continuation line. A parameter gets its own line only
when it cannot fit safely with a neighbor. Ruff remains the Python linter; do
not use Ruff format because its exploded layout forces one parameter per line.

Run `python tools/compact_signatures.py` after editing Python signatures. Quick
CI runs the corresponding `--check` command. Multiline annotations, defaults,
or comments that cannot move without changing tokens stay as written.

C++, CUDA, and native headers use the root `.clang-format`, with parameter and
argument bin-packing enabled. Format touched ranges or files, not unrelated
numerical bodies.

Introduce a named request dataclass or C++ struct only when its fields form one
durable domain contract with one owner. Do not add generic argument bags,
forwarding shims, or compatibility layers for layout alone. Native ABI, pybind,
and autograd companion signatures may remain flat when explicit argument order
is part of the checked contract.

## Comments and file headers
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

## Shared mathematics

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

## Duplication

Exact-token duplication of at least 100 tokens is inventoried by
`tools/duplication_report.py`, classified in
`docs/dev/audit/duplication-classification.json`, and checked by
`ci/check_duplication.py`. Share validation, indexing, packing, and result
assembly only inside the same domain owner and only when extraction preserves
validation order, error precedence, allocation, launch, synchronization, and
ABI order. Cross-owner lookalikes stay explicit.

Repeated primal, JVP, VJP, and backward arithmetic stays duplicated when
floating-point order, compiler output, or tape lifetime is part of the
numerical contract. Collapsing it is a separate numerical change with exactness,
compiler, resource, and performance evidence. Generic vector, complex, and
scalar helpers do not qualify for that exemption.

## Enforced governance

Quick CI runs the import graph, orphan-module, single-definition, shared-math,
source-prose, compact-signature, public-surface, binding-coverage, and
maintenance checks. Nightly CI adds the complete duplication classification
gate. These checks are the regression boundary; this document explains them but
does not replace them.
