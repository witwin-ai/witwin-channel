# ADR-044: Native CUDA translation-unit consolidation

- **Status:** Accepted
- **Date:** 2026-07-28
- **Kind:** Native physical layout and build policy
- **Related:** ADR-004, ADR-009, ADR-022, ADR-025, ADR-026, ADR-027, ADR-032

## Context

The production `_channel` target currently compiles 45 Channel-owned CUDA
translation units under `native/channel/kernels`. Most of the count came from a
bounded maintenance migration that split complete native operation families to
satisfy a 2,000-line recommendation and a 3,000-line hard limit. That size
policy no longer represents an owner or correctness boundary. It now obscures
operation-family locality and makes the source list substantially larger than
the stable native ownership surface.

The Phase-9 source inventory remains frozen historical evidence. This decision
does not rewrite that baseline and does not weaken its ABI, launch,
synchronization, tape, numerical-order, exactness, or performance evidence.

## Decision

There is no maximum line count for native `.cpp`, `.cu`, `.cuh`, or header
files. ADR-009's **Translation-unit budget** section is superseded only with
respect to the recommended and hard line counts and their debt allowlists. The
ADR-009 ownership priority and every numerical/fusion acceptance condition
remain in force.

Channel consolidates the 45 CUDA translation units to the 15 physical units
recorded in `docs/dev/audit/adr-044-native-tu-consolidation.json`. Physical
co-location is not kernel fusion and does not change semantic ownership:

1. every `channel_*` ABI entry retains one owner and the same schema;
2. kernel launches, explicit synchronizations, allocations, streams, tape
   lifetime, row identity, and reduction order stay unchanged;
3. body moves preserve numerical expressions; private-name collision repairs
   are limited to owner-local detail names and are recorded;
4. default precise CUDA sources remain precise;
5. `kirchhoff_table_ad.cu` and `mc_transmission_wall_product.cu` remain separate
   and retain `--fmad=false`;
6. RayD's pure-wedge `--use_fast_math` translation unit is untouched;
7. consolidation may not add compatibility symbols, dormant bindings, a second
   dispatcher, a fallback backend, or a new host/device boundary.

Owner-local collision repairs use provenance-section-scoped preprocessor aliases
that are undefined at the next section boundary. The compiler therefore sees
unique private names while frozen function-body source tokens remain unchanged.
The sole include-order repair clears the Windows RPC `small` macro before the
later Torch `CUDAGuard` include in `evaluated_paths.cu`; it is outside every
function body and carries no numerical behavior.

Shared `.cuh` files survive only while they have more than one production
consumer or own an explicit lockstep device contract. The consolidation must
move bodies into the owning `.cu`; it must not merely rename old `.cu` files to
include fragments.

## Acceptance

Each consolidation batch must preserve the pre-change ABI and kernel-launch
multisets, explicit synchronization count, registered binding coverage, and
focused primal/JVP/VJP behavior. A Release build records clean-build time,
per-unit compile time where available, compiler peak memory where available,
extension size, and `ptxas` register/shared-memory/spill diagnostics for changed
units. Any numerical output, launch/resource, compile-mode, or performance
change is a separate decision and is not accepted by this ADR.

The old split-specific tests are replaced by ledger-driven tests of the final
physical owners and invariant multisets. `ci/maintenance-budgets.json` retains
mandatory function-complexity policy but omits `limits.native_file_lines` and
`native_file_exemptions`.
## Local validation

Per the requested local scope, validation covers `sm_120-real` only. It does not
claim PTX, virtual-architecture, or multi-architecture coverage. The build used
the `witwin2` interpreter, Ninja, and Release mode, and completed a 62-step full
rebuild after the architecture change in 1,373.078 seconds. The resulting
`_channel.cp311-win_amd64.pyd` is 34,934,784 bytes with SHA-256
`c555b237323743a8b228983d9c8d388db0af86c087f31cb1889efc2b4b88f599` and
build fingerprint
`efc19ddfeadbb3196e9f0f0afbe74ae338cce2124ca07d80f2ba3584c79bf3e0`.

The focused native capacity, compact-path, field, scattering-chain, and BDPT
primal/JVP/VJP suite passed 117 tests, the direct Python/native AD wrapper
control-flow suite passed 11 tests, and 344 runtime/facade contract tests passed
after the remaining 19 hand-rolled native-symbol probes were routed through the
single `runtime.required_symbol` owner. The recorded single-definition debt is
therefore zero. The complete repository suite passed 2,678 tests, with 8 declared
skips and 1 declared expected failure. All applicable governance checks passed.
Statement coverage is 87.481429%, branch coverage is 67.113665%, and the governed
core statement aggregate is 90.468785%; no coverage threshold or exemption
changed. The duplication ledger contains 152 current classified regions, no
unclassified or stale regions, and 9.090609% coverage against the unchanged
10.211512% frozen ceiling.

The Ninja/toolchain log did not emit per-unit timing, compiler peak-memory, or
`ptxas` resource diagnostics, so this validation makes no new resource-usage
claim. It recorded no build error marker; emitted warnings were the existing
Windows code-page, conversion, and unused private-symbol warnings. Deleted
source paths remain only as historical evidence names and provenance labels;
they are not compatibility build inputs, aliases, re-exports, or dormant
owners.


