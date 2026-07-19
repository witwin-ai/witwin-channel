# Channel Native migration and runtime-dependency boundary

## Current decision

`witwin.channel_native` is the native entrypoint for the capabilities it
advertises. Repository-owned production Python must not import DrJit, Mitsuba,
Sionna, Python RayDN, or `witwin.channel`. The old Channel may remain in tests
and benchmarks as an offline correctness oracle; it must never be a production
fallback. The independent radar implementation is a separate product and is
not evidence for either Channel implementation.

The audited sibling roots (`core`, `genesis`, `maxwell`, `radar`, and `studio`)
contain no production Channel imports. This does not prove that external users,
deployed jobs, plugins, or private repositories have migrated.

The platform `core` package's `channel` and `all` extras now route to
`witwin-channel-native>=0.1,<0.2` (companion commit `9ee6655`) instead of the old
`witwin-channel` distribution. This establishes the repository-owned default
installation route; application-level canary/default-on state still requires
confirmation from each consumer owner.

## API surface changes

### ADR-013 coupled double diffraction (D->D)

`coupled_paths=True` now enables the uniform order-2 compensator family
{R->D, D->R, D->D} instead of the R->D / D->R pair alone. There is no new
`Config` field and no new public toggle: a partial family is non-uniform by
measurement, so the double-diffraction term (component id 7) shares the
existing `coupled_paths` gate, per-block candidate budget, and coupled
accumulator slot. Exported `coupled_paths=True` path tables now include cid 7
rows (kept distinct from cid 3/4 for audits), and the aggregated `coupled`
component map sums cids 3, 4, and 7. Coupled-off solves stay byte-identical.

The semantic capability manifest (`capabilities()`) gains
`coupled_double_diffraction: True` on every solver block that declares
reflection-diffraction coupling support (`path`, `deterministic`,
`montecarlo_bdpt`); `montecarlo_basic`, which does not support coupling, does
not expose the key. This is an intentional additive surface change under the
ADR-003 process; the curated `public-api-snapshot.json` function/class
contracts are unchanged because no public callable signature or `Config` field
changed.

## Rollout states

1. Inventory: route every real call to a supported Native capability or record
   an explicit unsupported product decision.
2. Shadow: execute old and Native implementations independently and record the
   comparison artifact below. Shadow failures must not trigger a production
   fallback.
3. Canary: make Native authoritative for a small declared cohort and retain the
   same correctness and operational evidence.
4. Default-on: the owning consumer makes Native authoritative. This repository
   exposes the Native entrypoint but cannot verify an application router or its
   rollout percentage; consumer-owner confirmation is required.
5. Delete: remove the old runtime integration only after every blocker below is
   closed.

## Shadow evidence artifact

Each maintained scenario must store versioned JSON under the release evidence
location chosen by CI. It must include: schema version, timestamp, release,
scenario/config/seed, Native and oracle commit/build identities, GPU/driver/
CUDA/OptiX/PyTorch metadata, cold and steady timing, peak memory, correctness
metrics and thresholds, pass/fail, and whether either side errored. Raw NPZ/JSON
outputs must be linked by content digest. No maintained Phase 10 shadow artifact
has been recorded in this repository yet; this is a required evidence contract,
not a fabricated run result.

The reduced three-way attempt on 2026-07-11 is recorded in
`path-threeway-shadow-attempt-2026-07-11.json`. Native timing and peak-memory
measurement completed, but both offline oracle processes failed during LLVM/
reference initialization, so the artifact is explicitly failed and does not
close the maintained shadow gate.

## Deletion blockers

Deletion remains blocked until all of the following are true:

- external consumers and private/deployed workloads have a signed inventory;
- maintained shadow and canary artifacts pass for the supported matrix;
- every consumer owner confirms Native is default-on with no production fallback;
- two consecutive release cycles complete without a fallback request;
- all P0/P1 items are closed or explicitly excluded by product decision;
- maintained correctness, performance, memory, cold-start and deployment gates pass;
- wheel smoke and the required GPU/SM matrix have runtime evidence;
- pipeline cache is either implemented and validated or explicitly removed from
  the release requirement.

As of 2026-07-11 the external audit, shadow/canary evidence, owner default-on
confirmation, two-release observation, wheel/SM evidence, and pipeline-cache
gate are not complete. Phase 10 therefore establishes the migration contract
and production dependency boundary but does not authorize deletion.

## Enforcement

Run the local contract with:

```powershell
python ci/check_production_dependencies.py
```

Sibling repositories can be audited without modifying them:

```powershell
python ci/check_production_dependencies.py --consumer-roots ..\core ..\genesis ..\maxwell ..\radar ..\studio
```

Consumer mode rejects the old Channel import only. This intentionally does not
classify Radar's independent DrJit/RayD tracer as a Channel runtime fallback.

## Public API additions (backward compatible)

### ADR-019: `montecarlo.bdpt` coherent combine (2026-07-18)

`witwin.channel_native.montecarlo.bdpt.Config` gains one field:

- `coherent: bool = False`

This is a purely additive, opt-in switch. The default (`False`) preserves the
existing power-domain incoherent accumulation BIT-IDENTICALLY, so no existing
caller, benchmark, or preset changes behaviour. Existing positional/keyword
construction of the config is unaffected (the field appends after `components`
with a default).

When set to `True`, BDPT sums the enumerated delta/UTD family (`los`,
`reflection`, `diffraction`, plus the `coupled_paths` compensator) coherently
per (tx, rx, component) and finalizes `|sum|^2`, tracking the deterministic
per-component coherent power. Coherent is refused for `transmission`/`scattering`
components (stochastic samplers, no coherent field) and for `ad_mode != 'none'`.
Result metadata records the active combine domain under `metadata["combine_domain"]`
(`"power"` or `"coherent"`). See
`docs/dev/standards/adr-019-bdpt-coherent-combine.md`.

The public-api snapshot updates only the `montecarlo.bdpt.Config`
`contract_sha256` (export count unchanged). No native ABI symbol is added; the
`bdpt_accumulate_connection_samples` binding gains defaulted `combine_domain` /
`coeff_real` / `coeff_imag` arguments (binding count unchanged at 193).

### ADR-021: multi-bounce coherent scattering (2026-07-18)

Three public `Config` classes gain purely additive, opt-in fields. Every default
preserves the existing behaviour BIT-IDENTICALLY, so no existing caller,
benchmark, or preset changes.

- `witwin.channel_native.deterministic.Config` gains four fields:
  `scattering_coherent: bool = False`, `scattering_chain_max_depth: int = 0`,
  `scattering_chain_samples_per_m2: float = 2.0`,
  `scattering_chain_max_rows: int = 256`.
- `witwin.channel_native.path.Config` gains three fields:
  `scattering_chain_max_depth: int = 0`,
  `scattering_chain_samples_per_m2: float = 2.0`,
  `scattering_chain_max_rows: int = 256`.
- `witwin.channel_native.montecarlo.bdpt.Config` gains one field:
  `max_scattering_order: int = 1`.

`scattering_chain_max_depth = 0` disables chain discovery (no allocation, launch,
or RNG); `scattering_coherent = False` keeps the incoherent power scattering
slot; `max_scattering_order = 1` keeps BDPT's terminal single-scatter behaviour.
See `docs/dev/standards/adr-021-multibounce-coherent-scattering.md` and
`docs/dev/plans/10a-scattering-v2-native-interfaces.md`.

The public-api snapshot updates three `contract_sha256` values only
(`path.Config`, `deterministic.Config`, `montecarlo.bdpt.Config`); the public
export count is unchanged at 37. Six new native ABI symbols are added (the ADR-021
chain forwards `scattering_chain_ensemble_eval` / `scattering_chain_realization_eval`
plus their `_backward`/`_jvp` companions), moving the binding count 193 -> 199
(`EXPECTED_NATIVE_BINDING_COUNT`, `EXPECTED_BINDING_COUNT`, and the phase-10
binding-ownership audit `expected_count`). ADR-021's D3 coherent combine adds NO
new primal symbol: it rides a defaulted `scattering_combine_domain` argument on
the existing `deterministic_accumulate_flat` op (and its `_backward`/`_jvp`),
mirroring ADR-019's `combine_domain`.
