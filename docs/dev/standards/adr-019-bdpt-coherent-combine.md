# ADR-019: BDPT opt-in coherent combine (DEFAULT OFF)

Status: Accepted.

## Context

The `montecarlo.bdpt` solver accumulates real per-path POWER only. Every
connection sample carries `contribution = |coeff|^2`
(`native/channel/kernels/bdpt_connect.cu:168-181`), and
the accumulator atomic-adds `contribution * mis_weight` into per-(tx, rx,
component) real matrices
(`native/channel/kernels/bdpt_connect.cu:964,1138,1189,1259`). The
connection-sample schema (`_BDPT_CONNECTION_SCHEMA`) carries no complex
coefficient, so paths that land in the same (tx, rx, component) bin combine
INCOHERENTLY (their powers add).

The deterministic solver already supports a coherent combine (`coherent: bool`
on its config): it sums the complex per-path field `PathFields.path_field` per
(tx, rx, component) and finalizes `|sum|^2`, so paths within a component
interfere. On the WS1 wedge fixture the two domains diverge measurably:

- deterministic incoherent diffraction (point rx): `2.1433029573358908e-08`
- deterministic coherent diffraction (point rx): `1.7121797313279785e-08`

BDPT had no coherent option. This ADR adds one as a NEW, switchable capability
without touching the existing power-domain default. The WS1 audit
(`artifacts/ws1-alignment/`) established that BDPT's standalone diffraction,
reflection, and coupled classes route through the shared enumerated engine as
unit-mass discrete connections (ADR-008/ADR-018); those enumerated rows already
carry the complex field `path_field` that the deterministic coherent
accumulator sums. On the wedge fixture the enumerated diffraction produces four
rows into the single (tx, rx) bin, with `|path_field|^2` matching each row's
`path_gain` to seven digits, so summing the complex field reproduces the
deterministic coherent value exactly.

The existing `accumulation_strategy` axis (int64 0/1/2: add / cell_reduce /
compact_atomic_add) is a PERFORMANCE choice for the power-domain reduction. It
is orthogonal to the combine algebra and is left fully untouched.

## Decision

Add `coherent: bool = False` to the `montecarlo.bdpt` config. DEFAULT OFF is
BIT-IDENTICAL to today's power-domain incoherent accumulation. When enabled,
the solve routes the coherent-eligible components through the shared enumerated
engine and accumulates their complex projected field coefficient as a phasor
per (tx, rx, component), finalizing `|sum|^2`.

### Native accumulate op: defaulted argument, not a sibling op or schema change

The coherent combine is implemented as a new defaulted `combine_domain`
argument (plus row-aligned `coeff_real` / `coeff_imag` tensors) on the EXISTING
`bdpt_accumulate_connection_samples` op, rather than a sibling op or a widened
connection-sample schema. Rationale:

1. It is the SAME reduction primitive over the SAME (tx, rx, component) bins
   with the same tape lifetime and launch structure, differing only in the
   combine algebra. Splitting it into a sibling op would duplicate the dispatch
   and validation and add an unowned ABI symbol; a defaulted argument keeps
   single ownership (`bdpt.connection_storage`) and one Python facade, and the
   native binding count stays 193.
2. The 12-field `_BDPT_CONNECTION_SCHEMA` and the field-hardcoded
   concat/compact/filter/copy/count kernels are frozen ABI. Widening the schema
   with `coeff_re`/`coeff_im` would touch every producer, every carry op, and
   every existing suite that constructs a connection dict, which conflicts with
   "default OFF, existing suites untouched, bit-identical." Instead the complex
   coefficient rides SEPARATE `coeff_real`/`coeff_imag` arguments to the
   accumulate op, sourced Python-side from the enumerated `path_field`. The
   coherent path concatenates those coefficients in the identical block order
   the native concat uses, so the coefficient rows align 1:1 with the samples.
3. `combine_domain == 0` (power, the default) is bit-identical: the coherent
   branch is never entered, the coefficient tensors are empty and untouched,
   and the existing power-domain kernels are byte-for-byte unchanged.

The coherent accumulator uses two new device kernels in
the `bdpt_connect_accumulation.cu` provenance section of
`bdpt_connect.cu`:
`bdpt_accumulate_connection_samples_coherent_kernel` (atomic-double complex sum
per bin) and `bdpt_finalize_coherent_accumulation_kernel` (`|sum|^2` per
component, plus `path_gain = sum` of the coherent component powers). Because the
coherent phasor sum is inherently an atomic complex reduction, it always uses
the atomic-double path regardless of `accumulation_strategy`; the power-domain
strategy code paths are not entered and stay orthogonal.

### Scope of coherent-eligible components

Only the enumerable delta/UTD family carries a complex field, so coherent is
scoped to `components subset of {los, reflection, diffraction}` (coupled folds
into the diffraction bucket when `coupled_paths=True`). BDPT's transmission and
scattering estimators are STOCHASTIC Monte Carlo samplers with no coherent
field; summing their per-sample power draws as phasors would be an invalid
estimator, so `coherent=True` with `transmission` or `scattering` in
`components` is REFUSED loudly in config validation.

Under `coherent=True` the whole solve routes through
`_collect_coherent_connection_samples`, which enumerates los / reflection /
diffraction (and the coupled compensator) as unit-mass discrete blocks carrying
`path_field`, concatenates them through the unchanged native connection concat,
concatenates the field coefficients in the same order, and accumulates with
`combine_domain='coherent'`. Paths within one component combine coherently;
components combine incoherently into `path_gain`, matching the per-component
coherent power the deterministic reference exposes and the acceptance gate
compares against.

### AD stance

The coherent path refuses `ad_mode != 'none'` loudly (ADR-017 precedent). BDPT
ships no AD in this release, so `ad_mode` is already forced to `'none'`; the
coherent-specific guard is explicit and documented for the record. Production AD
for the coherent combine (native forward/JVP/VJP companions) is a future ADR;
finite-difference or Torch-autograd reconstruction of the reduction is
forbidden.

## MIS implications

The coherent-eligible connections are unit-mass enumerated discrete paths
(`pdf = 1`, `mis_weight = 1`), exactly as reflection/diffraction/coupled are in
the power domain (ADR-018). The coherent accumulator sums the field with UNIT
weight, so the coherent estimate is MIS-invariant by construction:
`mis="none"`, `mis="balance"`, and `mis="power_heuristic"` yield the identical
coherent component power (verified bitwise on the wedge fixture).

## Acceptance gates

Measured on the WS1 wedge fixture (`tests/support/scenes.py::wedge_diffraction_scene`)
with the fresh `artifacts/cmake-p1` extension; guarded by
`tests/montecarlo/bdpt/test_coherent_combine.py`.

- (a) DEFAULT OFF bit-identical. BDPT `coherent=False` diffraction reproduces
  the deterministic incoherent value `2.1433029573358908e-08` bit-exactly
  (point rx) and is bit-for-bit reproducible across runs. The full existing
  BDPT suite stays green (only the accumulate facade contract test is updated
  for the new signature, and the coherent coverage added; no behavior weakened).
- (b) Coherent ON converges to the deterministic coherent component power. BDPT
  `coherent=True` diffraction is `1.7121797313279785e-08` versus the
  deterministic coherent reference `1.7121797313279785e-08`: measured ratio
  `1.00`, well inside the `[0.5x, 2x]` gate (ADR-018 gate parity). It is
  strictly below the incoherent estimate on this fixture (the four diffraction
  rows partially cancel), proving the phasor sum is active.
- (c) The three MIS modes stay consistent under coherent (bitwise equal).
- (d) Governance complete: config + validation, `coherent_combine` capability,
  intentional public-api snapshot update (BDPT `Config` contract sha256),
  migration note, binding manifest signature refresh (count unchanged at 193),
  owner-inventory body-hash refresh + digest, direct facade contract coverage,
  negative refusal tests, and metadata combine-domain record.

## Governance

- Public API: `ci/public-api-snapshot.json` `montecarlo.bdpt.Config`
  `contract_sha256` regenerated intentionally (the `coherent` field is a new
  public config attribute). Export count unchanged (37).
- Native binding: `bdpt_accumulate_connection_samples` gains `combine_domain`
  (default 0) and `coeff_real`/`coeff_imag` (default empty) arguments. No new
  ABI symbol; `EXPECTED_NATIVE_BINDING_COUNT` stays 193 and
  `ci/native-binding-manifest.json` is regenerated for the signature.
- Owner inventory: `docs/dev/audit/phase9-native-owner-inventory.json`
  `bdpt.connection_storage` multiset refreshes the
  `channel_bdpt_accumulate_connection_samples_cuda` body hash and adds the two new
  coherent kernels; `manifest_sha256` recomputed. `abi_owner` set unchanged.
- Contract coverage: unchanged (owner, E2E callers, and count for the
  accumulate op are all the same).
- No AGENTS.md/CLAUDE.md guardrail change: this feature lives entirely within
  the accepted architecture (native hot-path reduction, single owner, no
  solver-to-solver import, no fallback).

## Revisit condition

Re-evaluate when: coherent combine needs AD companions (new ADR); BDPT gains a
coherent estimator for transmission/scattering (would require a coherent field
carrier for those samplers); or cross-component coherent `path_gain` (full
field interference across component buckets) is required, which would change the
finalize from per-component `|sum|^2` to a single all-component field total.
