# ADR-039 evidence: the consumer now publishes the declared source amplitude

Accepted change record for `docs/dev/standards/adr-039-consumer-source-amplitude.md`.
Environment: `witwin2`, CUDA, developer-override extension built with
standalone CMake (Ninja, Release, `CMAKE_CUDA_ARCHITECTURES=120`).

## 1. Defect

`EndpointBatch.powers_w` is required for a source batch
(`witwin/channel/propagation/consumer.py`), reaches
`EnumeratedEndpointTensors.tx_power` and then every field kernel, and could
not change any value the consumer published. Measured before the change
(source at P=1 W versus P=4 W, identical geometry, depths 0 and 1):

```text
discovery |coefficient| ratio  : [1.0, 1.0]   expected 2.0
reevaluate |coefficient| ratio : [1.0, 1.0]   expected 2.0
```

## 2. Root cause

Not a selection bug and not a regression. The field transport kernels publish
both families on one launch
(`native/channel/kernels/field_transport.cu`):

```text
field_vector = carrier * tx_axis              unit excitation
coefficient  = carrier * projection           unit excitation
path_field   = coefficient * sqrt(max(P, 0))  excited
path_gain    = |path_field|^2                 excited
```

The consumer selected the unit-excitation pair, deliberately: the same commit
that introduced the selection (`88f8a35`) also wrote the "for unit source
amplitude" sentence into ADR-034 and the contract doc, and
`witwin/channel/propagation/fields.py` divides `sqrt(tx_power)` back out of the
already-powered diffraction field to keep `PathFields.coefficient`
unit-excitation. The convention was real; the defect was that a required
input had no effect and that two published surfaces on one contract disagreed
with the Deterministic and Monte Carlo results about what a published field
is.

## 3. Change

- `ScalarTransport.coefficient` publishes the native `path_field`. No new
  compute, no new saved tensor, no new launch.
- `Complex3Transport.field` publishes a new native output
  `path_field_vector = field_vector * sqrt(max(tx_power, 0))` from
  `native/channel/kernels/field_transport.cu`, with backward and JVP
  companions. One elementwise launch, only on a `complex3_transport` request.
- `JonesTransport`, `PathFields`, `PathResult`, `deterministic.Result`,
  `deterministic.PathTable`, `montecarlo.basic.Result` and the BDPT
  contributions are unchanged.
- `CONTRACT_VERSION` 2 -> 3; ADR-034 and the contract doc restate the
  convention; the Path solver metadata quotes
  `UNIT_EXCITATION_PHASE_CONVENTION` instead of contradicting its own
  `coefficient_semantics`.

## 4. Numbers

After the change, same probe:

```text
discovery |coefficient| ratio  : [2.0, 2.0]   expected 2.0
reevaluate |coefficient| ratio : [2.0, 2.0]   expected 2.0
reeval/discovery at P=1        : [1.0, 1.0]   expected 1.0
```

Surfaces that must not move, P=1 versus P=4, measured after the change:

| Surface | ratio | expected |
|---|---|---|
| `PathResult.a` | 1.0000 | 1.0 |
| `PathResult.field_xyz` | 1.0000 | 1.0 |
| `deterministic.Result.field` | 2.0000 | 2.0 |
| `deterministic.Result.path_gain` | 4.0000 | 4.0 |
| `deterministic.PathTable.coefficient` | 1.0000 | 1.0 |
| `deterministic.PathTable.field_real/imag` | 2.0000 | 2.0 |
| `montecarlo.basic.Result.path_gain` | 4.0000 | 4.0 |
| `JonesTransport.matrix` | bit-identical | excitation-free |

The native amplitude is exact for the powers used above
(`sqrt(4) = 2`, `sqrt(9) = 3` are exact in float32), so the direct contract
tests compare at `rtol=0, atol=0`.

## 5. Blast radius

- Numerically zero for every existing Channel test: all of them build
  `powers_w = torch.ones(...)`, and `sqrt(1) == 1`.
- Seven existing Channel test files changed, each for a stated reason, and two
  new files were added (section 6). Four of the changes are semantic:
  - `tests/propagation/consumer/test_public_contracts.py` (x2 assertions):
    `CONTRACT_VERSION` 2 -> 3.
  - `tests/test_stage1_phase3_governance.py`: the same version bump, with the
    reason recorded in the comment.
  - `tests/propagation/consumer/test_service_contract.py`
    `test_evaluate_consumes_request_batch_and_aliases_finalized_rows`: the
    published scalar transport is now aliased to `fields.path_field`.
  - `tests/propagation/consumer/test_service_contract.py`
    `test_reevaluate_reuses_frozen_topology_without_discovery`: the fake
    native result dict now carries `path_field`, which is what the route
    reads.

  Four are governance counts that follow the three new native symbols
  (binding universe 250 -> 253, `live_count_delta_from_plan13_baseline`
  39 -> 42): `tests/test_phase13_adr027_mc_activation.py`,
  `tests/test_phase13_adr027_penetration_activation.py`,
  `tests/test_phase13_adr027_penetration_governance.py`,
  `tests/test_phase13_phase11a_governance.py`.

  One is unrelated hygiene:
  `tests/propagation/consumer/test_public_contracts.py`
  `test_consumer_import_is_solver_neutral` now restores the consumer modules
  it pops from `sys.modules`. That is a pre-existing pollution bug it exposed,
  fixed in its own commit with its assertions unchanged.

  No tolerance was weakened.
- Radar (read-only here) `test_channel_does_not_apply_the_declared_transmit_power`
  pins the old convention and flips; `EFFECTIVE_TRANSMIT_POWER_W` and
  `test_the_composed_coefficient_is_the_bistatic_radar_equation` follow it.
  That is a separate radar-side change.

## 6. New coverage

- `tests/kernels/test_field_source_amplitude_scale.py` (9 tests): direct
  contract for the three native symbols, the exact amplitude, inert
  zero/negative-power rows, row-count refusal, autograd VJP/JVP agreement
  with the native companions, and the `tx_power` derivative refusal.
- `tests/propagation/consumer/test_source_amplitude.py` (21 tests):
  discovery and both reevaluation routes scale by `sqrt(powers_w)` per row
  from each row's own source, a distinct-power multi-source batch scales each
  row independently, a prepared topology carrying both LoS and reflection rows
  scales each of those rows by its own source (two sources at 4 W and 9 W, so a
  single global amplitude fails the test), complex3 projects onto the scalar
  coefficient, the Jones operator stays bit-identical under a power change, all
  three AD modes agree on the fixed primal, VJP gradients and JVP tangents
  carry the amplitude as a constant factor, and the Path/Deterministic surfaces
  keep their own conventions.

## 7. Validation

Targeted set `tests/propagation/consumer tests/kernels tests/ad tests/path
tests/deterministic`:

| | result |
|---|---|
| before | 902 passed, 7 skipped, 1 xfailed (163.70s) |
| after | 930 passed, 7 skipped, 1 xfailed (106.00s) |
| after the audit hardening (+2 tests) | 932 passed, 7 skipped, 1 xfailed (125.73s) |

Whole suite less `tests/performance` and the three Munich parity files:

| | result |
|---|---|
| before | 2401 passed, 6 skipped, 1 xfailed |
| after | 2429 passed, 6 skipped, 1 xfailed |

The `+28` is exactly the new coverage in section 6; no existing test changed
its outcome.

`python ci/run_ci_tier.py quick`: passing before and after.
`python ci/run_ci_tier.py cuda`: passing before and after, run because the
native surface and the AD dispatch changed. After the change its gates report
`2126 + 14 + 26 + 117` passed and `contract coverage passed (52 public
exports, 253 native bindings)`.

Rerun after the audit hardening, on the same environment: quick tier passes
(exit 0), cuda tier passes (exit 0) with
`2128 passed, 4 skipped` on the unit-contract-acceptance gate and
`14 + 26 + 117` on the solver-smoke, no-fallback and AD gates. The `+2` on the
acceptance gate is the new reflection multi-source case.

One intermittent, pre-existing artifact seen once while gathering these
numbers: three `tests/deterministic/test_transmission.py` tests failed in one
of three otherwise identical targeted runs and passed on repeat and in
isolation. It is not order-dependent (no random ordering plugin is installed)
and no run of the cuda tier reproduced it.

Pass `--basetemp` explicitly on Windows. Without it, and with a long temp
path, `tests/kernels/test_rayd_package_discovery.py` fails nine times on
`Filename too long` while cloning the locked RayD checkout. That is an
environment artifact, unrelated to this change, and reproduces on the
unmodified base commit.

## 8. Extension fingerprint

The native change requires a rebuilt extension. Built with standalone CMake
(not pip):

| | fingerprint |
|---|---|
| before | `3b6ddd1028e6e269701b459db1a56acf37d35c26136812eebaf48189dbdeaf4f` |
| after | `c2fde2675a43de52166a06b09f6dfa67603799783049a3725401ef65babd19e2` |

Any environment that pins `WITWIN_CHANNEL_EXPECTED_FINGERPRINT` must be
updated together with this change, and a released wheel must be rebuilt.
