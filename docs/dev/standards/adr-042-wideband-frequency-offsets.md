# ADR-042: Wideband frequency offsets on the fixed-topology consumer

- **Status:** Accepted
- **Date:** 2026-07-26
- **Kind:** Additive contract change. One new request field with a safe
  default, two new paired optional fields on each of two response types, seven
  new capability fields, one new module function, two new convention strings,
  one new diagnostics field, three new scene-dependent refusals. No native
  code, no ABI symbol, no kernel, no new production Torch physics, no change to
  reduction order, no change to any published number when the field is absent.
  The launch count per bucket DOES change, and this ADR states the new law out
  loud rather than claiming otherwise.
- **Related:** ADR-026 (resident scattering tables and their frequency-hashed
  lifetime), ADR-032 (the one-copy/one-synchronization budget this exists to
  preserve), ADR-036 (a declared limit must be discoverable, not learned from a
  rejection), ADR-037 (`row_valid` is the sole authority and is a mask over
  rows), ADR-038 (wrapper-level forward-AD liveness), ADR-039 (the source
  amplitude is applied exactly once, by the native owner), ADR-040 (frequency
  is deliberately not a world version domain, which is what makes replay across
  frequency legal), ADR-041 (slot batching, the axis this one composes with)
- **Closes:** Phase-8 work item 1, the Channel half

## Context

A `CompiledScene` is built at one `reference_frequency_hz`, and
`CompiledScene.require_reference_frequency` refuses a request that does not
match it by exact hex equality. Every published coefficient is therefore at one
frequency, and version 4 gave a caller who wanted another one a law instead of
an evaluation:

```text
narrowband_frequency_offset_law = "H(f_ref+df) = C(f_ref)*exp(-j*2*pi*df*delay_s)"
```

### The law is quantitatively wrong, and not because of dispersion

Three terms separate `H(f_ref + df)` from what that law returns.

1. **Spreading.** The free-space amplitude is `lambda/(4*pi*d)`, so the true
   magnitude ratio is exactly `f_ref/(f_ref+df)` with zero phase. The law drops
   it. Closed form, always known.
2. **Material selectivity.** A compiled wall is a layer stack, and a layer
   stack fringes. The fixture's 0.1 m `eps_r = 4` slab fringes every 755 MHz at
   the measured incidence. Measured on the default wall at 77 GHz with
   `df = 1 MHz` - a fractional offset of `1.3e-5` - the law is already off by
   **0.63% in magnitude and 15.0 mrad in phase**
   (`test_the_narrowband_law_is_measurably_wrong_at_one_megahertz`). Across a
   2.4 GHz sweep at 3 GHz the law is off by a factor of **10.5** at its worst,
   because it holds the wall reflectivity fixed across three nulls
   (`test_multilayer_sweep_crosses_fringes_and_falsifies_the_narrowband_law`).
3. **Dispersion.** `d(eps_r)/df` from a `witwin.core` `DispersionSpec`. The law
   is zeroth-order in it, and so is the compiled record.

The capability statement, published verbatim on the convention and in
`consumer/README.md`: *the narrowband law is exact to `O(df/f_ref)` in
spreading and `O(df/df_fringe)` in material response, and is zeroth-order in
dispersion. The wideband route removes terms 1 and 2 exactly and refuses
term 3.*

### The material record already knows how to do this

Compiling the same scene at 77 GHz and at 78 GHz produces `MaterialStore`s
whose `eps_r`, `mu_r`, `sigma_e`, `gain`, `thickness_m`, all four `layer_*`
columns, and every roughness, scattering, and XPD column are `torch.equal`.
Only a material carrying a `DispersionSpec` differs. Everything that genuinely
varies with frequency - the conductivity loss tangent
`eps_c = eps_r - j*sigma_e/(omega*eps0)`, the wavenumber `k(omega)`, the layer
electrical thicknesses, the entire Airy/Rouard recursion, the `lambda/(4*pi*d)`
amplitude, the `exp(-j*k*d)` carrier phase - is re-derived natively from the
frequency the launch receives.

So a per-offset response for a non-dispersive material needs no recompile, no
host material replay, and no new native symbol. Only the scalar `frequency_hz`
argument changes.

### Compiled frequency-dependent material record: primal and AD capability

| Material feature | Where evaluated | Primal at an offset | AD w.r.t. frequency |
|---|---|---|---|
| `eps_r`, `sigma_e` (homogeneous lossy) | native, at the passed omega | **supported** | supported (existing native companion) |
| perfect conductor | native | supported | supported (trivially) |
| multilayer stack (Airy / Rouard) | native, full recursion at the passed omega | **supported** | supported |
| surface roughness (Kirchhoff table) | native kernel, resident table keyed on a frequency-hashed material cache token | **refused (W4)** | refused |
| phase screen | same | **refused (W4)** | refused |
| `DispersionSpec` (`PowerLawDispersion`, Debye, ...) | Core `evaluate_at_frequency`, **frozen at compile** | **refused (W1)** | already refused |

One line: *the only thing a compiled material record freezes as a function of
frequency is the `DispersionSpec` evaluation; everything else is already a
native function of the frequency argument.*

## Decision

### 1. `frequency_offsets_hz`, a host-declared propagation-frequency grid

`FixedTopologyRequest` gains

```python
frequency_offsets_hz: tuple[float, ...] | None = None
```

Declaring `(df_0, ..., df_{F-1})` states that the caller wants the same frozen
rows evaluated at the `F` absolute frequencies `reference_frequency_hz + df_j`,
in the declared order. `None`, the default, is exactly the single-frequency
behaviour, bit for bit, with `F` absent rather than treated as `1`.

It is a **propagation-frequency grid and nothing else.** It names frequencies
at which a field is evaluated. It never accepts a subcarrier count, an FFT
size, a bandwidth, or a sample count, and no field with such a meaning may ride
in with it. `test_the_grid_is_a_propagation_frequency_grid_and_nothing_else`
asserts that over every dataclass on the boundary, because this is the field
through which Radar waveform policy is most likely to erode the Channel
contract.

It is a tuple of host floats, not a tensor, because it is a declaration in the
same class as `slot_count`: structurally non-differentiable, structurally
host-known (which is what makes the `F` loop legal), and float64-exact. A
tensor is refused with a `TypeError` that names the reason - a tangent with
respect to one grid point IS the `reference_frequency_hz` tangent evaluated at
that point, so the caller seeds `reference_frequency_hz`.

Structural refusals in `__post_init__`, before any native work: a tensor, an
empty tuple, a non-finite entry, a duplicate entry (which would produce
bit-identical columns and hide a caller bug), and `polarimetric_transport`
(naming `capabilities().wideband_responses`).

`PropagationRequest` (discovery) does **not** get the field. Discovery is
frequency-independent in practice and the inner loop a wideband caller runs is
fixed-topology. The consequence is a named deferral: wideband `transmission`
and `diffraction` are out of scope, because they are not freezable components.

### 2. The payload: additive, optional, paired with its grid

```python
ScalarTransport:
    coefficient: torch.Tensor                        # [K] complex64, unchanged
    coefficient_offsets: torch.Tensor | None = None  # [K, F] complex64
    frequency_offsets_hz: tuple[float, ...] | None = None

Complex3Transport:
    field: torch.Tensor                              # [K, 3] complex64, unchanged
    direction: torch.Tensor                          # [K, 3], geometry, unchanged
    field_offsets: torch.Tensor | None = None        # [K, F, 3] complex64
    frequency_offsets_hz: tuple[float, ...] | None = None
```

`JonesTransport` gains nothing. It is line-of-sight only and power-free, and
widening it is a separate decision rather than a side effect of this one; the
ADR-039 asymmetry is not "fixed" here either.

Seven laws, all published as data and all under test:

1. **Column law.** `coefficient_offsets[:, j]` is the response at
   `reference_frequency_hz + frequency_offsets_hz[j]`.
2. **Reference identity.** A `0.0` entry produces a column **bit-identical** to
   `coefficient` - `torch.equal`, not a tolerance. It re-launches at the same
   float32 frequency with the same inputs, so anything less would mean the
   wideband route is a different evaluation wearing the same name.
3. **Paired presence.** Payload and grid are both present or both absent, and
   `F == len(frequency_offsets_hz)`.
4. **Shared rows.** `row_valid` stays `[K]` and broadcasts over `F`. Row
   validity is a geometric fact - whether the stationary point exists at these
   endpoints - and cannot depend on frequency. Widening it to `[K, F]` is
   forbidden by ADR-037 and by physics.
5. **Geometry published once.** `path_length_m`, `delay_s`, `field_direction`,
   `interaction_positions_m`, and `interaction_normals` stay `[K]`-shaped and
   come from the reference evaluation. Offset columns recompute them natively
   and discard them; a test asserts they are `torch.equal`.
6. **Slot composition.** With `slot_count = T` the payload is
   `[T*K_frozen, F]`. Frequency is orthogonal to the ADR-041 slot law and does
   not touch `slot_pair_layout`. `replicate_over_slots` gets no frequency
   variant, because there is nothing to tile: the same rows are evaluated at
   `F` frequencies.
7. **Amplitude.** `sqrt(powers_w)` is applied by the native owner once per
   column, which is `F` scalings of `F` different columns and is correct
   (ADR-039). It is never lifted into a Torch `[K, F]` broadcast.

### 3. The service loop, and what holds the budget

`prepared_row_gather` (and, on the raw route, `fixed_los_gather`) runs **once,
above the column loop.** It is the sole owner of the validation D2H copy and
the synchronization, and running it once is exactly what holds ADR-032 at one
copy and one synchronization regardless of `F`.

The reference column runs exactly as before and produces the `[K]` payload, the
geometry, and `row_valid`. Each offset column calls the same
`evaluate_prepared` with `frequency_value = compiled.materials.frequency_hz +
df_j` and `frequency = reference_frequency_hz + df_j`, preserving the tensor
identity path so an AD seed reaches every column. The `[K, F]` payload is
`torch.stack` over the column outputs: **structural packing only.** No
offset-dependent phase, magnitude, or basis is applied in Torch anywhere.

**ADR-038 liveness is computed once, above the loop, and the same explicit flag
is passed to every column.** `field_free_space_ad` and
`field_reflection_sequence_ad` gained an optional `geometry_live` parameter for
this; when it is omitted they decide for themselves, exactly as before. The
consumer computes a two-flag `GeometryLiveness` record from the inputs every
column shares - the gathered endpoints for a zero-depth bucket, and those plus
the scene vertex table for a reflection bucket, since a differentiable mesh
makes the stationary point live even when the endpoints are primal. Recomputing
liveness inside the loop, or letting the first column decide for the rest, is
the exact shape of the defect ADR-038 removed, and it would show up as a
missing `delay_s` tangent on precisely the forward-only-dual requests Radar's
Doppler chain uses.

### 4. The launch-count law, stated out loud

ADR-041 could truthfully say "no change to the number of launches per bucket".
**This one cannot, and does not pretend otherwise.**

```text
launches       = (1 + F) * buckets * launches_per_bucket
D2H copies     = 1
synchronizations = 1
host reads     = 1
```

The leading `1` is the reference column, which also produces the shared
geometry and `row_valid`. Measured on the multi-endpoint fixture
(`buckets = 2`), by counting real field-operator invocations:

| `F` | measured launches | `(1+F)*buckets` | copies | syncs | host reads | ms | ms/column |
|---|---|---|---|---|---|---|---|
| absent | 2 | 2 | 1 | 1 | 1 | 1.604 | 1.6041 |
| 1 | 4 | 4 | 1 | 1 | 1 | 2.714 | 2.7143 |
| 8 | 18 | 18 | 1 | 1 | 1 | 9.830 | 1.2288 |
| 64 | 130 | 130 | 1 | 1 | 1 | 63.158 | 0.9868 |
| 256 | 514 | 514 | 1 | 1 | 1 | 265.523 | 1.0372 |

`test_the_launch_count_follows_the_published_law` asserts the equality rather
than describing it.

A follow-up transport-only column path would remove the redundant per-column
geometry work. It needs its own measurement and is out of scope here.

### 5. Three independent scene-dependent refusals

Each is enforced by its own function in `_preflight_wideband`, before any
native work, and each is reachable by a case that trips only that one. Folding
any of them into another would make one of the three limits undiscoverable,
which is the ADR-036 pattern.

- **W1, dispersive primal.** A non-`None` grid on a scene with a non-empty
  `compiled.materials.frequency_dependent` raises `NotImplementedError`, **at
  every AD mode including `"none"`.** The existing gate refuses a frequency
  *gradient* against a frozen record; the primal at an offset has the identical
  defect and was previously unreachable only because the compile-frequency
  mismatch rule forced a recompile. This is a required new refusal, not a
  restatement.
- **W2, resolution.** Every native field bridge takes a `double frequency_hz`
  and `static_cast<float>`s it at the launch, so two absolute frequencies
  inside one float32 ULP are the SAME launch and return bit-identical columns.
  A grid with a non-zero `|df_j|` below `native_frequency_resolution_hz(f_ref)`,
  or with two entries closer together than it, raises `ValueError` quoting the
  offending values and the resolution. At 77 GHz the resolution is 8192 Hz; at
  3 GHz, 256 Hz; at 1 GHz, 64 Hz.
- **W4, rough and phase-screen materials.** A non-`None` grid on a scene whose
  compiled materials carry roughness or a phase screen raises
  `NotImplementedError` naming the resident-table lifetime. The Kirchhoff
  tables and the phase-screen realization resources are keyed on a material
  cache token that hashes the compile frequency (ADR-026, Plan-13), so a table
  built at `f_ref` and used at `f_ref + df` is a frozen approximation of exactly
  the class W1 refuses.

W4 performs one reduced device read in the preflight - the same shape
`require_smooth_reflection_scene` already performs on the reflection route.
It is a refusal guard before any native work, not a hot-path transfer, and it
is not part of the per-call validation budget.

### 6. Capability, convention, constants, diagnostics

`PropagationCapabilities` gains `supports_wideband_offsets` (true),
`wideband_responses` (`{scalar_transport, complex3_transport}`),
`wideband_components` (`{los, reflection}`), `wideband_dispersive_materials`
(false), `wideband_rough_materials` (false), `max_frequency_offset_count`
(`None`: no contract bound, only launch time and device memory), and
`native_frequency_resolution_law`.

`consumer.native_frequency_resolution_hz(reference_frequency_hz)` returns one
float32 ULP at that frequency, so a caller computes the same number the refusal
quotes instead of rederiving it. It is the only new public export.

`PropagationConvention` gains `narrowband_frequency_offset_error_law`,
`wideband_offset_layout`, and `wideband_frequency_quantization_law`. The last
one publishes the resolution and the bound
`abs_phase_error_rad <= pi*resolution_hz*delay_s`. Channel does **not** compute
that bound: it needs `max(delay_s)`, which is a device reduction plus a host
read the ADR-032 budget does not have. The caller owns the budget check,
exactly as Radar already owns its declared aspect phase budget.

`constants.py`, the single owner of convention strings, gains
`NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW`. The existing law string is unchanged.

`PropagationDiagnostics` gains `frequency_column_count: int` (1 without a
grid), which is what makes the launch law auditable from a result.

`CONTRACT_VERSION` moves from 4 to 5.

## Evidence

All measurements on an SM120 device, CUDA-synchronized, `witwin2`.

**Reference identity is bitwise.** `torch.equal` on the `df == 0` column
against `coefficient`, and on `field_offsets[:, 0]` against `field`.

**Geometry is invariant.** `torch.equal` on `path_length_m`, `delay_s`,
`field_direction`, `interaction_positions_m`, and `interaction_normals` between
a wideband call spanning `+-400 MHz` and the single-frequency call.

**Line of sight matches the free-space closed form.** Magnitude ratio equals
`f_ref/(f_ref+df)` to `1e-6` relative, and
`arg H(f_ref+df) - arg H(f_ref) + 2*pi*df*tau` is zero to `1e-4` rad, over
`df in {-50, 0, +25, +50} MHz` at 1 GHz. Both are exact closed forms.

**Half-space Fresnel sweep.** A 0.3 m `sigma_e = 0.5` layer attenuates its own
round trip below `1e-10`, so the stack degenerates to the bare interface (the
test asserts that the stack and the interface agree to `1e-8` before using
either). Compared against `em_oracle.fresnel_interface` with
`medium_params(eps_r, sigma_e, mu_r, f)` at 17 offsets spanning `+-400 MHz`:
worst relative error on the complex coefficient **1.0e-5**, bound asserted at
`4e-5`. This isolates genuine frequency dependence from dispersion, because
`sigma_e` alone makes `eps_c(f)` frequency dependent with no `DispersionSpec`
anywhere.

**Multilayer slab sweep across three fringes.** 49 offsets spanning 2.4 GHz at
`f_ref = 3 GHz` on the 0.1 m `eps_r = 4` slab, against
`em_oracle.layer_stack_rt`: worst relative error **4.4e-5**, bound asserted at
`1e-4`. The analytic fringe period
`c/(2*Re(sqrt(eps_r))*d*cos(theta_t))` is 755 MHz at the measured incidence, so
the sweep crosses 3.2 fringes; the measured magnitude minima (after dividing
out the monotone spreading tilt) land in the SAME grid bins as the analytic
minima of `|r_TE(f)|`, indices `[9, 24, 40]`. **This is the fixture that
falsifies a narrowband implementation**: on the same grid the narrowband law is
off by a factor of 10.5 at its worst.

The `1e-4` bound is looser than the `5e-5` the design sketched, and the reason
is recorded rather than hidden: the worst case sits at a fringe null, where the
reference magnitude is small and a complex64 forward's relative error is
correspondingly large. The measured 4.4e-5 is reported next to the bound.

**Narrowband error bound.** At `df = 1 MHz` on the default wall at 77 GHz the
true-versus-law magnitude error is 6.3e-3 and the phase error is 1.50e-2 rad,
both asserted `> 5e-3`. This pins the claim that the law is quantitatively
wrong, so the wideband route has a measured reason to exist.

**AD at every column.** `tests/ad/test_wideband_frequency_ad.py` validates
`d/d(reference_frequency_hz)` at all four columns, on `{los, reflection}`, in
both `jvp` and `vjp`, against central differences taken by recompiling the scene
at `f +- h` and replaying the SAME frozen topology - legal precisely because
frequency is not a world version domain (ADR-040). The FD step and every offset
are quantized onto the float32 launch grid using
`native_frequency_resolution_hz`, so no difference is biased by a rounded
launch frequency.

The FD comparison is bounded at `1e-2` relative, because the FD oracle runs the
same float32 native forward and its measured noise floor is 5e-4 (line of
sight) and 2.2e-3 (reflection) at the best step tried. That is still five times
tighter than the suite-wide `REL_TOL_GENERAL` the existing single-frequency
frequency-AD test uses for the same quantity. The tight per-column claim is
carried instead by `test_forward_and_reverse_agree_column_by_column`, which
compares two independent native companions over the same launch at `1e-6` and
has no finite-difference noise in it at all. A fourth test asserts the columns
are not the same derivative, so the other three cannot pass on an
implementation that evaluated every column at the reference frequency.

**ADR-038 through the loop.** A forward-only dual on the sink positions - no
`requires_grad` anywhere - publishes a `delay_s` tangent equal to the
`path_length_m` tangent over `c` to `1e-6` relative, AND a complete `[K, F]`
payload tangent whose every entry is non-zero.

**Budget, flat in `F`.** `validation_d2h_copies == 1`,
`validation_d2h_bytes == 4`, `validation_sync_count == 1`,
`compact_count_d2h_copies == 0`, at `F in {1, 8, 64}`, with `pair_offsets` and
`pair_count` `torch.equal` to the single-frequency call. Measured host reads:
exactly 1, at every `F` up to 256.

**Cost, against the shape-B baseline.** Shape B is what a caller can do today
without this contract: one `CompiledScene` per offset and one `reevaluate` each.

| shape | `F` | total ms | ms/offset | host reads |
|---|---|---|---|---|
| B (per-offset compile + replay) | 8 | 13.325 | 1.666 | 8 |
| B | 64 | 157.465 | 2.460 | 64 |
| A (this ADR) | 8 | 9.328 | 1.166 | 1 |
| A | 64 | 69.125 | 1.080 | 1 |

2.3x faster per offset at `F = 64` and, more to the point, 64 host
synchronizations replaced by one.

**Slot composition.** At `slot_count = 4`, `F = 8`, the payload is
`[4*K_frozen, 8]`, the budget stays 1/1, and each slot's columns are
`torch.equal` to the same-slot single-slot result.

**Row validity.** With a wideband payload present, `row_valid` is `[K]`, and
every column of a `False` row is exactly zero - the zeros come out of the kernel
that owns the value, because the invalid row is made inert at the input
(ADR-037).

## Consequences

**The launch count grows linearly in `F`.** That is the price of the naive
per-offset shape and it is stated rather than buried. A caller sweeping a large
band pays `(1+F)` launches per bucket; replay is launch-bound, so wall time is
roughly linear in `F` at these row counts.

**Every offset column recomputes geometry it then discards.** The stationary
point re-solve and the free-space geometry are frequency-independent, so `F` of
the `(1+F)` geometry solves are redundant. Removing them means a transport-only
column entry point in the native field family, which is a native change with its
own AD family and its own ADR.

**Dispersive, rough, and phase-screen scenes are refused, not approximated.**
A wideband fixture on such a scene passes by refusing, with the quantified
explanation above. That is a complete answer under the "capability and
numerical differences explainable" criterion, and it is stated plainly rather
than presented as a numerical pass.

**Discovery has no frequency grid.** Wideband `transmission` and `diffraction`
are therefore out of scope, because `fixed_topology_components` is
`{los, reflection}` and those two are not freezable.

**`frequency_offsets_hz` does not make frequency a version domain.**
`WORLD_VERSION_DOMAINS` is untouched and `rediscovery_required` gains no
frequency term. `CompiledScene.require_reference_frequency` remains the
exact-hex frequency authority with its own failure mode; the grid is relative to
whatever that authority admitted.

## Alternatives rejected

- **Shape B: a caller loop over per-offset `CompiledScene`s.** Works today and
  needs no contract change, which is why it is the measured baseline. It costs
  `F` validation copies and `F` synchronizations - verbatim the anti-pattern
  ADR-041 exists to remove - and it rebuilds the Kirchhoff and phase-screen
  resources per offset because the material cache token hashes the frequency.
  Measured at 2.460 ms/offset and 64 host reads at `F = 64`, against 1.080
  ms/offset and 1 host read here.
- **Shape C: a native op taking an `[F]` offsets tensor.** The only shape that
  keeps the launch count flat, and the right *later* change. It needs a measured
  launch-bound profile, a new ABI symbol with a manifest entry, and
  primal+JVP+VJP in the same stage. Putting it in this change without that
  evidence would be a numerical and fusion-boundary change smuggled into a
  contract change.
- **The split envelope/delay form.** Publishing the envelope natively at the
  quantized offset and the delay phase exactly from `df` in host precision is
  strictly more accurate at large `tau`. It requires dividing out the reference
  phase, which is either a native kernel change (out of scope, see shape C) or a
  Torch multiply in `service.py`, which is production Torch physics and
  forbidden outright. The float32 grid is handled by the W2 refusal plus the
  published quantization law instead.
- **A tensor offset grid.** A per-grid-point tangent is identical to the
  `reference_frequency_hz` tangent at that point, so the tensor would carry no
  derivative the contract does not already support, while making the host-known
  `F` loop structurally illegal. Refused with a message that says so.
- **Widening `row_valid` to `[K, F]`.** Row validity is geometric. A frequency
  cannot make a stationary point stop existing.
- **A frequency variant of `replicate_over_slots` or `evaluate_time_varying`.**
  Slots tile rows; frequency does not. The same rows are evaluated at `F`
  frequencies, so there is nothing to replicate. A wideband time-varying CIR, if
  ever wanted, is `[T, K, F]` views over one replay from the same no-physics
  view layer.
- **Putting the grid on `PropagationRequest`.** Discovery is frequency
  independent in practice and would have to carry the same three refusals for a
  route whose cost is dominated by topology work. The deferral is recorded
  instead.
- **A wideband `JonesTransport`.** Line-of-sight only, power-free, and its
  fused primal route already refuses a tensor frequency. Keeping the wideband
  surface at two responses keeps the refusal set small and complete.
