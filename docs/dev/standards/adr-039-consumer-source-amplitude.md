# ADR-039: The propagation consumer publishes the declared source amplitude

- **Status:** Accepted
- **Date:** 2026-07-25
- **Kind:** Semantic change to a published quantity, plus one new native
  operation family. No existing kernel, launch, fusion boundary, reduction
  order, or solver result changes.
- **Related:** ADR-034 (supersedes its "for unit source amplitude" wording for
  the published scalar and complex3 transport), ADR-036, ADR-037, ADR-038

## Provenance

The survey of this defect hit a STOP condition: the power-free publication was
the convention ADR-034 had accepted, which made the change an owner decision
rather than a defect fix. This ADR was written and implemented after that STOP,
on the direction of the maintainer who owns the convention. Because it changes
a published quantity, it takes effect only when `claude/fix-powers-w` is merged
by that maintainer; the merge is the record of the decision, and no separate
approval file is kept.

## Context

`EndpointBatch.powers_w` is required for a source batch. Before this decision
it could not change any value the consumer published.

The field transport kernels emit two families on one launch:

```text
field_vector = carrier * tx_axis                 <- unit excitation
coefficient  = carrier * projection              <- unit excitation
path_field   = coefficient * sqrt(max(P, 0))     <- excited
path_gain    = |path_field|^2                    <- excited
```

`ScalarTransport.coefficient` published `coefficient` and
`Complex3Transport.field` published `field_vector`, so a caller that declared
`powers_w = 4.0` received exactly the same numbers as one that declared
`1.0`. Deterministic and both Monte Carlo solvers already published the
excited side; the Path solver publishes the unit-excitation side and says so
in its `coefficient_semantics` metadata.

A required input whose value provably cannot reach any output is the
"declared but unenforced" pattern ADR-036 rejects, and `service.py` already
calls it out by name for the polarimetric frozen inputs. Two responses on one
contract were also silently disagreeing with the amplitude string the package
publishes in its own phase convention,
`sqrt(tx_power)*wavelength/(4*pi*distance)`.

## Decision

The propagation consumer publishes the **excited** transport.

| Surface | Published quantity | Carries `sqrt(powers_w)` |
|---|---|---|
| `ScalarTransport.coefficient` | `path_field` | yes |
| `Complex3Transport.field` | `path_field_vector` | yes |
| `JonesTransport.matrix` | native/composed basis operator | no |

`JonesTransport` is unchanged. A `2 x 2` polarization-basis map is not a
transported field, the fused native LoS Jones owner takes no power input at
all, and the composed route is held bit-identical to it by ADR-037 section 6.
A caller that wants a powered response applies the amplitude to its own
source-basis excitation.

Nothing outside the consumer moves. `PathFields.coefficient` and
`PathFields.field_xyz` stay unit-excitation because Path, Deterministic and
BDPT share them, and `deterministic.PathTable` publishes both conventions
side by side; redefining them would double-count power for those readers.

## Native ownership

`ScalarTransport` needs no new compute: `path_field` is already produced on
the same launch, is already covered by every registered backward/JVP
companion, and adds zero saved tensors and zero launches.

`Complex3Transport` had no excited vector anywhere in the ABI. Applying
`sqrt(P)` in Torch is hot-path physics and is forbidden, so this ADR adds the
native owner of exactly that quantity:

```text
field_source_amplitude_scale(field_vector, tx_power)
    -> path_field_vector = field_vector * sqrt(max(tx_power, 0))
field_source_amplitude_scale_backward(tx_power, grad_path_field_vector)
field_source_amplitude_scale_jvp(tx_power, tangent_field_vector)
```

in `native/channel/kernels/fields.cu`, with the Python owner
`witwin.channel.kernels.fields`, which holds both the three native facades and
the differentiable wrapper.

Two properties make this a contained addition rather than a second physics
owner:

- The amplitude expression is the identical `sqrtf(fmaxf(tx_power, 0))` the
  transport kernels use, so the receiver projection of `path_field_vector` is
  the same quantity as `path_field`. The two are not bit-identical: this owner
  scales before the projection and the transport kernel scales after it, so
  they differ by float rounding order, which is why the consumer test that
  compares them uses the default tolerance rather than `rtol=0, atol=0`.
- The map is linear in the field vector and its amplitude is real, so the VJP
  and the JVP are the same scale and need no primal field input, no saved
  field tensor, and no reduction.

That shared amplitude expression is an ADR-004 lockstep duplicate. It is one
expression, well under the 100-token duplication gate, so it is recorded by
hand as `adr039_lockstep_duplicates` in
`docs/dev/audit/duplication-classification.json` together with the two tests
that hold the two sites together.

It costs one elementwise launch, and only on a `complex3_transport` request.
The alternative - threading a new `path_field_vector` output through the
free-space, reflection, wedge, coupled, transmission and deterministic
kernels and their backward/JVP companions - would have shifted the positional
output indices every AD wrapper depends on, for the same numbers.

`tx_power` remains a frozen primal. It has no native derivative in any field
companion, both new wrappers reject a gradient or tangent on it by name, and
the consumer preflight refusals in `service.py` are unchanged.

## Consequences

- `CONTRACT_VERSION` moves from 2 to 3.
- `EndpointBatch.powers_w` becomes load-bearing on the scalar and complex3
  responses. It stays inert for `polarimetric_transport` by the decision
  above, which the contract docstring now states.
- Every Channel test that exercised the consumer used `powers_w = 1.0`, so
  the numerical change is invisible to all of them. Seven existing test files
  changed, plus the two new ones: two files carry the `CONTRACT_VERSION` bump,
  two unit tests read `path_field` from the route or its fake native result
  dict, four governance files carry the binding-universe count of the three new
  symbols, and one test restores the modules it pops from `sys.modules`. The
  evidence record lists them individually.
- Radar's `test_channel_does_not_apply_the_declared_transmit_power` pins the
  old convention and will flip. Radar's own
  `batch.weight_includes_tx_power is True` is the belief that contradicted it,
  so `EFFECTIVE_TRANSMIT_POWER_W` and the bistatic-radar-equation composition
  test are the radar-side follow-up, in a separate change.
- Path solver metadata now quotes `UNIT_EXCITATION_PHASE_CONVENTION`. Its
  `free_space_amplitude` no longer states `sqrt(tx_power)*...` next to
  `coefficient_semantics = "unit_excitation..."`, which were two different
  amplitudes for one number.

## Alternatives rejected

- **Keep unit excitation and reject `powers_w`.** Numerically free and
  conformant with the original ADR-034 wording, but it moves the amplitude
  into every caller and leaves `ScalarTransport` disagreeing with the
  Deterministic and Monte Carlo results about what a published field is.
- **Scale only the scalar response.** One Python selection, no rebuild, but
  two responses on one contract would then disagree about power.
- **Multiply by `sqrt(P)` in Torch.** Forbidden hot-path physics.
