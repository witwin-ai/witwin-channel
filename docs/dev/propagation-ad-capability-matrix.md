# Channel propagation-consumer AD capability matrix

Authoritative for every `witwin.channel.propagation.consumer` AD cell, as
accepted by ADR-043. The machine-readable half of the same statement is the
capability record: `capabilities().component_ad_modes`,
`.component_material_leaves`, `.differentiable_geometry_outputs`,
`.direction_differentiable_components`, `.primal_only_ad_inputs`,
`.supports_higher_order_ad`, and `.ad_accounting`. If this document and the
record disagree, the record is the source of truth and this document is stale;
`tests/propagation/consumer/test_ad_capability_matrix.py` fails when they part.

Solver-boundary AD (`path`, `deterministic`, `montecarlo.*`) is a different
surface and stays in `docs/dev/ad-capability-boundary.md`.

## 1. Target states

There are exactly four, and "silent" is not one of them. A cell with no
declaration is a defect, not a fifth state.

| State | Meaning | Required evidence |
|---|---|---|
| `SUP` | Supported. A nonzero derivative is published and it is correct. | A named test at the boundary that publishes it, validated against finite differences, an independent float64 oracle, an analytic closed form, or a jvp/vjp adjoint identity. |
| `ZERO` | Structurally zero. The leaf does not enter this physics; exact zero is the correct and complete answer. | A named test asserting an exact zero (no graph, or a bit-exact zero), plus the leaf's absence from the component's published leaf list. |
| `REF` | Refused. Fails loudly before any numerical work and before any result object exists. | A named test asserting the raise, its owner, and that no result was produced. |
| `DECL` | Declared non-differentiable output. The published tensor deliberately carries no graph and no tangent. | The capability record names the field and the route, a test pins the declaration against the behaviour, and this document states it. Legal for outputs only, never for inputs. |

`mechanism` is one of `native-companion`, `native-declared`,
`torch-orchestration`, `host-declaration`. `torch-orchestration` is legal only
outside the hot path (result assembly, refusal checks); it is never physics.

`validation` is one of `fd`, `oracle-f64`, `analytic`, `adjoint`,
`declaration`, `refusal`.

## 2. Matrix

### 2.1 Geometry outputs, per route

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| reevaluate/prepared | out:field_direction | vjp | SUP | native-companion | native/channel/kernels/field_transport.cu, native/channel/kernels/field_transport.cu | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_arrival_direction_carries_a_reverse_gradient_matching_fd | fd |
| reevaluate/prepared | out:field_direction | jvp | SUP | native-companion | native/channel/kernels/field_transport.cu, native/channel/kernels/field_transport.cu | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_arrival_direction_tangent_agrees_with_the_reverse_gradient | adjoint |
| kernel/free-space | out:field_direction | both | SUP | native-companion | witwin/channel/propagation/fields.py | tests/ad/test_field_em_ad.py::test_free_space_direction_seed_satisfies_the_adjoint_identity | adjoint |
| kernel/reflection | out:field_direction | both | SUP | native-companion | witwin/channel/propagation/fields.py | tests/ad/test_field_em_ad.py::test_reflection_direction_seed_satisfies_the_adjoint_identity | adjoint |
| reevaluate/prepared | out:field_direction | both | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_direction_liveness_is_one_decision_for_the_whole_result | analytic |
| discovery | out:field_direction | both | DECL | native-declared | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_arrival_direction_stays_declared_dead_on_the_discovery_route | declaration |
| discovery | out:interaction_positions_m | both | DECL | native-declared | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_interaction_positions_are_declared_dead_on_the_discovery_route | declaration |
| reevaluate/prepared | out:interaction_positions_m | vjp | SUP | native-companion | witwin/channel/propagation/geometry.py | tests/propagation/consumer/test_fixed_reflection.py::test_the_specular_point_moves_with_the_sink_and_that_motion_is_differentiable | analytic |
| discovery | out:path_length_m, out:delay_s | both | SUP | native-companion | witwin/channel/propagation/fields.py | tests/propagation/consumer/test_e2e.py::test_fixed_los_reevaluation_vjp_and_jvp_end_to_end | fd |
| reevaluate/prepared | out:path_length_m, out:delay_s | jvp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_fixed_reflection.py::test_forward_mode_publishes_geometry_tangents_under_the_declared_convention | fd |
| reevaluate/prepared | out:interaction_normals | both | DECL | native-declared | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_interaction_positions_are_declared_dead_on_the_discovery_route | declaration |

### 2.2 Endpoint and frequency leaves

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| reevaluate/raw-los | sources.positions_m, sinks.positions_m | both | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_e2e.py::test_fixed_los_reevaluation_vjp_and_jvp_end_to_end | fd |
| reevaluate/prepared | sinks.positions_m | vjp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_fixed_reflection.py::test_the_specular_point_moves_with_the_sink_and_that_motion_is_differentiable | analytic |
| reevaluate/prepared | sources.positions_m, sinks.positions_m | jvp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_fixed_reflection.py::test_a_forward_only_dual_carries_full_geometry_tangents | fd |
| reevaluate/wideband | reference_frequency_hz | both | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/ad/test_wideband_frequency_ad.py::test_forward_and_reverse_agree_column_by_column | fd |
| reevaluate/slots | sinks.positions_m | vjp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_slot_batched_replay_carries_reverse_gradients | analytic |
| reevaluate/slots | sinks.positions_m | jvp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase7_slot_batching.py::test_forward_duals_survive_slot_replication | analytic |
| reevaluate/time-varying | sinks.positions_m | vjp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_time_varying_replay_carries_reverse_gradients | analytic |
| reevaluate/time-varying | sinks.positions_m | jvp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase7_time_varying_cir.py::test_time_varying_cir_is_a_valid_impulse_response | oracle-f64 |
| reevaluate/jones | sinks.positions_m | vjp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_reflection_jones.py::test_reflection_jones_reverse_mode_matches_central_differences | fd |
| reevaluate/jones | sinks.positions_m | jvp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_jones_operator_carries_forward_tangents | analytic |
| reevaluate/prepared | combined endpoint + material leaves | vjp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_combined_request_equals_the_sum_of_its_single_leaf_gradients | fd |
| discovery/transmission | sources.positions_m | vjp | SUP | native-companion | witwin/channel/propagation/fields.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_transmission_scene_carries_a_reverse_gradient_matching_fd | fd |
| discovery/transmission | sources.positions_m | jvp | SUP | native-companion | witwin/channel/propagation/fields.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_transmission_tangent_agrees_with_its_reverse_gradient | adjoint |

### 2.3 Material and mesh leaves

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| reevaluate/prepared | materials.eps_r (reflection) | vjp | SUP | native-companion | native/channel/kernels/field_transport.cu | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_reflection_scene_reads_the_per_face_material_leaves | analytic |
| reevaluate/prepared | materials.layer_eps_r (reflection) | vjp | ZERO | host-declaration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_layer_leaf_contributes_exactly_zero_to_a_reflection_scene | analytic |
| reevaluate/prepared | structures[i].vertices | vjp | SUP | native-companion | witwin/channel/scene/endpoints.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_mesh_vertex_gradient_matches_the_image_source_closed_form | analytic |
| reevaluate/prepared | materials.mu_r, materials.layer_mu_r | both | REF | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_primal_only_material_is_refused_before_any_native_work | refusal |

### 2.4 Primal-only inputs, refused on every route and both modes

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| discovery | sources.powers_w | both | REF | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_primal_only_input_is_refused_before_discovery_produces_a_result | refusal |
| discovery | sources.polarizations, sinks.polarizations | both | REF | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_primal_only_input_is_refused_before_discovery_produces_a_result | refusal |
| reevaluate/raw-los | sources.powers_w | both | REF | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_service_contract.py::test_fixed_primal_only_endpoint_ad_fails_before_gather | refusal |
| reevaluate/jones | sources.polarization_basis, sinks.polarization_basis | both | REF | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_reflection_jones.py::test_polarization_basis_gradients_are_rejected_before_any_native_work | refusal |
| all | the published refusal vocabulary | both | REF | host-declaration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_capability_record_names_every_pre_compute_refusal | declaration |

### 2.5 Diffraction

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| discovery/diffraction | every leaf | jvp | REF | host-declaration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_diffraction_ad_is_refused_before_any_native_work | refusal |
| discovery/diffraction | every leaf | vjp | REF | host-declaration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_diffraction_ad_is_refused_before_any_native_work | refusal |
| discovery/diffraction | primal reachability | none | REF | host-declaration | witwin/channel/interactions/diffraction.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_diffraction_primal_defect_is_pinned_rather_than_fixed | refusal |

### 2.6 Higher order

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| discovery | create_graph=True | vjp | REF | torch-orchestration | witwin/channel/runtime.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_create_graph_through_a_discovery_names_the_owner | refusal |
| reevaluate/prepared | create_graph=True | vjp | REF | torch-orchestration | witwin/channel/runtime.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_create_graph_through_a_reevaluation_names_the_owner | refusal |
| discovery | forward dual under ad_mode=vjp | vjp | REF | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_forward_over_reverse_request_is_refused_before_any_result | refusal |
| reevaluate/prepared | forward dual under ad_mode=vjp | vjp | REF | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_forward_over_reverse_reevaluation_is_refused_before_any_result | refusal |
| reevaluate/prepared | requires_grad primal under a forward dual | jvp | SUP | native-companion | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_requires_grad_primal_under_a_forward_dual_stays_supported | analytic |
| reevaluate/prepared | nested forward levels | jvp | REF | torch-orchestration | torch/autograd/forward_ad.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_nested_forward_levels_raise_from_torch_and_that_is_the_owner | refusal |
| kernel/reflection | create_graph=True | vjp | REF | torch-orchestration | witwin/channel/runtime.py | tests/ad/test_field_em_ad.py::test_double_backward_raises | refusal |
| kernel/geometry | create_graph=True | vjp | REF | torch-orchestration | witwin/channel/runtime.py | tests/ad/test_rayd_geometry_ad.py::test_double_backward_raises | refusal |
| kernel/reflection | composed functorch transforms | jvp | REF | torch-orchestration | witwin/channel/runtime.py | tests/ad/test_field_em_ad.py::test_composed_functorch_transforms_raise | refusal |

### 2.7 AD accounting

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| discovery | out:ad_companion_launches, out:ad_tape_bytes | both | SUP | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_discovery_route_publishes_the_ledger_it_already_built | analytic |
| reevaluate/prepared | out:ad_companion_launches, out:ad_tape_bytes | both | SUP | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_the_record_declares_ad_accounting_and_both_routes_publish_it | analytic |
| reevaluate/wideband | out:ad_companion_launches | vjp | SUP | torch-orchestration | witwin/channel/propagation/consumer.py | tests/propagation/consumer/test_phase9_ad_matrix.py::test_a_wideband_sweep_accounts_every_column_it_launches | analytic |

## 3. Tape ledger

`AdLaunchLedger` (`witwin/channel/runtime.py`) is the one
counter both consumer routes now publish. One `add()` per registered
differentiable native `Function` this call drives, and `tape_bytes` is the sum
of what those `Function`s retained for backward.

| family | tape owner | saved tensors | bytes formula | fwd launches | lifetime |
|---|---|---|---|---|---|
| free-space transport | `witwin/channel/propagation/fields.py::_FieldFreeSpaceAdFunction` | source, target, tx_power, tx_polarization, rx_polarization | `4*(K*3 + K*3 + K + K*3 + K*3)` | 1 per bucket per frequency column | released when the published tensors are released |
| reflection transport | `witwin/channel/propagation/fields.py::_FieldReflectionSequenceAdFunction` | source, target, interaction_positions, interaction_normals, tx_power, the three polarization/power tensors, eps_r, sigma_e, mu_r, gain, thickness | `4*(K*3*4 + K*4 + 2*K*D*3 + 5*K*D)` | 1 per bucket per frequency column | released when the published tensors are released |
| source amplitude | `witwin/channel/propagation/fields.py` | field_vector, tx_power | `4*(2*K*3 + K)` | 1 per bucket, complex3 response only | released with the published field |

`ad_tape_bytes` is gated to reverse mode by the consumer tape gate, which
reproduces the gate `witwin/channel/deterministic.py` applies: forward mode retains
nothing past the solve, so a `jvp` call reports zero tape however many
companions it launched. A wideband sweep of `F` offsets drives `1 + F` columns,
so its launch count is `(1 + F)` times the single-column count; that is the
honest law and `frequency_column_count` publishes the multiplier.

No tape object reaches a public result. `PropagationEvaluation`,
`FixedTopologyEvaluation`, and `TimeVaryingEvaluation` carry only tensors, ints,
strings, and frozen records; `AdLaunchLedger` is a plain `(int, int)` counter
that is read into `PropagationDiagnostics` and discarded.

## 4. Deferred

Each entry is a `DECL` or a recorded gap, with the reason and the follow-up.

- **`field_direction` on `transmission`, the wedge family, and the coupled
  RD/DD family.** `channel_field_transmission_sequence_backward/_jvp` and the
  wedge/coupled companions forward to `rayd::torch::*`, and RayD owns their
  direction seam. Adding the cotangent input and the tangent output there is a
  RayD change and needs its own RayD ADR. Until then
  `direction_differentiable_components` is `{los, reflection}` and a request
  that carries any other component publishes a fully detached `field_direction`
  for the whole result. This is why the set is published rather than inferred.
- **Discovery-route geometry liveness (`interaction_positions_m`,
  `field_direction`).** Discovery re-solves the topology, so the derivative is
  only defined between selection boundaries and Channel deliberately publishes
  no subgradient at one. Making it live also requires seeding
  `evaluated_paths_compact_finalize_backward` with direction and interaction
  cotangents. No consumer asks for it: the supported differentiable geometry
  route is `prepare_fixed_topology` + `reevaluate`.
- **Diffraction through `consumer.evaluate` (primal).**
  `service._solver_scene` builds `SolverScene(transmitters=(), receivers=())`
  because the consumer takes explicit endpoint batches, while
  `interactions/diffraction.py:469` indexes `tx_polarizations[tx_index]`. The
  result is an `IndexError` at every AD mode including `none`. ADR-043
  deliberately does not fix it: it is a primal reachability defect, and fixing
  it would silently re-open an AD column nobody has validated. The failure is
  pinned by exception type and site so a future fix is a deliberate decision.
- **Second-order AD anywhere.** `supports_higher_order_ad` is `False` and every
  composition is refused. A reparameterised second-order contract would be its
  own ADR.
