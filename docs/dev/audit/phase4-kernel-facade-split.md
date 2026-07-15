# Phase 4 kernel facade split audit

This is the body-preserving migration manifest for
`src/witwin/channel_native/core/kernels/ops.py`.  It records ownership, direct
dependencies, call sites, and compatibility hazards before the Phase 4 moves.
It is not authorization to change algorithms, signatures, launch order, or
validation behavior.

## Immediate boundary: runtime symbols

The first independently movable boundary is deliberately small:

- `runtime.symbols` owns native extension lookup and required/optional symbol
  lookup.  From `ops.py`, only `_required_native_op` belongs to that boundary;
  `native_extension` remains a same-object compatibility re-export.
- `validate_cuda_tensor` is a shared tensor contract, not symbol lookup.  Move
  it separately to a narrowly named validation/tensor-contract module.
- `_raydn_module_handle` belongs to the temporary RayD bridge ABI and
  `_raydn_scene_handle_id` belongs to scene-handle normalization.  Neither is a
  runtime-symbol concern.
- `noop_metadata` remains with kernel metadata.

After a function moves, its `__globals__` is the new owner module.  Therefore
patching `core.kernels.ops.native_extension` no longer changes the moved
function's lookup.  Tests must patch `runtime.symbols.native_extension` (or the
new canonical owner dependency) in the same migration commit.  Do not add a
reverse `sys.modules` lookup or copy a function body to preserve a private
monkeypatch target.

## Exact ownership manifest

Names separated by commas move together.  Autograd classes move with their
primal/backward/JVP wrappers so class globals continue to resolve inside one
owner.

### Shared runtime and metadata

- `runtime.symbols`: `_required_native_op`; compatibility re-export
  `native_extension`.
- `runtime` tensor validation contract: `validate_cuda_tensor`.
- `runtime.torch_compat` / narrowly scoped AD contract:
  `_ad_still_wrapped`, `_ad_raise_composed_transforms`, `_ad_native_tensor`,
  `_ad_native_tangent_or_none`, `_ad_checked_tangent`, `_ad_check_rows`,
  `_ad_check_active`, `_ad_check_optional_grad`, `_ad_check_tangent_vec3`,
  `_ad_active_ctx`, `_ad_frequency_value`, `_ad_frequency_tangent`,
  `_ad_frequency_grad`, `_ad_reject_fixed_inputs`,
  `_ad_reject_fixed_tangents`, `_ad_geometry_live`, `_ad_geometry_tangent`.
- `core.kernels.metadata`: `noop_metadata`.

`runtime.torch_compat` is the only final owner allowed to touch `torch._C`, but
the generic AD validation functions may be kept in a sibling runtime AD
contract module if that avoids turning `torch_compat.py` into a new utility
barrel.

### Scene kernels

`_raydn_scene_handle_id`, `raydn_scene_create`, `raydn_scene_edge_records`.

`_raydn_module_handle` stays with the temporary scene/RayD bridge adapter until
the legacy handle argument is removed end-to-end.  Geometry callers may depend
on that adapter; it must not be duplicated in geometry.

### Propagation geometry kernels

- Native geometry and stable aliases: `bdpt_visibility_forward` and
  `raydn_visibility_forward`; `bdpt_intersect_forward`;
  `bdpt_reflection_accumulation_forward` and
  `raydn_reflection_accumulation_forward`;
  `bdpt_diffraction_discover_edges` and `raydn_diffraction_discover_edges`;
  `bdpt_diffraction_discover_edges_counted` and
  `raydn_diffraction_discover_edges_counted`;
  `bdpt_diffraction_accumulation_forward` and
  `raydn_diffraction_accumulation_forward`.
- Trace and geometry AD: `raydn_trace_reflections_forward`,
  `raydn_reflection_epc_paths_forward`, `raydn_intersect_backward`,
  `raydn_intersect_jvp`, `_RaydnIntersectAdFunction`, `raydn_intersect_ad`,
  `raydn_trace_reflections_forward_tape`,
  `raydn_trace_reflections_backward`, `raydn_trace_reflections_jvp`,
  `_RaydnTraceReflectionsAdFunction`, `raydn_trace_reflections_ad`,
  `_epc_paths_frozen_winner_checks`,
  `raydn_reflection_epc_paths_backward`,
  `raydn_reflection_epc_paths_jvp`,
  `_RaydnReflectionEpcPathsAdFunction`,
  `raydn_reflection_epc_paths_ad`, `raydn_scene_face_normals_backward`,
  `raydn_scene_face_normals_jvp`, `_RaydnFaceNormalsAdFunction`,
  `raydn_face_normals_ad`, `raydn_coupled_rd_geometry_forward`,
  `raydn_diffraction_paths_order1_forward`.
- Pure geometry primitives: `core_diffraction_edge_count`,
  `bdpt_diffraction_edge_geometry`, `bdpt_surface_group_edge_candidates`,
  `mc_diffraction_edge_geometry`, `mc_surface_group_edge_candidates`,
  `deterministic_normalize_vec3`, `deterministic_reflect_points`,
  `deterministic_face_groups`, `deterministic_surface_face_groups`.

The old `bdpt_*` and `mc_*` prefixes on shared native geometry wrappers do not
make them solver-owned.  Each fused symbol has one geometry facade owner; old
names are same-object re-exports only.

### Propagation fields kernels

- Primal: `field_free_space`, `field_project_complex3`,
  `field_reflection_sequence`, `field_transmission_sequence`,
  `field_coupled_rd`, `field_diffraction_wedge`.
- Backward/JVP: `field_free_space_backward`, `field_free_space_jvp`,
  `field_reflection_sequence_backward`, `field_reflection_sequence_jvp`,
  `field_transmission_sequence_backward`,
  `field_transmission_sequence_jvp`.
- Autograd: `_FieldFreeSpaceAdFunction`, `field_free_space_ad`,
  `_FieldReflectionSequenceAdFunction`, `field_reflection_sequence_ad`,
  `_FieldTransmissionSequenceAdFunction`, `field_transmission_sequence_ad`,
  `_FieldDiffractionWedgeAdFunction`, `field_diffraction_wedge_ad`,
  `_FieldProjectComplex3AdFunction`, `field_project_complex3_ad`,
  `_FieldCoupledRdAdFunction`, `field_coupled_rd_ad`,
  `_CoupledRdPrepareAdFunction`, `coupled_rd_prepare_ad`.

### Materials and scattering kernels

- Materials: `_validate_layer_csr`, `bdpt_face_material_tensors`,
  `bdpt_face_material_tensors_from_host`, `mc_face_material_tensors`,
  `em_layer_stack_eval`, `em_layer_stack_backward`, `em_layer_stack_jvp`,
  `_EmLayerStackAdFunction`, `em_layer_stack_ad`.
- Scattering: `scattering_table_eval`, `scattering_table_pdf`,
  `scattering_table_sample`, `scattering_event_probabilities`.

Layer CSR validation must have one materials owner.  Fields may depend on it
for transmission input validation; it must not be copied into fields.

### Propagation topology/enumerated kernels

- Generic discrete/count/packing primitives: `deterministic_component_counts`,
  `deterministic_selected_edge_count`, `core_pack_int2`,
  `deterministic_diffraction_state_pack`,
  `deterministic_diffraction_state_pack_selected`,
  `mc_selected_edge_indices`.
- Path export and validators: `path_los_export`, `_validate_path_block`,
  `_validate_deterministic_topology_block`,
  `_validate_topology_extra_fields`, `_validate_path_reflection_candidates`.
- Concatenate/gather/filter/finalize: `path_concat_vec3`,
  `deterministic_concat_topology_blocks`,
  `deterministic_gather_topology_block`, `path_los_visibility_inputs`,
  `path_filter_los`, `path_filter_block`, `path_diffraction_block`,
  `path_merge_blocks`, `path_finalize_blocks`.
- Deterministic topology construction: `deterministic_los_topology_block`,
  `deterministic_topology_default_fields`,
  `deterministic_pad_topology_sequences`,
  `deterministic_topology_base_fields`, `deterministic_repeat_range`,
  `deterministic_face_anchor_points`,
  `deterministic_reflection_epc_input_batch`,
  `deterministic_face_sequence_chunk`,
  `deterministic_mapped_face_sequence_chunk`,
  `deterministic_reflection_order1_compact`,
  `deterministic_reflection_sequence_compact`,
  `deterministic_diffraction_order1_compact`, `deterministic_sort_order`.
- Candidate/path assembly: `path_reflection_candidates`,
  `path_diffraction_paths_order1`.

`deterministic_face_anchor_points` stays enumerated because its contract is
discrete batch construction, whereas normalize/reflect/group primitives are
geometry.  `propagation.enumerated` must remain unused by MC Basic and BDPT.

### Deterministic fields and accumulation

- Deterministic field assembly: `deterministic_los_field`,
  `deterministic_diffraction_vector_field`,
  `deterministic_reflection_field`,
  `deterministic_reflection_sequence_field`,
  `deterministic_delay_to_path_length`, `deterministic_pack_complex`,
  `deterministic_phase_from_field`, `deterministic_zero_field_phase`,
  `deterministic_phase_from_length`,
  `deterministic_field_from_power_phase`.
- Accumulation and AD: `deterministic_accumulate_flat`,
  `deterministic_accumulate_flat_backward`,
  `deterministic_accumulate_flat_jvp`,
  `_DeterministicAccumulateFlatAdFunction`,
  `deterministic_accumulate_flat_ad`.

These are solver-local deterministic ownership, not
`propagation.enumerated`.  Existing deterministic facade modules may re-export
shared field helpers during the migration, but there must be one body owner.

### MC Basic kernels

- LoS and component maps: `mc_los_path_gain_backward`,
  `mc_los_path_gain_jvp`, `_McLosPathGainAdFunction`,
  `mc_los_path_gain_ad`, `mc_finalize_component_maps`,
  `_McFinalizeComponentMapsAdFunction`, `mc_finalize_component_maps_ad`,
  `mc_los_component_maps_adjoint`, `_McLosGridMapsAdFunction`,
  `mc_los_grid_maps_ad`, `mc_zero_matrix`, `mc_point_component_power`,
  `mc_component_map_buffer`, `mc_store_component_map`,
  `mc_store_scaled_component_map`, `mc_los_component_maps`,
  `mc_los_component_maps_from_matrix`, `mc_apply_los_visibility`,
  `mc_los_visibility_inputs`.
- Sampling and packing: `mc_sample_directions`, `mc_transmitter_tensors`,
  `mc_pack_vec3`, `mc_receiver_grid_points`, `mc_reflection_launch_inputs`,
  `mc_diffraction_state_wi`, `mc_diffraction_state_pack`.
- Reflection: `mc_sionna_reflection_accumulate`,
  `mc_reflection_ad_max_depth`,
  `mc_sionna_reflection_accumulate_backward`,
  `mc_sionna_reflection_accumulate_jvp`, `_McReflectionMapAdFunction`,
  `mc_sionna_reflection_accumulate_ad`.
- Diffraction: `mc_sionna_diffraction_tape_accumulate`,
  `mc_sionna_diffraction_tape_accumulate_backward`,
  `mc_sionna_diffraction_tape_accumulate_jvp`,
  `_McDiffractionMapAdFunction`,
  `mc_sionna_diffraction_tape_accumulate_ad`.

The selected-edge, geometry, and material wrappers listed under shared owners
are intentionally excluded from MC Basic.

### BDPT kernels

- Launch/state validation: `bdpt_launch_state`,
  `_validate_bdpt_subpath_state`, `_validate_bdpt_connection_samples`,
  `_bdpt_mis_mode_id`, `bdpt_empty_subpath_state`,
  `bdpt_endpoint_subpath_state`, `bdpt_subpath_intersection_inputs`,
  `bdpt_reflected_light_subpath_state`,
  `bdpt_transmitted_light_subpath_state`.
- Connections/MIS: `bdpt_endpoint_connection_samples`,
  `bdpt_endpoint_connection_visibility_inputs`,
  `bdpt_accumulate_connection_samples`, `bdpt_filter_connection_samples`,
  `bdpt_count_valid_connection_samples`,
  `bdpt_compact_connection_samples`, `bdpt_concat_connection_samples`,
  `bdpt_connection_variance`, `bdpt_mis_weights`,
  `bdpt_diffraction_connection_samples_from_tape`,
  `bdpt_diffraction_point_connection_samples`.
- Buffers and maps: `bdpt_zero_matrix`,
  `bdpt_store_point_component_column`, `bdpt_finalize_point_components`,
  `bdpt_point_component_power`, `bdpt_transmitter_tensors`,
  `bdpt_host_vec3_tensor`, `bdpt_receiver_grid_points`, `bdpt_los_export`,
  `bdpt_los_component_maps`, `bdpt_los_component_maps_from_matrix`,
  `bdpt_los_visibility_inputs`, `bdpt_apply_los_visibility`,
  `bdpt_component_map_buffer`, `bdpt_store_component_map`,
  `bdpt_store_scaled_component_map`, `bdpt_finalize_component_maps`.
- Sampling/packing: `bdpt_sample_directions`,
  `bdpt_reflection_launch_inputs`, `bdpt_diffraction_state_wi`,
  `bdpt_selected_edge_indices`, `bdpt_diffraction_state_pack`,
  `bdpt_pack_vec3`.

Geometry, material, visibility/intersection, and accumulation native wrappers
already assigned to shared domain owners are intentionally excluded from this
BDPT set.  BDPT imports them; it does not own duplicate facades.

## Private-helper dependency edges

The following cross-owner edges must be made explicit imports when bodies move:

- Every domain facade uses `runtime.symbols._required_symbol` through its local
  dependency and most primal validators use the shared tensor contract.
- Scene/geometry functions use the temporary `_raydn_module_handle` bridge and
  scene-owned `_raydn_scene_handle_id`.
- All AD classes use the runtime AD/torch compatibility helpers.  Field,
  material, MC, and geometry AD classes must import those helpers; they must not
  import `core.kernels.ops`.
- Geometry `_RaydnFaceNormalsAdFunction` calls the geometry primitive
  `deterministic_normalize_vec3`.
- Materials `_EmLayerStackAdFunction` calls its own eval/backward/JVP functions;
  fields transmission validation imports materials `_validate_layer_csr`.
- MC `_McLosGridMapsAdFunction` calls MC LoS map/apply helpers;
  `_McLosPathGainAdFunction` imports topology-owned `path_los_export`.
- Topology validators form one internal chain:
  `_validate_deterministic_topology_block` -> `_validate_path_block` plus
  `_validate_topology_extra_fields`; reflection candidates use
  `_validate_path_reflection_candidates`.
- Deterministic accumulation AD calls only its local backward/JVP functions and
  the runtime AD helpers.
- BDPT connection functions call only BDPT validators/MIS-mode parsing plus
  shared runtime/material dependencies.

No moved function may resolve a sibling via the compatibility `ops` facade.

## Production import/call-site inventory

The inventory below is the direct production import surface at audit time.
Callers move to the canonical owner in the same domain commit.

- `core/diffraction_geometry.py`: `mc_diffraction_edge_geometry`.
- `core/material_runtime.py`: `mc_face_material_tensors`.
- `core/runtime/raydn.py`: `mc_pack_vec3`, `raydn_scene_create`,
  `raydn_scene_edge_records`.
- `core/scene_tensors.py`: `mc_receiver_grid_points`,
  `mc_transmitter_tensors`, `path_concat_vec3`.
- `deterministic/accumulation.py`: `deterministic_accumulate_flat`,
  `deterministic_accumulate_flat_ad`, `deterministic_pack_complex`,
  `deterministic_phase_from_field`, `deterministic_zero_field_phase`.
- `deterministic/field.py`: `deterministic_diffraction_vector_field`,
  `deterministic_field_from_power_phase`, `deterministic_los_field`,
  `deterministic_pack_complex`, `deterministic_phase_from_field`,
  `deterministic_phase_from_length`, `deterministic_reflection_field`,
  `deterministic_reflection_sequence_field`.
- `deterministic/solver.py`: `deterministic_component_counts`.
- `montecarlo/basic/backend.py`: `mc_los_path_gain_ad`,
  `mc_los_visibility_inputs`, `mc_zero_matrix`, `path_los_export`,
  `raydn_visibility_forward`.
- `montecarlo/basic/raydn_components.py`:
  `deterministic_diffraction_state_pack`,
  `deterministic_diffraction_state_pack_selected`, `mc_apply_los_visibility`,
  `mc_component_map_buffer`, `mc_diffraction_state_wi`,
  `mc_los_component_maps_from_matrix`, `mc_los_grid_maps_ad`,
  `mc_los_visibility_inputs`, `mc_reflection_launch_inputs`,
  `mc_sample_directions`, `mc_sionna_diffraction_tape_accumulate`,
  `mc_sionna_diffraction_tape_accumulate_ad`,
  `mc_sionna_reflection_accumulate`, `mc_sionna_reflection_accumulate_ad`,
  `mc_store_component_map`, `mc_store_scaled_component_map`,
  `mc_surface_group_edge_candidates`, `raydn_diffraction_accumulation_forward`,
  `raydn_diffraction_discover_edges`,
  `raydn_diffraction_discover_edges_counted`,
  `raydn_trace_reflections_forward`, `raydn_visibility_forward`.
- `montecarlo/basic/solver.py`: `mc_component_map_buffer`,
  `mc_finalize_component_maps`, `mc_finalize_component_maps_ad`,
  `mc_point_component_power`, `mc_reflection_ad_max_depth`, `mc_zero_matrix`.
- `montecarlo/bdpt/connections.py`: `bdpt_host_vec3_tensor`,
  `bdpt_receiver_grid_points`, `bdpt_transmitter_tensors`.
- `montecarlo/bdpt/mis.py`: `bdpt_mis_weights`.
- `montecarlo/bdpt/sampling.py`: `bdpt_launch_state`.
- `montecarlo/bdpt/subpaths.py`: `bdpt_empty_subpath_state`.
- `montecarlo/bdpt/solver.py`: `bdpt_accumulate_connection_samples`,
  `bdpt_compact_connection_samples`, `bdpt_concat_connection_samples`,
  `bdpt_connection_variance`, `bdpt_count_valid_connection_samples`,
  `bdpt_diffraction_accumulation_forward`,
  `bdpt_diffraction_connection_samples_from_tape`,
  `bdpt_diffraction_point_connection_samples`, `bdpt_diffraction_state_pack`,
  `bdpt_diffraction_state_wi`, `bdpt_endpoint_connection_samples`,
  `bdpt_endpoint_connection_visibility_inputs`,
  `bdpt_endpoint_subpath_state`, `bdpt_filter_connection_samples`,
  `bdpt_finalize_component_maps`, `bdpt_finalize_point_components`,
  `bdpt_intersect_forward`, `bdpt_los_component_maps_from_matrix`,
  `bdpt_reflected_light_subpath_state`, `bdpt_reflection_launch_inputs`,
  `bdpt_sample_directions`, `bdpt_selected_edge_indices`,
  `bdpt_subpath_intersection_inputs`, `bdpt_transmitted_light_subpath_state`,
  `bdpt_zero_matrix`, `em_layer_stack_eval`, `mc_component_map_buffer`,
  `mc_store_component_map`, `raydn_visibility_forward`.
- `montecarlo/scattering_events.py`: `raydn_visibility_forward`,
  `scattering_event_probabilities`, `scattering_table_sample`.
- `montecarlo/transmission.py`: `_ad_frequency_value`,
  `bdpt_intersect_forward`, `em_layer_stack_ad`, `em_layer_stack_eval`.
- `propagation/enumerated.py`: `em_layer_stack_eval`,
  `raydn_visibility_forward`.
- `scattering/tables.py`: `scattering_table_eval`, `scattering_table_pdf`,
  `scattering_table_sample`.
- `core/path_topology.py`: `_ad_frequency_value`, `bdpt_intersect_forward`,
  `coupled_rd_prepare_ad`, `deterministic_concat_topology_blocks`,
  `deterministic_delay_to_path_length`,
  `deterministic_diffraction_order1_compact`,
  `deterministic_diffraction_state_pack`,
  `deterministic_diffraction_vector_field`,
  `deterministic_face_anchor_points`, `deterministic_face_groups`,
  `deterministic_face_sequence_chunk`,
  `deterministic_gather_topology_block`,
  `deterministic_los_topology_block`,
  `deterministic_mapped_face_sequence_chunk`,
  `deterministic_normalize_vec3`, `deterministic_pack_complex`,
  `deterministic_pad_topology_sequences`, `deterministic_reflect_points`,
  `deterministic_reflection_epc_input_batch`,
  `deterministic_reflection_field`,
  `deterministic_reflection_order1_compact`,
  `deterministic_reflection_sequence_compact`,
  `deterministic_reflection_sequence_field`,
  `deterministic_selected_edge_count`, `deterministic_sort_order`,
  `deterministic_topology_base_fields`,
  `deterministic_topology_default_fields`, `field_coupled_rd`,
  `field_coupled_rd_ad`, `field_diffraction_wedge_ad`, `field_free_space`,
  `field_free_space_ad`, `field_project_complex3`,
  `field_project_complex3_ad`, `field_reflection_sequence`,
  `field_reflection_sequence_ad`, `field_transmission_sequence`,
  `field_transmission_sequence_ad`, `mc_sample_directions`,
  `mc_selected_edge_indices`, `path_los_export`, `path_los_visibility_inputs`,
  `raydn_coupled_rd_geometry_forward`,
  `raydn_diffraction_paths_order1_forward`, `raydn_face_normals_ad`,
  `raydn_reflection_epc_paths_ad`, `raydn_reflection_epc_paths_forward`,
  `raydn_trace_reflections_forward`, `raydn_visibility_forward`.

## Compatibility hazards

### Monkeypatch/global lookup

Tests patch `core.kernels.ops.native_extension` in the BDPT MIS/facade,
ops-facade, scattering-kernel, and scattering-sampling suites.  Those patches
must move to the canonical owner because same-object re-export does not change
`function.__globals__`.

Tests also patch sibling names looked up by moved functions, including
`path_los_export`, `deterministic_los_topology_block`,
`deterministic_diffraction_vector_field`,
`raydn_reflection_epc_paths_forward`, `deterministic_reflection_field`, and
`raydn_coupled_rd_geometry_forward`.  Patch the canonical module in each domain
commit.  Patching `_McDiffractionMapAdFunction.forward` through `ops` remains
valid only while `ops` re-exports the exact same class object.

The five `raydn_* = bdpt_*` aliases must remain object-identical.  Define each
pair once in the geometry owner and re-export it; wrappers would break identity,
stack traces, and monkeypatch behavior.

### AST body hashes and private Torch API

Use `tools.refactor_baseline.python_body_hashes`.  For a move, compare
`body_sha256` and `normalized_ast_sha256` by the old terminal qualified name
after stripping only the module prefix; the baseline path is expected to
change.  This covers functions and class methods.  Signature snapshots and
same-object alias assertions are separate gates.

There are 18 `_DisableFuncTorch` context expressions in the AD classes, plus
direct `_functorch` access in `_ad_still_wrapped` and `_ad_native_tensor`.
Replacing `torch._C...` expressions with calls into `runtime.torch_compat`
necessarily changes those AST bodies.  Do that as a dedicated compatibility
commit with an explicit before/after hash exception limited to those
expressions, run the supported Torch matrix, then freeze the new hashes before
moving any domain bodies.  Do not claim body equivalence for this mechanical
rewrite.

The current runtime-symbol extraction also changes `_required_native_op` from a
direct extension lookup to delegation.  Its hash exception must be documented
in that runtime-only commit; subsequent domain moves must preserve the new
frozen body.

## Independently reviewable migration order

1. Runtime symbols only; update symbol-lookup patch targets and freeze its new
   compatibility hash.
2. Runtime Torch compatibility and AD contracts only; allowlist exactly the
   private-API expression rewrites and run the Torch-version contract matrix.
3. Shared tensor validation and metadata, with no domain moves.
4. Scene kernels and the temporary RayD handle adapter.
5. Geometry kernels and aliases, including geometry AD and shared primitives.
6. Fields kernels and their AD classes.
7. Materials, then scattering, as two commits if either changes call sites
   outside its domain.
8. Topology/enumerated kernels; verify MC Basic and BDPT have zero imports from
   `propagation.enumerated`.
9. Deterministic field/accumulation kernels.
10. MC Basic kernels.
11. BDPT kernels.
12. Reduce `core.kernels.ops` to same-object re-exports under 300 lines, prove
    direct production imports are zero, then defer deletion to Phase 12.

For every domain commit: capture hashes/signatures/import graph first; move
bodies without edits; update production imports and canonical patch targets in
the same commit; assert aliases by `is`; run focused forward/JVP/VJP tests,
exact-golden/RNG/launch-ledger checks applicable to that domain, the full suite,
Ruff, and import-boundary gates.  A body hash mismatch is a logic change and
must be split out and explained.
