# Transmission / Diffuse Scattering Analysis And Implementation Plan (2026-04-01)

## Scope

This report analyzes the current `witwin.channel` codebase and proposes a concrete implementation plan for:

- transmission / specular-through-material propagation
- diffuse reflection / surface scattering

The goal is not to design an abstract "future architecture". The goal is to determine what the current code already supports, what is structurally missing, and what implementation order minimizes rework.

## Executive Summary

### Current conclusion

The current solver is structurally strong in three places:

1. The public orchestration is already clean: `Scene + Tracer + Result`.
2. Path-level outputs already reserve `TRANSMISSION` and `SCATTERING` interaction codes.
3. Reflection and diffraction already use material-aware Jones/vector transport, which is the right foundation for transmission.

The current solver is structurally blocked in four places:

1. `FieldMonitor` results are hard-coded to `los + reflection + diffraction_direct + diffraction_mixed + diffraction + total`.
2. LoS and visibility logic are binary occlusion tests, not attenuation-integrating traversals.
3. Scene runtime material tables only store `eps_r` and `sigma_e`; they do not store thickness, roughness, scattering gain/albedo, or lobe parameters.
4. Reflection discovery is image-source/specular-chain based, so diffuse scattering does not fit by "slightly modifying" reflection EPC.

### Recommended implementation order

1. Add material/runtime/config support for transmission and scattering parameters.
2. Implement transmission first, starting from a thin-slab model.
3. Extend `FieldMonitor` and `PathMonitor` to expose transmission explicitly.
4. Add transmission-aware visibility for diffraction and mixed families.
5. Add scattering as a separate solver family, not as a patch on the specular reflection solver.
6. Start scattering with first-order surface scattering only; defer mixed higher-order scattering until the first-order model is stable.

### Key recommendation

Do **not** start with full Snell refraction or arbitrary mixed scattering chains.

For the current codebase, the highest-leverage first release is:

- thin-slab transmission
- first-order diffuse / rough-surface scattering
- field and path outputs for both
- no higher-order scattering chains yet

That fits the existing architecture with the least disruption.

---

## 1. Current Code Snapshot

### 1.1 Solver orchestration

The top-level orchestration is in `witwin/channel/trace/tracer.py:27` (`class Tracer`) and `witwin/channel/trace/tracer.py:236` (`trace()`).

Today `Tracer.trace()` does exactly this:

- resolve monitors
- run `trace_field_monitor()` for each `FieldMonitor`
- run `trace_path_monitor()` for each `PathMonitor`
- pack everything into `Result`

This is a good extension point. Adding transmission and scattering does **not** require changing the public architecture.

### 1.2 Result schema

The structured field result types are hard-coded in:

- `witwin/channel/result.py:371` `MonitorField`
- `witwin/channel/result.py:383` `MonitorJones`
- `witwin/channel/result.py:395` `MonitorVector`

The field result payload assembled in `witwin/channel/monitors/field/trace_field.py:597` only contains:

- `los`
- `reflection`
- `diffraction_direct`
- `diffraction_mixed`
- `diffraction`
- `total`

This is the first major schema blocker. Even if a transmission or scattering solver existed tomorrow, the field result layer has nowhere to put it.

By contrast, the path result layer is much more extensible:

- `witwin/channel/result.py:504` `PathResult`
- interaction type slots are variable-depth tensors
- interaction codes already reserve transmission and scattering

### 1.3 Config surface

`TraceConfig` in `witwin/channel/config.py:189` currently exposes controls for:

- reflection ray counts / bounce limits
- reflection material override
- diffraction material override
- mixed diffraction budgets
- solver mode
- polarization

It does **not** expose:

- `enable_transmission`
- `max_transmissions`
- transmission material override
- thickness / slab control
- `enable_scattering`
- scattering ray count / sample count
- scattering lobe or roughness controls
- energy split or pruning thresholds for scattering

This means both features require real config work, not just new solver files.

### 1.4 Scene runtime material data

Per-triangle scene material data is compiled and stored in:

- `witwin/channel/scene/compile.py:30-31`
- `witwin/channel/scene/compile.py:205-210`
- `witwin/channel/scene/runtime.py:44`

Today the runtime table stores:

- `material_eps_r`
- `material_sigma_e`
- `material_specified`
- `material_structure_idx`

This is sufficient for Fresnel reflection and current diffraction face operators.

It is **not** sufficient for robust transmission/scattering because there is no per-triangle storage for:

- slab thickness
- interior path length model inputs
- roughness
- scattering strength / albedo
- scattering mode / lobe parameters

### 1.5 LoS and visibility semantics

LoS is binary:

- `witwin/channel/trace/los.py:9` `los_blocked()`
- `witwin/channel/trace/los.py:36` `compute_los_field()`

Current behavior:

- cast shadow/intersection
- if blocked -> zero
- if unblocked -> free-space field

The same binary visibility idea also appears in diffraction target visibility and reflection EPC validity. This is the second major blocker for transmission. Transmission requires attenuation accumulation across penetrated surfaces, not a single boolean.

### 1.6 Reflection architecture

Reflection is organized around specular path discovery and replay:

- `witwin/channel/trace/reflection/api.py:740` `discover_reflection_paths()`
- `witwin/channel/trace/reflection/api.py:1427` `compute_reflection_field()`
- `witwin/channel/trace/reflection/paths.py:11` image-source tolerance
- `witwin/channel/trace/reflection/paths.py:265` unique-path collection

The key structural fact is this:

- reflection discovery is image-source/canonicalized/specular
- receiver evaluation can use EPC of those specular chains

This is excellent for transmission-adjacent specular families.
It is a poor fit for diffuse scattering, where there is no unique image source and no exact specular replay.

### 1.7 Diffraction architecture

Diffraction is organized around edge state construction and accumulation:

- `witwin/channel/trace/diffraction/api.py:394` `compute_diffraction_field()`
- `witwin/channel/trace/diffraction/api.py:240` `compute_diffraction_order_breakdown()`
- `witwin/channel/trace/diffraction/builders/__init__.py`

Current mixed families explicitly assume reflection as the only non-diffraction interaction family:

- reflection prefix
- inserted reflection
- reflection suffix

That shows up in builder metadata such as:

- `field_component_ownership`
- `path_families`

inside `witwin/channel/trace/diffraction/builders/__init__.py:207` and `:246`.

This is the third major blocker. Mixed-family state builders currently know only `R` and `D`; they do not know `T` or `S`.

### 1.8 Path monitor interaction coding

The path monitor stack is the strongest existing extension point:

- `witwin/channel/monitors/path/trace_path.py:101`
- `witwin/channel/types.py:8-9`
- `witwin/channel/monitors/path/collectors.py:110`
- `witwin/channel/monitors/path/collectors.py:171`
- `witwin/channel/monitors/path/collectors.py:530`

Important facts:

- `InteractionType.TRANSMISSION = 4`
- `InteractionType.SCATTERING = 8`
- path metadata already labels these as reserved interaction types

This means path-level schema work is smaller than field-level schema work.

---

## 2. What The Current Code Already Gives Us

## 2.1 Material-aware Fresnel foundation

`witwin/channel/utils/material.py:46` already provides `fresnel_reflection()`.

Reflection and diffraction already use per-surface/per-face material resolution through:

- `witwin/channel/trace/materials.py:179` `resolve_surface_material()`
- `witwin/channel/trace/materials.py:246` `bounce_reflection_weight()`
- `witwin/channel/trace/diffraction/material_ops.py`

This is a strong foundation for transmission because:

- transmission shares the same complex permittivity inputs
- transmission needs the same angle-dependent interface math
- transmission can reuse the same Jones basis machinery

## 2.2 Jones/vector transport is already in place

The current code has already moved beyond scalar-only material handling:

- reflection scalar output is derived from Jones/vector transport
- diffraction face operators are Jones-aware
- monitor metadata already documents transport and scalar projection rules

That materially lowers the risk of adding transmission correctly. Transmission is physically another interface operator, not a new kind of state representation.

## 2.3 Shared reflection detail cache

`Tracer.trace()` caches reflection discovery detail and reuses it between field and path monitors.

That matters because transmission can likely reuse the same pattern:

- discover reusable specular-through-material chains once
- replay to field monitors and path monitors separately

This pattern is useful for transmission.
It is not useful for diffuse scattering.

## 2.4 Path padding, filtering, and CIR/CFR/taps are already solved

`PathResult` already supports:

- padded path tensors
- variable-depth interaction slots
- filtering by interaction type
- CIR / CFR / taps

Once transmission and scattering paths are collected, most of the downstream post-processing already works.

---

## 3. Structural Gaps And Blockers

## 3.1 Field result schema is too rigid

Today the field result type system assumes exactly three physical families:

- LoS
- reflection
- diffraction

Adding `transmission` and `scattering` requires updating:

- `MonitorField`
- `MonitorJones`
- `MonitorVector`
- `MonitorResult.from_payload()`
- `trace_field_monitor()`

This must be done before either new family can be exposed cleanly.

## 3.2 Binary visibility is incompatible with transmission

Transmission requires replacing "blocked or not" with "what was traversed and what attenuation/operator did it contribute".

This affects at least:

- LoS
- diffraction shadowing / segment visibility
- any future transmission-aware reflection or scattering families

## 3.3 Material tables do not carry enough data

Transmission needs at least one additional material concept beyond `eps_r` and `sigma_e`:

- thickness, or
- a rule for estimating effective slab thickness

Scattering needs at least:

- roughness or spread parameter
- scattering gain / albedo / coefficient
- possibly lobe type

Without this, both features would be forced into global defaults, which is not a good public design.

## 3.4 Reflection EPC is specular-only

This is the core architectural reason transmission is easier than scattering.

Specular transmission can still be represented as deterministic chains and replayed.
Diffuse scattering cannot.

So:

- transmission can reuse the reflection-style "discover then replay" structure
- scattering should be a separate "sample and accumulate" family

## 3.5 Mixed diffraction builders are reflection-centric

Current diffraction builders explicitly encode:

- direct TX source
- reflection-prefix source
- inserted reflection
- reflection suffix

They do not have a neutral "generic interaction chain" abstraction.

This means adding transmission to mixed diffraction is not a one-line extension. The builder/state schema itself must be widened.

## 3.6 No dedicated test surface for transmission/scattering

The test tree is strong for:

- reflection
- diffraction
- path monitor
- mixed R/D families

There is effectively no dedicated solver/test package yet for:

- transmission
- scattering

So implementation should include new test groups from the start, not as a later cleanup.

---

## 4. Transmission Analysis

## 4.1 Transmission should be split into three scopes

The word "transmission" can mean three different implementation levels in this codebase:

### Scope T1: attenuation-only thin-slab transmission

Behavior:

- ray direction unchanged
- each penetrated surface contributes a transmission operator / attenuation
- good for wall/floor penetration

Fit with current code:

- very good

Difficulty:

- medium

### Scope T2: deterministic specular transmission paths

Behavior:

- transmission becomes a first-class path family
- field monitors and path monitors expose it
- supports chains like `T`, `R-T`, `T-R`

Fit with current code:

- moderate to good

Difficulty:

- medium-high

### Scope T3: full refraction with direction change

Behavior:

- Snell-law direction changes
- medium entry/exit geometry matters
- interior travel path must be modeled

Fit with current code:

- weak for first implementation

Difficulty:

- high

### Recommendation

Start with T1, then T2.
Do not start with T3 unless your immediate product goal is glass-dominant refractive geometry.

## 4.2 Why transmission is feasible now

Transmission is compatible with the current architecture because:

1. Materials are already resolved per surface.
2. Jones/vector transport already exists.
3. Reflection path discovery/replay infrastructure is already deterministic and chain-based.
4. Path monitor interaction codes already reserve transmission.

Transmission mostly requires:

- new material operators
- multi-hit traversal / segment accumulation
- result/config/schema plumbing

It does **not** require inventing a new result model or a new state transport representation.

## 4.3 What must change for transmission

### A. Material helpers

Extend material utilities with:

- Fresnel transmission coefficients
- thin-slab transmission operator
- optional loss/phase through a slab

Expected files:

- `witwin/channel/utils/material.py`
- `witwin/channel/trace/materials.py`

### B. Scene/runtime material payload

Add per-structure/per-triangle support for:

- `thickness`
- optionally `transmission_gain`

This likely means extending what `structure.material.evaluate_static()` contributes into scene compilation.

Expected files:

- `witwin/channel/scene/compile.py`
- `witwin/channel/scene/runtime.py`

If `witwin.core.Material` cannot yet express these fields, decide this explicitly up front:

- preferred: extend `witwin.core.Material`
- temporary fallback: store them in `Structure.metadata`

The preferred option is the first one.

### C. LoS traversal

Replace binary LoS blocking with transmission-aware multi-hit traversal.

Expected files:

- `witwin/channel/trace/los.py`

Recommended behavior for the first version:

- march along the ray
- gather penetrated surfaces
- accumulate thin-slab transmission operator
- keep direction unchanged

### D. Field result exposure

Add:

- `field.transmission`
- `vector.transmission`
- `jones.transmission`

And update total composition:

- `total = los + reflection + transmission + diffraction + scattering`

Expected files:

- `witwin/channel/result.py`
- `witwin/channel/monitors/field/trace_field.py`

### E. Path collection

Add transmission path collection to the path monitor stack.

Expected files:

- `witwin/channel/monitors/path/collectors.py`
- `witwin/channel/monitors/path/trace_path.py`

Recommended first target:

- LoS-through-material transmission paths
- specular transmission chains only after the base path format is stable

### F. Mixed diffraction support

Only after base transmission is stable, extend diffraction state builders so that:

- transmission can appear in source prefixes
- transmission-aware visibility can attenuate edge-to-receiver and source-to-edge segments

Expected files:

- `witwin/channel/trace/diffraction/builders/__init__.py`
- `witwin/channel/trace/diffraction/geometry/*`
- `witwin/channel/trace/diffraction/api.py`
- `witwin/channel/trace/diffraction/field.py`

## 4.4 Transmission risk assessment

### Low risk

- Fresnel transmission math
- config plumbing
- result schema extension
- path type coding

### Medium risk

- multi-hit LoS traversal
- per-material thickness plumbing
- deterministic transmission path replay

### High risk

- full Snell refraction
- mixed T/R/D state explosion
- preserving AD stability across multi-interface branching

## 4.5 Recommended transmission MVP

The best first transmission release for this repo is:

1. thin-slab transmission operator
2. LoS-through-material attenuation
3. explicit `transmission` field component
4. path monitor transmission coding for direct-through-material paths
5. no full direction-changing refraction yet

That is useful, physically meaningful, and aligned with the current code.

---

## 5. Diffuse Reflection / Scattering Analysis

## 5.1 "Diffuse Reflection / Scattering" needs a strict definition here

In this codebase, "scattering" should be defined as:

- surface-originated non-specular outgoing directions
- driven by roughness / scattering material parameters
- accumulated as a separate path family

It should **not** initially mean:

- volumetric random media scattering
- arbitrary multiple scattering inside participating media
- a vague perturbation of specular reflection without energy accounting

## 5.2 Why scattering is harder than transmission

Transmission still preserves deterministic chain structure.

Diffuse scattering does not.

The current reflection solver depends on:

- canonicalized specular chains
- image sources
- EPC eligibility

That entire logic breaks once a surface emits a lobe of directions instead of a single specular direction.

Therefore the correct architectural conclusion is:

### Scattering should be a separate solver family

Do not try to retrofit diffuse scattering into:

- `discover_reflection_paths()`
- image-source deduplication
- exact specular replay

Instead add:

- a dedicated scattering tracer / collector
- dedicated sampling metadata
- dedicated path family ownership

## 5.3 Scattering implementation choices

There are three realistic designs.

### Scope S1: first-order surface scattering only

Behavior:

- sample a scattering lobe at a hit surface
- accumulate contributions to receivers
- no scattering after scattering
- no mixed D-S-D / R-S-R yet

Fit with current code:

- good

Difficulty:

- medium-high

### Scope S2: specular + scattering mixed paths

Behavior:

- allow chains like `R-S`, `T-S`, `S-R`

Fit with current code:

- moderate

Difficulty:

- high

### Scope S3: arbitrary higher-order scattering chains

Behavior:

- repeated scattering events
- Monte Carlo path tracing style transport

Fit with current code:

- poor as a first implementation

Difficulty:

- very high

### Recommendation

Start with S1 only.

That gives user-visible value without breaking the current deterministic specular architecture.

## 5.4 Recommended scattering architecture

### A. New scattering solver package

Recommended package:

- `witwin/channel/trace/scattering/`

Suggested modules:

- `api.py`
- `materials.py`
- `sampling.py`
- `field.py`
- `paths.py`

Reason:

- keeps layering clean
- avoids overloading `trace/reflection`
- lets scattering evolve independently of EPC logic

### B. Material model

For first-order scattering, add minimal parameters:

- `scattering_gain`
- `roughness`
- `scattering_model`

Possible first model choices:

1. Lambertian-like hemisphere model
2. directive lobe around the specular direction
3. radio rough-surface lobe with controllable spread

Recommendation:

- use a directional lobe around the specular direction, not pure Lambertian

Reason:

- pure Lambertian is simple but usually too unrealistic for radio surfaces
- a specular-centered lobe is a better bridge from current reflection physics

### C. Field monitor integration

For field monitors, scattering can be implemented as:

- sample scattering directions from hit surfaces
- trace those contributions to monitor receivers
- accumulate into `field.scattering`

This is similar in spirit to reflection field accumulation, but it should not rely on exact image-source replay.

### D. Path monitor integration

For path monitors:

- every accepted scattered contribution becomes a path entry
- `InteractionType.SCATTERING` is written into the proper slot

This fits the existing `PathResult` structure well.

### E. Result ownership

Extend result ownership rules to include:

- `a_scat`: owns all paths with at least one scattering event and no diffraction event initially
- later mixed ownership can be widened if needed

## 5.5 What must change for scattering

### A. Config

Add controls such as:

- `enable_scattering`
- `scattering_n_rays` or `scattering_n_samples`
- `max_scattering_events`
- `scattering_weight_threshold`
- optional `scattering_model`

### B. Scene/runtime material payload

Add per-triangle or per-structure fields:

- `roughness`
- `scattering_gain`
- `scattering_model`

### C. Field result schema

Add:

- `field.scattering`
- `vector.scattering`
- `jones.scattering`

### D. Path metadata

Add:

- scattering sampling provenance
- per-family counts
- sampling model labels

### E. Tests

New test groups should include:

- isotropic/hemispherical sanity checks
- energy monotonicity / threshold pruning
- roughness sensitivity
- path monitor interaction typing

## 5.6 Scattering risk assessment

### Low risk

- config/result schema extension
- interaction type coding
- first-order path metadata

### Medium-high risk

- sampling model correctness
- energy accounting
- variance / convergence behavior

### High risk

- mixed specular + diffuse higher-order chains
- differentiable scattering parameter gradients with acceptable noise
- performance on dense field monitors

## 5.7 Recommended scattering MVP

The best first scattering release for this repo is:

1. first-order surface scattering only
2. specular-centered lobe model
3. field and path outputs
4. no scattering-after-scattering
5. no mixed diffraction-scattering families yet

This is the smallest credible scattering feature.

---

## 6. Recommended Phased Implementation Plan

## Phase 0: Schema And Material Preparation

### Goal

Create the minimum infrastructure both features need.

### Changes

- extend `TraceConfig` with transmission/scattering controls
- extend scene runtime material tables for thickness and scattering params
- extend `MonitorField`, `MonitorJones`, `MonitorVector` with:
  - `transmission`
  - `scattering`
- update total composition semantics

### Files

- `witwin/channel/config.py`
- `witwin/channel/result.py`
- `witwin/channel/scene/compile.py`
- `witwin/channel/scene/runtime.py`
- `witwin/channel/monitors/field/trace_field.py`

### Tests

- result payload construction tests
- config round-trip tests
- scene material table compile/runtime tests

### Exit criteria

- transmission/scattering components exist in result objects
- new material fields can reach runtime tables
- no solver logic added yet

## Phase 1: Transmission MVP

### Goal

Ship useful penetration behavior fast.

### Changes

- add Fresnel transmission helpers
- add thin-slab transmission operator
- replace binary LoS blocking with transmission-aware traversal
- expose `field.transmission`
- expose direct transmission paths in `PathMonitor`

### Files

- `witwin/channel/utils/material.py`
- `witwin/channel/trace/materials.py`
- `witwin/channel/trace/los.py`
- `witwin/channel/monitors/field/trace_field.py`
- `witwin/channel/monitors/path/collectors.py`
- `witwin/channel/monitors/path/trace_path.py`

### Tests

- single wall attenuation
- multi-wall attenuation composition
- path monitor transmission interaction coding
- regression that existing pure-LoS and pure-reflection tests still pass when transmission is disabled

### Exit criteria

- blocked LoS behind penetrable structures no longer collapses to zero
- users can inspect transmission as a separate component

## Phase 2: Deterministic Specular Transmission Families

### Goal

Promote transmission from "LoS attenuation" to a first-class path family.

### Changes

- add transmission path discovery / replay
- allow deterministic specular-through-material chains
- extend field monitor accumulation to include those chains

### Suggested new package

- `witwin/channel/trace/transmission/`

### Tests

- single pane / slab direct transmission
- one transmission plus one reflection family
- path sorting / type slots for transmission chains

### Exit criteria

- `PathMonitor` can return non-LoS transmission paths
- `FieldMonitor` separates reflection vs transmission contributions

## Phase 3: Transmission-Aware Diffraction

### Goal

Make diffraction visibility and mixed families aware of penetration.

### Changes

- transmission-aware segment visibility
- transmission-aware prefix/source attenuation
- optional transmission in diffraction state construction

### Files

- `witwin/channel/trace/diffraction/api.py`
- `witwin/channel/trace/diffraction/field.py`
- `witwin/channel/trace/diffraction/geometry/*`
- `witwin/channel/trace/diffraction/builders/__init__.py`

### Tests

- wedge behind a penetrable slab
- transmission + diffraction coexistence
- mixed budget metadata remains coherent

### Exit criteria

- diffraction is no longer over-occluded by penetrable materials

## Phase 4: Scattering MVP

### Goal

Ship first-order surface scattering as a separate family.

### Changes

- add `trace/scattering/` package
- add scattering material model and sampling
- accumulate first-order scattering into field monitors
- collect first-order scattering paths

### Files

- `witwin/channel/trace/scattering/api.py`
- `witwin/channel/trace/scattering/materials.py`
- `witwin/channel/trace/scattering/sampling.py`
- `witwin/channel/trace/scattering/field.py`
- `witwin/channel/trace/scattering/paths.py`
- `witwin/channel/monitors/field/trace_field.py`
- `witwin/channel/monitors/path/trace_path.py`

### Tests

- first-order scattering non-zero sanity scene
- roughness / gain sensitivity
- path interaction typing uses `SCATTERING`
- total field includes scattering component only when enabled

### Exit criteria

- users can separately inspect `scattering` on field and path outputs
- implementation is first-order only

## Phase 5: Mixed Specular/Scattering Families

### Goal

Add limited mixed chains once first-order scattering is stable.

### Recommended first mixed families

- `R -> S`
- `T -> S`
- `S -> R`

Do **not** start with:

- `S -> S`
- `D -> S -> D`
- arbitrary recursive scattering

### Files

- scattering package
- path collectors
- result metadata

### Tests

- mixed-family ownership
- budget/pruning behavior
- convergence sanity

### Exit criteria

- mixed scattering families exist with explicit budget control

## Phase 6: Performance, Native Kernels, And Higher Order

### Goal

Only after correctness is stable:

- optimize hot loops
- decide whether native kernels are worth adding
- consider higher-order scattering or full refraction

### Important note

Neither full refraction nor higher-order scattering should block the earlier phases.

---

## 7. Suggested Public API Additions

## 7.1 TraceConfig

Recommended additions:

```python
enable_transmission: bool = False
max_transmissions: int = 2
transmission_material: Mapping[str, Any] | None = None
use_scene_materials_for_transmission: bool = True
transmission_weight_threshold: float = 0.0
default_material_thickness: float | None = None

enable_scattering: bool = False
scattering_n_samples: int = 2048
max_scattering_events: int = 1
scattering_material: Mapping[str, Any] | None = None
use_scene_materials_for_scattering: bool = True
scattering_weight_threshold: float = 0.0
scattering_model: str = "specular_lobe"
```

## 7.2 Result field components

Recommended field components after Phase 0:

- `los`
- `reflection`
- `transmission`
- `diffraction_direct`
- `diffraction_mixed`
- `diffraction`
- `scattering`
- `total`

This is the smallest clean extension of the current schema.

## 7.3 Path interaction codes

No enum redesign is needed immediately because the reserved values already exist:

- `TRANSMISSION = 4`
- `SCATTERING = 8`

---

## 8. File-Level Impact Summary

## High-probability modified files

- `witwin/channel/config.py`
- `witwin/channel/result.py`
- `witwin/channel/scene/compile.py`
- `witwin/channel/scene/runtime.py`
- `witwin/channel/utils/material.py`
- `witwin/channel/trace/materials.py`
- `witwin/channel/trace/los.py`
- `witwin/channel/monitors/field/trace_field.py`
- `witwin/channel/monitors/path/collectors.py`
- `witwin/channel/monitors/path/trace_path.py`
- `witwin/channel/trace/diffraction/api.py`
- `witwin/channel/trace/diffraction/field.py`
- `witwin/channel/trace/diffraction/builders/__init__.py`

## Likely new packages

- `witwin/channel/trace/transmission/`
- `witwin/channel/trace/scattering/`
- `tests/transmission/`
- `tests/scattering/`

---

## 9. Testing Strategy

## 9.1 Transmission tests

Add dedicated tests for:

- normal-incidence slab transmission
- oblique-incidence slab transmission
- LoS through one wall
- LoS through multiple walls
- transmission path typing in `PathMonitor`
- transmission + reflection coexistence
- transmission + diffraction coexistence

## 9.2 Scattering tests

Add dedicated tests for:

- first-order scattering non-zero sanity
- scattering disabled/enabled toggles
- roughness sensitivity
- scattering path typing
- field vs path consistency on simple scenes

## 9.3 Regression focus

Existing reflection and diffraction regressions must continue to pass with:

- `enable_transmission=False`
- `enable_scattering=False`

That should be the default stabilization mode during rollout.

---

## 10. Final Recommendation

If the goal is to add both features with the least rework, the best plan is:

1. Do transmission first.
2. Start transmission as thin-slab, not full Snell refraction.
3. Treat scattering as a separate solver family, not as a reflection patch.
4. Start scattering as first-order only.
5. Only after both are stable, widen diffraction mixed-family builders.

### In concrete terms

Recommended delivery sequence:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 4
5. Phase 3
6. Phase 5
7. Phase 6

Reason:

- Phase 0/1/2 unlock practical transmission quickly.
- Phase 4 gives a usable scattering feature without destabilizing mixed-family state logic.
- Phase 3 and Phase 5 are the expensive combinatorial stages and should only happen once base families are stable.

### Bottom line

The current codebase is already well prepared for transmission.
It is only partially prepared for scattering.

So the right implementation strategy is:

- **Transmission as the next feature**
- **Scattering as the feature after that**
- **Mixed T/R/D/S chains only after both base families are individually stable**

---

## 11. Related Existing Internal Docs

These older docs remain useful as background, but they no longer match the current codebase exactly:

- `docs/dev/archive/superseded/transmission-refraction-analysis.md`
- `docs/dev/plans/path-monitor-design.md`
- `docs/dev/archive/superseded/jones-vector-material-model-plan.md`

The main differences versus those older docs are:

- `PathMonitor` is now implemented, not just planned
- field/path orchestration is already shared in `Tracer.trace()`
- Jones/material transport is more mature than before
- the remaining blockers are now mostly schema, traversal, and family-builder issues
