# Research-Oriented Feature Roadmap

## Purpose

This document captures the research-facing feature roadmap for the channel solver and RayD-backed runtime. It is not a user-facing inventory. The public feature inventory remains in `FEATURE_LIST.md`.

The goal is not to clone Sionna RT or Wireless InSite feature-for-feature. The goal is to build a GPU-native, differentiable, radio-first platform that is stronger in the areas that matter most for current academic work:

- Mixed interaction chains
- Differentiable inverse problems
- Path-level interpretability and auditability
- Dynamic scene optimization
- RF digital twin workflows

## Architecture Readiness Assessment

### What The Current Architecture Already Supports Well

- Stable public decomposition through `Scene + Tracer + Result`.
- RayD-native query abstraction through `query_backend="rayd"`.
- RayD-native wedge compilation/runtime through `wedge_backend="rayd"`.
- GPU-first packed wedge and edge data caches.
- Differentiable runtime vertex updates without forcing topology rebuilds.
- Diffraction state arrays with path history, ownership, and audit metadata.
- Explicit solver controls for path budgeting and approximation guardrails.
- Clear separation between declarative scene assembly and runtime compilation.

These are strong foundations for research-oriented extensions. In particular, backend neutrality, state audits, and gradient-preserving scene updates are already aligned with future academic needs.

### What The Current Architecture Does Not Yet Isolate Cleanly Enough

- Interaction families are still encoded mostly inside diffraction-specific builders rather than inside a general mixed-interaction path graph or interaction registry.
- Reflection, diffraction, and future transmission/scattering interactions do not yet share a unified path-state representation.
- Research workflows such as calibration, inverse solving, benchmark export, and uncertainty propagation are not first-class APIs.
- Dynamic scene studies still rely on re-running the current tracer rather than a dedicated time-dependent path identity and event model.
- Material modeling is still relatively narrow compared with what is needed for inverse problems and higher-frequency studies.
- The public tracer path still hardcodes some solver choices, including 2D reflection-mode usage in mixed-path tracing.

### Bottom-Line Assessment

The architecture is directionally correct for academic expansion.

It already contains the most valuable long-term foundations:

- Backend neutrality
- GPU-first data flow
- Differentiable scene updates
- Auditable path states

However, the next stage should shift from "adding more diffraction logic" to "building a general research platform layer" on top of the current solver core.

## Priority Tiers

### Tier P0

Features that most directly strengthen the platform against Sionna RT in academic use cases.

- General 3D mixed-interaction solver
- Inverse-problem and calibration toolkit
- Public benchmark and reproducibility suite
- Dynamic path identity tracking
- Path-level interpretable outputs

### Tier P1

Features that strongly increase publication value and research adoption.

- RIS and metasurface co-design
- Point-cloud and reconstructed-scene support
- Wideband, dispersive, and THz modeling
- ISAC and sensing-oriented outputs
- Dataset generation and supervision export

### Tier P2

Features that deepen the platform and create defensible long-term differentiation.

- Uncertainty-aware RT and calibration
- Multi-GPU and city-scale execution
- Learned surrogate and multi-fidelity solver modes
- Differentiable diffuse scattering and rough surfaces
- Antenna and array-body co-optimization

## Detailed Feature List

### 1. General 3D Mixed-Interaction Path Solver

#### Motivation

The strongest research differentiator is not isolated reflection or isolated diffraction. It is a solver that can model and audit mixed interaction chains in a controllable and differentiable way.

#### Target Capabilities

- Native 3D support for mixed `R`, `D`, `T`, and scattering chains.
- Explicit support for:
  - `R -> D`
  - `D -> R`
  - `D -> R -> D`
  - `R -> D -> R`
  - Higher-order alternating chains
  - Future `T` insertion and suffix/prefix variants
- Unified path-state ownership across all interaction types.
- Exact path-sequence export for every retained state.
- Interaction-depth budgets per family instead of diffraction-only controls.

#### Required Architecture Changes

- Introduce a general path-state representation independent of diffraction-only semantics.
- Move from diffraction-builder-centric expansion to an interaction registry or path-graph scheduler.
- Separate path generation, path pruning, and field accumulation into explicit stages.

#### Why Academia Cares

- Better modeling of realistic mixed urban and indoor paths.
- Easier publication story than pure reflection or first-order diffraction.
- Strong fit for optimization, interpretability, and ablation studies.

### 2. Inverse-Problem And Calibration Toolkit

#### Motivation

Academic demand is shifting from forward simulation to scene, material, and device inference from measurements.

#### Target Capabilities

- Optimize geometry from CIR/CFR/RSS/phase observations.
- Optimize material parameters such as relative permittivity, conductivity, and roughness.
- Joint optimization of:
  - transmitter pose
  - receiver pose
  - object pose
  - material parameters
  - RIS parameters
- Gradient-safe losses for sparse, noisy, and partial observations.
- Calibration workflows for real datasets.

#### Required Architecture Changes

- Add inverse-problem APIs instead of requiring ad hoc scripts.
- Define observation adapters for CIR, CFR, path lists, radio maps, and sensing targets.
- Standardize differentiable parameter containers for geometry and materials.
- Add optimization-ready batching and checkpointing support.

#### Why Academia Cares

- RF digital twins
- Localization
- Material estimation
- Scene calibration
- Joint communication-and-sensing inference

### 3. Public Benchmark And Reproducibility Suite

#### Motivation

Feature parity claims are weak. Reproducible benchmarks create platform authority.

#### Target Capabilities

- Canonical geometry benchmarks:
  - single wedge
  - double wedge
  - triple wedge
  - mixed reflection-diffraction scenes
  - slanted-edge scenes
- External comparisons against:
  - Sionna RT
  - local reference formulas
  - measured scenes when available
- Metrics for:
  - path discovery
  - complex field error
  - delay error
  - angle error
  - gradient agreement
  - memory
  - runtime
- Public benchmark manifests and versioned outputs.

#### Required Architecture Changes

- Stable export schema for path lists and audit metadata.
- Dedicated benchmark runners instead of scattered test scripts.
- Dataset-compatible serialization for path-level outputs.

#### Why Academia Cares

- Reproducibility
- Fair comparisons
- Easier adoption in papers and theses

### 4. Dynamic Scene And Path Identity Engine

#### Motivation

Mobility, blockage, V2X, UAV, and robotics research all need time-consistent path structure, not only repeated static snapshots.

#### Target Capabilities

- Persistent path IDs across timesteps.
- Birth/death logging for paths.
- Event-aware path transitions:
  - visibility changes
  - edge changes
  - reflection-surface changes
- Support for moving objects, blockers, vehicles, and articulated actors.
- Differentiable motion optimization.
- Time-indexed channel and path audit export.

#### Required Architecture Changes

- Add a time-dependent scene/runtime layer.
- Track correspondence between path states across frames.
- Separate dynamic topology changes from dynamic vertex changes.

#### Why Academia Cares

- Mobility and Doppler studies
- Dynamic blockage
- ISAC
- V2X and robotics digital twins

### 5. Path-Level Interpretable Outputs

#### Motivation

Most tools produce channels. Researchers increasingly need explanations of where those channels come from.

#### Target Capabilities

- Path attribution by interaction family and object.
- Edge importance maps.
- Surface responsibility maps.
- Per-path contributions to:
  - total field
  - delay spread
  - angular spread
  - Doppler spread
- Geometry sensitivity maps showing which scene elements dominate gradients.
- Semantic radio maps with path-cause annotations.

#### Required Architecture Changes

- Enrich existing audit structures with object and semantic references.
- Add object-level accumulation and contribution summaries.
- Define structured explanation outputs in `Result`.

#### Why Academia Cares

- Better debugging
- Better ablations
- Better physical interpretation
- More compelling publications

### 6. RIS And Metasurface Co-Design

#### Motivation

RIS remains a high-volume research area, but most stacks still treat it as a specialized add-on rather than a general optimization target.

#### Target Capabilities

- RIS placement optimization.
- RIS orientation and aperture optimization.
- Multi-mode reradiation modeling.
- Wideband and near-field RIS support.
- Joint optimization of:
  - RIS phase profile
  - base station beamforming
  - user placement
  - scene geometry
- Differentiable RIS losses for coverage and link metrics.

#### Required Architecture Changes

- Add RIS objects to the declarative scene model.
- Expose RIS parameters as differentiable controls.
- Integrate reradiation as a first-class interaction family.

#### Why Academia Cares

- Massive literature volume
- Strong optimization story
- Easy benchmark story against existing stacks

### 7. Point Cloud, Reconstructed Scene, And Semantic Scene Support

#### Motivation

Academic workflows increasingly start from imperfect reconstructions rather than curated CAD scenes.

#### Target Capabilities

- Direct scene ingestion from point clouds.
- Point cloud to surface/wedge candidate extraction.
- Support for reconstructed triangle meshes and semantic meshes.
- Material priors from semantic labels.
- Optional hybrid scene representations:
  - point cloud
  - mesh
  - semantic surface graph

#### Required Architecture Changes

- Add new geometry adapters beyond clean triangle meshes.
- Add scene preprocessing pipelines for edge and wedge extraction from noisy geometry.
- Support uncertainty tags on reconstructed geometry.

#### Why Academia Cares

- Real-world deployment
- RF digital twins
- Robotics and embodied AI

### 8. Wideband, Dispersive, And THz Modeling

#### Motivation

Many research questions now sit outside narrowband sub-6 assumptions.

#### Target Capabilities

- Frequency-dependent materials.
- Dispersion-aware path accumulation.
- Wideband-consistent path parameter export.
- THz-specific effects:
  - molecular absorption
  - stronger roughness sensitivity
  - frequency-selective reflection and diffraction
- Unified APIs for sub-6, mmWave, and THz studies.

#### Required Architecture Changes

- Replace scalar material records with frequency-aware material models.
- Add vectorized multi-frequency tracing and accumulation modes.
- Separate narrowband path discovery from wideband field evaluation where possible.

#### Why Academia Cares

- THz communications
- integrated sensing
- wideband localization
- material-sensitive propagation studies

### 9. ISAC And Sensing-Oriented Outputs

#### Motivation

Researchers often need radar-like outputs and target-aware path decomposition, not only communication channels.

#### Target Capabilities

- Range-Doppler-angle exports.
- Path-to-target attribution.
- Micro-Doppler support for moving articulated objects.
- Sensing-aware output heads for target detection and localization tasks.
- Differentiable sensing losses.

#### Required Architecture Changes

- Add sensing-result schemas to `Result`.
- Standardize target objects and sensing query APIs.
- Track path-target interaction metadata.

#### Why Academia Cares

- ISAC
- radar-communication co-design
- robotics perception

### 10. Dataset Generation And Supervision Export

#### Motivation

If researchers use the platform to generate training and evaluation datasets, the platform gains long-term ecosystem leverage.

#### Target Capabilities

- Batch scene randomization.
- Structured export of:
  - path lists
  - CIR/CFR
  - AoA/AoD
  - Doppler
  - semantic/object labels
  - calibration ground truth
- Export formats for PyTorch pipelines.
- Reproducible dataset manifests.

#### Required Architecture Changes

- Standardized export module and schemas.
- Scene sampling and randomized configuration tools.
- Efficient batched tracing APIs.

#### Why Academia Cares

- Learning-based channel prediction
- inverse problems
- sensing supervision
- reproducible evaluation

### 11. Uncertainty-Aware Ray Tracing

#### Motivation

Real scenes, materials, and measurements are uncertain. Current ray tracing pipelines usually ignore this.

#### Target Capabilities

- Parameter uncertainty propagation.
- Path existence confidence.
- Confidence bands on radio maps and path metrics.
- Monte Carlo and local linearization uncertainty modes.
- Posterior estimation after calibration.

#### Required Architecture Changes

- Add uncertainty-aware parameter containers.
- Extend result schemas to include confidence outputs.
- Support repeated trace/evaluate workflows with shared path infrastructure.

#### Why Academia Cares

- Trustworthy digital twins
- robust planning
- scientific reporting with uncertainty bounds

### 12. Antenna And Array Co-Design

#### Motivation

Most channel stacks treat antenna patterns as fixed inputs. Research users increasingly need antenna and array optimization inside the propagation loop.

#### Target Capabilities

- Differentiable antenna pose and orientation.
- Array geometry optimization.
- Polarization-aware array modeling.
- Near-field array behavior.
- Optional approximations for coupling-sensitive studies.

#### Required Architecture Changes

- Expand the declarative scene model to treat antennas as richer objects.
- Promote polarization and array metadata to first-class inputs.
- Add channel-to-array projection modules beyond scalar field accumulation.

#### Why Academia Cares

- array calibration
- beam management
- near-field MIMO
- integrated RIS and array studies

### 13. Differentiable Diffuse Scattering And Rough Surfaces

#### Motivation

Diffuse scattering is important, but academic users increasingly want roughness and scattering parameters to be optimized or inferred.

#### Target Capabilities

- Differentiable roughness parameters.
- Anisotropic scattering models.
- Mixed scattering-diffraction chains.
- Multi-frequency rough-surface behavior.
- Learned scattering priors as optional plug-ins.

#### Required Architecture Changes

- Add a first-class scattering family to the interaction scheduler.
- Extend material records and accumulation models.
- Add audit semantics for scattered paths.

#### Why Academia Cares

- mmWave and THz studies
- urban realism
- inverse material estimation

### 14. Multi-Fidelity And Learned Surrogate Modes

#### Motivation

Researchers need both exactness and scale. A single solver mode is not enough.

#### Target Capabilities

- Explicit exact, bounded, and surrogate-assisted modes.
- Learned refinement of selected path families.
- Adaptive path budgeting based on geometry and sensitivity.
- Distillation of expensive mixed-path solvers into fast approximators.

#### Required Architecture Changes

- Generalize current solver-mode controls into a policy layer.
- Add hooks for learned models without polluting the core solver.
- Record fidelity provenance in all outputs.

#### Why Academia Cares

- large-scale studies
- training-data generation
- real-time planning

### 15. Large-Scene And Multi-GPU Execution

#### Motivation

A serious alternative to commercial planning tools eventually needs scale.

#### Target Capabilities

- Multi-GPU path tracing and accumulation.
- Spatial tiling and out-of-core scene handling.
- Incremental scene updates in large environments.
- Batched user/receiver evaluation at scale.

#### Required Architecture Changes

- Separate geometry residency from scene ownership.
- Add distributed accumulation and batching infrastructure.
- Define large-scene cache invalidation rules.

#### Why Academia Cares

- city-scale experiments
- campus-scale digital twins
- high-throughput dataset generation

## Recommended Architecture Refactors Before Major Feature Expansion

### Refactor 1: Generalize Path States

Move from diffraction-centric state arrays toward a general interaction-state model that can represent reflection, diffraction, transmission, scattering, and RIS reradiation without special-case ownership rules.

### Refactor 2: Add A Research Workflow Layer

Introduce dedicated modules for:

- calibration
- optimization
- benchmark runners
- dataset export
- uncertainty evaluation

This avoids turning `Tracer` into a catch-all object.

### Refactor 3: Expose 3D Mixed Tracing Modes Explicitly

Current mixed-path tracing should evolve from fixed internal mode choices to explicit solver controls that support 2D and 3D path families consistently.

### Refactor 4: Upgrade Material Modeling

Replace narrow scalar material records with richer, frequency-aware, optimization-friendly material objects.

### Refactor 5: Add Stable Scene Element IDs

Future interpretability, dynamic tracking, and inverse calibration all benefit from persistent object, face, edge, and wedge identities.

## Suggested Near-Term Execution Order

### Phase A

- General path-state abstraction
- 3D mixed-interaction tracing
- benchmark/export infrastructure

### Phase B

- inverse-problem toolkit
- dynamic path identity engine
- richer material parameterization

### Phase C

- RIS co-design
- reconstructed-scene support
- sensing outputs

### Phase D

- uncertainty propagation
- surrogate modes
- large-scene execution

## Success Criteria

The platform can claim clear academic differentiation when it can demonstrate all of the following:

- Stronger mixed interaction support than Sionna RT in public benchmarks.
- Better path-level interpretability and auditability than mainstream alternatives.
- End-to-end differentiable optimization for geometry, materials, and RIS controls.
- Reproducible benchmark and dataset tooling used by external researchers.
- A clear RF digital twin story from reconstructed scenes to calibrated channels.
