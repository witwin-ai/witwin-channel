# Changelog

All notable changes to `witwin-channel` are documented in this file.

## [Unreleased]

## [0.6.0] - 2026-11-12

Channel 0.6.0 is one breaking release. The radiomap solvers move into one
package, BDPT moves to an experimental tier, and the coherent solvers
(`radiomap.Coherent`, `path`) gain a set of physics fixes. Correctness is
checked against full-wave (Tidy3D) reference fields in
`tests/accuracy/test_fullwave.py`, not against other ray tracers; the cases,
their metrics and the remaining gaps are in `docs/dev/fullwave-validation.md`.
Upgrading from 0.5: [`docs/migration-0.6.md`](docs/migration-0.6.md).

### Breaking changes

- **The radiomap package (ADR-074).** `witwin.channel.deterministic` and
  `witwin.channel.montecarlo.basic` are replaced by `witwin.channel.radiomap`.
  `radiomap.solve(scene, config, *, reference_frequency_hz)` picks the
  estimator from the configuration type: `radiomap.Coherent` (was
  `deterministic.Config`) returns a `radiomap.CoherentRadioMap` (was
  `deterministic.Result`), and `radiomap.Stochastic` (was
  `montecarlo.basic.Config`) returns a `radiomap.RadioMap` (was
  `montecarlo.basic.Result`); any other config raises `TypeError` before any
  work. `deterministic.PathTable` is `radiomap.PathTable`. Every old
  configuration has an exact new spelling with bit-identical results.
- **The experimental tier (ADR-074).** `witwin.channel.montecarlo.bdpt` is now
  `witwin.channel.experimental.bdpt` with the same names. `experimental.*` is
  importable and tested but outside the public API snapshot, so its names,
  options and results may change in any release.
- `radiomap.Coherent(combine="field" | "power")` replaces
  `deterministic.Config(coherent=True | False)`; `coherent=` is refused.
  `scattering_coherent` and BDPT's `coherent` flag are unchanged.
- Result metadata uses one vocabulary in every solver:
  `metadata["estimator"]` is `"coherent"` (`path`, `radiomap.Coherent`) or
  `"stochastic"` (`radiomap.Stochastic`, BDPT) instead of `"enumerated"` /
  `"sampled"`, and `metadata["combine"]` is `"field"` / `"power"` / `"none"`
  instead of `"coherent"` / `"incoherent"` / `"none"`. The Coherent
  `metadata["coherent"]` key is removed (read `metadata["combine"]`), and
  `diagnostics["coherent"]` is `diagnostics["combine"]`.
- `PathTable.interaction_position` and `PathTable.interaction_normal` (`(K, 3)`)
  are removed. Use `interaction_positions[:, 0]` and `interaction_normals[:, 0]`,
  which hold the same values bit for bit.
- Removed import paths (`witwin.channel.deterministic`,
  `witwin.channel.montecarlo`, `.montecarlo.basic`, `.montecarlo.bdpt`) raise
  `ImportError` naming the replacement. There are no compatibility aliases.
- **One grid axis order (ADR-061).** Every solver returns a single
  `ReceiverGrid` as `[T, rows, cols]`, Core's row-major order. Before,
  Deterministic and BDPT returned `[T, cols, rows]` and MC Basic a flat
  `path_gain` in column order, so `path_gain[:, k]` named different receivers
  per solver.
- **Coincident endpoints are refused (ADR-067).** A point receiver or any
  `ReceiverGrid` sample within `max(wavelength / (4 pi), 2^-20 * coordinate
  magnitude)` of a transmitter (6.8 mm at 3.5 GHz) raises `ValueError` naming
  the receiver, the grid cell `(row, column)` and the transmitter, before
  endpoint upload. Before, only a point receiver at exactly the transmitter
  position was refused; a grid cell on a transmitter returned near-zero values
  in Deterministic/Path and a 4.6e7 path gain in MC Basic/BDPT. The
  point-receiver message reads "coincides with transmitter". Migration: shift
  a radio-map grid origin by half a spacing when a sample falls on a
  transmitter.
- Propagation consumer `CONTRACT_VERSION` moves from 6 to 7 (see Added). A
  consumer that pins `CONTRACT_VERSION == 6` or the exact capability field set
  must accept 7 and the new fields; requests that do not set `discovery` keep
  their rows.
- Solver configurations are validated more strictly before any native work:
  every scalar field is checked against its declared type (int fields reject
  `bool`, `float` and `str`; float fields must be finite, so NaN no longer
  passes a `<= 0` check), and every range or vocabulary error names the
  owning class and field (`radiomap Coherent.max_depth ...`).

### Added

- `witwin.channel.importers` (ADR-048): `load_mitsuba` imports Mitsuba 3 /
  Sionna RT XML scenes (ply, obj, serialized, rectangle and cube shapes,
  `to_world` transforms, ITU and explicit radio materials); `load_mesh`,
  `read_ply`, `read_obj` and `read_gltf` import PLY, OBJ and glTF 2.0 files.
  Duplicate vertices are welded by default and reported. Output is plain
  `witwin.core` structures.
- `witwin.channel.presets` (ADR-048): `itu_material` builds ITU-R P.2040
  materials as dispersive `PhysicalMaterial` values with enforced frequency
  ranges (`dispersive=False` freezes them at one frequency for wideband
  replay); `tr38901_pattern` and `dipole_pattern` build Core custom antenna
  patterns.
- `examples/` with runnable radiomap, path/CIR/CFR, material-calibration and
  moving-transmitter scripts, and user guides under `docs/user/` (quickstart,
  scene import, materials and antennas, troubleshooting).
- `Config.discovery` on `path` and `radiomap.Coherent` (`"auto"`,
  `"exhaustive"`, `"rays"`) selects the reflection candidate source; result
  metadata publishes `discovery` (requested source and source per depth) and
  `complete` (true only when every reflection depth came from a provably
  complete source; `"rays"` is never complete).
- Result-semantics metadata (ADR-062): every solver publishes `estimator`,
  `quantity` (per component, `point_sample` or `cell_average`), `combine` and
  `complete`.
- Typed result metadata: `path.PathMetadata`,
  `radiomap.CoherentRadioMapMetadata`, `radiomap.RadioMapMetadata` and
  `experimental.bdpt.Metadata` are `TypedDict`s naming the stable keys of
  each solver's `Result.metadata` (including the `kernel` record). At run time
  the metadata is the same plain `dict`. Each `solve` annotates
  `reference_frequency_hz: float | torch.Tensor`, and
  `PathResult.from_ragged(metadata=...)` accepts any `Mapping`.
- Propagation consumer contract 7: `PropagationRequest.discovery` (same
  vocabulary, default `"auto"`), `PropagationEvaluation.complete` and
  `PropagationCapabilities.discovery_modes`.
- `propagation.consumer.check_endpoint_separation(sources, sinks,
  reference_frequency_hz=...)` refuses a source/sink pair closer than the
  coincidence tolerance, naming the pair, with one one-byte device read.
  `evaluate` and `reevaluate` never read endpoint coordinates, so separation is
  a documented caller precondition there. `CONTRACT_VERSION` and every
  existing schema are unchanged by it.
- `PropagationDiagnostics.transmission_dropped_rows` and
  `metadata["transmission"]["dropped_rows_over_max_depth"]` (Path, Coherent,
  Stochastic, BDPT) count source-sink pairs whose straight segment crosses more
  walls than `max_depth` (see Fixed).
- `radiomap.Stochastic` (MC Basic) solves point receivers, several receiver
  grids and grids mixed with point receivers (ADR-055). Reflection rays come
  from a seeded, randomly rotated Fibonacci lattice folded into the
  half-spaces that reach the scene, and RayD traces a bounded final segment
  instead of an unbounded extra bounce (ADR-054); the seed now changes the
  reflection estimate. Point receivers take reflection and diffraction in
  capture spheres.
- `experimental.bdpt`: a fused native wavefront sampler with counter-based
  random numbers and sorted accumulation (ADR-049); rough-face Kirchhoff
  scattering and export rows on the wavefront; `ad_mode="jvp"` / `"vjp"`
  through the wavefront for material, roughness/table, frequency and
  transmitter-power leaves (endpoint-position and mesh-vertex leaves are
  refused); several receiver grids and grids mixed with point receivers.
- Scenes request a RayD trace backend (OptiX or pure CUDA, ADR-035) and report
  the selected one; the GPU is checked at scene compile.
- `witwin-channel-doctor` (also `python -m witwin.channel doctor`) and
  deployment diagnostics schema v2, derived from the build's cubin/PTX
  targets, the active GPU, the CUDA driver and the OptiX probe.
- Full-wave reference suite: 15 new Tidy3D reference maps (lossy and
  multilayer walls, a second frequency, concrete and dielectric cubes, a PEC
  trihedral, double edges, a wall with a column, a four-block canyon) and the
  gate in `tests/accuracy/test_fullwave.py`, with a ceiling per metric and a
  strict xfail per known gap.
- Developer tooling: `scripts/dev_build.sh` / `scripts/dev_env.sh`, a `dev`
  extra, `tools/relock_rayd.py`, a Dockerfile and `CONTRIBUTING.md`; generated
  native binding manifest and source inventory checks.

### Fixed

Each numerical fix below has a design record under `docs/dev/standards/` with
its full-wave before/after evidence.

- **Reflection in-face test (D1, ADR-068).** The containment slack was
  barycentric (1e-4), so it grew with the face: a reflection point 3.96 mm past
  the edge of a 40 m wall still counted. It is now a metric slack of 8 float
  roundings, shared by every reflection validator. `double_edge_z` RSB excess
  4.32 -> -0.03 dB.
- **Oblique face term (D2, ADR-072).** The diffraction face coefficients use
  the 3D incidence cosine and a reciprocal specular dyadic in the edge-fixed
  bases instead of the edge-plane angle and a diagonal operator.
  `cube_concrete_z` RSB excess 3.80 -> 0.47 dB.
- **Layered face term (D3, ADR-071).** The diffraction face term reflects with
  the face's slab or layer stack, the coefficient the specular path uses,
  instead of a half-space Fresnel coefficient. The residual reflected field at
  the RSB drops from 0.05 / 0.38-0.42 / 0.54-0.60 (slab / glass / three-layer
  wall) to 0.002.
- **Chain completeness (D4, ADR-069).** Interaction chains carry up to
  `max_depth` reflections around the diffraction (they stopped one short), and
  both legs of a chain must leave the wedge into free space (9,803 rows ran
  through a solid cube). `wall_column_z` RSB excess 5.35 -> 0.70 dB.
  `capabilities()["solvers"]["deterministic"]["max_reflections_in_coupled_path"]`
  is 5.
- **Coincident edges (D5, ADR-073).** Where separate structures share an edge
  (plate junctions, building bases on a ground), each edge's wedge is built at
  compile time from every face that meets the edge line, across structures,
  with each face's own material; edges with no free sector are dropped.
  `trihedral_x` shadow 3.04 dB / NMSE 0.908 -> 1.67 dB / 0.115.
- **GO-boundary sample (D6, ADR-070).** A receiver exactly on a reflection
  shadow boundary keeps the side the geometric tests choose; only the pole term
  takes the one-sided limit, and near-pole quantities come from one small
  angle. On-boundary |T| 3.7e-5 -> 8.2e-4 (lit 8.2e-4).
- **D-D leg along a sheet (ADR-075).** A double-diffraction leg that runs along
  a plate to a junction wedge is evaluated on the junction's side (two thirds
  of those legs took the wrong side). `trihedral_z` shadow NMSE 0.462 -> 0.107.
- **Penetrable-wedge incident terms (ADR-076).** With the `transmission`
  component on, order-1 diffraction weights its incident-boundary terms by
  `(I - T)`, `T` the transmission paths' wall product through the wedge, so the
  edge compensates only the GO step the transmitted field leaves.
  `cube_dielectric_z` ISB excess 4.45 -> -0.38 dB; no PEC case moves.
- **Diffraction edge set (ADR-057).** The edge set is the scene's geometric
  wedges, not the triangulation: subdividing a mesh no longer adds rows or
  changes per-row gains, and boundary half-planes get a back-face material and
  an orientation independent of vertex order. Half-plane edges that rest on
  another surface are dropped.
- **UTD coefficient (ADR-050).** The shared RayD coefficient matches
  Kouyoumjian-Pathak (finite-edge aperture phase argument, face axes at normal
  exit, reciprocal and label-free face angles, the dipole `sin(theta)` source),
  with a full-wave corner refit; MC face angles are reciprocal.
- **Multilayer reflection (ADR-051).** Every reflection kernel evaluates a
  multilayer material as its whole layer stack; it read only the first layer.
- **Layout-independent diffraction (ADR-058).** A scene with exactly one
  `ReceiverGrid` replaced the diffraction component power with a per-cell
  vector sum, so a grid cell and a point receiver at the same position
  disagreed by up to 2.47x. Diffraction now accumulates the same way for every
  receiver layout.
- **Reflection path validity (ADR-059, ADR-060).** One rule set validates every
  specular path (minimum segment length 1e-5 of the scene diagonal, no
  repeated plane group); degenerate paths (two reflection points on a shared
  edge, repeated reflections on one plane) are no longer reported.
- **MC diffraction seed streams (ADR-066).** Nearby seeds no longer reuse one
  set of Keller-cone draws; the samplers hash the lane first.
- `transmission` no longer destroys the CUDA context when a straight segment
  crosses more walls than `max_depth` (or has non-finite or huge endpoints).
  Over-depth pairs are dropped and counted; genuine device failures raise
  `CapacityError` naming the failure bits.
- `coupled_paths=True` no longer device-asserts when the coupled D-D bisection
  cannot certify a stationary point; that candidate is dropped and counted.
- Consumer discovery with `diffraction` raised `IndexError`; the order-1
  diffraction topology now takes the request's source polarizations.
- The consumer refuses a scene edited in place after compile (it returned the
  old answer with every row valid), endpoint batches on another GPU, and
  out-of-range frozen faces.
- Compile range-checks mesh face and UV indices and refuses non-finite
  vertices, naming the structure and row, before RayD sees the mesh;
  degenerate and duplicate faces are dropped.
- Kirchhoff tables cached on a compiled scene are rebuilt when a roughness or
  layer store tensor is replaced, written in place, marked `requires_grad` or
  given a forward-AD tangent. `k0 * rms_height > 32` raises the
  `kirchhoff_domain_exceeded` error at once instead of building for minutes.
- BDPT under `ad_mode="jvp"` / `"vjp"` keeps the autograd graph across
  concatenated connection blocks, and its capability record states the AD it
  implements.
- Binding a solver scene is host-only, so a `workspace_limit_bytes` refusal
  happens before any CUDA work.
- `witwin-channel-doctor` keeps the PyTorch identity when the runtime probe
  fails, instead of reporting a CPU-only PyTorch.
- Channel's source build accepts the installer-generated bytecode in a PyPI
  `rayd-torch` source bundle.
- `read_obj` keeps only the referenced vertices in each material group.
- Aggregated outputs are bit-identical run to run (ADR-077). Float atomics in the
  `radiomap.Coherent` cell accumulator (all six slots, including coupled and
  rough scattering) and its JVP, and in the `radiomap.Stochastic` diffraction
  and reflection maps, their JVPs, component power totals and
  `grad_frequency`, are replaced by order-independent integer accumulation.
  The result no longer depends on thread order, chunking, receiver windows or
  row order; a windowed solve is bitwise the whole solve. Values move by at
  most 6.4e-7 of the map maximum against the old run-to-run mean, which used to
  vary by up to 1.1e-6. Per-path columns are unchanged.

### Performance

Launch, sync and copy counts are the evidence; timings are from a shared GPU.

- One fused field launch evaluates LoS, reflection (every depth, slab, layer
  stack and rough factor) and order-1 diffraction rows (ADR-065). Kernels per
  warm solve: Coherent 64 x 64 grid 193 -> 130, Path 213 -> 149.
- Reflection candidates come from an image tree with conservative beam pruning
  and an image-method receiver test (ADR-060); depth-3 reflection on a 64 x 64
  grid of the 4 x 4 synthetic city drops from 418 ms / 2.15 GB to about
  14 ms / 0.1 GB.
- Interaction-chain discovery joins transmitter and receiver image-tree halves
  on a shared edge (ADR-064) and tests the deepest receiver half without
  materializing it, with algebraic cone decisions that fall back to the
  trigonometric test near the threshold (bit-identical rows). The coupled
  depth-3 budget case: kernels 28,100 -> 5,563, syncs 1,585 -> 284, peak
  4,163 -> 1,903 MiB.
- A coupled `radiomap.Coherent` solve over more than 32,768 receivers without
  path export, `max_paths`, AD or scattering runs in receiver windows: the
  canyon at 104k receivers drops from out of memory (> 7.9 GiB) to 4.12 GiB.
- Reflection and diffraction discovery are batched over every transmitter (one
  RayD call per stage).
- AD frequency cotangents use a fixed-partition device sum (bitwise
  reproducible; 7.9 ms instead of 73-81 ms at 16M rows); the shared-leaf
  transmission VJP is row-parallel (ADR-053).
- `radiomap.Stochastic`: native scattering receiver pairs replace the chunked
  Torch all-pairs code (ADR-056); grid maps are deposited in row-major order;
  warm solves reuse face material tensors (64^2 case: kernels 25 -> 11, syncs
  32 -> 1, H2D copies 36 -> 6); the reflection VJP evaluates one
  multi-tangent Fresnel dual; Kirchhoff tables are cached by roughness, layers
  and frequency, optionally on disk (ADR-052).
- BDPT accumulation uses a sorted segmented reduction (`O(rows log rows)`,
  bitwise reproducible) instead of an `O(cells x rows)` loop.
- Compiled runtimes are keyed on Channel's structure state and the cache is
  bounded by bytes; moving an antenna is a cache hit. Per-face material and id
  tables stay device resident on the compiled scene, the endpoint store is
  built once per endpoint token, and a 1 x 1 LoS Path solve makes 2
  host-to-device copies instead of 8.
- The primal fixed-topology wideband replay evaluates every frequency column
  of a bucket in one `[K, F]` launch.

### Changed

- `coupled_paths=True` adds interaction chains (ADR-064): reflections with one
  diffraction anywhere, up to `max_depth` interactions, plus double
  diffraction. The 1,000,000 candidate cap is gone and
  `coupled_candidate_limit` only sizes validator launches. Metadata geometry
  tags read `native_interaction_chains` (Path) and
  `native_interaction_chains_plus_dd` (Coherent).
- Reflection rows of a depth no longer depend on `max_depth`: the
  100,000-sequence switch between exhaustive and ray discovery is gone.
- Numerical results of `radiomap.Coherent` and `path` change wherever the fixes
  above apply (diffraction, coupled chains, multilayer and lossy faces), and
  the diffraction coefficient now depends on whether the solve carries the
  `transmission` component (ADR-076).
- Scene metadata no longer selects the diffraction edge table (ADR-073). A
  scene carrying `metadata["mitsuba"]["merge_shapes"]` (possible only by hand
  since `Scene.load_mitsuba` was removed; `importers.load_mitsuba` never sets
  it) used to diffract from RayD's raw imported edges, which count the
  coincident edges of different meshes as separate wedges. It now diffracts
  from the merged coincident-edge wedges like every other scene, so its
  diffraction rows and values change in `path`, `radiomap.Coherent` and
  `radiomap.Stochastic` (Munich, 32 x 32 grid: 45,429 -> 26,909 edges,
  28,910 -> 19,162 depth-1 rows). The key is now ignored.
- Error messages name the public solvers (`radiomap.Coherent`,
  `radiomap.Stochastic`, `experimental.bdpt`) instead of "deterministic",
  "Monte Carlo basic" or "MC basic"; for example "radiomap.Stochastic does not
  support directional antenna patterns". No message cites an ADR, plan or
  gate.
- The consistency gate between `radiomap.Stochastic` and incoherent
  `radiomap.Coherent` covers reflection only; diffraction is checked against
  the full-wave references instead (ADR-063).
- Internal native bindings and facades with no production caller were removed
  (including the BDPT event-composed sampler the wavefront replaced and the
  scatter-chain realization op), and so were the internal re-export aliases.
  A chain-scattering solve (`scattering_chain_max_depth >= 1`) with
  `ad_mode != "none"` is still refused before native work.
- `capabilities()["solvers"]` keeps its keys (`deterministic`,
  `montecarlo_basic`, `montecarlo_bdpt`): they name capability records, not
  import paths.

### Dependencies

- RayD 0.10.3, source-linked at the locked commit in
  `dependencies/rayd.lock.json` (was RayD 0.8.0). RayD is still not a runtime
  dependency.
- Wheels for CPython 3.10 to 3.13 on `manylinux_2_28_x86_64` and `win_amd64`.
  PyPI wheels pin `torch==2.10.0`; wheels for PyTorch 2.11 carry a local
  version (`0.6.0+torch211`) and are attached to the GitHub release.
- x86_64 wheels carry SASS for SM70, SM75, SM80, SM86, SM89, SM90, SM100 and
  SM120 plus `compute_120` PTX; SM87 and SM101 (aarch64-only Jetson parts) are
  dropped.
- `witwin>=0.4,<0.5` (unchanged).

### Known limitations

- Remaining full-wave gaps, each a strict xfail in
  `tests/accuracy/test_fullwave.py` (measured / target p95 boundary-jump
  excess): `canyon_x` plane 0 RSB 2.98 / 1.5 dB and plane 1 RSB 2.32 / 1.5 dB;
  `canyon_z` RSB 3.98 / 1.5 dB (plane 0) and 4.95 / 1.5 dB (plane 1);
  `canyon_z` plane 0 ISB 2.25 / 1.5 dB; `canyon_x` plane 0 ISB 3.24 / 1.5 dB;
  `double_edge_y` plane 1 RSB 2.82 / 1.5 dB.
- Transmission-then-diffraction is not modeled: the step of a field
  transmitted through one block at another block's edge is not compensated
  (it sets the `canyon_x` RSB gap). Second-order chain boundaries set the
  `canyon_z` RSB gap.
- Model-class residuals with their own ceilings: layered-wall transmission is
  a thin-sheet model (no refraction or lateral shift); the deep shadow of a
  concrete cube is 1.5-1.8 dB too bright (Luebbers-type incident terms);
  antennas are far-field.
- The penetrable-wedge weighting applies to `radiomap.Coherent` and `path`;
  `radiomap.Stochastic` and BDPT diffraction keep the opaque coefficient.
  Double diffraction and Monte Carlo use a slab view of multilayer faces, and
  the face layer tables carry no derivative through the wedge.
- `radiomap.Stochastic` reflection is scalar: against incoherent
  `radiomap.Coherent` it differs by 6-12 % at steep incidence and 19-33 % near
  the Brewster angle (strict xfails in
  `tests/accuracy/test_enumerated_sampled_consistency.py`).
- `radiomap.Stochastic` LoS and reflection project the field onto the
  transmitter polarization only, while `radiomap.Coherent`, `path` and
  `experimental.bdpt` also project onto the receiver's polarization. Under a
  raised transmitter, near its foot (large zenith angles), Stochastic is up to
  about 12 dB brighter (strict xfail `los_steep` in
  `tests/accuracy/test_enumerated_sampled_consistency.py`).
- `radiomap.Stochastic` diffraction on a receiver grid counts Keller-cone rays
  where they cross the grid plane. A vertical edge keeps its cone on the
  incident elevation, so with the transmitter at or near the grid height the
  shadow behind vertical edges receives little or no diffraction power (zero in
  62 % of a three-cube grid at equal heights, against 6 % for Coherent). A
  raised transmitter reduces but does not remove the gap (strict xfail
  `test_mc_basic_diffraction_reaches_the_deep_shadow_of_a_vertical_edge`).
- `experimental.bdpt` refuses reflection off multilayer materials, and its
  mixed reflection+transmission chains have no deterministic reference.
- The propagation consumer discovers diffraction rows at `ad_mode="none"` only,
  and its fixed-topology route (`reevaluate`, `evaluate_time_varying`, wideband
  offsets) replays LoS and reflection rows only.
- Coincident-edge wedges are probed at the edge midpoint, so a partly
  overlapping coincident edge is not split. A double-diffraction leg between
  the two free edges of one plate keeps its previous evaluation.
- Depth-3 coupled solves cost more than in 0.5 because the chains are now
  complete (about 2.5-3.5x the incomplete enumeration on the canyon).
  Path-export and AD coupled solves above 32,768 receivers still hold every
  row in memory.

- `grad_frequency` of `radiomap.Coherent` and `path` still varies run to run
  (up to about 1e-6 relative; 6e-6 with rough scattering), because the
  frequency cotangent is accumulated with float atomics inside RayD's VJP
  kernels. Leaf gradients (positions, powers, material and table cotangents)
  and the `experimental.bdpt` accumulators also keep float atomics (ADR-077).

## [0.5.0] - 2026-07-30

### Changed

- **Breaking (ADR-047).** Channel now source-links the released RayD 0.8.0
  commit and validates its API-8 eight-header integration set, published source
  manifest, canonical `torch/` build tree, and bundled
  `src/field_transport_ad.cuh` and `src/transmission_device.cuh` source
  seams. The retired RayD 0.7 include and source-tree spellings have no
  compatibility alias.
- Require the released `witwin==0.4.0` logical-world package and pin its release
  commit in the native wheel workflow. The Channel distribution version is
  0.5.0.

- **Breaking (ADR-039).** The propagation consumer publishes the declared
  source amplitude. `ScalarTransport.coefficient` and `Complex3Transport.field`
  now carry `sqrt(sources.powers_w)` of each row's own transmitting endpoint;
  `JonesTransport` stays excitation-free because a polarization-basis map is
  not a transported field. `CONTRACT_VERSION` moves from 2 to 3. A caller that
  already multiplied by `sqrt(powers_w)` itself must stop doing so.
- The Path solver's `phase_convention` metadata quotes the unit-excitation
  free-space amplitude, matching its own `coefficient_semantics`.
- **Breaking (ADR-040).** `reevaluate` refuses a frozen topology whose world
  moved since discovery. A moved `topology_version`, `material_version`, or
  `assignment_version` always raises; a moved `geometry_version` raises unless
  the request declares `world_motion="fixed_winner_replay"`. Previously a
  stale replay returned a full-strength old answer with no signal.
  `CONTRACT_VERSION` moves from 3 to 4.

### Added

- Native `field_source_amplitude_scale` and its backward/JVP companions, the
  owner of the excited complex3 field vector.
- **ADR-040.** `consumer.WorldProvenance`, stamped onto
  `PropagationTopology.provenance` by `evaluate` and forwarded verbatim by
  `prepare_fixed_topology`; `consumer.rediscovery_required(compiled, prepared)`,
  a host-only signal naming the version domain that moved;
  `FixedTopologyRequest.world_motion`; `CompiledScene.time_s`, the compiled
  snapshot instant, carried for reporting only. No call signature changes and
  no device work: the freshness check is four host integer comparisons.
- **ADR-041.** Slot-batched fixed-topology reevaluation and the time-varying
  channel impulse response. `FixedTopologyRequest.slot_count` (default `1`, so
  every existing call is unchanged) declares that the frozen rows and both
  endpoint batches are `slot_count` block-diagonal slots stacked slot-major,
  and `consumer.replicate_over_slots(prepared, slot_count, *, source_count,
  sink_count)` builds that topology by index arithmetic. `pair_count` becomes
  `slot_count * source_count * sink_count`, linear in the slot count, and a
  whole frame costs one launch per bucket, one four-byte validation copy, and
  one synchronization instead of one of each per instant.
  `consumer.evaluate_time_varying(compiled, TimeVaryingRequest)` publishes
  `delay_s`, `path_length_m`, the transport, and `row_valid` as `[T, K]` views
  over that single replay, with `times_s: float64[T]` and the frozen per-slot
  `pair_offsets`. `PropagationConvention.slot_pair_layout` states the pairing
  law without redefining `pair_layout`, and
  `PropagationCapabilities.supports_slot_batching` / `max_slot_count` disclose
  it. `slot_count > 1` requires a `PreparedFixedTopology`: the raw
  zero-interaction route builds its pair segmentation inside a native gather
  over the full outer product and refuses slot batching by name.
  `CONTRACT_VERSION` is unchanged at 4.
- **ADR-042.** Wideband frequency offsets on the fixed-topology route.
  `FixedTopologyRequest.frequency_offsets_hz` (default `None`, so every
  existing call is unchanged bit for bit) is a host tuple of propagation
  frequency offsets, and declaring it publishes the same frozen rows evaluated
  natively at each absolute frequency `reference_frequency_hz + df_j`.
  `ScalarTransport` gains `coefficient_offsets` (`[K, F]` complex64) and
  `Complex3Transport` gains `field_offsets` (`[K, F, 3]`), each paired with a
  `frequency_offsets_hz` echo of the grid it was evaluated on; a `0.0` entry
  reproduces the reference coefficient bitwise. `row_valid` stays `[K]`,
  geometry is published once, and the validation budget stays at one four-byte
  copy and one synchronization however large `F` is - the launch count is what
  grows, as `(1 + F) * buckets * launches_per_bucket`, reported by the new
  `PropagationDiagnostics.frequency_column_count`.
  `consumer.native_frequency_resolution_hz(f_ref)` publishes the float32 launch
  resolution a caller needs to build a resolvable grid (8192 Hz at 77 GHz).
  Seven new capability fields, two new convention strings, and
  `constants.NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW` quantify what the
  narrowband law this replaces actually costs. Dispersive scenes, rough or
  phase-screen scenes, and unresolvable grids are three independent fail-loud
  refusals before any native work; `polarimetric_transport` and a tensor grid
  are refused structurally. No native code, no ABI symbol, no new production
  Torch physics. `CONTRACT_VERSION` moves from 4 to 5.
- **Breaking (ADR-043).** The propagation-consumer AD capability matrix is
  published rather than inferred, and `CONTRACT_VERSION` moves from 5 to 6.
  `PropagationCapabilities` gains `component_material_leaves`,
  `differentiable_geometry_outputs`, `direction_differentiable_components`,
  `primal_only_ad_inputs`, `supports_higher_order_ad`, and `ad_accounting`, and
  `component_ad_modes` narrows `diffraction` to `{"none"}` because the consumer
  cannot produce a diffraction row at any AD mode. `PropagationDiagnostics`
  gains `ad_companion_launches` and `ad_tape_bytes`, published on both consumer
  routes with the reverse-only tape gate the solver metadata layer already
  applied.
  `PropagationGeometry.field_direction` now carries a real derivative on the
  fixed-topology route for `los` and `reflection` rows, in both AD modes, from
  new optional `grad_direction` / `direction` seams on the two Channel-owned
  native field transports; liveness is one decision for the whole result. A
  caller that relied on it being detached now receives a graph-bearing tensor,
  with bit-for-bit unchanged primal values.
  Every input named by `primal_only_ad_inputs` is now refused before any native
  work on every response and every route, instead of raising from inside
  `backward()` after a complete result had been published. Second-order AD is
  refused everywhere: `create_graph=True` raises from inside the backward it
  asked to differentiate, naming the owner, and `ad_mode="vjp"` with a forward
  dual is refused at the preflight. No published primal number changes; no new
  launch, transfer, or synchronization on any route.

## [0.4.0] - 2026-07-23

### Added

- Native Windows and `manylinux_2_28_x86_64` wheels for CPython 3.11 and
  PyTorch 2.10.
- Native CUDA SASS for SM70, SM75, SM80, SM86, SM87, SM89, SM90, SM100,
  SM101, and SM120, plus `compute_120` PTX forward compatibility.
- GitHub-hosted, release-gated wheel builds with final binary architecture and
  package-layout verification.

### Changed

- Source-link the locked, clean RayD 0.7.0 checkout at
  `49c58c4cb8212f6babb920cc88fb937509826cc5`.
- Group compatible CUDA code-generation targets and cache C/C++ compilation
  to reduce hosted-runner time without reducing release architecture coverage.
- Restrict paid native CI to explicit manual dispatches and published GitHub
  Releases. Ordinary pushes, pull requests, and schedules do not trigger it.

### Removed

- Obsolete self-hosted quick, CUDA, nightly, and evidence-only release
  workflows.

[Unreleased]: https://github.com/witwin-ai/witwin-channel/compare/witwin-channel-v0.6.0...HEAD
[0.6.0]: https://github.com/witwin-ai/witwin-channel/releases/tag/witwin-channel-v0.6.0
[0.5.0]: https://github.com/witwin-ai/witwin-channel/releases/tag/witwin-channel-v0.5.0
[0.4.0]: https://github.com/witwin-ai/witwin-channel/releases/tag/witwin-channel-v0.4.0
