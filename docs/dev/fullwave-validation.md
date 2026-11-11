# Deterministic versus full-wave validation

This workflow compares the channel deterministic solver with a Tidy3D
full-wave reference on the same geometry, material, source polarization,
frequency, and observation plane.

## Cases

The versioned manifest is
`benchmarks/scenarios/fullwave_validation.v1.json`. It contains:

- `single_cube` using the relative layout of the original channel single-cube
  example;
- `three_cube` using the three centers and transmitter layout of the original
  channel three-cube example;
- `metal`, represented by `PerfectConductor` in channel and
  `PECMedium` in Tidy3D;
- `dielectric`, represented by a single-layer `PhysicalSurface` per crossed
  interface in channel and a true finite dielectric volume in Tidy3D.

The geometry is scaled by 0.1 and the current carrier is 5 GHz. The single-cube
analysis plane uses 256 by 256 samples and a 6.25 mm full-wave grid (about 9.6
cells per free-space wavelength). The domain origin is chosen so the Maxwell
`Ez` Yee nodes coincide with those 256 receiver cell centers; jump metrics must
not be computed from a half-cell-interpolated field. The scale, grid, PML, and
source layout are part of the case fingerprint, so a result from a different
electrical problem cannot be compared silently.

For `three_cube`, "from the original channel example" describes geometry
provenance only: the bounds, transmitter, cube centres, observation height, and
cube size are scaled by 0.1 from
`channel/examples/deterministic_radiomap_three_cubes.py`. The validation case
uses a 5 GHz z-polarized source and PEC cubes, whereas the original example uses
1 GHz and a high-permittivity material. It is therefore not an
electromagnetically similar reproduction of the original example.

The legacy `three_cube` receiver grid retains the original field example's
256 by 256 sampling over a 2 m span, so its 7.8125 mm pitch does **not**
coincide with the configured 6.25 mm full-wave Yee grid. It remains valid for
deterministic-only visualization, but full-wave jump statistics must use the
versioned `three_cube_320` case: the same physical scene (cubes, transmitter,
observation plane, materials, 5 GHz) sampled at 320 by 320 cells over the same
`[-1, 1]^2` analysis bounds, with the full-wave domain origin shifted to
`x/y in [-1.396875, 1.403125]`, `z in [-0.496875, 0.903125]` so every receiver
cell center coincides with a Maxwell `Ez` Yee node (x/y at integer node
offsets, the `z=0.1` plane at a half-integer offset, exactly like the
single-cube case). Neither case may be silently interpolated to claim
grid-coincident jump statistics.

The scenario model rejects analysis bounds, transmitters, monitor planes, or
cube geometry that overlap the configured PML. For metal cases, receiver cells
inside the PEC volume are invalid observation points and must be excluded from
field-error and ISB/RSB jump statistics.

The manifest and saved NPZ coordinates use SI metres. The Tidy3D adapter
converts lengths from metres to micrometres and conductivity from S/m to S/um
at its boundary, then converts monitor coordinates back to metres.

The dielectric comparison intentionally exposes a model difference:
channel transmission uses a straight thin-sheet interface chain,
whereas Tidy3D solves the finite volume and includes refraction, internal
multiple scattering, and near-field coupling.

Tidy3D `PECMedium` is the reviewed strict no-transmission reference. The current
witwin-maxwell public material API has no PEC volume medium; using infinite
permittivity to force a zero electric update coefficient is useful for local
diagnostics but is not accepted as final ground truth until witwin-maxwell has
an explicit PEC voxel mask and reflection/transmission regression tests.

## Reproduced local PEC experiment

The reviewed local experiment uses the `single_cube` / `metal` manifest case
with the following frozen setup:

| Parameter | Value |
| --- | --- |
| Carrier | 5 GHz (`lambda_0` about 59.96 mm) |
| Cube | centre `(0, 0, 0.15)` m; side length 0.2 m |
| Material | no-transmission PEC |
| Transmitter | `(-0.2, -0.5, 0.42)` m; z-polarized point source |
| Analysis plane | `z=0.10` m; x/y bounds `[-0.8, 0.8]` m |
| Analysis samples | 256 by 256 cell centres |
| FDTD domain | x/y `[-0.996875, 1.003125]` m; z `[-0.296875, 0.603125]` m |
| FDTD grid | uniform 6.25 mm; 320 by 320 by 144 cells |
| PML | 12 cells (75 mm) on every side |
| Maxwell source | ideal `PointDipole`, `Ez`, Gaussian pulse, `fwidth=1 GHz` |
| Maxwell sampling | Hanning DFT at 5 GHz with source-spectrum normalization |
| Runtime | 12 transient plus 12 steady cycles; 16,587 time steps in the recorded run |
| Display | 256 by 256 data; Matplotlib PNG at 6144 by 3456 pixels |

The transmitter is about 108 mm inside the upper PML interface. Raising it to
0.50 m would leave only 28 mm in the original domain and therefore requires a
larger z domain; the frozen case deliberately uses 0.42 m.

The local witwin-maxwell PEC is implemented as `eps_r=inf`, which sets the
electric update coefficient to zero in the voxelized cube. The experiment
checks that the interior/exterior field-peak ratio is below `1e-7`. This is a
useful local diagnostic, but Tidy3D `PECMedium` remains the strict ground truth
until witwin-maxwell exposes a first-class PEC volume.

### Environment

The recorded run used Windows, an NVIDIA GPU, Visual Studio C++, the `witwin2`
Conda environment, and a local witwin-maxwell checkout. Run from the repository
root. Configure the local Maxwell checkout and an output directory:

```powershell
conda activate witwin2
$env:WITWIN_MAXWELL_SOURCE = (Resolve-Path 'E:\Code\witwin-maxwell')
$env:WITWIN_FULLWAVE_OUTPUT_DIR = Join-Path (
  Get-Location
) 'artifacts\fullwave\single-cube-metal-z042'
```

The experiment creates the output directory. An installed channel
extension needs no extra configuration. For a developer build,
point the validated loader at a `.pyd` and its sidecar fingerprint:

```powershell
$nativeDir = Resolve-Path 'artifacts\cmake-ad'
$env:WITWIN_CHANNEL_DEVELOPER_OVERRIDE = '1'
$env:WITWIN_CHANNEL_EXTENSION_PATH = (
  Get-ChildItem $nativeDir -Filter '_channel*.pyd' | Select-Object -First 1
).FullName
$env:WITWIN_CHANNEL_EXPECTED_FINGERPRINT = (
  Get-Content "$nativeDir\_channel.build-fingerprint" -Raw
).Trim()
```

### Run

Generate the PEC cube fields, then the independent empty-space calibration:

```powershell
python benchmarks\fullwave_validation\experiments\run_maxwell_single_cube.py
python benchmarks\fullwave_validation\experiments\run_empty_baseline.py
```

Each command runs both channel and witwin-maxwell. The two FDTD runs
took about 70 seconds each on the recorded machine. Generate the 6K Matplotlib
comparison:

```powershell
python benchmarks\fullwave_validation\experiments\plot_single_cube_comparison.py
```

The output directory then contains:

- `visual-deterministic-metal-centered-5ghz-256.npz`;
- `visual-maxwell-metal-centered-5ghz-256.npz`;
- `visual-deterministic-empty-5ghz-256.npz`;
- `visual-maxwell-empty-5ghz-256.npz`;
- `single-cube-metal-centered-tx-z042-5ghz-256-empty-calibrated-6k.png`.

Validate the harness with:

```powershell
python -m pytest tests\validation\test_fullwave_validation.py -q
python -m ruff check --no-cache benchmarks\fullwave_validation tests\validation
```

### Calibration and interpretation

The deterministic map is a scalar channel coefficient `h`; witwin-maxwell
exports the Cartesian field component `Ez`. Their absolute source units differ,
and their complex phases are not the same observable. Do not fit a global
complex scalar to these maps: low complex coherence can drive that scalar
towards zero and make the deterministic plot artificially dark.

Instead, the experiment computes one positive amplitude scale from the empty
scene, excluding the nominal cube footprint from the calibration mask:

```text
s_empty = sqrt(sum(|Ez_empty|^2) / sum(|h_empty|^2))
```

That scale is frozen and applied to the PEC cube without refitting.

**Observable semantics (since the 2026-07 UTD continuity fixes):** the
deterministic export is now the receiver-polarization projection of the
coherent field vector (`p̂_rx·E⃗`, i.e. `Ez` for the ẑ-polarized benchmark
receivers), and every component transports the transmitter polarization as an
unnormalized transverse projection (short-dipole sin θ pattern). The exported
scalar is therefore the *same physical observable* as Maxwell `Ez`, including
the axial dipole null; the old "scalar 1/r envelope" caveats below no longer
apply, and the empty-scene scale must be re-derived whenever the observable
changes (the sin θ pattern lowers `|h_empty|`, raising `s_empty`).

Recorded metrics, original code vs the post-fix solver (see
`docs/dev/audit/utd-continuity-fix-design.md` for the fix inventory; the
before column is reproduced from the recorded run to full precision):

| Metric | Original | Post-fix |
| --- | ---: | ---: |
| Empty-space amplitude scale | 39.569651 | 62.2108 |
| PEC envelope NMSE | 0.217382 | 0.04352 |
| PEC magnitude correlation | 0.447111 | 0.8456 |
| PEC magnitude RMSE | 5.063745 dB | 2.8733 dB |
| PEC complex coherence (after one global phase) | 0.8074 | 0.8849 |
| ISB p95 excess jump | +6.956 dB | **-0.099 dB** |
| RSB p95 excess jump | +1.266 dB | +0.157 dB |
| Deterministic max adjacent jump | 156.9 dB | 29.8 dB (physical null) |

The historically quoted complex coherence `2.0069e-5` used the naive phase
convention; under the conjugate convention with a single global phase removed
the original map already scored 0.8074. Continuity regression tests live in
`tests/deterministic/test_field_continuity.py`.

## Workflow

Run commands from the repository root in the `witwin2` environment.

Prepare and inspect a Tidy3D simulation without creating a cloud task:

```powershell
python -m benchmarks.fullwave_validation prepare `
  --scenario single_cube --material metal `
  --output-dir artifacts/fullwave/single-cube-metal
```

Run the deterministic solver and save the common reference format:

```powershell
python -m benchmarks.fullwave_validation solve-deterministic `
  --scenario single_cube --material metal `
  --output artifacts/fullwave/single-cube-metal/deterministic.npz
```

Creating a Tidy3D cloud task is explicit because it may consume credits:

```powershell
python -m benchmarks.fullwave_validation solve-tidy3d `
  --scenario single_cube --material metal --submit `
  --output artifacts/fullwave/single-cube-metal/tidy3d.npz `
  --tidy3d-data-output artifacts/fullwave/single-cube-metal/tidy3d-data.hdf5
```

An existing downloaded `SimulationData` file can be imported without
submitting another task:

```powershell
python -m benchmarks.fullwave_validation solve-tidy3d `
  --scenario single_cube --material metal `
  --simulation-data artifacts/fullwave/single-cube-metal/tidy3d-data.hdf5 `
  --output artifacts/fullwave/single-cube-metal/tidy3d.npz
```

Compare the results:

```powershell
python -m benchmarks.fullwave_validation compare `
  --deterministic artifacts/fullwave/single-cube-metal/deterministic.npz `
  --fullwave artifacts/fullwave/single-cube-metal/tidy3d.npz `
  --output artifacts/fullwave/single-cube-metal/comparison.json
```

Repeat the same commands for `three_cube` and/or `dielectric`.

## Three-cube full-wave reference (`three_cube_320`, recorded 2026-07-17)

The recorded local witwin-maxwell experiment for the Yee-locked three-cube
case uses the `three_cube_320` / `metal` manifest entry (case fingerprint
`077bcc039733863ac3af47584f5a760bc6fe9f594f7e23464015dcc03b90c86a`) with the
same PEC voxelization, source, DFT sampling, and runtime policy as the
single-cube experiment. The FDTD domain is 448 by 448 by 224 cells; the
recorded runs took 22,991 time steps (about 355 s PEC, 311 s empty). Guards:
receiver/monitor alignment below 1e-7 m, all three PEC interiors field-free
(peak ratio below 1e-7).

```powershell
conda activate witwin2
$env:WITWIN_MAXWELL_SOURCE = (Resolve-Path 'E:\Code\witwin-maxwell')
$env:WITWIN_FULLWAVE_OUTPUT_DIR = Join-Path (
  Get-Location
) 'artifacts\fullwave\three-cube-metal-320'

python benchmarks\fullwave_validation\experiments\run_maxwell_three_cube.py
python benchmarks\fullwave_validation\experiments\run_three_cube_empty_baseline.py
python benchmarks\fullwave_validation\experiments\run_three_cube_deterministic.py
python benchmarks\fullwave_validation\experiments\compare_three_cube_fullwave.py
```

The output directory then contains the FDTD references
(`visual-maxwell-metal-three-cube-5ghz-320.npz`, the empty pair), the
deterministic coupled-OFF/ON solves plus full path tables, the comparison
report `three_cube_320_comparison.json`, gap maps, and the rendered
`three_cube_320_p1_comparison.png`.

Recorded arbiter metrics (empty-scene amplitude scale `s_empty = 62.5931`,
consistent with the single-cube 62.2108; conjugate phase convention; PEC
volumes masked):

| Metric | coupled OFF | coupled ON |
| --- | ---: | ---: |
| Envelope NMSE | 0.0899 | 0.0934 |
| Magnitude correlation | 0.8019 | 0.7946 |
| Magnitude RMSE | 4.03 dB | 3.94 dB |
| Coherence (one global phase) | 0.8475 | 0.8455 |
| ISB p95 excess | +0.39 dB | +2.04 dB |
| RSB p95 excess | +3.55 dB | +3.53 dB |
| Flagship row (y=0.459) max adjacent jump | 39.3 dB | 9.2 dB |
| FDTD same row max adjacent jump | 13.5 dB (physical null) | — |

Interpretation (full analysis in ADR-011's arbiter addendum): FDTD shows **no
step** at the flagship occlusion boundary and substantial energy throughout
the reflected-shadow sector where coupled-OFF is 35-53 dB low; coupled-ON
fills that sector to within 3-20 dB and removes the unphysical 39 dB step.
Coupled-ON's residual liabilities are its own enumeration-existence seams: at
face-edge-exit boundaries the coupled term degenerates into an anti-phase
duplicate of order-1 diffraction (deep cancellation nulls, worst
`(x=0.053, y=0.453)` at -59.9 dB vs truth), and ISB edges touching
coupled-active cells carry +3.92 dB p95 excess versus +1.11 dB with coupled
OFF. Those seams are the P2 (D-to-D) acceptance targets.

Frozen regression thresholds for `three_cube_320` / `metal` with the
benchmark default (coupled ON): envelope NMSE < 0.12, magnitude correlation
> 0.75, coherence after one global phase > 0.80, ISB p95 excess < 3.0 dB,
RSB p95 excess < 4.5 dB. Any tightening must come with a new recorded run;
never weaken these to pass a change.

**ADR-013 D->D update (2026-07-18):** with double diffraction in the
coupled family the recorded coupled-ON numbers improve to envelope NMSE
0.0922, coherence 0.8467, ISB p95 excess +1.08 dB, RSB p95 excess
**+0.43 dB**, blockage-boundary steps 0.5/0.7/3.6 dB (truth 1.3/1.5/1.9),
warm solve 2.6 s; the single-cube benchmark is unchanged-or-better (NMSE
0.0358, coherence 0.8896). Full gate table and the lambda/20 near-field
residuals deferred to P4 are recorded in ADR-013's measured-acceptance
section.

### Three-cube deterministic visualization

The following commands generate and plot only the deterministic legacy
`three_cube` / `metal` result; the plotting script neither loads nor creates a
full-wave result:

```powershell
conda run -n witwin2 python -m benchmarks.fullwave_validation solve-deterministic `
  --scenario three_cube --material metal `
  --output artifacts/fullwave/three-cube-metal/deterministic.npz

conda run -n witwin2 python `
  benchmarks/fullwave_validation/experiments/plot_three_cube_deterministic.py `
  --deterministic artifacts/fullwave/three-cube-metal/deterministic.npz `
  --output artifacts/fullwave/three-cube-metal/three-cube-deterministic-components.png
```

The four panels share the total-field peak and a `[-60, 0]` dB display range.
Receiver samples inside any PEC cube are masked, and all three cube outlines and
the transmitter marker are derived from the versioned case specification.

The historical limitation this figure originally exposed — the deterministic
solver dropping coupled reflection-diffraction component IDs 3/4 and taking a
35.99 dB step at the flagship occlusion boundary — is resolved: ADR-011/012
added deterministic `coupled_paths` with RX-chunked streaming and stationary
coupled-leg semantics, and the benchmark backend enables it by default. The
remaining higher-order boundaries (D-to-D blockage birth, coupled sector
edges, order-2 R-R-D) are quantified against the `three_cube_320` full-wave
reference above and are the plan-09 P2/P3 targets.

## Metrics and jump checks

Absolute source amplitudes differ between a one-watt channel transmitter and a
discretized point dipole. `compare_fields` retains the global complex-scalar
metric for references that export the same physical complex observable.
`compare_magnitudes` is the correct envelope metric when source units or field
observables differ. It accepts a fixed positive `amplitude_scale`; the local
Maxwell experiment derives that scale from an independent empty scene.

Jump checks use deterministic component support, not a gradient heuristic:

- ISB edges are adjacent receiver cells where LoS support turns on or off;
- RSB edges are adjacent receiver cells where reflection support turns on or
  off.

For each set of edges, the report compares adjacent-cell total-field magnitude
jumps in dB for deterministic and full-wave fields. Median, p95, maximum, and
p95 excess are reported. Full-wave results are interpolated in complex form to
the deterministic grid before any metric is computed. For jump qualification,
prefer a grid origin that makes the two sample sets exactly coincident;
interpolation can smooth a discontinuity. `compare_fields` and
`analyze_boundaries` accept a `valid_mask` for excluding conductor interiors
and other invalid observation cells.

The workflow diagnoses discontinuities; it does not yet define pass/fail
thresholds. Thresholds should be frozen only after references for all four
cases have been generated at a reviewed grid-convergence level.

## Tidy3D reference set and accuracy gate (`fullwave_reference.v1.json`)

The cases above compare against local witwin-maxwell runs whose arrays are not
in the repository. The committed reference set is a separate, small group of
synthetic scenes solved once with Tidy3D 2.11.2 and stored as calibrated field
maps under `tests/data/fullwave/<case>.npz`, so the comparison needs no
full-wave solver to reproduce.

| Case | Geometry | Polarization | Planes `z` (m) | Role |
| --- | --- | --- | --- | --- |
| `beam_z`, `beam_y` | PEC beam 0.18 x 0.5 x 0.18 m: four 90-degree wedges with 8.3-wavelength edges | z, y | -0.1, 0.12 | fit |
| `halfplane_z`, `halfplane_y` | zero-thickness PEC plate 0.5 x 0.5 m in `x = 0` | z, y | -0.05, -0.2 | fit |
| `cube_z` | manifest `single_cube` layout, PEC | z | 0.1, 0.3 | fit |
| `cube_x` | same, x-polarized | x | 0.1, 0.3 | holdout |
| `three_cube_z` | manifest `three_cube_320` layout, PEC | z | 0.1 | holdout |
| `cube_dielectric_z` | `single_cube`, eps_r 4, sigma 0.01 S/m | z | 0.1, 0.3 | holdout |
| `empty_z`, `empty_x` | free space | z, x | -0.1 | calibration |

All runs use 5 GHz, a uniform 2 mm grid (2.5 mm for `three_cube_z`, about
lambda/24 to lambda/30), 12 PML layers outside the domain, a Gaussian-pulse
`PointDipole` and colocated monitors. Stored maps keep every fourth colocated
grid node (8 mm or 10 mm pitch), so no interpolation is involved. The ten
tasks cost 1.35 FlexCredits.

**Calibration.** The two free-space runs fit one complex constant `C` per
dipole axis between the conjugated Tidy3D phasor and the exact field of a unit
dipole (`exp(-jkr)/r` times the transverse projection, with its `1/(kr)` and
`1/(kr)^2` terms). The ratio is constant to +-0.4 % in magnitude; its phase
drifts by about -8 degrees per metre (z) and -1 degree per metre (x), the FDTD
dispersion at this grid. Stored fields are `conj(E_tidy3d) / C`, so they are in
the same units as the deterministic field divided by its own LoS calibration
(the LoS term alone equals the unit-dipole far field to 5e-5). Nothing is
fitted per case.

**Regions.** `classify_regions` labels each sample: invalid (inside or within
one pitch of a block, two pitches of a plate), corner (the Keller stationary
point of a silhouette edge seen by both antennas lies within
`0.15 sqrt(lambda L)` of an edge end), transition (within half a projected
Fresnel width `sqrt(lambda s (s + s') / s')` of a change of LoS or specular
reflection visibility), otherwise lit or shadow by LoS visibility.

**Metrics.** `region_errors` reports per region the RMS and mean of the dB
magnitude error (both maps clamped 40 dB below the plane's 99th percentile) and
the complex NMSE with no fitted scale or phase.

**Gate.** `tests/accuracy/test_fullwave.py` (cuda tier, about 10 s) solves
every fit and holdout case with the deterministic solver (`max_depth` 2, 3 for
`three_cube_z`, coupled paths on, no ISB taper) and checks each region's RMS dB
error and complex NMSE, and each plane's ISB/RSB jump p95 excess, against
ceilings set from the measured errors plus headroom (ADR-050 amendment). The
jump check matters: a wider corner sigmoid lowered the region RMS but put
several-dB steps on shadow boundaries, which RMS alone did not flag.

Regenerating the set (needs the Tidy3D API key; never uploads repository
content, only the synthetic simulations):

```bash
python benchmarks/fullwave_validation/experiments/build_reference_set.py estimate --tasks tasks.json --data-dir out
python benchmarks/fullwave_validation/experiments/build_reference_set.py submit --tasks tasks.json --data-dir out --max-credits 20
python benchmarks/fullwave_validation/experiments/build_reference_set.py extract --tasks tasks.json --data-dir out
```

## Coverage cases (`role: coverage`)

The fit and holdout cases cover free-standing PEC wedges, a half-plane and
cubes. The coverage cases test the rest of what the deterministic solver claims:

- lossy and multilayer wall reflection and transmission;
- reflection up to third order;
- double diffraction;
- reflection-diffraction chains;
- lossy dielectric wedges;
- a ground plane;
- horizontal and vertical polarization;
- two more frequencies.

The 23 Tidy3D tasks cost 3.60 FlexCredits. They are 15 stored maps, 5 free-space
calibrations, a 1 mm convergence run, and two discarded trihedral runs (described
below).

| Case | Scene | Claim under test |
| --- | --- | --- |
| `wall_concrete_z`, `_y` | infinite 0.1 m concrete slab (eps_r 5.24, 0.163 S/m), through the PML | reflection off a lossy wall at 0-80 degrees, TE and TM (the TM case crosses Brewster at 66 degrees); thin-wall transmission |
| `wall_layered_z`, `_y` | infinite symmetric wall: 2 / 15 / 2 cm with eps_r 3 / 6.5 / 3 | multilayer reflection and transmission |
| `wall_concrete_z_2g4` | concrete wall at 2.4 GHz | frequency dependence of material and spreading |
| `cube_concrete_z`, `_x` | single-cube layout, concrete | lossy dielectric wedge diffraction |
| `trihedral_z`, `_x` | inner corner of three finite PEC plates | reflection depth 1-3 with polarization, plate edges |
| `double_edge_y`, `_z` | two parallel PEC screens, infinite in y | double diffraction behind both screens |
| `wall_column_z`, `_y` | infinite PEC column 0.2 m in front of an infinite concrete wall | reflection-diffraction chains |
| `canyon_z_3g5`, `_x_3g5` | four concrete blocks with crossing streets on a concrete ground, 3.5 GHz | a small city: everything above at once |

**Grids and calibration.**
- Dielectric walls refine only the axis normal to the wall:
  - 1 mm or 0.5 mm inside the slab, 2 mm elsewhere.
  - Range ends lie on the 2 mm lattice, and extraction keeps only the decimated
    lattice nodes, so the maps are still uniform and interpolation-free.
- Every structure that crosses the PML stands for an unbounded one:
  - In Channel it becomes a 40 m block.
  - `classify_regions` ignores the edges and edge ends outside the simulated domain.
- 5 GHz cases use the existing `empty_z` / `empty_x` calibration.
  - The refined grid (`empty_z_refined`) shifts that constant by 1.7 degrees and
    leaves its magnitude unchanged.
- 3.5 GHz and 2.4 GHz cases have their own free-space runs. Their constant is flat
  to +-0.4 % and their phase drift is -4.4 degrees/m (z) and -3.6 degrees/m.
- A 1 mm run of `wall_concrete_z` (`wall_concrete_z_fine`, map not stored) agrees
  with the 2 mm reference to 0.09-0.11 dB RMS and 0.1 % NMSE in every region.
  This converges both the dielectric interface and the slab interior.

**Grid guard.**
- `build_tidy3d_simulation` refuses a case whose 2 mm lattice, zero-thickness plate
  or monitor plane is not a grid node.
- The guard exists because of the first trihedral run. Its z extent, 0.6 m, came
  out as 300.0000001 cells, so Tidy3D stretched the grid to 301 cells. The z = 0
  ground plate then fell between nodes and vanished: the stored field had no
  ground reflection at all.
- Those maps were discarded and the case was rerun.

### Results

**Results** (`max_depth` 3 for the trihedral, screens, wall-column and canyon
cases, 2 otherwise; coupled paths on). Entries are RMS dB error / complex NMSE.

| Case / plane | lit | shadow | transition | corner |
| --- | --- | --- | --- | --- |
| `wall_concrete_z` -0.06 / -0.18 | 0.68 / 0.013, 1.10 / 0.007 | 0.83 / 0.046, 0.68 / 0.052 | - | - |
| `wall_concrete_y` -0.06 / -0.18 | 1.11 / 0.009, 0.35 / 0.002 | 0.72 / 0.042, 0.97 / 0.057 | - | - |
| `wall_layered_z` | 0.84 / 0.014, 1.15 / 0.009 | 1.67 / 0.242, 1.42 / 0.349 | - | - |
| `wall_layered_y` | 1.26 / 0.009, 0.76 / 0.004 | 1.98 / 0.143, 2.75 / 0.247 | - | - |
| `wall_concrete_z_2g4` | 1.03 / 0.054, 1.09 / 0.018 | 0.88 / 0.031, 0.71 / 0.026 | - | - |
| `cube_concrete_z` 0.1 | 0.37 / 0.006 | 3.91 / 0.164 (+1.9 dB bias) | 0.93 / 0.022 | 1.74 / 0.013 |
| `cube_concrete_x` 0.1 | 0.37 / 0.003 | 3.44 / 0.168 (+1.7 dB bias) | 0.97 / 0.035 | 1.23 / 0.007 |
| `trihedral_z` 0.08 / 0.26 | 0.69 / 0.003, 0.71 / 0.008 | 0.32, 1.06 | 0.44, 0.57 | 0.77, 0.77 |
| `trihedral_x` 0.08 / 0.26 | 0.79 / 0.006, 0.97 / 0.008 | 4.23 / 1.65, 3.67 / 0.61 | 1.57, 1.57 | 0.77, 4.33 / 1.34 |
| `double_edge_y` -0.15 / 0.1 | 0.70, 0.27 | 1.62 / 0.134, 0.16 | -, 2.26 | - |
| `double_edge_z` -0.15 / 0.1 | 1.59, 0.11 | 1.95 / 0.077, 0.07 | -, 1.49 | - |
| `wall_column_z` / `_y` | 1.09, 1.12 | 2.94 / 0.140, 1.85 / 0.079 | 0.74, 1.47 | - |
| `canyon_z_3g5` 0.06 / 0.16 | 1.77 / 0.039, 1.54 / 0.060 | 4.23 / 1.56, 2.51 / 0.52 | 2.85, 2.34 | 2.22, 2.23 |
| `canyon_x_3g5` 0.06 / 0.16 | 2.61 / 0.030, 3.06 / 0.045 | 1.50 / 1.04, 1.71 / 0.43 | 2.95, 2.62 | 3.22, 2.26 |

### Validated claims

- **Multi-bounce specular reflection.**
  - Inside the trihedral corner, Channel's LoS-plus-reflection field equals
    eight-image theory to 1e-9 NMSE at depths 1, 2 and 3.
  - The full-wave field agrees with that image field to 0.3-1.2 % NMSE in the
    inner 0.44 m square. The remainder is diffraction from the finite plates.
  - The lit regions agree to 0.7-1.0 dB.
- **Lossy wall reflection, TE and TM, through Brewster:** 0.35-1.1 dB.
- **Thin-wall transmission, magnitude and phase.**
  - Shadow NMSE is 3-6 %.
  - An independent straight-ray layer-stack model, which carries the
    `exp(+j k d cos(theta))` carrier correction for the slab thickness, gives the
    same NMSE.
  - Without that correction the NMSE would be 2.8, so Channel's phase reference
    is right.
- **Double diffraction behind two screens:** 1.6-2.0 dB.
- **Frequency scaling at 2.4 and 3.5 GHz:** no frequency-specific error appears.

### Model-class residuals (tolerances, not defects)

- **Layered-wall transmitted region:** 1.4-2.8 dB, NMSE 0.14-0.35.
  - The independent straight-ray layer-stack model gives the same numbers to 0.01 dB.
  - The difference from full-wave is the thin-sheet model: no refraction, no
    lateral displacement of the internal multiple reflections.
- **Concrete-cube deep shadow:** +1.7-1.9 dB too bright.
  - Full-wave puts the concrete shadow 2.2 dB (z) and 3.1 dB (x) below the PEC
    cube's. Channel's finite-conductivity wedge coefficient puts it only 0.7 and 1.3 dB
    below.
  - Luebbers-type face coefficients leave the incident-shadow terms PEC-like.
- **Wall planes under the transmitter:** errors from the dipole's near field and
  axis null. Channel's antennas are far-field.
- **Canyon:** `max_depth` 4 lowers the shadow error further than 3 does.

### Defects found

- **D1: reflection in-face tolerance scales with the face size.**
  - Location: `native/channel/kernels/paths.cu`, `kReflectionBaryEps = 1e-4` on barycentrics.
  - A hit up to 1e-4 x the face size past a boundary edge still counts, so on large
    faces the reflection support overshoots the edge and misses the UTD RSB term.
  - Measured on the screens:

    | Screen size | Overshoot past the edge |
    | ---: | ---: |
    | 40 m | 3.96 mm |
    | 4 m | 0.38 mm |
    | 0.4 m | 0 |

  - Effect: the `double_edge` RSB p95 jump excess is 6.3 / 4.3 dB. With a 1.2 m
    screen it is 2.4 / 0.0 dB.
- **D2: dielectric UTD face reflection uses the edge-plane angle.**
  - Location: `utd::reciprocal_face_cosines`, which uses `(|sin phi| + |sin phi'|) / 2`.
  - Geometric-optics reflection uses the 3D incidence angle, `sin(beta0) sin(phi')`
    at the boundary.
  - Channel-only probe on a concrete half-plane, sampled across its RSB:

    | Edge type | Incidence | Residual jump, fraction of R |
    | --- | --- | ---: |
    | PEC | any skew | 0.002 |
    | Concrete | no skew | 0.002 |
    | Concrete | 45-degree skew | 0.26-0.28 |

- **D3: UTD face operator uses half-space Fresnel; reflection uses the layer stack.**
  - The UTD face operator evaluates half-space Fresnel from one (eta, sigma) per face.
  - Geometric-optics reflection uses the full layer stack.
  - RSB residual, no skew:

    | Face material | Residual jump, fraction of R |
    | --- | ---: |
    | 0.1 m concrete slab | 0.05 |
    | 6 mm glass | 0.38-0.42 |
    | the layered wall | 0.54-0.60 |

  - Effect, with D2: `cube_concrete` RSB excess 3.8 / 2.1 dB, against 0.8 / 0.05 dB for PEC.
- **D4: coupled chains do not compensate a reflected path blocked by another object.**
  - Where another object shadows a reflected path, the coupled R-D/D-R component
    is continuous: its step is 0.3-0.8 % of R. The total field therefore steps by
    the full |R|.
  - Measured in a Channel-only probe of the wall-column alley. A PEC wall behaves
    the same.
  - Effect: `wall_column` RSB excess 5.4 / 2.4 dB; canyon RSB excess 2.8-8.6 dB.
- **D5: coincident edges of separate structures are diffracted independently.**
  - Each separate structure's coincident boundary edges are diffracted as their
    own half-planes or convex wedges. Examples: plates meeting at a corner, and
    building bases standing on a ground.
  - `trihedral_x`:

    | Region | Separate plates | One welded mesh |
    | --- | ---: | ---: |
    | shadow | 4.23 dB / 1.65 | 1.61 dB / 0.10 |
    | corner | 4.33 dB / 1.34 | 2.15 dB / 0.49 |

  - `canyon_z` shadow: 4.23 dB / 1.56 as built, 2.92 dB / 0.85 with the block
    bases hidden 5 cm below the ground.
  - City imports that place building meshes on a ground mesh have this geometry.
- **D6 (minor): a receiver exactly on an RSB loses half the reflected field.**
  - At that receiver the reflection counts as visible while the diffraction takes
    the far-side value, so the total is off by R/2 on that one sample.
- **Unexplained: canyon LoS boundaries.**
  - At the LoS boundaries of the blocks standing on the ground, the field is
    compensated only to 25 % of the LoS (median, Channel-only probe). On the free
    concrete cube it is 4 %.
  - Canyon ISB excess is 2.0-3.7 dB. The cause is not isolated.

### Fixes and their full-wave effect

D1-D4 and D6 are fixed; D5 is designed but not implemented (awaiting the owner's decision). Each has its own
design record: `docs/dev/standards/adr-0xx-fw-d1-reflection-face-containment.md`, `-fw-d2-oblique-face-dyadic`,
`-fw-d3-layered-face-term`, `-fw-d4-chain-completeness`, `-fw-d5-coincident-edges` (proposed) and
`-fw-d6-boundary-sample`; the RayD side is in RayD ADR-0047 and ADR-0048 amendments.

| Defect | Fix | Gate metric (before -> after; target) |
| --- | --- | --- |
| D1 | in-face slack is a distance (8 float roundings of the coordinates), shared by every reflection validator | `double_edge_z` RSB 4.32 -> -0.03 dB (1.5); `double_edge_y` transition 2.26 -> 1.03 dB (1.5) |
| D2 | face coefficients at the 3D incidence cosine; reciprocal specular dyadic in the edge-fixed bases | `cube_concrete_z` RSB 3.80 -> 0.47 dB (1.8); `cube_concrete_x` RSB 2.13 -> 0.05 dB (1.05) |
| D3 | face terms reflect with the face's slab or layer stack (the specular kernels' coefficient) | RSB residual at normal incidence 0.05 / 0.38-0.42 / 0.54-0.60 -> 0.002 (slab / glass / layered) |
| D4 | chains carry `max_depth` reflections; chain legs must leave the wedge into free space | `wall_column_z` RSB 5.35 -> 0.70 dB, `wall_column_y` 2.37 -> 0.98 dB (1.5); `canyon_x` plane 0 RSB 4.52 -> 1.14 dB |
| D6 | the pole term alone takes the one-sided limit, on the side the geometric tests choose; near-pole quantities from one small angle | on-boundary receiver: \|T\| 3.7e-5 -> 8.2e-4 (lit 8.2e-4) |

Remaining strict xfails (measured / target): `double_edge_y` plane 1 RSB 2.84 / 1.5 dB (the RSB is continuous;
an interference null next to the transmitter sits 2-3 mm off its full-wave position on the four worst edges);
`trihedral_x` shadow and corner (D5); canyon shadow (D5), RSB (5.55 / 4.65 dB on `canyon_z`, 1.96 on `canyon_x`
plane 1; second-order chain boundaries) and ISB (2.25-3.44 dB). The **unexplained canyon ISB** is resolved as a
measurement effect: the probe that reported 25 % counted support edges at block faces (receivers inside the
blocks); excluding the block interiors, the LoS steps are compensated to 1.4 % (y steps) and 2.3 % (x steps) of
the LoS, and the ground-reflection steps that coincide with them to 0.0-0.5 %. The remaining ISB excess is at
samples where the total is 15-20 dB below the LoS (deep cancellation near the block corners), where a few
percent of the LoS is a large dB jump. Hiding the block bases (the D5 proxy) does not reduce it (0.22 -> 0.32
median residual with the interior-inclusive probe).

### Gate

**What the gate checks.**
- `tests/accuracy/test_fullwave.py` gives every coverage region and boundary a
  ceiling under the same rule as the fit cases.
- Each metric that one of the defects above explains is not given a wider
  ceiling. Instead it runs as a strict xfail in
  `test_deterministic_fullwave_known_gaps`, against the error the model class
  reaches without that defect. There are 23 such metrics.
- Each xfail reason records the measured value and its diagnosis.
- A fix makes its xfail pass unexpectedly. The fixer then moves that metric into
  the ordinary ceilings.

**Cost.** The gate solves 23 cases in about 12 s.

**Harness tests.** `tests/validation/test_fullwave_reference_set.py` checks:
- the stored labels and fingerprints;
- slab and refined-grid construction;
- the grid guard (with Tidy3D installed).
