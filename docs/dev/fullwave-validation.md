# Deterministic versus full-wave validation

This workflow compares the channel-native deterministic solver with a Tidy3D
full-wave reference on the same geometry, material, source polarization,
frequency, and observation plane.

## Cases

The versioned manifest is
`benchmarks/scenarios/fullwave_validation.v1.json`. It contains:

- `single_cube` using the relative layout of the original channel single-cube
  example;
- `three_cube` using the three centers and transmitter layout of the original
  channel three-cube example;
- `metal`, represented by `PerfectConductor` in channel-native and
  `PECMedium` in Tidy3D;
- `dielectric`, represented by a single-layer `PhysicalSurface` per crossed
  interface in channel-native and a true finite dielectric volume in Tidy3D.

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
channel-native transmission uses a straight thin-sheet interface chain,
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

The experiment creates the output directory. An installed channel-native
extension needs no extra configuration. For a developer build,
point the validated loader at a `.pyd` and its sidecar fingerprint:

```powershell
$nativeDir = Resolve-Path 'artifacts\cmake-ad'
$env:WITWIN_CHANNEL_NATIVE_DEVELOPER_OVERRIDE = '1'
$env:WITWIN_CHANNEL_NATIVE_EXTENSION_PATH = (
  Get-ChildItem $nativeDir -Filter '_channel_native*.pyd' | Select-Object -First 1
).FullName
$env:WITWIN_CHANNEL_NATIVE_EXPECTED_FINGERPRINT = (
  Get-Content "$nativeDir\_channel_native.build-fingerprint" -Raw
).Trim()
```

### Run

Generate the PEC cube fields, then the independent empty-space calibration:

```powershell
python benchmarks\fullwave_validation\experiments\run_maxwell_single_cube.py
python benchmarks\fullwave_validation\experiments\run_empty_baseline.py
```

Each command runs both channel-native and witwin-maxwell. The two FDTD runs
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
