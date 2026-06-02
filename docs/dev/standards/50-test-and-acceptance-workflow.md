# Channel Test And Acceptance Workflow

Status: Active
Category: Standard
Last reviewed: 2026-04-03

## Suite Entry

Run tests from the `channel/` subproject root.

```bash
conda activate witwin2
cd channel
python -m pytest tests
```

## Gates

- `--gpu`: enable tests that require CUDA, RayD, or GPU-resident torch tensors.
- `--acceptance`: enable end-to-end acceptance and reference-validation coverage.

## Marker Policy

- `gpu`: runtime depends on CUDA.
- `acceptance`: broader end-to-end or parity validation.
- `validation`: solver/reference comparison or exported-audit checks.

## Recommended CI / Local Order

### Fast regression pass

```bash
python -m pytest tests -m "not acceptance" --gpu
```

### Scene migration acceptance

```bash
python -m pytest tests/test_core_scene_migration.py --gpu --acceptance
```

### Reference validation

```bash
python -m pytest tests/test_validation_references.py --gpu --acceptance
python -m pytest tests/test_validation_state_audit.py --gpu --acceptance
```

## Required Acceptance Signals

- Declarative `witwin.core` scenes initialize `Tracer` successfully.
- Core-scene runtime topology matches the normalized raw-mesh path.
- Runtime vertex updates refresh RayD and diffraction caches correctly.
- Small-grid tracer output matches between raw and declarative scene construction.
- Validation sweeps continue to export complete diffraction state audits.
