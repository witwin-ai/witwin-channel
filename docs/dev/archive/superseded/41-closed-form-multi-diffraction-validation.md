# Closed-Form Multi-Diffraction Validation

Status: Superseded
Category: Archive
Last reviewed: 2026-05-16

> **Archive note:** the `witwin/channel/validation.py`, `samples/save_wedge_validation_suite.py`, and `tests/test_validation_references.py` harness referenced below was removed during the channel package rewrite and has not been re-ported under `witwin/channel/deterministic/` or `witwin/channel/montecarlo/`. Treat this document as a literature-ladder snapshot only; a fresh standard will replace it once the harness is rebuilt.

## Scope

This repository now contains a validation harness for canonical
double-wedge, triple-wedge, and first-order overlap scenes. The current
implementation now includes:

- an explicit pair-expansion complex-field reference for canonical
  double diffraction on the double-wedge case
- an explicit triplet-expansion complex-field reference for canonical
  triple diffraction on the triple-wedge case
- a first-order overlap harness against the local Sionna RT utility
  implementation for PEC-like shared-path cases
- the existing solver-side order sweeps and exported bundles

Its purpose is now to lock down:

- canonical scene definitions
- order sweeps for `max_diffractions = 1, 2, 3`
- explicit order-2 double-diffraction reference comparison
- explicit order-3 triple-diffraction reference comparison
- first-order external-overlap comparison against local Sionna RT utilities
- exported field maps and line cuts
- a stable data format for future closed-form/reference comparisons

The current implementation lives in:

- `witwin/channel/validation.py`
- `samples/save_wedge_validation_suite.py`
- `tests/test_validation_references.py`

## Literature Ladder

These are the main references to encode and validate against next:

1. Double diffraction baseline:
   - Tiberio, Manara, Pelosi, Kouyoumjian (1989), closed-form double
     diffraction expressions for two wedges.
   - Metadata: https://arpi.unipi.it/handle/11568/10805
2. Uniform double-wedge coefficient:
   - Schneider and Luebbers (1991), "A general, uniform double wedge
     diffraction coefficient".
   - Metadata: https://pascal-francis.inist.fr/vibad/index.php?action=getRecordDetail&idt=19461695
3. Arbitrary-configuration double wedge:
   - Albani (2005), "A uniform double diffraction coefficient for a pair of
     wedges in arbitrary configuration".
   - Metadata: https://usiena-air.unisi.it/handle/11365/24951
4. Triple diffraction:
   - Carluccio, Puggelli, Albani (2012), "A UTD Triple Diffraction Coefficient
     for Straight Wedges in Arbitrary Configuration".
   - Metadata: https://usiena-air.unisi.it/handle/11365/43815
5. Secondary survey source used to cross-check the validation roadmap:
   - TUM dissertation (2023) discussing higher-order wedge diffraction
     references and transition-region limitations.
   - PDF: https://mediatum.ub.tum.de/doc/1691763/1691763.pdf

## What Must Be Validated

When a reference implementation is added, every case must be
checked at the complex-field level, not just in power:

- geometry convention for `phi`, `phi_prime`, wedge ordering, and path sequence
- distance/spreading factor used by the closed-form coefficient
- transition-function behavior near SB and RB overlaps
- limiting behavior when a higher-order path degenerates to a lower-order one
- line cuts through boundary regions and interior shadow regions
- order-by-order incremental field contribution

## Current Harness

The current harness exports, for each canonical case:

- diffraction power map for each order sweep
- total-field power map for each order sweep
- one fixed line cut through the field
- incremental order differences
- path-level diffraction state audits with edge history and incident data
- metadata in JSON and NumPy NPZ form

The current validation helpers also expose:

- `evaluate_closed_form_double_diffraction_reference()` for explicit
  order-2 double-diffraction pair expansion on the canonical double-wedge case
- `evaluate_closed_form_triple_diffraction_reference()` for explicit
  order-3 triple-diffraction triplet expansion on the canonical triple-wedge case
- `compare_first_order_overlap_against_sionna()` for direct first-order
  overlap checks against local Sionna RT utilities on the single-wedge case

Run:

```powershell
conda activate witwin2
python samples\save_wedge_validation_suite.py
```

Default outputs:

- `figures/validation/double_wedge_solver_bundle.png`
- `figures/validation/double_wedge_solver_bundle.npz`
- `figures/validation/triple_wedge_solver_bundle.png`
- `figures/validation/triple_wedge_solver_bundle.npz`

## Remaining Gap

The next validation step is to move beyond the current explicit double/triple
expansions and first-order overlap coverage by adding:

- literature-derived arbitrary-configuration double/triple coefficients where
  they materially differ from the current explicit scalar expansion references
- boundary-region overlays and error summaries integrated into the saved bundle
- additional transition-region checks against published higher-order formulas
