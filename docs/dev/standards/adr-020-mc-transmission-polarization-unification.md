# ADR-020: Monte Carlo transmission polarization unification

Status: Accepted.

## Context

The WS1 alignment audit found that both Monte Carlo solvers evaluated specular
transmission with a polarization-agnostic model that diverges from the
deterministic/Path reference on any polarized oblique-incidence wall.

The deterministic and Path solvers evaluate pure transmission through the shared
enumerated engine: the straight-segment topology
(`propagation/enumerated/transmission.py`) is filled by the native full-Jones
layer-stack field kernel `field_transmission_sequence`
(`native/.../field_transport_transmission.cu`), which propagates the complex
TE/TM Jones coefficients of the incident polarization through each wall and
projects the result on the receiver antenna. On a lossy wall at oblique
incidence this yields the polarized transmittance, e.g. `T_te` for a
receiver-projected TE component, distinct from the unpolarized TE/TM mean.

Both Monte Carlo solvers instead scaled a Line-of-Sight-style contribution by the
unpolarized power transmittance `0.5*(cap_T_te + cap_T_tm)`:

- `montecarlo.basic` transmission radiomap: `transmission_component_map` ->
  `straight_transmission_chains` (`montecarlo/events/transmission.py`) marched the
  Tx->cell segment and multiplied the LoS matrix by
  `unpolarized_power_budgets(stack)` (the TE/TM mean) per wall.
- `montecarlo.bdpt` standalone transmission:
  `_transmission_straight_connection_samples`
  (`montecarlo/bdpt/connections.py`) marched the Tx->Rx segment through the same
  `straight_transmission_chains` helper and scaled the endpoint-connection LoS
  contribution by that mean, reclassified as component id 5.

The native layer-stack op `em_layer_stack_eval` already returns the polarized
complex Jones coefficients and the separate `cap_T_te` / `cap_T_tm` power
budgets; only the Python glue collapsed them to a mean.

Measured before this change, on the polarized-incidence fixture below (a lossy
wall, thickness 0.1 m, eps_r 4.0, sigma_e 0.05, at 45 degree oblique incidence,
3 GHz, transmitter polarization `[1, -1, 2]`):

- `T_te = 2.32429e-01`, `T_tm = 3.37313e-01`, unpolarized mean `2.84871e-01`.
- Intra-solver transmission/LoS ratio (isolates the transmittance model from each
  solver's LoS convention):
  - deterministic: `2.32431e-01` (receiver-projected full Jones)
  - montecarlo.basic: `2.84873e-01` (unpolarized mean)
  - montecarlo.bdpt: `2.84873e-01` (unpolarized mean)
- Cross-solver transmission component ratio to deterministic:
  - montecarlo.basic: `1.8384`
  - montecarlo.bdpt: `1.2256`

The BDPT diffraction realignment (ADR-018) already established the precedent that
a delta-like discrete path class BDPT was estimating with a physically wrong
heuristic should route through the shared enumerated engine as a unit-mass
discrete connection (ADR-008). Pure specular transmission is delta-like in
exactly that sense: for every (transmitter, receiver) pair the direct segment
either penetrates a fixed set of thin_sheet walls or it does not, and the
enumerated engine already evaluates its field with the deterministic full-Jones
kernel.

## Decision

Unify the transmission MODEL for `montecarlo.basic` and `montecarlo.bdpt` onto
the full-Jones layer-stack evaluation the enumerated engine uses. Each solver's
ESTIMATOR DOMAIN stays per-solver.

### montecarlo.bdpt: enumerated discrete connection (ADR-018 precedent)

`montecarlo.bdpt.pipeline` gains `_transmission_discrete_connection_samples`,
which calls the shared `_single_class_discrete_connection_samples` helper with
`component="transmission"`, `component_id=5`. That helper calls the public
`evaluate_enumerated_paths({"transmission"})` read-only (ADR-008), selects the
rows with `component_id == 5`, and packs them with
`_evaluated_connection_samples(..., component_out=5)`. Each selected row becomes a
single discrete connection with unit forward/reverse mass (`pdf = 1`,
`mis_weight = 1`), identical to the reflection, standalone-diffraction, and
coupled discrete-connection blocks. `_collect_connection_samples` appends this
block in place of `_transmission_straight_connection_samples` and increments
`launch_count` by 1 (was 2). `transmission_chain_count` (metadata
`straight_chain_paths`) becomes the number of enumerated transmission connection
rows.

The obsolete Python connection builder
`_transmission_straight_connection_samples` is deleted from
`montecarlo.bdpt.connections`, together with its now-unused imports
(`straight_transmission_chains`, and the `_TRANSMISSION_COMPONENT_ID` constant).

Consequently BDPT standalone transmission reproduces the deterministic component
power exactly (same native full-Jones enumerated field, receiver-antenna
projection included), and inherits the enumerated route's grid-radiomap
accumulation through the same connection-sample accumulator reflection already
uses.

### montecarlo.basic: incident-projected power-domain transmittance

`montecarlo.basic` keeps its power-domain radiomap estimator (the transmission
map is the analytic per-cell LoS gain scaled by the through-wall power
transmittance product). The per-wall transmittance in `straight_transmission_chains`
changes from the unpolarized mean to the Jones-derived power projected on the
incident polarization:

```
t_wall = f_te * cap_T_te + f_tm * cap_T_tm
```

where `(f_te, f_tm)` are the incident TE/TM power fractions of the transmitter
polarization in the wall's plane-of-incidence basis
(`incident_te_tm_fractions`, sharing the s/p convention with the scattering event
glue `te_tm_incident_power` and the native layer-stack kernel). Because the
radiomap accumulates total transmitted power, the projection is on the incident
polarization only, with no receiver-antenna projection (that is the estimator
domain difference from the deterministic/BDPT connection value). At normal
incidence the plane of incidence is degenerate; there `cap_T_te == cap_T_tm`, so
the split is irrelevant and the result is unchanged.

`straight_transmission_chains` gains a required `polarization` argument;
`transmission_component_map` threads the per-transmitter polarization vector into
it. Live-transmitter AD is preserved: the incident direction (and therefore the
TE/TM fractions) already move with the transmitter through the differentiable
straight-segment direction, so the polarized transmittance carries the same
transmitter-position gradient the incidence cosine already carried, plus the new
plane-of-incidence dependence. The projection weights are pure Torch on the
already-tracked direction and the frozen (detached) hit normal and polarization,
so the material and frequency gradients still flow only through the native
`em_layer_stack_ad` budgets, matching finite differences.

### Scope boundary: what does NOT change

- The BDPT event-selected shooting sampler (mixed reflection+transmission chains)
  is unchanged. It already applies the exact native full-Jones transmission field
  through `bdpt_transmitted_light_subpath_state`; its use of
  `unpolarized_power_budgets` is only the two-way event PROBABILITY
  `p_t = T/(R+T)`, which is the plan-sanctioned use of the mean and is disjoint
  from the pure-transmission class (it emits only chains carrying both a
  reflection and a transmission mask bit, never pure single-wall transmission),
  so there is no double count with the enumerated pure-transmission block.
- `unpolarized_power_budgets` and `transmission_event_probability` remain for the
  sampler's event-probability glue.
- No native ABI symbol is added or removed. `em_layer_stack_eval`,
  `em_layer_stack_ad`, and `field_transmission_sequence` are already bound and
  already have production callers. The native binding manifest, owner inventory,
  and binding count are unchanged. This is a Python-only realignment.
- Coherent combine (ADR-019) still refuses transmission at config validation; it
  is out of scope here.

## Compute-policy note

The MC-basic per-wall projection `f_te*cap_T_te + f_tm*cap_T_tm` is evaluated in
Torch over the native `em_layer_stack_eval` / `em_layer_stack_ad` outputs. This
is the same class of plan-sanctioned Monte Carlo event glue as the pre-existing
`unpolarized_power_budgets` mean it replaces and the scattering
`te_tm_incident_power` projection: the RF physics (the complex Jones coefficients
and the power budgets) is computed natively; the projection weights are
polarization bookkeeping over native outputs. No new Torch physics path is
introduced; an accepted one is refined. BDPT introduces no Torch physics at all
(it routes to the native enumerated field).

## Expected numerical change and acceptance

After this change, on the fixture above:

- Intra-solver transmission/LoS ratio:
  - deterministic: `2.32431e-01` (unchanged)
  - montecarlo.bdpt: `2.32431e-01` (now equals the deterministic value exactly;
    receiver-projected full Jones)
  - montecarlo.basic: `2.67391e-01` (incident-projected power
    `2/3 * T_te + 1/3 * T_tm`, distinct from both the deterministic
    receiver-projected value and the retired mean)
- Cross-solver transmission component ratio to deterministic:
  - montecarlo.bdpt: `1.0000` (was `1.2256`)
  - montecarlo.basic: `1.7256` (was `1.8384`; the residual is the pre-existing
    LoS-convention difference `mc_los/det_los ~= 1.50`, not the transmittance
    model)

Acceptance gates (all met):

- BDPT and MC-basic standalone transmission component within `[0.5x, 2x]` of the
  deterministic reference on the polarized-incidence wall fixture. Guarded by
  `tests/acceptance/test_adr020_transmission_polarization.py`.
- BDPT reproduces the deterministic transmission component power to tight
  tolerance (shared enumerated full-Jones field); MC-basic reproduces the
  incident-projected `f_te*T_te + f_tm*T_tm`, verified distinct from the mean.
- No MC sampling semantics change: transmission is a deterministic straight-march
  in both solvers, so visibility and pdf are untouched; the event-selected
  shooting sampler (visibility/pdf/random-number consumption) is unchanged.
- All existing transmission and transmission-AD suites stay green or are updated
  with justification (listed below).

### Updated tests (encoded the unpolarized mean)

Both existing lossy-wall power-ratio tests were written at exact normal
incidence, where `T_te == T_tm` and the polarized projection equals the mean, so
they still pass unchanged numerically; their expected expression is updated from
`0.5*(T_te + T_tm)` to the polarized `T_te` (with an assertion that
`T_te == T_tm` at normal incidence) to reflect the unified model rather than a
coincidental equality:

- `tests/montecarlo/bdpt/test_transmission.py::test_bdpt_lossy_wall_transmission_power_ratio_matches_stack`
- `tests/montecarlo/basic/test_basic_transmission.py::test_lossy_wall_attenuates_by_stack_power_transmittance`

No other suite encoded the mean. All other existing transmission tests use vacuum
walls (unit transmittance) or normal incidence, where the change is a numerical
no-op, and the transmission-AD tests (oblique lossy multilayer wall) compare
against finite differences of the actual forward, so they remain green with the
polarized forward.

### New tests

- `tests/acceptance/test_adr020_transmission_polarization.py`: the polarized
  oblique wall fixture across all three solvers, both a pure-TE case (all solvers
  reproduce `T_te`, distinctly below the mean) and a mixed-polarization case
  (BDPT matches deterministic; MC-basic is incident-projected).

## Enforcement

The BDPT -> enumerated import edge already exists for reflection, standalone
diffraction, and coupled paths and is admitted by `mc-enum-001` in
`ci/import_graph_allowlist.json` (ADR-008/ADR-018). Standalone transmission
reuses the same public `evaluate_enumerated_paths` entry imported on the same
line from the same module, so no new import edge is created and the allowlist is
unchanged. This ADR is the record that the admitted edge now also carries
standalone transmission.

## Revisit condition

Re-evaluate if MC-basic ever needs a receiver-antenna-projected transmission
radiomap (which would move it onto the same connection-domain estimator as BDPT),
or if the mixed reflection+transmission shooting sampler is folded into an
enumerated coupled-transmission class.
