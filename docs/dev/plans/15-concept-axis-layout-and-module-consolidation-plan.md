# Plan 15: Concept-Axis Layout and Module Consolidation

Status: **proposed — no work started, and none is to start without approval.**
The target layout in §3 has been reviewed and found sound. Execution is
deliberately deferred: prerequisites P1–P4 gate most steps, and three steps that
are technically unblocked (§5.1) are also on hold pending that approval.

Evidence base: `docs/dev/audit/python-surface-bloat-and-ad-substrate-audit.md`
(§0.3 unifying diagnosis, §I.8 ceremony surface, §I.9 file organization, §V.7
interaction-indexed field stage). This plan does not repeat that evidence; it
turns it into a target layout and a sequence.

### Decision log

| Date | Decision |
| --- | --- |
| 2026-07-27 | Target layout (§3) reviewed and accepted in principle. |
| 2026-07-27 | Solver files consolidated to `path.py`, `deterministic.py`, `montecarlo/basic.py`, `montecarlo/bdpt.py`. The flat `montecarlo.py` / `bdpt.py` form was considered and **not** adopted, because it breaks two public import paths for no additional flatness (§4.3). |
| 2026-07-27 | Execution deferred. No step begins until this plan is accepted. |
| 2026-07-27 | **Step 1 executed independently** (see revision note below). Remaining steps still deferred. |
| 2026-07-27 | **Phases 0, 1, 2 executed and verified.** Node-id set 2648 → 2670: **+22 additions, zero removals**. Full suite 2660 passed / 9 skipped / 1 xfailed / **0 failed**. See §5.3. |

### 5.3 Execution record — phases 0–2

| Phase | Outcome |
| --- | --- |
| 0 Policy | `file_lines` retired as an optional gate (not a raised number). `function_complexity: 15` and `native_file_lines: 3000/2000` stay mandatory. +4 tests. |
| 1a Ceremony | `component_status` 4 sites → 1; component depth rule 4 copies → 1, both owned by `components.py`. A **fifth** copy of the count-refined transmission/scattering rule was found in `deterministic/pipeline.py` and absorbed. |
| 1b Config | Shared bases for `path`/`deterministic` and `montecarlo.basic`/`bdpt`. Public config names, fields, defaults and messages unchanged; `ci/public-api-snapshot.json` untouched. |
| 2a Governance | New `ci/check_single_definition.py`, wired into `quick`. 3 protected concepts, 37 pre-existing `required_symbol` duplicates recorded as a **ratchet that may only shrink**. +18 tests. |
| 2b Governance | One of the two active import-graph debts repaid (`deployment.py` rewired). `mc_enumerated_dependency=1` remains — it is the ADR-008 sanctioned BDPT enumerated-oracle dependency and must stay. |

**Correction to audit §I.8.** The audit reported "four implementations of component
availability status". Two of them (`montecarlo/basic/metadata.py:22`,
`montecarlo/bdpt/metadata.py:75`) were four-line pass-through wrappers already
delegating to the canonical owner, not re-implementations. Only
`path/metadata.py:11` was a genuine second implementation, and it differed
substantively: it interleaved a `config.max_depth < 1` refusal per component and
resolved transmission/scattering to `enabled` / `enabled_no_paths` /
`not_requested` from exported path counts. Both behaviours were preserved, the
second as a separate `apply_exported_path_counts` owner.

**Stale exemptions retired as a consequence, not as a concession.** The phase-1
extractions dropped four exempted functions below complexity 15, which the
checker correctly reports as `stale-exemption` violations. All four were removed
(32 → 28); `montecarlo/bdpt/config.py::Config.__post_init__` had its ceiling
lowered 26 → 19. `test_current_baseline_is_exact_and_passes` asserts every
ceiling equals its measured complexity, so these are tightenings, not waivers.

**Cost note.** Suite runtime 295 s → 365 s (+24%), entirely from the 22 new
governance tests. No solver path changed.

**Audit outcome.** Four independent passes: three Claude lenses and one Codex
adversarial review. Codex verified 1,181 solver/component/depth metadata cases
and 6,245 config validation cases matched HEAD exactly, and found nothing. The
Claude `dropped-guard` lens found the `stale-exemption` red gate that the Codex
pass missed — because the Codex focus text steered it at metadata and config
equivalence and never had it run the gate. **Independent models do not
automatically give independent coverage; the brief decides the coverage.**

### Revision note — 2026-07-27, second pass

All measurements were re-taken against the tree at `73a553f`. Three landings
changed the baseline after this plan was first drafted:

| Commit | Effect on this plan |
| --- | --- |
| `9cc19d7` + `209cc33` — delete the dormant ADR-029/030/031 capacity artifacts | **Step 1 is done**, and went further than this plan scoped: 72 files changed, 23,362 deletions. It also removed `propagation/models/reflection.py` and `propagation/topology/kernels/reflection.py`, which this plan had counted as reflection-concept files. |
| `1606f0d` / `037ce18` / `637a8d4` — ADR-043 propagation AD capability matrix | Added `consumer/_ad_policy.py` (193) and grew `consumer/contracts.py` 1,103 → 1,200, `service.py` 1,087 → 1,102, `_fixed_reflection.py` 599 → 636. §3 and §4 sizes updated. |
| `73a553f` — ci: reject an unreachable production module | **Adjacent to step 3 but not the same rule.** It rejects a *resurrected dead module*; it does not reject a *second definition of a live concept*. Step 3 remains open — the four `component_status` implementations and the four copies of the component depth rule are all still present. |

**Net baseline change: 49,917 lines / 196 files → 52,201 / 187.** The package grew
by 2,284 lines net: ~2,000 lines of dead code came out and roughly 4,300 lines of
ADR-043 and related work went in. The consolidation argument is unaffected — if
anything the ceremony surface it targets is now more concentrated, since
`consumer/contracts.py` gained 97 lines and a twelfth sibling module appeared.

---

## 1. Context

`src/witwin/channel` is **52,201 lines across 187 files**, 133 of them at
directory depth 3–4. One domain concept is still spread across many files —
reflection lives in nine — and **41 of 187 files (22%)** are named after an
artifact category (`config`, `metadata`, `contracts`, `capacity`, `schema`,
`models`, `result`, `pipeline`, …) rather than after a domain thing.

Reflection today:

```
propagation/topology/discovery/reflection.py       discovery
propagation/geometry/reflection.py                 geometry
propagation/enumerated/reflection.py               enumerated orchestration
propagation/consumer/_fixed_reflection.py          frozen replay
propagation/fields/kernels/functional.py           field         (shared file)
propagation/fields/kernels/autograd.py             field AD      (shared file)
path/config.py + deterministic/config.py           its config fields
path | montecarlo.basic | montecarlo.bdpt /metadata.py   its metadata (three copies)
```

Two of the original eleven (`models/reflection.py`,
`topology/kernels/reflection.py`) were removed by the ADR-029/030/031 cleanup;
they were capacity artifacts, not reflection physics, so the scatter is smaller
but its shape is unchanged.

The layout materializes a matrix, concepts × (stages + artifact kinds), with one
file per non-empty cell. The change axis, however, is the concept axis: the
planned work (multi-order diffraction rebuild, T-matrix scattering fed from
Maxwell) consists of "everything about one concept" changes. The layout is
orthogonal to how the code actually changes.

Intended outcome: **187 → ~40 files, max depth 2**, with each interaction owning
one file, every native facade in one package, and no artifact-category modules.

---

## 2. Prerequisites

### P1 — Decide the field-stage index (audit §V.7). **Gates steps 6–8.**

Whether field kernels stay indexed by path word (`R`, `D`, `RD`, `DD`, …) or
become indexed by interaction. `interactions/reflection.py` is only a reasonable
size after this decision goes the second way; before it, the per-concept file
still has to carry word-specific field dispatch.

Needs its own ADR (touches the ADR-009 fusion boundary).

### P2 — Remove the Python file-size budget. **RESOLVED 2026-07-27.**

`ci/maintenance-budgets.json` enforces:

```json
"file_lines": { "hard": 2000, "recommended": 1200 }
```

**Owner decision: there is no maximum file-line limit.** The `file_lines` gate is
retired. CLAUDE.md forbids weakening a maintenance budget "just to make a change
pass," so this is recorded as a deliberate policy amendment with its reason: the
old limit encodes "a person cannot hold more than N lines in their head at once,"
and this codebase is now primarily read by agents. The argument applies to
**file size only**.

**`function_complexity: 15` stays.** It is about testability and defect density,
not reading capacity, and nothing about agent-assisted reading weakens it. The
three existing `file_exemptions` become dead entries and are removed with the
gate; the 32 `function_exemptions` are untouched.

**Consequence — this unblocks more than it looks.** The original plan gated
step 7 (`interactions/`) on P1 with the reasoning that
"`interactions/reflection.py` only stays a reasonable size after P1." With no
size limit, that reasoning no longer holds. **The concept-major file layout is
now separable from the field-stage restructure**: the layout can be reached by
pure code motion, and P1 decides only whether the *field kernels* are later
reindexed from path-word to interaction. Phase 5 below exploits this.

### P3 — Amend CLAUDE.md and ADR-036 for the ownership axis. **Gates step 6.**

The current ownership list asserts two axes at once — `scattering` (a concept)
alongside `propagation.topology` / `.geometry` / `.fields` (stages). That is why
`scattering` exists both as `scattering/` and as
`propagation/enumerated/scattering.py`. The concept axis must be stated as
primary for interactions, with stages retained only for the concept-agnostic
pipeline.

### P4 — Repay import-graph debt before moving the files that carry it.

`ci/check_import_graph.py` carries a `FROZEN_BASELINE_DIGEST` and states:

> "Entries may be removed from the active allowlist, but **relocating or
> replacing an entry is rejected**."

A file carrying an import-graph debt therefore cannot simply move — the debt must
be resolved first.

**This is a much smaller gate than it first appeared.** Measured at `73a553f`:

| Debt group | baseline (frozen history) | **allowed (active)** |
| --- | --- | --- |
| `existing_boundary` | 8 | **1** |
| `solver_to_solver` | 1 | **0** |
| `mc_enumerated_dependency` | 1 | **1** |

**Only two active debts exist.** The baselines are frozen history and do not
constrain movement; the two live entries do. P4 is therefore hours of work, not
weeks, and could be folded into step 1's follow-up rather than treated as a
phase.

**Still verify the baseline against reality.** It contains **8** references to
`witwin.channel.core`, a namespace that was dissolved (the checker itself carries
`_DISSOLVED_PREFIXES = ("witwin.channel.core",)`). The frozen baseline has
drifted from the tree; someone must decide whether it is re-baselined or the
stale entries are formally removed. See open question 4.

---

## 3. Target layout

```
witwin/channel/
├── __init__.py            public root: build_info · capabilities · pipeline_cache_key
│                          runtime_diagnostics · Complex3State · JonesState
├── constants.py           EM constants, phase convention                        ~90
├── field_state.py         Complex3State / JonesState native ABI                 ~80
├── components.py       ★  component registry: name · mask · depth_rule ·
│                          ad_modes · config_fields · status                     ~400
│                          (absorbs three metadata.py; anchors on ADR-043)
├── capabilities.py        solver capability + ADR-043 AD matrix                 ~300
├── deployment.py          build / runtime reporting                            ~220
├── runtime.py          ★  extension load · ABI · symbols · buffers · torch_compat
│                          memory budget · capacity transaction · autograd and
│                          tensor contracts · profiling      (12 files → 1)    ~1,400
│
├── scene/
│   ├── compiler.py        compile · CompiledScene · stores · tensors          ~1,100
│   ├── endpoints.py       endpoints · antenna · receiver geometry · AD seam      ~560
│   └── resources.py       scattering resources · edge policy/selection · rayd    ~790
│
├── materials.py           contracts + encoding + evaluation      (8 → 1)         ~360
├── scattering.py          tables + phase screen + energy         (10 → 1)        ~840
│
├── interactions/       ★★ the concept axis — one file per interaction
│   ├── __init__.py        registry; each interaction declares itself
│   ├── los.py                                                                    ~200
│   ├── reflection.py      discovery + geometry + operator + depth rule            ~800
│   ├── diffraction.py                                                            ~550
│   ├── transmission.py    (absorbs montecarlo/events/transmission.py)             ~600
│   ├── scattering_events.py  (absorbs montecarlo/events/scattering.py)          ~1,900
│   └── coupled.py         RD / DD — non-local, keeps a dedicated owner            ~700
│
├── kernels/            ★  every native facade (the analogue of gsplat _wrapper.py)
│   ├── __init__.py        declarative AD op registry (audit Tier 1)
│   ├── fields.py
│   ├── geometry.py
│   ├── topology.py
│   ├── scattering.py
│   ├── materials.py
│   └── montecarlo.py      mc_* and bdpt_* map/sampling/transmission facades
│
├── pipeline/              concept-agnostic stage machinery
│   ├── topology.py        concatenate · export · compaction driving              ~660
│   ├── geometry.py        endpoint / visibility / reevaluate plumbing            ~560
│   ├── fields.py          evaluate_path_fields                                 ~1,300
│   ├── chain.py        ★  the §V.7 composer (new; absorbs much of fields.py)        —
│   └── enumerated.py      engine                                                 ~470
│
├── consumer/              13 files / 4,872 lines today
│   ├── contracts.py       wire contracts only (validator + capability table out) ~900
│   ├── policy.py          ADR-043 AD matrix + admission (`_ad_policy` + capability) ~500
│   ├── service.py         evaluate / reevaluate — ONE pipeline, not two         ~1,000
│   └── replay.py          prepared · fixed · wideband · time-varying · jones    ~1,500
│
├── path.py                public entry — unchanged import path                 ~1,900
├── deterministic.py       public entry — unchanged import path                 ~1,110
└── montecarlo/
    ├── __init__.py
    ├── basic.py           public entry `witwin.channel.montecarlo.basic`       ~1,930
    └── bdpt.py            public entry `witwin.channel.montecarlo.bdpt`        ~3,700
```

**≈40 files, max depth 2.**

ADR-043 added `consumer/_ad_policy.py` after this layout was first drafted. It is
capability/admission policy, not a wire contract and not replay machinery, so it
gets its own destination (`consumer/policy.py`) rather than being folded into
`contracts.py` — which is the module §I.8 of the audit already flags as a
grab-bag.

---

## 4. Solver consolidation: measured

Question addressed: how much do `montecarlo.basic` and `montecarlo.bdpt` share,
and can each solver be a single file?

### 4.1 Sharing between the two Monte Carlo solvers

| | Lines | Files |
| --- | --- | --- |
| `montecarlo/basic/` | 4,607 | 14 |
| `montecarlo/bdpt/` | 6,527 | 18 |
| `montecarlo/events/` | 1,118 | 3 |

(`bdpt/` lost `subpaths.py` and one capacity module in the ADR-029/030/031
cleanup; `basic/` is unchanged.)

**They do not import each other.** Neither appears in the other's import list;
the solver-isolation rule holds here.

**Genuinely shared: only `montecarlo/events/` (1,118 lines).**
`basic/rayd_components.py:40-41` imports `events.scattering.scattering_map_matrix`
and `events.transmission`; `bdpt/connections.py` and `bdpt/pipeline.py` import the
same.

**Key finding: `events/` is not a Monte Carlo concept.**
`propagation/enumerated/scattering.py` imports it too. It is scattering and
transmission *event physics* that three callers share. In the target layout it
belongs in `interactions/scattering_events.py` and `interactions/transmission.py`,
**not under `montecarlo/`**. This removes the last reason for `montecarlo/` to be
a multi-level package.

**Duplicated rather than shared:** `basic/kernels/maps.py` (25 `mc_*` ops) and
`bdpt/kernels/maps.py` (19 `bdpt_*` ops) have **9 same-named ops**, 70% identical
after prefix normalization (~140 lines verbatim), already accidentally drifted —
one copy hand-rolls the symbol lookup and omits an output validation the other
has. See audit §I.2.

### 4.2 Can each solver be one file?

Re-measured at `73a553f`, i.e. with the dead-code removal already applied.

| Target | Package today | After kernels move to `kernels/` | After Tier 1 registry | Verdict |
| --- | --- | --- | --- | --- |
| `path.py` | 1,902 (8 files) | 1,902 | 1,902 | **yes, now** |
| `deterministic.py` | 1,540 (8 files) | **~1,110** | ~1,110 | **yes, now** |
| `montecarlo/basic.py` | 4,607 (14 files) | **~1,930** | ~1,930 | **yes, once kernels move** |
| `montecarlo/bdpt.py` | 6,527 (18 files) | ~5,030 | **~3,700** | **yes, after Tier 1** |

There is no file-size ceiling to land under: P2 retired the `file_lines` gate
entirely rather than raising it. The line counts are recorded as facts about the
resulting modules, not as headroom against a limit.

BDPT is the only one that resists, and only because ~1,800 lines of it are the
hand-written AD shells (`autograd.py` 903, `paths_ad.py` 546,
`autograd_accumulate.py` 351) that the Tier 1 registry collapses.

### 4.3 On `montecarlo.py` / `bdpt.py` as literal top-level files

Technically possible, but it **breaks two public import paths**:
`witwin.channel.montecarlo.basic` → `witwin.channel.montecarlo`, and
`witwin.channel.montecarlo.bdpt` → `witwin.channel.bdpt`. Both are in
`ci/public-api-snapshot.json`.

**Recommendation: keep `montecarlo/basic.py` and `montecarlo/bdpt.py`.** This
yields the same flatness (2 files instead of 34, depth 2) with zero public break.
Collapsing `path/` → `path.py` and `deterministic/` → `deterministic.py` is
transparent to importers and costs nothing.

If the flat `montecarlo.py` / `bdpt.py` form is wanted anyway, CLAUDE.md permits
it — "Public API changes require an intentional `ci/public-api-snapshot.json`
update and migration note" — but it should be a deliberate break, not a side
effect of a layout change.

---

## 5. Migration sequence

Each step lists its gate. **The order is load-bearing**: steps 6–8 redo
themselves if run before 1–5.

| # | Step | Gate | Status | Effect |
| --- | --- | --- | --- | --- |
| 1 | Delete dead code (audit §I.3) | none | **DONE** (`9cc19d7`, `209cc33`) | 72 files changed, 23,362 deletions incl. tests. Went beyond the scoped inventory. |
| 2 | Ceremony merge (audit §I.8) | none | open | three `metadata.py` → `components.py`; four `config.py` → shared base; ≈ −450 lines, −4 files. **No manifest change.** |
| 3 | Add the single-definition CI rule | none | open | a domain concept (component status, depth rule, symbol lookup) may have one definition site |
| 4 | Repay import-graph debt | P4 | open, small | **2 active entries only** — hours, not a phase |
| 5 | Move `events/` to `interactions/` | 2–4 | open | removes the shared-package reason for `montecarlo/` nesting |
| 6 | P1 decision + Tier 1 registry | P1, P2 | open | `kernels/` collapses 35 files → 7; BDPT drops below the file ceiling |
| 7 | Build `interactions/` | P1, P3, 6 | open | the concept axis; reflection 9 files → 1 |
| 8 | Collapse `pipeline/`, `consumer/`, solvers | 6, 7 | open | reach ≈40 files |

**Step 3 is not satisfied by `73a553f`.** That check rejects an *unreachable
production module* — a resurrected dead module. It does not reject a *second
definition of a live concept*. Verified still present at `73a553f`:

```
components.py:28                 component_availability_status()   ← canonical owner
path/metadata.py:11              _component_status()
montecarlo/basic/metadata.py:22  component_status()
montecarlo/bdpt/metadata.py:75   component_status()
```

and the component depth rule in **four** files (`path/metadata.py`,
`montecarlo/basic/metadata.py`, `montecarlo/bdpt/metadata.py`, and now
`deterministic/pipeline.py` as well). The duplication this step targets is
untouched, and has grown by one site since the first draft.

Step 3 matters more than its size suggests. Without it the duplication regrows:
`required_symbol` is hand-rolled at 37 sites and `component_availability_status`
at 3, in both cases with the canonical owner already present.

### 5.1 Steps unblocked today (on hold)

With step 1 landed, steps 2–4 depend on **no** prerequisite decision. They are
Python-internal work and touch no manifest at all.

| Step | Effect | Manifest churn |
| --- | --- | --- |
| 2 — ceremony merge | −450 lines, −4 files | **none** |
| 3 — single-definition CI rule | +1 check | **none** |
| 4 — repay 2 active import debts | 2 allowlist entries removed | allowlist only |
| **Total** | **≈ −450 lines, −4 files, +1 gate** | none |

Step 2 is the cheapest demonstration that the concept axis is the right one, and
it is reversible: it moves no public symbol and changes no number.

Step 3 should land **with or before** step 2, not after. Step 2 removes three
duplicate `component_status` implementations; without the rule that forbids a
fourth, the next solver-touching change reintroduces one. The evidence that this
happens is in the tree: `required_symbol` is hand-rolled at 37 sites, and the
component depth rule gained a fourth copy (`deterministic/pipeline.py`) between
the first and second drafts of this plan.

**These are on hold along with the rest.** They are listed separately so that
approval can be granted for them alone without committing to P1–P3.

---

## 5.2 Executable phases

Every phase below is **pure code motion plus import rewiring**. No phase may
change a number, a reduction order, a launch configuration, a kernel, or a
public symbol's behaviour. That property is what makes the acceptance gate
simple: the test suite is the numerical contract, and it must be bit-identical
in outcome before and after.

### Verification harness (established once, used by every phase)

```bash
export WITWIN_CHANNEL_DEVELOPER_OVERRIDE=1
export WITWIN_CHANNEL_EXTENSION_PATH=<repo>/.codex_tmp/phase11-objbuild/_channel.cp311-win_amd64.pyd
export WITWIN_CHANNEL_EXPECTED_FINGERPRINT=$(cat <same dir>/_channel.build-fingerprint)
export RAYD_SOURCE_DIR=E:/Code/witwin-platform/RayD     # ← REQUIRED, see below
PY=C:/Users/Asixa/miniconda3/envs/witwin2/python.exe

$PY -m pytest tests -q -p no:cacheprovider --basetemp=<tmp>     # numerical contract
$PY ci/run_ci_tier.py quick                                     # architecture gates
```

Confirmed working at `73a553f`: extension loads, `uses_rayd_native=True`,
`cuda_available=True`, `optix_available=True`, torch 2.10.0.

**`RAYD_SOURCE_DIR` is not optional and its absence is a trap.** Without it, eight
tests fail — all of them RayD native-boundary and phase-governance locks:

```
tests/test_field_transport_native_boundaries.py   (3)
tests/test_field_wedge_native_boundaries.py       (1)
tests/test_phase13_phase10b_governance.py         (1)
tests/test_phase13_phase8a_governance.py          (2)
tests/test_scattering_rayd_native_boundaries.py   (1)
```

With it set, the same 29 tests in those files pass in 3.1 s. These are precisely
the tests that lock the native surface a layout refactor must not disturb, so
running without the variable both hides the guard and invites an agent to "fix" a
test that was never broken. Every phase agent must be given the variable
explicitly.

**Baseline artifact**: the pass/fail node-id set and the CI-quick gate results
are captured before phase 0 and are the comparison target for every later phase.
A phase is accepted only when the node-id set is identical — not merely "still
green", because a rename that silently drops a test also stays green.

Baseline at `73a553f`, harness complete: **2638 passed, 9 skipped, 1 xfailed,
0 failed**, ~5 min. (An earlier capture missing `RAYD_SOURCE_DIR` reported
2630/8-failed; that run is void.)

### Phase table

| Phase | Content | Parallel within | Depends on | Gate |
| --- | --- | --- | --- | --- |
| **0** | Baseline capture; retire `file_lines` in `ci/maintenance-budgets.json` (P2) | — | — | baseline recorded; `quick` green |
| **1** | Ceremony merge: `component_status` ×4 → `components.py`; component depth rule ×4 → `components.py`; `path`/`deterministic` config base; `mc basic`/`bdpt` config base | 2 tasks | 0 | full tests node-identical |
| **2** | Governance: single-definition CI rule; repay the 2 active import-graph debts | 2 tasks | 1 | `quick` green; new rule fails on a planted duplicate |
| **3** | Artifact-category collapse, by package: `runtime/`→`runtime.py`; `materials/`→`materials.py`; `scattering/`→`scattering.py`; `propagation/models/`→owning stages; `scene/`→3 files; `consumer/`→4 files | **6 tasks** | 2 | per-package targeted tests, then full tests node-identical |
| **4** | Solver collapse: `path/`→`path.py`; `deterministic/`→`deterministic.py`; `montecarlo/basic/`→`basic.py`; `montecarlo/bdpt/`→`bdpt.py` | **4 tasks** | 3 | full tests node-identical; public import paths unchanged |
| **5** | Concept axis (move-only, **no longer gated on P1** — see P2): `montecarlo/events/`→`interactions/`; per-concept merge of discovery + geometry + enumerated into `interactions/<name>.py` | **6 tasks** | 3, 4 | full tests node-identical; `cuda` tier green |
| **6** | `kernels/` consolidation (move-only): every domain `kernels/` package → one top-level `kernels/` | 6 tasks | 5 | `native-binding-manifest.json` + `contract-coverage-manifest.json` updated in the same commit; `cuda` tier green |
| **7** | `pipeline/` assembly and final tree; delete empty packages | 3 tasks | 6 | full tests node-identical; `quick` + `cuda` green |
| **A** | **Adversarial audit** — runs against every phase, not once at the end | N per phase | each phase | see brief below |

### Adversarial audit brief

Pure code motion fails in specific, enumerable ways. Each phase's audit agents
are told to **refute** the claim "this change is behaviour-preserving" by hunting
for exactly these:

1. **Dropped validation** — a `raise`, `assert`, `validate_cuda_tensor`, or
   `require_*` present before the move and absent after.
2. **Changed evaluation order** — statements reordered inside a merged function;
   module-level initialisation now running earlier, later, or twice.
3. **Changed module-level side effects** — a registration, a cache construction,
   or an import-time check that now fires at a different time or not at all.
4. **Silently narrowed `__all__` / public surface** — a name that was importable
   and now is not, or vice versa.
5. **Symbol identity change** — two call sites that shared one function object
   now getting two, or a dataclass rebuilt so `isinstance` fails across modules.
6. **Test set shrinkage** — a test file merged or renamed such that its node-ids
   vanish from the baseline set.
7. **Import-time cycle or ordering hazard** introduced by consolidation.

An audit finding is only accepted when the auditor names the pre-move
`file:line`, the post-move `file:line`, and the concrete input that would behave
differently.

### What is deliberately NOT in these phases

- **The Tier 1 declarative AD registry.** Phase 6 moves the kernel facades; it
  does not restructure the 47 hand-written `autograd.Function` classes. That is a
  behaviour-bearing change and needs its own ADR and its own evidence.
- **The interaction-indexed field stage (P1 / audit §V.7).** Phase 5 achieves the
  concept-major *layout*; it does not reindex the field kernels. P1 remains open
  and is still gated on the UTD-locality question (audit appendix item 7c).
- **Any resolution of audit Tier 3** (the Torch-physics policy breaches in
  `enumerated/scattering.py` and `scene/antenna.py`). Those are wrong-result
  problems, not layout problems, and moving them must not be confused with fixing
  them.

## 6. Governance impact

Four machine-checked artifacts index code by path and must migrate with it:

| Artifact | Impact |
| --- | --- |
| `ci/import_graph_allowlist.json` | frozen digest; entries cannot be relocated, only removed → **P4**. Only **2 active entries**, so this is a small gate. 8 stale `witwin.channel.core` baseline references remain. |
| `ci/contract-coverage-manifest.json` | keyed on full Python owner paths (e.g. `witwin.channel.propagation.enumerated.capacity._EvaluatedPathsCapacityPackFunction.forward`) |
| `ci/native-binding-manifest.json` | same |
| `ci/maintenance-budgets.json` | 3 file exemptions + 32 function exemptions, all exact paths |

Two further items:

- **ADR-008's BDPT exception** permits `montecarlo.bdpt.pipeline` to call the
  public `evaluate_enumerated_paths` through `propagation.__getattr__`. That entry
  point becomes `pipeline/enumerated.py`; the exception must be re-pointed, not
  quietly widened.
- **`ci/public-api-snapshot.json`** is untouched under the recommendation in §4.3.

---

## 7. Non-goals

- No numerical change. No kernel is touched, no reduction order altered, no
  launch configuration changed. If a step would change a number, it is out of
  scope and belongs in its own ADR with exactness evidence.
- No new fallback, compatibility shim, or re-export layer. Files move; import
  paths change; callers are updated in the same commit.
- Not a rewrite of the CUDA. See audit §V.8: Channel's 36k lines of `.cu` are
  orchestration and stay CUDA.
- `interactions/scattering_events.py` is sized at ~1,900 lines assuming the
  current content. Audit §I.5 flags `enumerated/scattering.py`'s
  `realization_coherent` as the largest policy breach in the package (Torch
  physics and host loops on a production path). Resolving that is **audit Tier 3**
  and is independent of this plan; if it lands first, this file drops to ~1,200.

---

## 8. Open questions

1. **P1's answer.** If UTD diffraction is not expressible as a local 2×2 operator
   once its spreading factor and transition function are included, the
   interaction-indexed field stage loses most of its benefit and step 7 should be
   rescoped to a file move without a kernel restructure. See audit appendix item 7(c).
2. **Whether `pipeline/` should be one file.** Five files at depth 2 versus one
   `pipeline.py` at ~2,500 lines. The five-file form is proposed because the
   stages have genuinely different owners and because `chain.py` is expected to
   absorb most of `fields.py` after P1.
3. **Whether `consumer/` collapses to one file.** ~3,150 lines total. Feasible
   under P2's raised ceiling, but `service.py` currently contains two parallel
   replay pipelines (audit §I.6); merging those first would decide the answer.
4. **The import-graph baseline drift** noted in P4 needs an owner: it references a
   dissolved namespace, so someone must decide whether the frozen digest is
   re-baselined or the stale entries are formally removed.
