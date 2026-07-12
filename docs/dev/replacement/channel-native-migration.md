# Channel Native migration and runtime-dependency boundary

## Current decision

`witwin.channel_native` is the native entrypoint for the capabilities it
advertises. Repository-owned production Python must not import DrJit, Mitsuba,
Sionna, Python RayDN, or `witwin.channel`. The old Channel may remain in tests
and benchmarks as an offline correctness oracle; it must never be a production
fallback. The independent radar implementation is a separate product and is
not evidence for either Channel implementation.

The audited sibling roots (`core`, `genesis`, `maxwell`, `radar`, and `studio`)
contain no production Channel imports. This does not prove that external users,
deployed jobs, plugins, or private repositories have migrated.

The platform `core` package's `channel` and `all` extras now route to
`witwin-channel-native>=0.1,<0.2` (companion commit `9ee6655`) instead of the old
`witwin-channel` distribution. This establishes the repository-owned default
installation route; application-level canary/default-on state still requires
confirmation from each consumer owner.

## Rollout states

1. Inventory: route every real call to a supported Native capability or record
   an explicit unsupported product decision.
2. Shadow: execute old and Native implementations independently and record the
   comparison artifact below. Shadow failures must not trigger a production
   fallback.
3. Canary: make Native authoritative for a small declared cohort and retain the
   same correctness and operational evidence.
4. Default-on: the owning consumer makes Native authoritative. This repository
   exposes the Native entrypoint but cannot verify an application router or its
   rollout percentage; consumer-owner confirmation is required.
5. Delete: remove the old runtime integration only after every blocker below is
   closed.

## Shadow evidence artifact

Each maintained scenario must store versioned JSON under the release evidence
location chosen by CI. It must include: schema version, timestamp, release,
scenario/config/seed, Native and oracle commit/build identities, GPU/driver/
CUDA/OptiX/PyTorch metadata, cold and steady timing, peak memory, correctness
metrics and thresholds, pass/fail, and whether either side errored. Raw NPZ/JSON
outputs must be linked by content digest. No maintained Phase 10 shadow artifact
has been recorded in this repository yet; this is a required evidence contract,
not a fabricated run result.

The reduced three-way attempt on 2026-07-11 is recorded in
`path-threeway-shadow-attempt-2026-07-11.json`. Native timing and peak-memory
measurement completed, but both offline oracle processes failed during LLVM/
reference initialization, so the artifact is explicitly failed and does not
close the maintained shadow gate.

## Deletion blockers

Deletion remains blocked until all of the following are true:

- external consumers and private/deployed workloads have a signed inventory;
- maintained shadow and canary artifacts pass for the supported matrix;
- every consumer owner confirms Native is default-on with no production fallback;
- two consecutive release cycles complete without a fallback request;
- all P0/P1 items are closed or explicitly excluded by product decision;
- maintained correctness, performance, memory, cold-start and deployment gates pass;
- wheel smoke and the required GPU/SM matrix have runtime evidence;
- pipeline cache is either implemented and validated or explicitly removed from
  the release requirement.

As of 2026-07-11 the external audit, shadow/canary evidence, owner default-on
confirmation, two-release observation, wheel/SM evidence, and pipeline-cache
gate are not complete. Phase 10 therefore establishes the migration contract
and production dependency boundary but does not authorize deletion.

## Enforcement

Run the local contract with:

```powershell
python ci/check_production_dependencies.py
```

Sibling repositories can be audited without modifying them:

```powershell
python ci/check_production_dependencies.py --consumer-roots ..\core ..\genesis ..\maxwell ..\radar ..\studio
```

Consumer mode rejects the old Channel import only. This intentionally does not
classify Radar's independent DrJit/RayD tracer as a Channel runtime fallback.
