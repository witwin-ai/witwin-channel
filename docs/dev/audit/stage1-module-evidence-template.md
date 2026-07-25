# Stage-I module evidence template

Use this template once per large module, after implementation is complete.
Targeted tests during development are recorded in the phase commit message or
working notes; they do not trigger repeated full acceptance.

## Identity

- phase:
- repository:
- commit:
- dependency lock:
- build fingerprint:
- machine/GPU:
- Python/Torch/CUDA/compiler:
- clean checkout or source artifact:

## Scope

- intended changes:
- explicitly unchanged owners:
- deleted duplicate/legacy owners:
- deferred cross-phase findings:

## Functional and contract evidence

| Gate | Command/artifact | Result | Notes |
| --- | --- | --- | --- |
| targeted unit/contract | | | |
| public/import boundary | | | |
| exact identity/order | | | |
| invalidation/reuse | | | |
| fail-loud/no partial result | | | |
| package-neutral probe | | | |

## Numerical and AD evidence

- deterministic inputs and random seeds:
- reference/baseline commit:
- exact fields/hashes:
- tolerance-based fields and justification:
- primal/JVP/VJP row-identity evidence:
- unsupported capability preflight evidence:
- atomic or stochastic nondeterminism disclosure:

## Performance and resource evidence

- warmup count:
- measured repeats:
- CUDA event and host timing protocol:
- explicit synchronization points:
- median/p95/variance:
- peak-memory reset/sample protocol:
- launch-ledger source:
- D2H/sync ledger:
- cold-start isolation:
- baseline comparison and thresholds:

## Packaging evidence

- clean wheel/sdist build:
- packaged extension fingerprint:
- ABI/binding/source manifests:
- import/native-load smoke:
- Python/Torch rows:
- Linux manylinux environment:
- native SASS/PTX inventory, including SM87:

## Adversarial findings

| Severity | Finding | In phase? | Resolution or next phase |
| --- | --- | --- | --- |
| | | | |

## Acceptance

- [ ] all in-scope findings resolved
- [ ] no tolerance, allowlist, or budget weakened
- [ ] no new fallback or duplicate owner
- [ ] working tree clean except the intended phase
- [ ] phase commit is independent and reproducible
