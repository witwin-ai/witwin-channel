# Channel AD decision and migration contract

## Decision

The first Channel replacement release is primal-only. The public
capability manifest reports `supports_ad=False`, and Path, Deterministic,
Monte Carlo Basic, and Monte Carlo BDPT accept only `ad_mode="none"`.
Requesting `jvp`, `vjp`, `forward`, `reverse`, or any other AD mode fails while
constructing the solver configuration, before scene compilation, native
extension dispatch, or GPU allocation.

This is a product-scope decision, not a claim that RF channel derivatives are
unimportant. It prevents a partial derivative implementation from silently
omitting topology, visibility, polarization, material, or stochastic terms.

## Repository audit

The 2026-07-11 audit searched the repository-owned production consumer roots
`core`, `genesis`, `maxwell`, `radar`, and `studio`. It found no imports or
calls that couple those packages to legacy Channel AD or Channel AD.

The legacy `channel/examples`, `channel/tests`, `channel/tutorials`, and
`channel/scripts` reference roots contain 19 files with an explicit
`dr.backward(...)`, `ad=True`, or equivalent AD request. These are examples,
tests, tutorials, and validation tools. They remain offline migration oracles;
they are not production dependencies. External consumers are not visible to
this repository and must be audited before a release removes legacy Channel.

Channel also contains two low-level MC LoS derivative primitives:
`mc_los_path_gain_backward` and `mc_los_path_gain_jvp`. They are exercised by
kernel tests against the free-space analytic derivative and central finite
differences. No public solver calls them. Their presence does not imply
fixed-topology solver AD and does not alter the public capability.

## First-release contract

- `capabilities()["supports_ad"]` is false.
- Every solver-specific capability reports `supports_ad=False` and
  `ad_modes=["none"]`.
- `ad_mode="none"` means primal evaluation only. Solver metadata reports
  `ad_status="none"` and no AD tape or derivative launch.
- Non-primal modes fail at configuration construction. There is no zero-gradient
  fallback and no implicit finite-difference execution.
- The low-level LoS derivative kernels are experimental validation primitives,
  not a stable public optimization API.

## Analytic and finite-difference oracle

For free-space LoS power,

`G(tx, rx, P) = P * (c / (4*pi*f))^2 / ||tx-rx||^2`.

The maintained kernel test compares the experimental JVP/VJP primitives with
the analytic derivatives and a centered finite difference at fixed topology.
This oracle validates the derivative primitive only. It must not be used to
claim derivatives for visibility decisions, reflection/diffraction topology,
materials, polarization frames, arrays, or Monte Carlo sampling.

## Future migration gate

AD can be advertised only after a separate capability version completes both
stages below:

1. A PyTorch-native, topology-fixed JVP/VJP contract with declared
   differentiable inputs, stable path identity, analytic and finite-difference
   tests, and legacy Channel reference comparisons.
2. An explicit estimator for topology and visibility discontinuities. These
   discontinuities must not be presented as ordinary continuous gradients.

Until then, optimization workflows must stay on legacy Channel as an offline
tool or use an application-owned finite-difference loop around primal Channel
Native solves. Such a loop is outside the Channel solver contract.
