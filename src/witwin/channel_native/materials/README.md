# Materials owner

## Ownership

`materials` owns material models, Material ABI v3 encoding, layer-CSR
validation, frequency evaluation, loader parsing, and native electromagnetic
layer-stack facades. Under ADR-024 and the completed Phase 6A pin/switch, RayD
is the numerical source owner of the resident layer-stack
primal/backward/JVP family; the model, ABI/CSR, cache, resource, validation,
and facade contracts remain here. Scene stores cache encoded records;
propagation and solvers consume them without redefining material semantics.

## Public entry points

`witwin.channel_native.materials` exposes the model classes in its `__all__`:
dispersion, dielectric/conductor, physical layer, roughness, phase-screen, and
surface-assignment types. The root package re-exports the five legacy material
classes frozen in the public API snapshot. Encoding, evaluation, and kernels
are internal.

## Dependency rules

Models and encoders do not import solvers or mutable scene runtime. Kernel
facades may depend on runtime symbol/tensor contracts and the typed RayD RF
entry, but may not reproduce its numerical implementation. Scene may consume
materials; materials cannot reach back into compiled scene or solver policy.

## Numerical and AD contract

Under `e^{+j wt}`, passive complex relative permittivity has non-positive
imaginary part, folded into non-negative equivalent conductivity. Layer records
use ABI v3 CSR and SI units. Layer order, roughness axes, frequency, dtype, and
device are contractual; tabulated data never extrapolates.

### AD contract

Layer-stack evaluation has native backward and JVP companions for supported
continuous inputs. Compiled records are frozen at the primal frequency, so
frequency AD through frequency-dependent material records is rejected.
Unsupported leaves fail instead of being detached or approximated.

## Forbidden fallback

Missing kernels, invalid CSR, unsupported frequencies, or stale ABI records do
not fall back to legacy scalar fields, CPU/PyTorch evaluation, vacuum/default
materials, or zero coefficients.

## Maintenance

Export changes require `ci/public-api-snapshot.json` and a migration note.
The completed kernel migration ledger is archived at
`docs/dev/audit/phase12-ops-migration-ledger.json`. ABI changes require a
versioned migration and fixtures; dependency changes must satisfy the
import-graph manifest.
