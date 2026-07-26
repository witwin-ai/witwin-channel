# Changelog

All notable changes to `witwin-channel` are documented in this file.

## [Unreleased]

### Changed

- **Breaking (ADR-039).** The propagation consumer publishes the declared
  source amplitude. `ScalarTransport.coefficient` and `Complex3Transport.field`
  now carry `sqrt(sources.powers_w)` of each row's own transmitting endpoint;
  `JonesTransport` stays excitation-free because a polarization-basis map is
  not a transported field. `CONTRACT_VERSION` moves from 2 to 3. A caller that
  already multiplied by `sqrt(powers_w)` itself must stop doing so.
- The Path solver's `phase_convention` metadata quotes the unit-excitation
  free-space amplitude, matching its own `coefficient_semantics`.

### Added

- Native `field_source_amplitude_scale` and its backward/JVP companions, the
  owner of the excited complex3 field vector.

## [0.4.0] - 2026-07-23

### Added

- Native Windows and `manylinux_2_28_x86_64` wheels for CPython 3.11 and
  PyTorch 2.10.
- Native CUDA SASS for SM70, SM75, SM80, SM86, SM87, SM89, SM90, SM100,
  SM101, and SM120, plus `compute_120` PTX forward compatibility.
- GitHub-hosted, release-gated wheel builds with final binary architecture and
  package-layout verification.

### Changed

- Source-link the locked, clean RayD 0.7.0 checkout at
  `49c58c4cb8212f6babb920cc88fb937509826cc5`.
- Group compatible CUDA code-generation targets and cache C/C++ compilation
  to reduce hosted-runner time without reducing release architecture coverage.
- Restrict paid native CI to explicit manual dispatches and published GitHub
  Releases. Ordinary pushes, pull requests, and schedules do not trigger it.

### Removed

- Obsolete self-hosted quick, CUDA, nightly, and evidence-only release
  workflows.

[0.4.0]: https://github.com/witwin-ai/witwin-channel/releases/tag/witwin-channel-v0.4.0
