# ADR-0xx: Kirchhoff table cache keyed on roughness, layers and frequency

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `kirchhoff-table-cache`.
- **Scope:** `witwin.channel.scene.resources.build_kirchhoff_table` (compile-time table
  construction). No numerical change.

## Context

A Kirchhoff ensemble table depends only on the surface roughness
(`rms_height_m`, both correlation lengths, principal axis), the layer stack and
the frequency. It was rebuilt for every `CompiledScene` instance: 1.16 s for an
isotropic material and 11.5 s (101 MB) for an anisotropic one, again after every
endpoint move that produced a new compiled scene, and again in every process
(finding P-16).

## Decision

- The float64 host build (`_kirchhoff_table_arrays`) is cached in a process-wide
  LRU of 32 entries keyed on `(rms_height_m, corr_x, corr_y, principal_axis,
  layers, frequency_hz)` with exact float equality. The cache holds the float32
  host arrays; each `build_kirchhoff_table` call uploads fresh device tensors, so
  no two tables share storage and nothing a caller does to one table can reach
  another.
- Domain refusals (`kirchhoff_domain_exceeded`) run before the lookup and are
  never cached.
- Setting `WITWIN_CHANNEL_KIRCHHOFF_CACHE_DIR` also persists entries as
  `kirchhoff-<digest>.npz`, where the digest covers the key and a table-format
  tuple (`kirchhoff-v1` plus the four grid sizes). A corrupt or unreadable entry
  is rebuilt and rewritten; writes go through a temporary file and `os.replace`.
  Loading uses `allow_pickle=False`.
- `clear_kirchhoff_cache()` empties the in-process cache.
- The AD path (`_maybe_differentiate_table`) is unchanged: it replaces `f_te`/`f_tm`
  of the freshly uploaded table, never the cached arrays.

## Evidence

`tests/scattering/test_kirchhoff_cache.py`: a second build with the same key does
not call the host builder and returns bitwise-equal tensors in distinct storage; a
cached table equals an uncached rebuild bit for bit; changing frequency, layers or
roughness rebuilds; a disk round trip loads without rebuilding; a corrupt disk
entry rebuilds.

## Consequences

Bit-identical tables; a moved endpoint or a second scene with the same material
skips the host build. The disk cache must be bumped (`_KIRCHHOFF_FORMAT`) whenever
the table construction changes.
