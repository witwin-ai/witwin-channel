# Scene version traversal cost

## Why this is on the hot path

`Scene.topology_version`, `.geometry_version`, `.material_version`, and
`.assignment_version` are computed properties. Each walks every structure,
collects the identity, device, dtype, shape, stride, `requires_grad`, and
`_version` of every tensor leaf it reaches, and hashes the result.

`witwin.channel.scene.compile` reads all four **before** it consults its cache:

```python
versions = _versions(scene_or_snapshot)
key = _cache_key(scene_or_snapshot, versions, _frequency_identity(...))
cached = _REGISTRY.get(key)
if cached is not None and cached.source is scene_or_snapshot:
    return cached
```

and every solver entry point calls `compile` on each `solve`. A cache hit
therefore still pays four full scene walks. An optimization loop that solves
repeatedly against one unchanged scene pays it on every iteration.

This is the cost of the Stage-I ownership move. Before it, Channel's scene
stored `_geometry_version: int` and bumped it in `with_*`, which read in O(1)
but could not see an in-place tensor mutation. The walk buys that detection.

## What actually cost the time

Profiling the four properties over a 512-structure scene, the hashing was not
the problem. `repr` plus `blake2b` was 3% of the total. The cost was per-node
type dispatch:

| | share |
|---|---|
| `_tensor_states` recursion | 30% |
| `isinstance` (632,550 calls) | 11% |
| `typing.Mapping.__instancecheck__` / `__subclasscheck__` | 12% |
| `hasattr` (169,010 calls) | 5% |
| `dataclasses.is_dataclass` | 5% |
| `repr` | 3% |

`typing.Mapping` in an `isinstance` check routes through the `typing` generic
alias machinery on top of the underlying ABC check, which is why it alone cost
four times as much as the hashing.

## What changed

The traversal now resolves each node's kind once per class into a dict, uses
`collections.abc.Mapping`, memoizes device and dtype strings, caches dataclass
field names, and skips scalar leaves without a recursive call.

The emitted state tuples are unchanged, so the version values are unchanged.
`tests/test_scene_version_traversal.py` pins the dispatch contract, and the
equivalence was verified by running the pre-optimization implementation and the
current one over the same live objects across every dispatch branch, with and
without identity, and requiring identical output.

## Measured

Four version properties, one read, median of 20:

| structures | before | after | speedup |
|---:|---:|---:|---:|
| 16 | 1.35 ms | 0.26 ms | 5.2x |
| 64 | 5.91 ms | 1.00 ms | 5.9x |
| 256 | 26.4 ms | 4.00 ms | 6.6x |
| 1024 | 112 ms | 16.8 ms | 6.7x |
| 4096 | 455 ms | 82.0 ms | 5.6x |

Warm-cache `witwin.channel.scene.compile`, median of 15:

| structures | before | after | speedup |
|---:|---:|---:|---:|
| 16 | 1.02 ms | 0.27 ms | 3.8x |
| 64 | 4.58 ms | 1.40 ms | 3.3x |
| 256 | 23.8 ms | 4.16 ms | 5.7x |
| 1024 | 87.0 ms | 16.9 ms | 5.2x |

RTX 5080, CPython 3.11.14, Torch 2.10.0. The walk is still linear in structure
count at roughly 16 us per structure, down from roughly 110 us.

## What is still open

This is a constant-factor fix. A warm `compile` on a 1024-structure scene is
still ~17 ms of host work, against a recorded propagation solve of ~3.5 ms.
Removing the walk from the cache-hit path entirely - memoizing the version
against a cheap mutation signal the scene owns - is a separate change, because
it alters the ADR-034 invalidation contract rather than just its cost.

Note also that the Stage-I Phase-3 performance evidence cannot see any of this:
its benchmark compiles `Scene()`, an empty scene, once outside the timing loop.
There is no gate covering per-solve compile cost against scene size.
