# Phase 12 final architecture and release acceptance

Phase 12 implementation closes at `1a774d9ffed36240d320102316d2f587c7ab32b1`. The complete release matrix was executed at `812e195ee79f1567ae93a0a3c69dd2a761570266`; the later commits add import/duplication governance and deduplicate native host wrappers without changing public ABI or CUDA kernels.

The user requested that additional testing stop during the final full-suite rebind. Consequently, this report does not claim that the runtime baseline, wheel, full pytest result, or complete manifest were rebound to `1a774d9`. No release tag was created.

## Architecture result

| Metric | Phase 0 | Final | Result |
|---|---:|---:|---|
| Production files / lines | 107 / 58,896 | 231 / 63,448 | More explicit domain files; +7.73% lines |
| Largest Python file | 11,596 | 1,835 | -84.18% |
| Largest native TU | 4,270 | 1,848 | -56.72% |
| Python complexity >=15 | 39 | 37 | Small reduction |
| Python functions >=100 lines | 37 | 41 | Did not improve |
| `core.kernels.ops` fan-in | 17 | 0 | Module deleted |
| `core.path_topology` fan-in | 7 | 0 | Module deleted |
| Solver-to-solver imports | 2 | 0 | Eliminated |
| Native bindings / duplicate names | 174 / 0 | 174 / 0 | Exact ABI surface retained |
| Tracked tree | 211.299 MiB | 10.421 MiB | -95.07% |
| Tracked `build-witwin3` | 316 files | 0 | Eliminated |

The global maximum fan-in increased from 17 to 31 because the new graph has explicit shared owners such as `scene.models`, runtime symbols, and tensor contracts. This is not represented as a global fan-in reduction; the improvement is removal of the untyped `ops` bottleneck and solver cross-imports.

The committed token-region duplication gate and classification ledger pass at the final SHA. The eight native whole-function duplicate groups of at least 100 tokens found at `812e195` were reduced to zero by sharing internal helpers behind 16 unchanged ABI wrappers. Existing long-function and maximum-complexity debt remains governed by maintenance budgets; the report does not claim those metrics improved.

## Acceptance evidence

At `812e195`:

- Full pytest: 1,628 passed, 1 skipped, 1 xfailed.
- Coverage: 85.52% statements, 66.02% branches, 90.08% core statements.
- Runtime: 8/8 result, metadata, input, and launch aggregates exact; worst median/p95/peak ratios were 0.3183 / 0.2704 / 1.0.
- Full statistics: 3/3 cases passed over 16 frozen seeds.
- Phase-E full: 35 measurements, 385/385 SM120 checks passed, all eight 100M preflights rejected before launch. Frozen Munich and San Francisco asset hashes matched.
- Wheel smoke passed for the 4,841,218-byte wheel, SHA-256 `c38e07e2aacefb4895526c64c3dd40462d879354d61684693c27659dbd217203`.
- The complete immutable manifest is `docs/dev/baselines/812e195ee79f1567ae93a0a3c69dd2a761570266/manifest.json`.

At final implementation `1a774d9`:

- Clean locked Release build: dirty=false, SM120, RayD `6047089cc7a41661402a02d40c96b9117e03a135`, fingerprint `887ea73b1b9e49505a4a5d84dc900252941ed0a058f370be196f3f29cc73a414`.
- CTest: 1/1 passed.
- Native host-wrapper and affected solver tests: 244 passed.
- Binding/owner audit: 16 passed; contract coverage remained 37 public exports and 174 native bindings.
- Ruff, duplication gate, and diff checks passed.

## Closure boundary

The final native change only consolidates identical host wrappers; it preserves exported names, validation order, tensor layouts, and kernel launch parameters. Nevertheless, the plan's strict release discipline requires evidence to be SHA-bound. Because the user stopped additional testing, a release tag should not be cut from `1a774d9` until full pytest, runtime capture, wheel build/smoke, Phase-E, and the complete immutable manifest are rebound to that SHA.
