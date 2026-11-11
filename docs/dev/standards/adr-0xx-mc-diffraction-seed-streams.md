# ADR-0xx: Independent MC diffraction sample streams for nearby seeds

- **Status:** Accepted.
- **Date:** 2026-11-11.
- **Decision ID:** `mc-diffraction-seed-streams`.
- **Scope:** `native/channel/kernels/diffraction.cu` (`dfr_uniform`, the point-receiver Keller-cone
  sampler) and RayD ADR-0049 (the diffraction accumulation sample tape that MC Basic grid maps
  use). Numerical change: MC Basic diffraction samples move for every seed.

## Context

Both samplers drew `hash(lane ^ stream_constant ^ seed)`. Seeds that differ only below the lane
count XOR-permute one set of hash inputs, so MC Basic diffraction estimates for seeds `1..K` reused
the same cone azimuths and edge offsets. Measured on the MC Basic diffraction scene of
`tests/ad/test_mc_basic_ad.py` at 4096 samples: 200 adjacent seeds gave an eps_r gradient mean of
`-1.79e-7 +- 0.04e-7`, 200 seeds spread over the high bits `-2.27e-7 +- 0.05e-7`, and 262144
samples `-2.20e-7`. A first fix premixed the user seed (splitmix64 to 31 bits) in Python; that
still shared a whole draw set with probability `samples / 2^31` per seed pair.

## Decision

Hash the lane first: `hash(hash(lane) ^ stream_constant ^ seed)`, in Channel's `dfr_uniform` and
in RayD's samplers (RayD ADR-0049), so a seed pair shares a draw only where
`hash(l1) ^ hash(l2)` equals the seed difference, about `samples^2 / 2^32` lanes. The Python
premix is removed: the raw seed reaches the kernels again, as before the first fix.

## Evidence

`tests/ad/test_mc_basic_ad.py::test_diffraction_seeds_closer_than_the_sample_count_draw_independent_streams`:
128 adjacent seeds (all below the 4096-lane count) and 128 seeds spread over the high bits estimate
the same gradient mean within four combined standard errors; with the old hash the gap was about
seven. Measured values are in the commit message and the Monte Carlo README.
