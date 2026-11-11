# `witwin.channel.presets`

ITU-R P.2040 material presets and antenna-pattern factories. Every function
returns `witwin.core` values. User guide:
[Materials and antennas](../user/materials-and-antennas.md).

```python
from witwin.channel.presets import (
    ITU_MATERIALS, ItuBand, ItuBandDispersion, dipole_pattern, itu_material,
    itu_material_names, itu_properties, tr38901_pattern,
)
```

## `itu_material`

```text
itu_material(
    name: str, frequency_hz: float | tuple[float, float], *, dispersive: bool = True,
    thickness_m: float = 0.1, material_name: str | None = None, **material_kwargs,
) -> witwin.core.PhysicalMaterial
```

| Parameter | Description |
| --- | --- |
| `name` | ITU material name (`itu_material_names()`); case, spaces, hyphens and an `itu_` prefix are normalized. |
| `frequency_hz` | Compile frequency, or `(low_hz, high_hz)` band; must lie inside one ITU range. |
| `dispersive` | `True`: carry the power law as an `ItuBandDispersion`. `False`: evaluate once at `frequency_hz` (a single frequency). |
| `thickness_m` | Slab thickness. |
| `material_name` | `PhysicalMaterial.name`; default `itu_<name>`. |
| `**material_kwargs` | Passed to `PhysicalMaterial` (`scattering_coefficient`, `roughness_front`, ...). |

Raises `ValueError` for an unknown name, an out-of-range frequency or band, or
a band with `dispersive=False`; `TypeError` for a non-numeric frequency.

## `itu_properties`

`itu_properties(name, frequency_hz) -> tuple[float, float]` returns
`(eps_r, sigma_e)` at one frequency, with the same validation.

## `ITU_MATERIALS`, `ItuBand`, `itu_material_names`

`ITU_MATERIALS` is a read-only mapping from name to a tuple of non-overlapping
`ItuBand(min_frequency_hz, max_frequency_hz, a, b, c, d)` rows
(`eps_r = a * f_GHz**b`, `sigma = c * f_GHz**d`). `ItuBand.contains(f)` and
`ItuBand.evaluate(f)` test and evaluate one row. `itu_material_names()` returns
the sorted names.

## `ItuBandDispersion`

A frozen subclass of `witwin.core.PowerLawDispersion` with
`reference_frequency_hz=1e9` and the band's exponents, plus
`min_frequency_hz`, `max_frequency_hz` and `material`. Its `scale(frequency)`
raises `ValueError` for a Python-number frequency outside the range, so Core's
`evaluate_at_frequency` and Channel's scene compilation reject it. A tensor
frequency is not read back to the host and is not range-checked.

## `tr38901_pattern`

```text
tr38901_pattern(
    *, max_gain_dbi: float = 8.0, beamwidth_deg: float = 65.0, side_lobe_db: float = 30.0,
) -> witwin.core.AntennaPattern
```

Returns `AntennaPattern(kind="custom")` for the 3GPP TR 38.901 Table 7.3-1
element: `A = -min(-(A_V + A_H), side_lobe_db)` with
`A_V = -min(12 ((theta - 90) / beamwidth)^2, side_lobe_db)` and
`A_H = -min(12 (phi / beamwidth)^2, side_lobe_db)`, scalar field amplitude
`sqrt(10**((max_gain_dbi + A) / 10))`. Boresight is local `+x`, zenith local
`+z`.

## `dipole_pattern`

`dipole_pattern(kind="half_wave" | "short") -> witwin.core.AntennaPattern`
returns a z-oriented dipole normalized to its directivity (1.641 or 1.5).

Patterns are honoured by `witwin.channel.path`; the other solvers accept only
isotropic endpoints.
