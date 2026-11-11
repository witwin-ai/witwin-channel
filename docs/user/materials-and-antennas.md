# Materials and antennas

Materials and antennas are `witwin.core` values (`PhysicalMaterial`,
`AntennaState`, `AntennaPattern`). `witwin.channel.presets` builds common ones
for you; it never defines new material or antenna types.

## ITU-R P.2040 materials

```python
from witwin.channel.presets import itu_material, itu_material_names, itu_properties

concrete = itu_material("concrete", 3.5e9)                   # dispersive, valid 1-100 GHz
glass = itu_material("glass", (2e9, 6e9), thickness_m=0.01)  # valid for a band
ground = itu_material("medium_dry_ground", 3.5e9, dispersive=False)
print(itu_material_names())
print(itu_properties("brick", 28e9))  # (eps_r, sigma_e in S/m)
```

ITU-R P.2040-3 Table 3 models the real relative permittivity and the
conductivity as power laws in frequency (in GHz):

```text
eps_r(f) = a * f_GHz ** b        sigma(f) = c * f_GHz ** d   [S/m]
```

| Name | a | b | c | d | Valid range (GHz) |
| --- | ---: | ---: | ---: | ---: | --- |
| `concrete` | 5.24 | 0 | 0.0462 | 0.7822 | 1-100 |
| `brick` | 3.91 | 0 | 0.0238 | 0.16 | 1-40 |
| `plasterboard` | 2.73 | 0 | 0.0085 | 0.9395 | 1-100 |
| `wood` | 1.99 | 0 | 0.0047 | 1.0718 | 0.001-100 |
| `glass` | 6.31 / 5.79 | 0 | 0.0036 / 0.0004 | 1.3394 / 1.658 | 0.1-100 / 220-450 |
| `ceiling_board` | 1.48 / 1.52 | 0 | 0.0011 / 0.0029 | 1.075 / 1.029 | 1-100 / 220-450 |
| `chipboard` | 2.58 | 0 | 0.0217 | 0.78 | 1-100 |
| `plywood` | 2.71 | 0 | 0.33 | 0 | 1-40 |
| `marble` | 7.074 | 0 | 0.0055 | 0.9262 | 1-60 |
| `floorboard` | 3.66 | 0 | 0.0044 | 1.3515 | 50-100 |
| `metal` | 1 | 0 | 1e7 | 0 | 1-100 |
| `very_dry_ground` | 3 | 0 | 0.00015 | 2.52 | 1-10 |
| `medium_dry_ground` | 15 | -0.1 | 0.035 | 1.63 | 1-10 |
| `wet_ground` | 30 | -0.4 | 0.15 | 1.30 | 1-10 |

Names are case-insensitive; spaces, hyphens and an `itu_` prefix are accepted.

**Frequency ranges are enforced.** `itu_material(name, frequency_hz)` raises
`ValueError` naming the valid ranges when `frequency_hz` (or every frequency of
a `(low_hz, high_hz)` band) is not inside one ITU range.

**Dispersion.** By default the material carries the ITU law as a Core
`PowerLawDispersion` (reference 1 GHz, exponents `b` and `d`), so
`scene.compile(..., reference_frequency_hz=f)` and every solver evaluate it at
the compile frequency `f`, and refuse an `f` outside the material's range.
Channel compiles one frequency per scene: a dispersive material is frozen at the
compile frequency, which means

- wideband replay with `frequency_offsets_hz` refuses scenes with dispersive
  materials (`NotImplementedError`), and
- a gradient with respect to the frequency is not available through them.

For those uses pass `dispersive=False`: the law is evaluated once at
`frequency_hz` and the material is a plain `PhysicalMaterial`. Across a band
`df` the frozen value then differs from the ITU law by roughly
`b * df / f` in `eps_r` and `d * df / f` in `sigma`.

`thickness_m` (default 0.1 m, as in Sionna) sets the slab thickness used by
transmission. Other keywords pass through to `PhysicalMaterial`, for example
`scattering_coefficient=0.3` or `roughness_front=SurfaceRoughness(...)`.
For an ideal conductor use `PhysicalMaterial.perfect_conductor()` instead of
`metal`.

## Antenna patterns

Core already provides `"isotropic"`, `"vertical"` (a z-directed short dipole
with unit peak gain) and `"horizontal"` patterns, plus `AntennaPattern(kind=
"custom", custom=callable)` for anything else. The presets add:

| Factory | Pattern | Peak gain |
| --- | --- | --- |
| `tr38901_pattern()` | 3GPP TR 38.901 Table 7.3-1 element: 65 degree beamwidths, 30 dB side-lobe and front-to-back limit | 8 dBi |
| `dipole_pattern("half_wave")` | `cos(pi/2 cos(theta)) / sin(theta)` along local z | 2.15 dBi |
| `dipole_pattern("short")` | `sin(theta)` along local z | 1.76 dBi |

```python
from witwin.core import AntennaState
from witwin.channel.presets import tr38901_pattern

sector = AntennaState(
    antenna_id, "tx", position, power_w=1.0, pattern=tr38901_pattern(),
    orientation=(yaw, 0.0, 0.0),        # boresight is the local +x axis
    polarization=(0.0, 0.0, 1.0),
)
```

The TR 38.901 element looks along its local +x axis with zenith along local +z;
rotate it with `AntennaState.orientation` (yaw, pitch, roll). The patterns are
normalized to their directivity, so their mean power over the sphere is 1, the
same as `"isotropic"`; compared with an isotropic link the received power rises
by the peak gain in the boresight direction. They return a scalar field
amplitude; the polarization of the endpoint still comes from
`AntennaState.polarization`.

Directional patterns are honoured by the Path solver. The Deterministic and
Monte Carlo solvers accept only isotropic endpoints and raise a `ValueError`
for a directional pattern.
