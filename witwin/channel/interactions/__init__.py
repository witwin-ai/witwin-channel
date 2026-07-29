# Copyright Xingyu Chen.
# One module per RF interaction concept; intentionally no eager imports.

"""One module per RF interaction concept; intentionally no eager imports.

Each concept owns its discovery, geometry and enumerated orchestration in a
single module: ``los``, ``reflection``, ``diffraction``, ``transmission``,
``scattering`` and ``coupled``. ``transmission`` and ``scattering`` also own
the specular-transmission and Kirchhoff scattering *event* helpers the two
Monte Carlo solvers share, which used to sit under ``montecarlo/events/``
despite having a third, enumerated consumer.

This package publishes no surface of its own. Native evaluation stays with the
kernel facades in ``witwin.channel.kernels``, the typed row contracts stay in
``witwin.channel.propagation.rows``, and every consumer imports the concept
module it needs directly, so each name keeps exactly one import path.
"""