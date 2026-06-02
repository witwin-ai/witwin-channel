"""Native kernel interface layer for witwin.channel.

Kernel modules are organized by lifecycle:

- ``scene_build``: scene compilation and geometry preprocessing kernels.
- ``trace``: monitor-agnostic solver kernels and state utilities.
- ``monitors``: monitor accumulation kernels and shared tiling planners.

Callers should import from the concrete sub-package, for example::

    from witwin.channel.kernels.trace.utd import utd_accumulate_forward
"""