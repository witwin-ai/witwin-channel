"""Native kernel interface layer for witwin.channel.deterministic.

Each kernel lives in its own flat sub-package under ``kernels/``. Callers
import from the concrete kernel package, for example::

    from witwin.channel.deterministic.kernels.utd import utd_accumulate_forward
"""
