"""Shared native-kernel wrappers consumable by both solver packages.

Modules under this subpackage own the Python entrypoints for native CUDA
kernels that are used by more than one solver (deterministic / montecarlo /
path). They are deliberately kept here rather than inside any one solver
package so the Python import graph between the solvers stays acyclic.

The compiled artifacts live in :mod:`witwin.channel._native`, built from the shared
native target sources and ``witwin/channel/core/kernels/``. This keeps both
Python ownership and the binary off the solver packages.
"""
