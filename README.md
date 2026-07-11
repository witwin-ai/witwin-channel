# witwin-channel-native

Greenfield DrJit-free Torch/CUDA RF channel runtime under the `witwin.channel_native` namespace.

Current implementation plan:

- `docs/dev/plans/00-channel-native-greenfield-plan.md`

Native geometry dependency:

- RayD source tree (`backends/torch`), built in the same CMake graph as
  `_channel_native`

The default source location is `../../RayDi` relative to this repository. Set a
different checkout explicitly when configuring:

```powershell
cmake -S . -B build -DRAYD_SOURCE_DIR=E:/Code/RayDi
cmake --build build --config Release --target _channel_native
```

This integration builds and links `rayd_torch_native_core` directly. It does
not build/import RayD's Python module, use the Torch dispatcher, or load a
second DSO with `GetProcAddress`/`dlsym`. The legacy `ext/raydn` directory is no
longer part of the build graph and is retained only as migration history until
its remaining audit references are archived.
