# witwin-channel-native

Greenfield DrJit-free Torch/CUDA RF channel runtime under the `witwin.channel_native` namespace.

Current implementation plan:

- `docs/dev/plans/00-channel-native-greenfield-plan.md`

Vendored native dependency:

- `ext/raydn`

The upstream worktree path is currently `E:\Code\RayDTorch`, but that codebase
now uses the RayDN names: Python package `raydn`, native module `_raydn`, and
Torch dispatcher namespace `torch.ops.raydn.*`.
