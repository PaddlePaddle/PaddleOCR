fix(parseq_head): remove redundant cpu/cuda round-trip that crashes on non-CUDA backends

In the ParseQ `refine_iters` loop, `tgt_padding_mask` was explicitly moved to
CPU with `.cpu()` and then back to CUDA with `.cuda()`. This round-trip is
unnecessary — `paddle.cumsum` and dtype casting work correctly on any device
and the tensor is already on the correct device (`tgt_in` is built on the model
device).

On non-CUDA backends (CPU-only builds, Ascend NPU via Paddle custom device,
XPU, etc.), the `.cuda()` call raises because Paddle is not compiled with CUDA
support, crashing inference whenever `refine_iters > 0`.

Fix: drop the redundant `.cpu().cuda()` round-trip and keep `tgt_padding_mask`
on its current device. The resulting tensor is identical to before on CUDA, and
now works on non-CUDA backends.

## What was verified
- Logic equivalence on CUDA: removing `.cpu().cuda()` produces the same bool
  mask (`cumsum(axis=-1) > 0` then `astype("float32") == 1.0`), just without the
  device transfer.
- This is a static/device-agnostic change; I could not run a full Paddle OCR
  model here (no Paddle GPU/NPU runtime in my environment).
