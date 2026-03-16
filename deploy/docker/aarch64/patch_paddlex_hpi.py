#!/usr/bin/env python3
"""Patch PaddleX's hpi.py to add aarch64 architecture support.

The suggest_inference_backend_and_config function rejects all non-x86_64
architectures.  This patch adds aarch64 handling that prefers ONNX Runtime
(with auto paddle2onnx conversion) as the inference backend.
"""

import pathlib
import sys

SITE_PACKAGES = (
    pathlib.Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
HPI_PATH = SITE_PACKAGES / "paddlex" / "inference" / "utils" / "hpi.py"

OLD = '''\
        if arch == "x86_64":
            key = "cpu_x64"
        else:
            return None, f"{repr(arch)} is not a supported architecture."'''

NEW = '''\
        if arch == "x86_64":
            key = "cpu_x64"
        elif arch in ("aarch64", "arm64"):
            # aarch64/arm64: no model-specific prior knowledge yet.
            # Prefer ONNX Runtime via ultra-infer (ENABLE_ORT_BACKEND=ON).
            # See: https://github.com/PaddlePaddle/PaddleOCR/issues/17590
            aarch64_backends = [
                b for b in ("onnxruntime", "paddle") if b in available_backends
            ]
            if not aarch64_backends:
                return None, "No suitable inference backend for aarch64."
            if hpi_config.backend is not None:
                if hpi_config.backend in aarch64_backends:
                    return hpi_config.backend, {}
                return (
                    None,
                    f"Inference backend {repr(hpi_config.backend)}"
                    " is not available on aarch64.",
                )
            return aarch64_backends[0], {}
        else:
            return None, f"{repr(arch)} is not a supported architecture."'''


def main():
    if not HPI_PATH.exists():
        print(f"ERROR: {HPI_PATH} not found", file=sys.stderr)
        sys.exit(1)

    src = HPI_PATH.read_text()
    if OLD not in src:
        if "aarch64" in src:
            print("hpi.py already patched for aarch64, skipping")
            return
        print(f"ERROR: cannot find patch target in {HPI_PATH}", file=sys.stderr)
        sys.exit(1)

    HPI_PATH.write_text(src.replace(OLD, NEW))
    print(f"Patched {HPI_PATH} for aarch64 support")


if __name__ == "__main__":
    main()
