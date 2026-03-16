#!/usr/bin/env python3
"""Test PaddleOCR on Linux aarch64 (ARM64).

Verifies that both model loading and inference work without SIGSEGV.
This script exercises the two known crash sites in PaddlePaddle's
pre-built aarch64 wheels:

  1. Model loading  — PIR executor std::filesystem::path corruption
  2. Inference call — null pointer dereference in native kernels

The fix routes inference through ONNX Runtime via PaddleX HPI, and
disables PIR via FLAGS_enable_pir_in_executor=0.
"""

import platform
import sys
import time


def print_env():
    import os

    print("=" * 60)
    print("PaddleOCR aarch64 test")
    print("=" * 60)
    print(f"Platform:     {sys.platform}")
    print(f"Machine:      {platform.machine()}")
    print(f"Python:       {sys.version}")
    print(
        f"PIR executor: {os.environ.get('FLAGS_enable_pir_in_executor', '(not set)')}"
    )
    print(f"PIR API:      {os.environ.get('FLAGS_enable_pir_api', '(not set)')}")
    print(f"MKL-DNN flag: {os.environ.get('FLAGS_use_mkldnn', '(not set)')}")
    print()


def check_deps():
    print("[1/4] Checking dependencies...")
    errors = []

    try:
        import paddle

        print(f"  paddlepaddle: {paddle.__version__}")
    except ImportError as e:
        errors.append(f"  paddlepaddle: MISSING ({e})")

    try:
        import paddlex

        print(f"  paddlex:      {paddlex.__version__}")
    except ImportError as e:
        errors.append(f"  paddlex:      MISSING ({e})")

    try:
        import paddleocr

        print(f"  paddleocr:    {paddleocr.__version__}")
    except ImportError as e:
        errors.append(f"  paddleocr:    MISSING ({e})")

    try:
        import onnxruntime as ort

        print(f"  onnxruntime:  {ort.__version__}")
    except ImportError as e:
        errors.append(f"  onnxruntime:  MISSING ({e})")

    try:
        import paddle2onnx

        print(f"  paddle2onnx:  {paddle2onnx.__version__}")
    except ImportError as e:
        errors.append(f"  paddle2onnx:  MISSING ({e})")

    if errors:
        print("\n  MISSING dependencies:")
        for err in errors:
            print(f"    {err}")
        sys.exit(1)

    print("  All dependencies OK")
    print()


def test_model_loading():
    """Test model loading (crash site 1: PIR executor SIGSEGV)."""
    print("[2/4] Testing model loading (crash site 1: PIR)...")
    from paddleocr import PaddleOCR

    t0 = time.time()
    ocr = PaddleOCR(device="cpu")
    elapsed = time.time() - t0
    print(f"  Model loaded successfully in {elapsed:.1f}s")
    print()
    return ocr


def test_inference(ocr):
    """Test inference (crash site 2: null pointer dereference)."""
    print("[3/4] Testing inference (crash site 2: native kernels)...")
    import numpy as np
    from PIL import Image, ImageDraw

    # Create a test image with text-like content
    img = Image.new("RGB", (200, 60), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), "Hello OCR", fill="black")
    img_array = np.asarray(img, dtype=np.uint8)

    t0 = time.time()
    results = ocr.predict(img_array)
    elapsed = time.time() - t0
    print(f"  Inference completed in {elapsed:.1f}s")

    if results:
        for res in results:
            rec_texts = res.get("rec_texts", res.get("rec_text", []))
            if rec_texts:
                print(f"  Detected text: {rec_texts}")
    else:
        print("  No text detected (expected for simple test image)")

    print()
    return True


def test_predict_with_file(ocr):
    """Test inference with a file path input."""
    print("[4/4] Testing inference with generated file input...")
    import tempfile

    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (300, 80), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "PaddleOCR aarch64 OK", fill="black")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        tmp_path = f.name

    t0 = time.time()
    results = ocr.predict(tmp_path)
    elapsed = time.time() - t0
    print(f"  File inference completed in {elapsed:.1f}s")

    import os

    os.unlink(tmp_path)
    print()
    return True


def main():
    print_env()
    check_deps()

    try:
        ocr = test_model_loading()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    try:
        test_inference(ocr)
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    try:
        test_predict_with_file(ocr)
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("=" * 60)
    print("ALL TESTS PASSED — PaddleOCR works on aarch64!")
    print("=" * 60)


if __name__ == "__main__":
    main()
