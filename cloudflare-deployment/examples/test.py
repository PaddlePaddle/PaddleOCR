#!/usr/bin/env python3
"""
Python test script for PaddleOCR API on Cloudflare Containers
"""
import os
import sys
import time
import base64
import json
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)

# Configuration
WORKER_URL = os.getenv("WORKER_URL", "https://paddleocr-service.your-subdomain.workers.dev")


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Test 1: Health Check")
    print("="*60)

    try:
        response = requests.get(f"{WORKER_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✓ Health check passed")
            print(json.dumps(data, indent=2))
            return True
        else:
            print("✗ Health check failed")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_worker_root():
    """Test worker root endpoint"""
    print("\n" + "="*60)
    print("Test 2: Worker Root")
    print("="*60)

    try:
        response = requests.get(f"{WORKER_URL}/", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_ocr_base64(image_path: Optional[Path] = None):
    """Test OCR with base64 encoded image"""
    print("\n" + "="*60)
    print("Test 3: OCR with Base64 Image")
    print("="*60)

    if image_path is None or not image_path.exists():
        print("⚠ No test image provided. Skipping base64 test.")
        return False

    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()

        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Send request
        print(f"Sending {len(base64_image)} bytes of base64 data...")
        start_time = time.time()

        response = requests.post(
            f"{WORKER_URL}/ocr/base64",
            json={"image": base64_image},
            timeout=60
        )

        elapsed = (time.time() - start_time) * 1000

        print(f"Status Code: {response.status_code}")
        print(f"Total Request Time: {elapsed:.2f}ms")

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                print(f"✓ OCR processing succeeded")
                print(f"Processing Time: {data.get('processing_time_ms', 0):.2f}ms")
                print(f"Detected {len(data.get('results', []))} text regions:")

                for i, result in enumerate(data.get('results', [])[:5]):  # Show first 5
                    print(f"\n  {i+1}. Text: '{result['text']}'")
                    print(f"     Confidence: {result['confidence']:.4f}")
                    print(f"     BBox: {result['bbox']}")

                if len(data.get('results', [])) > 5:
                    print(f"\n  ... and {len(data['results']) - 5} more")

                return True
            else:
                print(f"✗ OCR processing failed: {data.get('error')}")
                return False
        else:
            print(f"✗ Request failed: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_ocr_file_upload(image_path: Optional[Path] = None):
    """Test OCR with file upload"""
    print("\n" + "="*60)
    print("Test 4: OCR with File Upload")
    print("="*60)

    if image_path is None or not image_path.exists():
        print("⚠ No test image provided. Skipping file upload test.")
        return False

    try:
        # Open and upload file
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}

            print(f"Uploading {image_path.name}...")
            start_time = time.time()

            response = requests.post(
                f"{WORKER_URL}/ocr",
                files=files,
                timeout=60
            )

            elapsed = (time.time() - start_time) * 1000

        print(f"Status Code: {response.status_code}")
        print(f"Total Request Time: {elapsed:.2f}ms")

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                print(f"✓ OCR processing succeeded")
                print(f"Processing Time: {data.get('processing_time_ms', 0):.2f}ms")
                print(f"Image Size: {data.get('image_size')}")
                print(f"Detected {len(data.get('results', []))} text regions")

                # Show full results
                for i, result in enumerate(data.get('results', [])):
                    print(f"\n  {i+1}. Text: '{result['text']}'")
                    print(f"     Confidence: {result['confidence']:.4f}")

                return True
            else:
                print(f"✗ OCR processing failed: {data.get('error')}")
                return False
        else:
            print(f"✗ Request failed: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def benchmark_performance(image_path: Optional[Path] = None, num_requests: int = 5):
    """Run performance benchmark"""
    print("\n" + "="*60)
    print(f"Test 5: Performance Benchmark ({num_requests} requests)")
    print("="*60)

    if image_path is None or not image_path.exists():
        print("⚠ No test image provided. Skipping benchmark.")
        return False

    times = []
    successes = 0

    for i in range(num_requests):
        print(f"\nRequest {i+1}/{num_requests}...", end=' ')

        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}

                start_time = time.time()
                response = requests.post(f"{WORKER_URL}/ocr", files=files, timeout=60)
                elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200 and response.json().get('success'):
                times.append(elapsed)
                successes += 1
                print(f"✓ {elapsed:.2f}ms")
            else:
                print(f"✗ Failed")
        except Exception as e:
            print(f"✗ Error: {e}")

        # Small delay between requests
        if i < num_requests - 1:
            time.sleep(0.5)

    if times:
        print(f"\n" + "-"*60)
        print(f"Success Rate: {successes}/{num_requests}")
        print(f"Average Time: {sum(times)/len(times):.2f}ms")
        print(f"Min Time: {min(times):.2f}ms")
        print(f"Max Time: {max(times):.2f}ms")
        print(f"Median Time: {sorted(times)[len(times)//2]:.2f}ms")
        return True

    return False


def main():
    """Run all tests"""
    print("="*60)
    print("PaddleOCR API Testing Suite")
    print(f"Target: {WORKER_URL}")
    print("="*60)

    # Find test image
    test_image = None
    possible_paths = [
        Path("test-image.jpg"),
        Path("examples/test-image.jpg"),
        Path("../examples/test-image.jpg"),
    ]

    for path in possible_paths:
        if path.exists():
            test_image = path
            print(f"Found test image: {test_image}")
            break

    if test_image is None:
        print("\n⚠ Warning: No test image found.")
        print("Some tests will be skipped.")
        print("Create a test image at: examples/test-image.jpg")

    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("Worker Root", test_worker_root()))
    results.append(("OCR Base64", test_ocr_base64(test_image)))
    results.append(("OCR File Upload", test_ocr_file_upload(test_image)))
    results.append(("Performance Benchmark", benchmark_performance(test_image)))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL/SKIP"
        print(f"{test_name:.<40} {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    print("\n" + "="*60)
    print("Next Steps:")
    print("- Check logs: wrangler tail")
    print("- View API docs: " + WORKER_URL + "/docs")
    print("- Monitor dashboard: https://dash.cloudflare.com/")
    print("="*60)


if __name__ == "__main__":
    main()
