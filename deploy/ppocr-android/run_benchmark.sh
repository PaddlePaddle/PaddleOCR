#!/bin/bash
# PP-OCRv6 Android Speed Benchmark
# Usage: JAVA_HOME=/opt/android-jdk ./run_benchmark.sh [iterations] [warmup]
#   e.g. JAVA_HOME=/opt/android-jdk ./run_benchmark.sh 20 5
#   defaults: 10 iterations, 3 warmup

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ITERATIONS="${1:-50}"
WARMUP="${2:-30}"

echo "========================================"
echo "  PP-OCRv6 Android Speed Benchmark"
echo "========================================"
echo "  Warmup:   ${WARMUP}"
echo "  Iterations: ${ITERATIONS}"
echo "========================================"
echo ""

adb logcat -c 2>/dev/null || true

echo "[1/2] Running speed benchmark (warmup + ${ITERATIONS} measured iterations)..."
echo "----------------------------------------"
./gradlew :ppocr-sdk:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.paddle.ocr.benchmark.OCRBenchmarkTest#testLatencyBenchmark \
    -Pandroid.testInstrumentationRunnerArguments.warmup="${WARMUP}" \
    -Pandroid.testInstrumentationRunnerArguments.iterations="${ITERATIONS}" \
    2>&1
echo "----------------------------------------"
echo ""

echo "[2/2] Speed benchmark results:"
echo "========================================"
adb logcat -d -s System.out:I 2>/dev/null | grep "OCRBenchmark" | sed 's/^.*System.out/I/'
echo "========================================"
echo ""
echo "  Done"
