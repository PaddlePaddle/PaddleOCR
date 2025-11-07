#!/bin/bash
# Test script for PaddleOCR API on Cloudflare Containers

# Configuration
WORKER_URL="${WORKER_URL:-https://paddleocr-service.your-subdomain.workers.dev}"

echo "Testing PaddleOCR API at: $WORKER_URL"
echo "================================================"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Health Check
echo -e "\n${YELLOW}Test 1: Health Check${NC}"
echo "GET $WORKER_URL/health"
RESPONSE=$(curl -s "$WORKER_URL/health")
if echo "$RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "$RESPONSE"
fi

# Test 2: Worker Root
echo -e "\n${YELLOW}Test 2: Worker Root${NC}"
echo "GET $WORKER_URL/"
RESPONSE=$(curl -s "$WORKER_URL/")
echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"

# Test 3: OCR with Base64 Image (simple test image)
echo -e "\n${YELLOW}Test 3: OCR with Base64 Image${NC}"
# Create a simple test image with text (you can replace this with actual base64)
# This is a placeholder - in real use, encode your test image
echo "Note: For actual testing, replace BASE64_IMAGE with your encoded image"
echo 'Example: BASE64_IMAGE=$(base64 -i your-image.jpg)'

# Test 4: OCR with File Upload
echo -e "\n${YELLOW}Test 4: OCR with File Upload${NC}"
# Check if test image exists
if [ -f "test-image.jpg" ]; then
    echo "POST $WORKER_URL/ocr (uploading test-image.jpg)"
    RESPONSE=$(curl -s -X POST "$WORKER_URL/ocr" \
        -F "file=@test-image.jpg")

    if echo "$RESPONSE" | grep -q "success"; then
        echo -e "${GREEN}✓ OCR processing succeeded${NC}"
        echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"

        # Extract processing time
        PROC_TIME=$(echo "$RESPONSE" | jq -r '.processing_time_ms' 2>/dev/null)
        if [ ! -z "$PROC_TIME" ]; then
            echo -e "${GREEN}Processing time: ${PROC_TIME}ms${NC}"
        fi
    else
        echo -e "${RED}✗ OCR processing failed${NC}"
        echo "$RESPONSE"
    fi
else
    echo -e "${YELLOW}⚠ test-image.jpg not found. Skipping file upload test.${NC}"
    echo "Create a test image with: convert -size 200x50 -pointsize 24 -gravity center label:'Hello World' test-image.jpg"
fi

# Performance Test
echo -e "\n${YELLOW}Test 5: Performance Benchmark (5 requests)${NC}"
if [ -f "test-image.jpg" ]; then
    TOTAL_TIME=0
    SUCCESS_COUNT=0

    for i in {1..5}; do
        echo -n "Request $i: "
        START=$(date +%s%N)
        RESPONSE=$(curl -s -X POST "$WORKER_URL/ocr" -F "file=@test-image.jpg")
        END=$(date +%s%N)

        ELAPSED=$(( ($END - $START) / 1000000 )) # Convert to milliseconds

        if echo "$RESPONSE" | grep -q "success"; then
            echo -e "${GREEN}${ELAPSED}ms${NC}"
            TOTAL_TIME=$((TOTAL_TIME + ELAPSED))
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo -e "${RED}Failed${NC}"
        fi

        # Small delay between requests
        sleep 0.5
    done

    if [ $SUCCESS_COUNT -gt 0 ]; then
        AVG_TIME=$((TOTAL_TIME / SUCCESS_COUNT))
        echo -e "\n${GREEN}Average response time: ${AVG_TIME}ms${NC}"
        echo -e "${GREEN}Success rate: $SUCCESS_COUNT/5${NC}"
    fi
fi

echo -e "\n================================================"
echo "Testing complete!"
echo ""
echo "Next steps:"
echo "1. Review the responses above"
echo "2. Check logs with: wrangler tail"
echo "3. View metrics in Cloudflare dashboard"
echo "4. Add your test images to examples/ directory"
