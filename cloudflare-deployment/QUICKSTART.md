# Quick Start Guide - PaddleOCR on Cloudflare Containers

Get PaddleOCR running on Cloudflare's global network in under 10 minutes.

## Prerequisites Checklist

- [ ] Cloudflare account with Containers access (Public Beta)
- [ ] Workers Paid plan ($5/month)
- [ ] Docker Desktop installed and running
- [ ] Node.js 18+ installed
- [ ] Wrangler CLI installed (`npm install -g wrangler`)

## Step-by-Step Deployment

### 1. Verify Docker (2 minutes)

```bash
# Check Docker is running
docker ps

# If error, start Docker Desktop application
```

### 2. Install Dependencies (1 minute)

```bash
cd cloudflare-deployment

# Install Wrangler (if not already installed)
npm install -g wrangler

# Or install locally
npm install
```

### 3. Authenticate with Cloudflare (1 minute)

```bash
# Login to Cloudflare
wrangler login

# This will open browser for authentication
# After login, verify:
wrangler whoami
```

### 4. Configure Your Deployment (1 minute)

Edit `wrangler.toml` and update:

```toml
name = "paddleocr-service"  # Change to your preferred name

[vars]
OCR_LANG = "en"  # Change to your primary language (ch, en, french, etc.)
```

### 5. Deploy to Cloudflare (5-10 minutes)

```bash
wrangler deploy
```

**What happens:**
1. ⏳ Building Docker image (3-4 min)
2. ⏳ Pushing to Cloudflare registry (2-3 min)
3. ⏳ Deploying Worker & Durable Object (1 min)
4. ✅ Deployment complete!

**Expected output:**
```
Published paddleocr-service
  https://paddleocr-service.your-subdomain.workers.dev
```

### 6. Test Your Deployment (1 minute)

```bash
# Save your Worker URL
WORKER_URL="https://paddleocr-service.your-subdomain.workers.dev"

# Test health check
curl $WORKER_URL/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "ocr_ready": true,
  "config": {
    "lang": "en",
    "ocr_version": "PP-OCRv4",
    "cpu_threads": 4
  }
}
```

### 7. Run Your First OCR (1 minute)

```bash
# Download a test image
curl -o test.jpg "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/doc/imgs_en/img_12.jpg"

# Run OCR
curl -X POST $WORKER_URL/ocr \
  -F "file=@test.jpg" | jq '.'
```

**Expected response:**
```json
{
  "success": true,
  "results": [
    {
      "text": "ACKNOWLEDGEMENTS",
      "confidence": 0.99,
      "bbox": [[10,20], [200,20], [200,50], [10,50]]
    }
  ],
  "processing_time_ms": 1234.5
}
```

## You're Done! 🎉

Your PaddleOCR service is now live on Cloudflare's global network!

## What's Next?

### View API Documentation
Open in browser: `https://paddleocr-service.your-subdomain.workers.dev/docs`

### Run Comprehensive Tests
```bash
cd examples

# Update WORKER_URL in test scripts
export WORKER_URL="https://paddleocr-service.your-subdomain.workers.dev"

# Run tests
python test.py
# or
./test.sh
```

### Monitor Your Service
```bash
# Stream live logs
wrangler tail

# View metrics in dashboard
open https://dash.cloudflare.com/
```

### Optimize Configuration

Edit `config.py` to tune performance:

```python
OCR_CONFIG = {
    "lang": "en",              # Change language
    "ocr_version": "PP-OCRv4", # Use PP-OCRv3, PP-OCRv4, or PP-OCRv5
    "cpu_threads": 4,          # Match your vCPU count
    "det_limit_side_len": 960, # Lower for faster inference
    "rec_batch_num": 6,        # Lower to save memory
}
```

Then redeploy:
```bash
wrangler deploy
```

### Add Custom Domain

In `wrangler.toml`:
```toml
routes = [
  { pattern = "ocr.yourdomain.com", custom_domain = true }
]
```

Deploy:
```bash
wrangler deploy
```

## Common Issues

### "Docker daemon not running"
→ Start Docker Desktop application

### "Authentication failed"
→ Run `wrangler login` again

### "Container instance limit exceeded"
→ Reduce `max_instances` in `wrangler.toml`

### "Out of memory"
→ Reduce `rec_batch_num` in `config.py`

### Slow first request (30-60s)
→ Normal! This is cold start (model loading)
→ Keep warm with periodic health checks

## Cost Estimate

**For 1000 OCR requests/month:**
- ~$6/month for compute
- $5/month Workers Paid plan
- **Total: ~$11/month**

*Includes free tier: 25 GiB-hours memory + 375 vCPU-minutes daily*

## Performance Expectations

| Scenario | Time |
|----------|------|
| Cold start (first request) | 30-60s |
| Warm inference | 500-2000ms |
| Simple image | 500-1000ms |
| Complex document | 2-5s |

## Need Help?

- 📖 **Full Documentation:** See [README.md](README.md)
- 🔧 **PaddleOCR Issues:** https://github.com/PaddlePaddle/PaddleOCR/issues
- 💬 **Cloudflare Community:** https://community.cloudflare.com/
- 📊 **Cloudflare Status:** https://www.cloudflarestatus.com/

## Architecture Overview

```
Internet
   ↓
Cloudflare Edge (Worker)
   ↓
Durable Object (Container Manager)
   ↓
Container Instance (8GB RAM, 4 vCPU)
   ↓
FastAPI (Port 8000)
   ↓
PaddleOCR v3.3.0
   ↓
PP-OCRv4 Models
```

Your requests are routed to the nearest Cloudflare location (320+ cities worldwide) for lowest latency!

---

**Ready for production?** See [README.md](README.md) for advanced features:
- Result caching with KV storage
- Image storage with R2
- Rate limiting & authentication
- Multi-language support
- Performance optimization
