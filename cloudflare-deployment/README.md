# PaddleOCR on Cloudflare Containers

Deploy PaddleOCR as a globally distributed OCR service on Cloudflare's edge network.

## Features

- **Modern PaddleOCR v3.3.0** with PP-OCRv4/v5 support
- **109+ languages** supported
- **FastAPI** async web framework
- **Optimized for 8GB RAM, 4 vCPU** container instances
- **Global deployment** across Cloudflare's 320+ locations
- **Scale-to-zero pricing** - only pay for actual usage
- **REST API** with file upload and base64 support

## Architecture

```
User Request → Cloudflare Worker → Durable Object → Container (FastAPI + PaddleOCR)
```

- **Worker**: Entry point, routing, and orchestration
- **Durable Object**: Container lifecycle management
- **Container**: PaddleOCR inference service on port 8000

## Prerequisites

### 1. Cloudflare Account
- Cloudflare account with Containers access (Public Beta)
- Workers Paid plan ($5/month minimum)

### 2. Local Development Tools
- **Docker Desktop** (required for building images)
  - macOS: https://docs.docker.com/desktop/install/mac-install/
  - Windows: https://docs.docker.com/desktop/install/windows-install/
  - Linux: https://docs.docker.com/desktop/install/linux-install/

- **Wrangler CLI** (Cloudflare's deployment tool)
  ```bash
  npm install -g wrangler
  # or
  yarn global add wrangler
  ```

- **Node.js** 18+ (for Wrangler)
  ```bash
  node --version  # Should be 18.0.0 or higher
  ```

### 3. Authentication
```bash
# Login to Cloudflare
wrangler login

# Verify authentication
wrangler whoami
```

## Quick Start

### 1. Clone and Navigate
```bash
cd cloudflare-deployment
```

### 2. Verify Docker is Running
```bash
docker ps
# Should show list of running containers (can be empty)
# If error, start Docker Desktop
```

### 3. Deploy to Cloudflare
```bash
wrangler deploy
```

This will:
1. Build the Docker image locally
2. Push image to Cloudflare's container registry
3. Deploy the Worker and Durable Object
4. Configure global routing

**First deployment takes 5-10 minutes** due to:
- Docker image build (~3-4 min)
- Image push to registry (~2-3 min)
- Worker deployment (~1 min)

### 4. Test Your Deployment
```bash
# Get your deployment URL from wrangler output
# Example: https://paddleocr-service.your-subdomain.workers.dev

# Health check
curl https://paddleocr-service.your-subdomain.workers.dev/health

# Test with sample image (see examples/test.sh)
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ocr_ready": true,
  "config": {
    "lang": "ch",
    "ocr_version": "PP-OCRv4",
    "cpu_threads": 4
  }
}
```

### 2. OCR from File Upload
```bash
POST /ocr
Content-Type: multipart/form-data

FormData:
  - file: image file (JPEG, PNG, BMP, TIFF, WebP)
  - lang: (optional) language code
```

**Example:**
```bash
curl -X POST https://your-worker.workers.dev/ocr \
  -F "file=@sample.jpg"
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "text": "Hello World",
      "confidence": 0.98,
      "bbox": [[10,20], [100,20], [100,50], [10,50]]
    }
  ],
  "processing_time_ms": 234.5,
  "image_size": [800, 600]
}
```

### 3. OCR from Base64
```bash
POST /ocr/base64
Content-Type: application/json

{
  "image": "base64_encoded_image_data",
  "lang": "en"  // optional
}
```

**Example:**
```bash
curl -X POST https://your-worker.workers.dev/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgA..."
  }'
```

## Configuration

### Environment Variables (wrangler.toml)

```toml
[vars]
# Language: ch, en, french, german, korean, japan, arabic, etc.
OCR_LANG = "ch"

# OCR Version: PP-OCRv3, PP-OCRv4, PP-OCRv5
OCR_VERSION = "PP-OCRv4"

# Use text orientation classifier (slower but more accurate)
USE_ANGLE_CLS = "true"

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "INFO"

# Max image size in bytes (10MB default)
MAX_IMAGE_SIZE = "10485760"
```

### Instance Configuration

In `wrangler.toml`:

```toml
[[containers]]
instance_type = "large"  # 8GB RAM, 4 vCPU
max_instances = 10       # Max concurrent containers
```

**Available instance types:**
- `dev`: 256 MB RAM, 1/16 vCPU
- `basic`: 1 GB RAM, 1/4 vCPU
- `standard`: 4 GB RAM, 1/2 vCPU
- `large`: 8 GB RAM, 4 vCPU ✓ (recommended)

## Performance Expectations

With **8GB RAM, 4 vCPU CPU-only** configuration:

| Scenario | Expected Time |
|----------|--------------|
| **Cold start** (first request) | 30-60 seconds |
| **Warm inference** (subsequent) | 500-2000ms |
| **Simple image** (few characters) | 500-1000ms |
| **Complex image** (many lines) | 1000-3000ms |
| **Large document** (full page) | 2000-5000ms |

**Optimization tips:**
- Keep containers warm with periodic health checks
- Use smaller images when possible
- Batch process multiple images in parallel
- Consider caching results in KV storage

## Cost Estimation

**Cloudflare Containers Pricing (2025):**
- Memory: $0.0000025 per GiB-second
- CPU: $0.000020 per vCPU-second
- Disk: $0.00000007 per GB-second

**Example: 1000 OCR requests/day**
- Average processing: 2 seconds per image
- Instance: 8 GB RAM, 4 vCPU

**Monthly cost:**
```
Memory: 1000 * 2s * 8 GB * $0.0000025 * 30 days = $1.20
CPU:    1000 * 2s * 4 vCPU * $0.000020 * 30 days = $4.80
Total:  ~$6/month + Workers Paid Plan ($5/month) = $11/month
```

**Note:** First 25 GiB-hours memory and 375 vCPU-minutes included free daily.

## Local Development

### 1. Test Locally with Docker
```bash
# Build image
docker build -t paddleocr-local .

# Run container
docker run -p 8000:8000 paddleocr-local

# Test in another terminal
curl http://localhost:8000/health
```

### 2. Test with Wrangler Dev
```bash
# Start local development server
wrangler dev

# Note: Container must be built first
# Access at http://localhost:8787
```

## Deployment Updates

### Update Code Only
```bash
# Modify app.py or config.py
wrangler deploy
```

### Update Docker Image
```bash
# Modify Dockerfile or requirements.txt
wrangler deploy
# Wrangler will rebuild and push new image
```

### Update Configuration
```bash
# Modify wrangler.toml
wrangler deploy
```

## Monitoring

### View Logs
```bash
# Stream real-time logs
wrangler tail

# Filter by log level
wrangler tail --format pretty
```

### Cloudflare Dashboard
- Analytics: https://dash.cloudflare.com/
- Container instances, request rates, errors
- CPU/Memory usage metrics

## Supported Languages

PaddleOCR supports 109+ languages:

**Common languages:**
- `ch` - Chinese & English
- `en` - English
- `french` - French
- `german` - German
- `korean` - Korean
- `japan` - Japanese
- `arabic` - Arabic
- `cyrillic` - Cyrillic
- `devanagari` - Devanagari

**Full list:** https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md

## Troubleshooting

### Docker Build Fails
```bash
# Ensure Docker is running
docker ps

# Clean Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t paddleocr-local .
```

### Deployment Fails
```bash
# Check authentication
wrangler whoami

# Verify wrangler.toml syntax
wrangler deploy --dry-run

# Check account limits
wrangler deployments list
```

### Container Crashes
```bash
# Check logs
wrangler tail

# Common issues:
# - Out of memory (reduce batch size in config.py)
# - Model download timeout (redeploy to retry)
# - Invalid image format (check ALLOWED_EXTENSIONS)
```

### Slow Performance
- **Cold starts:** First request takes 30-60s (models loading)
- **Warm requests:** Should be 500-3000ms
- **Solutions:**
  - Send periodic health checks to keep warm
  - Increase `max_instances` in wrangler.toml
  - Use smaller model versions
  - Reduce `det_limit_side_len` in config.py

## Advanced Features

### Add Result Caching (KV Storage)

1. Create KV namespace:
```bash
wrangler kv:namespace create OCR_CACHE
```

2. Add to `wrangler.toml`:
```toml
[[kv_namespaces]]
binding = "OCR_CACHE"
id = "your-kv-namespace-id"
```

3. Implement caching in `app.py` using image hash as key

### Add Image Storage (R2)

1. Create R2 bucket:
```bash
wrangler r2 bucket create paddleocr-images
```

2. Add to `wrangler.toml`:
```toml
[[r2_buckets]]
binding = "OCR_IMAGES"
bucket_name = "paddleocr-images"
```

### Custom Domain

```bash
# Add custom domain in wrangler.toml
routes = [
  { pattern = "ocr.yourdomain.com", custom_domain = true }
]

wrangler deploy
```

## Security Considerations

1. **Authentication:** Add API key validation in Worker
2. **Rate Limiting:** Implement in Worker layer
3. **Input Validation:** Already implemented (file size, type)
4. **CORS:** Configure `allow_origins` in app.py
5. **Secrets:** Use `wrangler secret put` for API keys

Example rate limiting:
```javascript
// In src/index.js
const RATE_LIMIT = 100; // requests per minute
// Implement using Durable Objects storage
```

## Migration from Existing Deployment

If you have existing PaddleOCR deployment:

1. **HubServing → FastAPI:** API endpoints are similar but improved
2. **Model compatibility:** Same model files, newer versions
3. **Request format:** Compatible with base64 input
4. **Response format:** Enhanced with more metadata

## Resources

- **PaddleOCR Documentation:** https://github.com/PaddlePaddle/PaddleOCR
- **Cloudflare Containers Docs:** https://developers.cloudflare.com/containers/
- **Wrangler CLI:** https://developers.cloudflare.com/workers/wrangler/
- **API Documentation:** https://your-worker.workers.dev/docs (FastAPI auto-generated)

## Support

- **PaddleOCR Issues:** https://github.com/PaddlePaddle/PaddleOCR/issues
- **Cloudflare Community:** https://community.cloudflare.com/
- **This Deployment:** File issues in your repository

## License

PaddleOCR is licensed under Apache License 2.0.
See the main PaddleOCR repository for details.
