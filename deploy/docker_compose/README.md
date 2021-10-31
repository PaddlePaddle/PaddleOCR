English | [简体中文](README_cn.md)

# Deploy paddleocr web service with Docker

## Quick Start

### Install Docker

- Linux

```bash
sudo curl -fsSL "https://get.docker.com" | sh
```

- Windows and Mac

See [Install Docker Engine](https://docs.docker.com/engine/install/)

### Install Docker Compose

- Linux

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o "/usr/local/bin/docker-compose"
```

- Windows and Mac

See [Install Docker Compose](https://docs.docker.com/compose/install/)

### Run paddleocr web service

**确保工作目录是此文件所处目录**

```bash
# run in front
docker-compose up
# run in background
docker-compose up -d
```

**Note**: The default web service port is `5000`, you can change it by changing environment variable `FLASK_PORT` defined in `docker-compose.yaml`

### Test service

#### Detect text from remote image url

- curl example

```bash
curl "http://localhost:5000/api/ocr_dec" -X POST -d "img_url=https://ai.bdstatic.com/file/5419067D0B374C12A8CFB5C74684CC06"
```

- python exmaple

```python
import requests
remote_img_url = 'https://ai.bdstatic.com/file/5419067D0B374C12A8CFB5C74684CC06'
data = {
    'img_url': remote_img_url
}
api_url = 'http://localhost:5000/api/ocr_dec'
# or
# api_url = 'http://0.0.0.0:5000/api/ocr_dec'

response = requests.post(api_url, data=data)
json = response.json()
print(json)

```

#### Detect text from base64

- python example

```python
import base64
import requests
local_img_path = './test.jpg'
with open(local_img_path, 'rb') as f:
    base64_bytes = base64.encodebytes(f.read())
    # or
    # base64.encodebytes(f.read())
api_url = 'http://localhost:5000/api/ocr_dec'
# or
# api_url = 'http://0.0.0.0:5000/api/ocr_dec'
data = {
    'img_base64': base64_bytes
    # or
    # 'img_base64': base64_bytes.decode()
}
response = requests.post(api_url, data=data)
json = response.json()
print(json)

```

## Project file description

- `server.py`
  A simple flask app for ocr web service
- `examples`
  Examples of curl and python to call the ocr web service
