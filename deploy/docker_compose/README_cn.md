[English](README.md) | 简体中文

# Docker 部署 paddleocr web 服务

## 快速开始

### 安装 Docker

- Linux

```bash
sudo curl -fsSL "https://get.docker.com" | sh
```

- Windows and Mac

参考 [Install Docker Engine](https://docs.docker.com/engine/install/)

### 安装 Docker Compose

- Linux

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o "/usr/local/bin/docker-compose"
```

- Windows and Mac

参考 [Install Docker Compose](https://docs.docker.com/compose/install/)

### 启动服务

**确保工作目录是此文件所处目录**

```bash
# 前台测试服务
docker-compose up
# 后台运行服务
docker-compose up -d
```

**注意**：web 服务的默认端口为 5000，可在 `docker-compose.yaml` 中设置环境变量 `FLASK_PORT` 的值来改变

### 测试服务

#### 从远程图片链接中识别文本

- curl 例子

```bash
curl "http://localhost:5000/api/ocr_dec" -X POST -d "img_url=https://ai.bdstatic.com/file/5419067D0B374C12A8CFB5C74684CC06"
```

- python 例子

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

#### 从 base64 中识别文本

- python 例子

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

## 项目文件说明

- `server.py`

该文件是 flask web api 服务的入口文件，里面设定了一个简单的 ocr 识别调用 api

- `examples`

存放 curl 以及 python 调用服务的例子
