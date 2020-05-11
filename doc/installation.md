### 2.1 快速安装

我们提供了PaddleOCR开发环境的docker，您可以pull我们提供的docker运行PaddleOCR的环境。

1. 准备docker环境。第一次使用这个镜像，会自动下载该镜像，请耐心等待。
```
# 切换到工作目录下
cd /home/Projects
# 创建一个名字为pdocr的docker容器，并将当前目录映射到容器的/data目录下
sudo nvidia-docker run --name pdocr -v $PWD:/data --network=host -it paddlepaddle/paddle:1.7.2-gpu-cuda10.0-cudnn7  /bin/bash
```

2. 克隆PaddleOCR repo代码
```
apt-get update
apt-get install git
git clone https://github.com/PaddlePaddle/PaddleOCR
```

3. 安装第三方库
```
cd PaddleOCR
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
