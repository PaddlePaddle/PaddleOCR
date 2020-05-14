## 快速安装

建议使用我们提供的docker运行PaddleOCR，有关docker使用请参考[链接](https://docs.docker.com/get-started/)。
1. 准备docker环境。第一次使用这个镜像，会自动下载该镜像，请耐心等待。
```
# 切换到工作目录下
cd /home/Projects
# 首次运行需创建一个docker容器，再次运行时不需要运行当前命令
# 创建一个名字为pdocr的docker容器，并将当前目录映射到容器的/paddle目录下
sudo nvidia-docker run --name pdocr -v $PWD:/paddle --network=host -it  hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7-dev  /bin/bash

# ctrl+P+Q可退出docker，重新进入docker使用如下命令
sudo nvidia-docker container exec -it pdocr /bin/bash

```

2. 克隆PaddleOCR repo代码
```
git clone https://github.com/PaddlePaddle/PaddleOCR
```

3. 安装第三方库
```
cd PaddleOCR
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
