## 快速安装

建议使用我们提供的docker运行PaddleOCR，有关docker使用请参考[链接](https://docs.docker.com/get-started/)。
1. 准备docker环境。第一次使用这个镜像，会自动下载该镜像，请耐心等待。
```
# 切换到工作目录下
cd /home/Projects
# 首次运行需创建一个docker容器，再次运行时不需要运行当前命令
# 创建一个名字为ppocr的docker容器，并将当前目录映射到容器的/paddle目录下

如果您的机器安装的是CUDA9，请运行以下命令创建容器
sudo nvidia-docker run --name ppocr -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7-dev /bin/bash

如果您的机器安装的是CUDA10，请运行以下命令创建容器
sudo nvidia-docker run --name ppocr -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7-dev /bin/bash

您也可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取与您机器适配的镜像。

# ctrl+P+Q可退出docker，重新进入docker使用如下命令
sudo nvidia-docker container exec -it ppocr /bin/bash
```

2. 安装PaddlePaddle Fluid v1.7(暂不支持更高版本,适配工作进行中)
```
pip3 install --upgrade pip

如果您的机器安装的是CUDA9，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==1.7.2.post97 -i https://pypi.tuna.tsinghua.edu.cn/simple

如果您的机器安装的是CUDA10，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==1.7.2.post107 -i https://pypi.tuna.tsinghua.edu.cn/simple

更多的版本需求，请参照[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。
```

3. 克隆PaddleOCR repo代码
```
git clone https://github.com/PaddlePaddle/PaddleOCR
```

4. 安装第三方库
```
cd PaddleOCR
pip3 install -r requirments.txt
```
