---
comments: true
---

## 快速安装

经测试PaddleOCR可在glibc 2.23上运行，您也可以测试其他glibc版本或安装glic 2.23
PaddleOCR 工作环境

- PaddlePaddle 2.0.0
- python3.7
- glibc 2.23
- cuDNN 7.6+ (GPU)

建议使用我们提供的docker运行PaddleOCR，有关docker、nvidia-docker使用请参考[链接](https://www.runoob.com/docker/docker-tutorial.html/)。

*如您希望使用 mac 或 windows直接运行预测代码，可以从第2步开始执行。*

### 1. （建议）准备docker环境

第一次使用这个镜像，会自动下载该镜像，请耐心等待

```bash linenums="1"
# 切换到工作目录下
cd /home/Projects
# 首次运行需创建一个docker容器，再次运行时不需要运行当前命令
# 创建一个名字为ppocr的docker容器，并将当前目录映射到容器的/paddle目录下

如果您希望在CPU环境下使用docker，使用docker而不是nvidia-docker创建docker
sudo docker run --name ppocr -v $PWD:/paddle --network=host -it paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 /bin/bash

如果使用CUDA10，请运行以下命令创建容器，设置docker容器共享内存shm-size为64G，建议设置32G以上
sudo nvidia-docker run --name ppocr -v $PWD:/paddle --shm-size=64G --network=host -it paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 /bin/bash

您也可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取与您机器适配的镜像。

# ctrl+P+Q可退出docker 容器，重新进入docker 容器使用如下命令
sudo docker container exec -it ppocr /bin/bash
```

### 2. 安装PaddlePaddle 2.0

```bash linenums="1"
pip3 install --upgrade pip

# 如果您的机器安装的是CUDA9或CUDA10，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple

# 如果您的机器是CPU，请运行以下命令安装
python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple

# 更多的版本需求，请参照[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。
```

### 3. 克隆PaddleOCR repo代码

```bash linenums="1"
#【推荐】
git clone https://github.com/PaddlePaddle/PaddleOCR

# 如果因为网络问题无法pull成功，也可选择使用码云上的托管：

git clone https://gitee.com/paddlepaddle/PaddleOCR

# 注：码云托管代码可能无法实时同步本github项目更新，存在3~5天延时，请优先使用推荐方式。
```

### 4. 安装第三方库

```bash linenums="1"
cd PaddleOCR
pip3 install -r requirements.txt
```

注意，windows环境下，建议从[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)下载shapely安装包完成安装，
直接通过pip安装的shapely库可能出现`[winRrror 126] 找不到指定模块的问题`。
