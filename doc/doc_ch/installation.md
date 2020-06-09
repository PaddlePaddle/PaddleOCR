## 快速安装

经测试PaddleOCR可在glibc 2.23上运行，您也可以测试其他glibc版本或安装glic 2.23
PaddleOCR 工作环境
- PaddlePaddle1.7
- python3
- glibc 2.23

建议使用我们提供的docker运行PaddleOCR，有关docker使用请参考[链接](https://docs.docker.com/get-started/)。

*如您希望使用 mac 或 windows直接运行预测代码，可以从第2步开始执行。*

1. （建议）准备docker环境。第一次使用这个镜像，会自动下载该镜像，请耐心等待。
```
# 切换到工作目录下
cd /home/Projects
# 首次运行需创建一个docker容器，再次运行时不需要运行当前命令
# 创建一个名字为ppocr的docker容器，并将当前目录映射到容器的/paddle目录下

如果您希望在CPU环境下使用docker，使用docker而不是nvidia-docker创建docker
sudo docker run --name ppocr -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7-dev /bin/bash

如果您的机器安装的是CUDA9，请运行以下命令创建容器
sudo nvidia-docker run --name ppocr -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7-dev /bin/bash

如果您的机器安装的是CUDA10，请运行以下命令创建容器
sudo nvidia-docker run --name ppocr -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7-dev /bin/bash

您也可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取与您机器适配的镜像。

# ctrl+P+Q可退出docker，重新进入docker使用如下命令
sudo docker container exec -it ppocr /bin/bash
```

注意：如果docker pull过慢，可以按照如下步骤手动下载后加载docker,以cuda9 docker为例，使用cuda10 docker只需要将cuda9改为cuda10即可。
```
# 下载CUDA9 docker的压缩文件，并解压
wget https://paddleocr.bj.bcebos.com/docker/docker_pdocr_cuda9.tar.gz
# 为减少下载时间，上传的docker image是压缩过的，需要解压使用
tar zxf docker_pdocr_cuda9.tar.gz
# 创建image
docker load < docker_pdocr_cuda9.tar
# 完成上述步骤后通过docker images检查是否加载了下载的镜像
docker images
# 执行docker images后如果有下面的输出，即可按照按照 步骤1 创建docker环境。
hub.baidubce.com/paddlepaddle/paddle   latest-gpu-cuda9.0-cudnn7-dev    f56310dcc829
```

2. 安装PaddlePaddle Fluid v1.7(暂不支持更高版本,适配工作进行中)
```
pip3 install --upgrade pip

如果您的机器安装的是CUDA9，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==1.7.2.post97 -i https://pypi.tuna.tsinghua.edu.cn/simple

如果您的机器安装的是CUDA10，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==1.7.2.post107 -i https://pypi.tuna.tsinghua.edu.cn/simple

如果您的机器是CPU，请运行以下命令安装

python3 -m pip install paddlepaddle==1.7.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

更多的版本需求，请参照[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。
```

3. 克隆PaddleOCR repo代码
```
【推荐】git clone https://github.com/PaddlePaddle/PaddleOCR

如果因为网络问题无法pull成功，也可选择使用码云上的托管：

git clone https://gitee.com/paddlepaddle/PaddleOCR

注：码云托管代码可能无法实时同步本github项目更新，存在3~5天延时，请优先使用推荐方式。
```

4. 安装第三方库
```
cd PaddleOCR
pip3 install -r requirments.txt
```
