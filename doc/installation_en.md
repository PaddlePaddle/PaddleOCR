## Quick installation

After testing, paddleocr can run on glibc 2.23. You can also test other glibc versions or install glic 2.23 for the best compatibility.

PaddleOCR working environment:
- PaddlePaddle1.7
- python3
- glibc 2.23

It is recommended to use the docker provided by us to run PaddleOCR, please refer to the use of docker [link](https://docs.docker.com/get-started/).

1. (Recommended) Prepare a docker environment. The first time you use this image, it will be downloaded automatically. Please be patient.
```
# Switch to the working directory
cd /home/Projects
# You need to create a docker container for the first run, and do not need to run the current command when you run it again
# Create a docker container named ppocr and map the current directory to the /paddle directory of the container

#If you want to use docker in a CPU environment, use docker instead of nvidia-docker to create docker
sudo docker run --name ppocr -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7-dev /bin/bash
```
If you have cuda9 installed on your machine, please run the following command to create a container:
```
sudo nvidia-docker run --name ppocr -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7-dev /bin/bash
```
If you have cuda10 installed on your machine, please run the following command to create a container:
```
sudo nvidia-docker run --name ppocr -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7-dev /bin/bash
```
You can also visit [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) to get the image that fits your machine.
```
# ctrl+P+Q to exit docker, to re-enter docker using the following command:
sudo docker container exec -it ppocr /bin/bash
```

Note: If the docker pull is too slow, you can download and load the docker image manually according to the following steps. Take cuda9 docker for example, you only need to change cuda9 to cuda10 to use cuda10 docker:
```
# Download the CUDA9 docker compressed file and unzip it
wget https://paddleocr.bj.bcebos.com/docker/docker_pdocr_cuda9.tar.gz
# To reduce download time, the uploaded docker image is compressed and needs to be decompressed
tar zxf docker_pdocr_cuda9.tar.gz
# Create image
docker load < docker_pdocr_cuda9.tar
# After completing the above steps, check whether the downloaded image is loaded through docker images
docker images
# If you have the following output after executing docker images, you can follow step 1 to create a docker environment.
hub.baidubce.com/paddlepaddle/paddle   latest-gpu-cuda9.0-cudnn7-dev    f56310dcc829
```

2. Install PaddlePaddle Fluid v1.7 (the higher version is not supported yet, the adaptation work is in progress)
```
pip3 install --upgrade pip

# If you have cuda9 installed on your machine, please run the following command to install
python3 -m pip install paddlepaddle-gpu==1.7.2.post97 -i https://pypi.tuna.tsinghua.edu.cn/simple

# If you have cuda10 installed on your machine, please run the following command to install
python3 -m pip install paddlepaddle-gpu==1.7.2.post107 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.


3. Clone PaddleOCR repo code
```
# Recommend
git clone https://github.com/PaddlePaddle/PaddleOCR

# If you cannot pull successfully due to network problems, you can also choose to use the code hosting on the cloud:

git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The cloud-hosting code may not be able to synchronize the update with this GitHub project in real time. There might be a delay of 3-5 days. Please give priority to the recommended method.
```

4. Install third-party libraries
```
cd PaddleOCR
pip3 install -r requirments.txt
```
