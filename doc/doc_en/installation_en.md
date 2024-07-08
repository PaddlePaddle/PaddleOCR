## QUICK INSTALLATION

After testing, PaddleOCR can run on glibc 2.23. You can also test other glibc versions or install glibc 2.23 for the best compatibility.

PaddleOCR working environment:
- PaddlePaddle 2.0.0
- Python 3.7
- glibc 2.23

It is recommended to use the docker provided by us to run PaddleOCR. Please refer to the docker tutorial [link](https://www.runoob.com/docker/docker-tutorial.html/).

*If you want to directly run the prediction code on Mac or Windows, you can start from step 2.*

**1. (Recommended) Prepare a docker environment. For the first time you use this docker image, it will be downloaded automatically. Please be patient.**
```
# Switch to the working directory
cd /home/Projects
# You need to create a docker container for the first run, and do not need to run the current command when you run it again
# Create a docker container named ppocr and map the current directory to the /paddle directory of the container

#If using CPU, use docker instead of nvidia-docker to create docker
sudo docker run --name ppocr -v $PWD:/paddle --network=host -it  paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82  /bin/bash
```

With CUDA10, please run the following command to create a container.
It is recommended to set a shared memory greater than or equal to 32G through the --shm-size parameter:
```
sudo nvidia-docker run --name ppocr -v $PWD:/paddle --shm-size=64G --network=host -it paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 /bin/bash
```
You can also visit [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) to get the image that fits your machine.
```
# ctrl+P+Q to exit docker, to re-enter docker using the following command:
sudo docker container exec -it ppocr /bin/bash
```

**2. Install PaddlePaddle 2.0**
```
pip3 install --upgrade pip

# If you have cuda9 or cuda10 installed on your machine, please run the following command to install
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple

# If you only have cpu on your machine, please run the following command to install
python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple
```
For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.


**3. Clone PaddleOCR repo**
```
# Recommend
git clone https://github.com/PaddlePaddle/PaddleOCR

# If you cannot pull successfully due to network problems, you can switch to the mirror hosted on Gitee:

git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The mirror on Gitee may not keep in synchronization with the latest update with the project on GitHub. There might be a delay of 3-5 days. Please try GitHub at first.
```

**4. Install third-party libraries**
```
cd PaddleOCR
pip3 install -r requirements.txt
```
