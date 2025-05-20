---
comments: true
---

# Ascend NPU PaddlePaddle Installation Tutorial

Currently, PaddleOCR supports the Ascend 910B chip (more models are under support. If you have a related need for other models, please submit an issue to inform us). The Ascend driver version is 23.0.3. Considering the differences in environments, we recommend using the <b>Ascend development image provided by PaddlePaddle</b> to complete the environment preparation.

## 1. Docker Environment Preparation
* Pull the image. This image is only for the development environment and does not contain a pre-compiled PaddlePaddle installation package. The image has CANN-8.0.0, the Ascend operator library, installed by default.
```bash
# For X86 architecture
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-x86_64-gcc84
# For Aarch64 architecture
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-aarch64-gcc84
```
* Start the container with the following command. ASCEND_RT_VISIBLE_DEVICES specifies the visible NPU card numbers.
```bash
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-$(uname -m)-gcc84 /bin/bash
```
## 2. Install Paddle Package
* Download and install the Python wheel installation package
```bash
# Note: You need to install the CPU version of PaddlePaddle first
python -m pip install paddlepaddle==3.0.0.dev20250430 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu
python -m pip install paddle-custom-npu -i https://www.paddlepaddle.org.cn/packages/nightly/npu
```
* CANN-8.0.RC2 does not support some versions of numpy and opencv, it is recommended to install the specified versions.
```bash
python -m pip install numpy==1.26.4
python -m pip install opencv-python==3.4.18.65
```
* Set environment variables on the arm machine (not required for x86 environment)
```bash
# Solve the error reported by libgomp on the arm machine
# "libgomp cannot allocate memory in static TLS block"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD
```
* After verifying that the installation package is installed, run the following command
```bash
python -c "import paddle; paddle.utils.run_check()"
```
The expected output is as follows

```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
