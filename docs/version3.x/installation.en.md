---
comments: true
---

# Installation


# 1. Install PaddlePaddle Framework

When installing PaddlePaddle, you can choose to install it via Docker or pip.

## 1.1 Installing PaddlePaddle via Docker
<b>If you choose to install via Docker</b>, please refer to the following commands to use the official Docker image of the PaddlePaddle framework to create a container named `paddleocr` and map the current working directory to the `/paddle` directory inside the container:

If your Docker version >= 19.03, please use:

```bash
# For CPU users:
docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# For GPU users:
# gpu，requires GPU driver version ≥450.80.02 (Linux) or ≥452.39 (Windows)
docker run --gpus all --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# gpu，requires GPU driver version ≥550.54.14 (Linux) or ≥550.54.14 (Windows)
docker run --gpus all --name paddleocr -v $PWD:/paddle  --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
```

* If your Docker version <= 19.03 and >= 17.06, please use:

<details><summary> Click Here</summary>

<pre><code class="language-bash"># For CPU users:
docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# For GPU users:
# CUDA 11.8 users
nvidia-docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# CUDA 12.3 users
nvidia-docker run --name paddleocr -v $PWD:/paddle  --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
</code></pre></details>

* If your Docker version <= 17.06, please update your Docker.


* Note: For more official PaddlePaddle Docker images, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/docker/linux-docker.html)

## 1.2 Installing PaddlePaddle via pip
<b>If you choose to install via pip</b>, please refer to the following commands to install PaddlePaddle in your current environment using pip:

```bash
# CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，requires GPU driver version ≥450.80.02 (Linux) or ≥452.39 (Windows)
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，requires GPU driver version ≥550.54.14 (Linux) or ≥550.54.14 (Windows)
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```


Note: For more PaddlePaddle Wheel versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip.html).

After installation, you can verify if PaddlePaddle is successfully installed using the following command:

```bash
python -c "import paddle; print(paddle.__version__)"
```
If the installation is successful, the following content will be output:

```bash
3.0.0
```

## 1.3 Installation of PaddlePaddle Wheel Package for Windows with NVIDIA 50 Series GPUs

The standard installation of PaddlePaddle does not fully support NVIDIA 50 series GPUs on Windows operating systems. Therefore, we provide a specially adapted PaddlePaddle package for this hardware environment. Please select the corresponding wheel file according to your Python version for installation.

```bash
# python 3.9
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp39-cp39-win_amd64.whl

# python 3.10
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp310-cp310-win_amd64.whl

# python 3.11
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp311-cp311-win_amd64.whl

# python 3.12
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp312-cp312-win_amd64.whl
```
**Note:** The currently released PaddlePaddle wheel package for Windows systems with 50 series GPUs has known issues with text recognition model training, and related functionalities are still being adapted and improved.


# 2. Install PaddleOCR

If you only want to use the inference capabilities of PaddleOCR, please refer to [Install Inference Package](#21-install-inference-package); if you want to perform model training, exporting, etc., please refer to [Install Training Dependencies](#22-install-training-dependencies). It is allowed to install both the inference package and training dependencies in the same environment without the need for environment isolation.

## 2.1 Install Inference Package

Install the latest version of the PaddleOCR inference package from PyPI:

```bash
# If you only want to use the basic text recognition feature (returning text position coordinates and content)
python -m pip install paddleocr
# If you want to use all functionalities, such as document parsing, document understanding, document translation, and key information extraction
# python -m pip install "paddleocr[all]"
```

Or install from source (default is the development branch):

```bash
# If you only want to use the basic text recognition feature (returning text position coordinates and content)
python -m pip install "paddleocr@git+https://github.com/PaddlePaddle/PaddleOCR.git"
# If you want to use all functionalities, such as document parsing, document understanding, document translation, and key information extraction
# python -m pip install "paddleocr[all]@git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

In addition to the `all` dependency group demonstrated above, PaddleOCR also supports installing specific optional features by specifying other dependency groups. The available dependency groups provided by PaddleOCR are as follows:

| Dependency Group Name | Corresponding Functionality |
| - | - |
| `doc-parser` | Document parsing: can be used to extract layout elements such as tables, formulas, stamps, images, etc. from documents; includes models like PP-StructureV3 |
| `ie` | Information extraction: can be used to extract key information from documents, such as names, dates, addresses, amounts, etc.; includes models like PP-ChatOCRv4 |
| `trans` | Document translation: can be used to translate documents from one language to another; includes models like PP-DocTranslation |
| `all` | Complete functionality |

The general OCR pipeline (e.g., PP-OCRv3/v4/v5) and the document image preprocessing pipeline can be used without installing any additional dependency groups. Apart from these two pipelines, each remaining pipeline belongs to one and only one dependency group. You can refer to the usage documentation of each pipeline to determine which group it belongs to. For individual functional modules, installing any dependency group that includes the module will enable access to its core functionality.

## 2.2 Install Training Dependencies

To perform model training, exporting, etc., first clone the repository to your local machine:

```bash
# Recommended method
git clone https://github.com/PaddlePaddle/PaddleOCR

# (Optional) Switch to a specific branch
git checkout release/3.2

# If you encounter network issues preventing successful cloning, you can also use the repository on Gitee:
git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The code hosted on Gitee may not be synchronized in real-time with updates from this GitHub project, with a delay of 3~5 days. Please prioritize using the recommended method.
```

Run the following command to install the dependencies:

```bash
python -m pip install -r requirements.txt
```
