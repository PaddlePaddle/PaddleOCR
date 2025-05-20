---
comments: true
---

# Installation

# 1. Install PaddlePaddle Framework

Please refer to the [PaddlePaddle Official Website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip.html) to install PaddlePaddle framework version `3.0` or above. **Using the official PaddlePaddle Docker image is recommended.**

# 2. Install PaddleOCR

If you only want to use the inference capabilities of PaddleOCR, please refer to [Install Inference Package](#21-install-inference-package); if you want to perform model training, exporting, etc., please refer to [Install Training Dependencies](#22-install-training-dependencies). It is allowed to install both the inference package and training dependencies in the same environment without the need for environment isolation.

## 2.1 Install Inference Package

Install the latest version of the PaddleOCR inference package from PyPI:

```bash
python -m pip install paddleocr
```

Or install from source (default is the development branch):

```bash
python -m pip install "git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

## 2.2 Install Training Dependencies

To perform model training, exporting, etc., first clone the repository to your local machine:

```bash
# Recommended method
git clone https://github.com/PaddlePaddle/PaddleOCR

# (Optional) Switch to a specific branch
git checkout release/3.0

# If you encounter network issues preventing successful cloning, you can also use the repository on Gitee:
git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The code hosted on Gitee may not be synchronized in real-time with updates from this GitHub project, with a delay of 3~5 days. Please prioritize using the recommended method.
```

Run the following command to install the dependencies:

```bash
python -m pip install -r requirements.txt
```
