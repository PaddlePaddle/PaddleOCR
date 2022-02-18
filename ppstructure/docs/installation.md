- [快速安装](#快速安装)
  - [1. PaddlePaddle 和 PaddleOCR](#1-paddlepaddle-和-paddleocr)
  - [2. 安装其他依赖](#2-安装其他依赖)
    - [2.1 版面分析所需  Layout-Parser](#21-版面分析所需--layout-parser)
    - [2.2  VQA所需依赖](#22--vqa所需依赖)

# 快速安装

## 1. PaddlePaddle 和 PaddleOCR

可参考[PaddleOCR安装文档](../../doc/doc_ch/installation.md)

## 2. 安装其他依赖

### 2.1 版面分析所需  Layout-Parser

Layout-Parser 可通过如下命令安装

```bash
pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
```
### 2.2  VQA所需依赖
* paddleocr

```bash
pip3 install paddleocr
```

* PaddleNLP
```bash
git clone https://github.com/PaddlePaddle/PaddleNLP -b develop
cd PaddleNLP
pip3 install -e .
```
