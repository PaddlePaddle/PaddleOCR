# OCR模型自动压缩示例

目录：
- [1. 简介](#1简介)
- [2. Benchmark](#2Benchmark)
- [3. 自动压缩流程](#自动压缩流程)
  - [3.1 准备环境](#31-准备准备)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4. 预测部署](#4预测部署)
  - [4.1 Python预测推理](#41-Python预测推理)
  - [4.2 PaddleLite端侧部署](#42-PaddleLite端侧部署)
- [5. FAQ](5FAQ)


## 1. 简介
本示例将以图像分类模型PPOCRV3为例，介绍如何使用PaddleOCR中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化训练和蒸馏。

## 2. Benchmark

### PPOCRV4
| 模型 | 策略 | Metric | GPU 耗时(ms) | ARM CPU 耗时(ms) | 配置文件 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|




### PPOCRV3
| 模型 | 策略 | Metric | GPU 耗时(ms) | ARM CPU 耗时(ms) | 配置文件 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 中文PPOCRV3-det | Baseline | 84.57 | - | - | - | [Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) |
| 中文PPOCRV3-det | 量化+蒸馏 | 85.01 | - | - | [Config](./configs/ppocrv3_det_qat_dist.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/OCR/PPOCRV3_det_QAT.tar) |
| 中文PPOCRV3-rec | Baseline | 76.48 | - | - | - | [Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) |
| 中文PPOCRV3-rec | 量化+蒸馏 | 73.23 | - | - | [Config](./configs/ppocrv3_rec_qat_dist.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/OCR/PPOCRV3_rec_QAT.tar) |
> PPOCRV3-det 的测试指标为 hmean，PPOCRV3-rec的测试指标为 accuracy.




## 3. 自动压缩流程

### 3.1 准备环境

- python >= 3.6
- PaddlePaddle >= 2.4 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.4

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle==2.4.1
# GPU 以Ubuntu、CUDA 11.2为例
python -m pip install paddlepaddle-gpu==2.4.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

安装paddleslim：
```shell
pip install paddleslim
```

安装其他依赖：
```shell
pip install scikit-image imgaug
```


下载PaddleOCR:
```shell
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
```
> 你需要下载到 Paddleslim/example/auto_compression/ 目录下并运行 pip install -r requirements.txt 安装依赖。下载 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR.git) 的目的只是为了直接使用 PaddleOCR 中的 Dataloader 组件和精度评估模块，不涉及模型组网等。通过 `pip install paddleocr` 安装的 paddleocr 只有预测代码，没有数据集读取和精度评估的部分，因此需要下载 PaddleOCR 库。

### 3.2 准备数据集
公开数据集可参考[OCR数据集](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/dataset/ocr_datasets.md)，然后根据程序运行过程中提示放置到对应位置。

> 注意：使用不同的数据集需要修改配置文件中`dataset`中数据路径和数据处理部分。

### 3.3 准备预测模型
预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

> 注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

可在[PaddleOCR模型库](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)中直接获取Inference模型，具体可参考下方获取中文PPOCRV3模型示例：

```shell
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xf ch_PP-OCRv3_det_infer.tar
```

```shell
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xf ch_PP-OCRv3_rec_infer.tar
```

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口 ```paddleslim.auto_compression.AutoCompression``` 对模型进行量化训练和蒸馏。配置config文件中模型路径、数据集路径、蒸馏、量化和训练等部分的参数，配置完成后便可开始自动压缩。

**单卡启动**

```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --save_dir='./save_quant_ppocr_det/' --config_path='./configs/ppocrv3_det_qat_dist.yaml'
```

**多卡启动**

若训练任务中包含大量训练数据，如果使用单卡训练，会非常耗时，使用分布式训练可以达到几乎线性的加速比。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./save_quant_ppocr_det/' --config_path='./configs/ppocrv3_det_qat_dist.yaml'
```
多卡训练指的是将训练任务按照一定方法拆分到多个训练节点完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。服务节点在收到所有训练节点传来的梯度后，会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。多卡训练一轮训练能训练```batch size * num gpus```的数据，比如单卡的```batch size```为32，单轮训练的数据量即32，而四卡训练的```batch size```为32，单轮训练的数据量为128。

注意 ```learning rate``` 与 ```batch size``` 呈线性关系，这里单卡 ```batch size``` 8，对应的 ```learning rate``` 为0.00005，那么如果 ```batch size``` 增大4倍改为32，```learning rate``` 也需乘以4；多卡时 ```batch size``` 为8，```learning rate``` 需乘上卡数。所以改变 ```batch size``` 或改变训练卡数都需要对应修改 ```learning rate```。


**验证精度**

根据训练log可以看到模型验证的精度，若需再次验证精度，修改配置文件```./configs/ppocrv3_det_qat_dist.yaml```中所需验证模型的文件夹路径及模型和参数名称```model_dir, model_filename, params_filename```，然后使用以下命令进行验证：

```shell
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path='./configs/ppocrv3_det_qat_dist.yaml'
```

## 4.预测部署
### 4.1 Python预测推理

环境配置：若使用 TesorRT 预测引擎，需安装 ```WITH_TRT=ON``` 的Paddle，下载地址：[Python预测库](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html#python)

Python预测引擎推理可参考[基于Python预测引擎推理](https://github.com/PaddlePaddle/PaddleOCR/blob/9cdab61d909eb595af849db885c257ca8c74cb57/doc/doc_ch/inference_ppocr.md)

### 4.2 PaddleLite端侧部署
PaddleLite端侧部署可参考：
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleOCR/tree/9cdab61d909eb595af849db885c257ca8c74cb57/deploy/lite)

## 5.FAQ
