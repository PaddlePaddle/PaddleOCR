# OCR模型自动压缩示例

目录：
- [OCR模型自动压缩示例](#ocr模型自动压缩示例)
  - [1. 简介](#1-简介)
  - [2. Benchmark](#2-benchmark)
    - [PPOCRV4\_det](#ppocrv4_det)
    - [PPOCRV4\_rec](#ppocrv4_rec)
  - [3. 自动压缩流程](#3-自动压缩流程)
    - [3.1 准备环境](#31-准备环境)
    - [3.2 准备数据集](#32-准备数据集)
      - [3.2.1 PPOCRV4\_det\_server数据集预处理](#321-ppocrv4_det_server数据集预处理)
    - [3.3 准备预测模型](#33-准备预测模型)
  - [4.预测部署](#4预测部署)
      - [4.1 Paddle Inference 验证性能](#41-paddle-inference-验证性能)
        - [4.1.1 使用测试脚本进行批量测试：](#411-使用测试脚本进行批量测试)
        - [4.1.2 基于压缩模型进行基于GPU的批量测试：](#412-基于压缩模型进行基于gpu的批量测试)
        - [4.1.3 基于压缩前模型进行基于GPU的批量测试：](#413-基于压缩前模型进行基于gpu的批量测试)
        - [4.1.4 基于压缩模型进行基于CPU的批量测试：](#414-基于压缩模型进行基于cpu的批量测试)
    - [4.2 PaddleLite端侧部署](#42-paddlelite端侧部署)
  - [5.FAQ](#5faq)
    - [5.1 报错找不到模型文件或者数据集文件](#51-报错找不到模型文件或者数据集文件)
    - [5.2 软件环境一致，硬件不同导致精度差异很大？](#52-软件环境一致硬件不同导致精度差异很大)


## 1. 简介
本示例将以图像分类模型PPOCRV3为例，介绍如何使用PaddleOCR中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化训练和蒸馏。

## 2. Benchmark

### PPOCRV4_det
| 模型 | 策略 | Metric(hmean) | GPU 耗时(ms) | ARM CPU 耗时(ms) | 配置文件 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 中文PPOCRV4-det_mobile | Baseline | 72.71 | 5.7 | 92.0 | - | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/det/ch_PP-OCRv4_det_infer.tar) |
| 中文PPOCRV4-det_mobile | 量化+蒸馏 | 71.10 | 2.3 | 94.1 | [Config](./configs/ppocrv4/ppocrv4_det_qat_dist.yaml) | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/det/det_mobile_qat_3090.zip) |
| 中文PPOCRV4-det_server | Baseline | 79.82 | 32.6 | 844.7 | - | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/det/ch_PP-OCRv4_det_server_infer.tar) |
| 中文PPOCRV4-det_server | 量化+蒸馏 | 79.27 | 12.3 | 635.0 | [Config](./configs/ppocrv4/ppocrv4_rec_server_qat_dist.yaml) | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/det/det_server_qat_3090.zip) |
> - GPU测试环境：RTX 3090, cuda11.7+tensorrt8.4.2.4+paddle2.5
> - CPU测试环境：Intel(R) Xeon(R) Gold 6226R，使用12线程测试
> - PPOCRV4-det_server在不完整的数据集上测试，数据处理流程参考[ppocrv4_det_server数据集预处理](#321-ppocrv4_det_server数据集预处理)，仅为了展示自动压缩效果，指标并不具有参考性，模型真实表现请参考[PPOCRV4介绍](../../../doc/doc_ch/PP-OCRv4_introduction.md)

| 模型 | 策略 | Metric(hmean) | GPU 耗时(ms) | ARM CPU 耗时(ms) | 配置文件 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 中文PPOCRV4-det_mobile | Baseline | 72.71 | 4.7 | 198.4 | - | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/det/ch_PP-OCRv4_det_infer.tar) |
| 中文PPOCRV4-det_mobile | 量化+蒸馏 | 71.38 | 3.3 | 205.2 | [Config](./configs/ppocrv4/ppocrv4_det_qat_dist.yaml) | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/det/det_server_qat_v100.zip) |
| 中文PPOCRV4-det_server | Baseline | 79.77 | 50.0 | 2159.4 | - | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/det/ch_PP-OCRv4_det_server_infer.tar) |
| 中文PPOCRV4-det_server | 量化+蒸馏 | 79.81 | 42.4 | 1834.8 | [Config](./configs/ppocrv4/ppocrv4_rec_server_qat_dist.yaml) | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/det/det_server_qat_v100.zip) |
> - GPU测试环境：Tesla V100, cuda11.7+tensorrt8.4.2.4+paddle2.5.2
> - CPU测试环境：Intel(R) Xeon(R) Gold 6271C，使用12线程测试
> - PPOCRV4-det_server在不完整的数据集上测试，数据处理流程参考[ppocrv4_det_server数据集预处理](#321-ppocrv4_det_server数据集预处理)，仅为了展示自动压缩效果，指标并不具有参考性，模型真实表现请参考[PPOCRV4介绍](../../../doc/doc_ch/PP-OCRv4_introduction.md)

### PPOCRV4_rec
| 模型 | 策略 | Metric(accuracy) | GPU 耗时(ms) | ARM CPU 耗时(ms) | 配置文件 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 中文PPOCRV4-rec_mobile | Baseline | 78.92 | 1.7 | 33.3 | - | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/rec/ch_PP-OCRv4_rec_infer.tar.gz) |
| 中文PPOCRV4-rec_mobile | 量化+蒸馏 | 78.41 | 1.4 | 34.0 | [Config](./configs/ppocrv4/ppocrv4_rec_qat_dist.yaml) | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/rec/rec_mobile_qat.tar.gz) |
| 中文PPOCRV4-rec_server | Baseline | 81.62 | 4.0 | 62.5 | - | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/rec/ch_PP-OCRv4_rec_server_infer.tar.gz) |
| 中文PPOCRV4-rec_server | 量化+蒸馏 | 81.03 | 2.0 | 64.4 | [Config](./configs/ppocrv4/ppocrv4_rec_server_qat_dist.yaml) | [Model](https://paddle-ocr-models.bj.bcebos.com/ppocrv4_qat/rec/rec_server_qat.tar.gz) |
> - GPU测试环境：Tesla V100, cuda11.2+tensorrt8.0.3.4+paddle2.5
> - CPU测试环境：Intel(R) Xeon(R) Gold 6271C，使用12线程测试


## 3. 自动压缩流程

### 3.1 准备环境

- PaddlePaddle == 2.5 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim == 2.5
- PaddleOCR == develop

安装paddlepaddle：
```shell
# CPU
python -m pip install paddlepaddle==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# GPU 以Ubuntu、CUDA 10.2为例
python -m pip install paddlepaddle-gpu==2.5.1.post102 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

安装paddleslim 2.5：
```shell
pip install paddleslim@git+https://gitee.com/paddlepaddle/PaddleSlim.git@release/2.5
```

安装其他依赖：
```shell
pip install scikit-image imgaug
```


下载PaddleOCR:
```shell
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/
pip install -r requirements.txt
```

### 3.2 准备数据集
公开数据集可参考[OCR数据集](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/dataset/ocr_datasets.md)，然后根据程序运行过程中提示放置到对应位置。

#### 3.2.1 PPOCRV4_det_server数据集预处理
PPOCRV4_det_server在使用原始数据集推理时，默认将输入图像的最小边缩放到736，然而原始数据集中存在一些长宽比很大的图像，比如13:1，此时再进行缩放就会导致长边的尺寸非常大，在实验过程中发现最大的长边尺寸有10000+，这导致在构建TensorRT子图的时候显存不足。

为了能顺利跑通自动压缩的流程，展示自动压缩的效果，因此需要对原始数据集进行预处理，将长宽比过大的图像进行剔除，处理脚本可见[ppocrv4_det_server_dataset_process.py](./ppocrv4_det_server_dataset_process.py)。


> 注意：使用不同的数据集需要修改配置文件中`dataset`中数据路径和数据处理部分。

### 3.3 准备预测模型
预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

> 注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

可在[PaddleOCR模型库](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/models_list.md)中直接获取Inference模型，具体可参考下方获取中文PPOCRV4模型示例：

```shell
https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
tar -xf ch_PP-OCRv4_rec_infer.tar
```

```shell
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
tar -xf ch_PP-OCRv4_det_infer.tar
```

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口 ```paddleslim.auto_compression.AutoCompression``` 对模型进行量化训练和蒸馏。配置config文件中模型路径、数据集路径、蒸馏、量化和训练等部分的参数，配置完成后便可开始自动压缩。

**单卡启动**

```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --save_dir='./save_quant_ppocrv4_det/' --config_path='./configs/ppocrv4/ppocrv4_det_qat_dist.yaml'
```

**多卡启动**

若训练任务中包含大量训练数据，如果使用单卡训练，会非常耗时，使用分布式训练可以达到几乎线性的加速比。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./save_quant_ppocrv4_det/' --config_path='./configs/ppocrv4/ppocrv4_det_qat_dist.yaml'
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

#### 4.1 Paddle Inference 验证性能

输出的量化模型也是静态图模型，静态图模型在GPU上可以使用TensorRT进行加速，在CPU上可以使用MKLDNN进行加速。

TensorRT预测环境配置：
1. 如果使用 TesorRT 预测引擎，需安装 ```WITH_TRT=ON``` 的Paddle，上述paddle下载的2.5满足打开TensorRT编译的要求。
2. 使用TensorRT预测需要进一步安装TensorRT，安装TensorRT的方式参考[TensorRT安装说明](../../../docs/deployment/installtrt.md)。

以下字段用于配置预测参数：

| 参数名 | 含义 |
|:------:|:------:|
| model_path | inference 模型文件所在目录，该目录下需要有文件 .pdmodel 和 .pdiparams 两个文件 |
| model_filename | inference_model_dir文件夹下的模型文件名称 |
| params_filename | inference_model_dir文件夹下的参数文件名称 |
| dataset_config | 数据集配置的config  |
| image_file | 待测试单张图片的路径，如果设置image_file，则dataset_config将无效。   |
| device | 预测时的设备，可选：`CPU`, `GPU`。  |
| use_trt | 是否使用 TesorRT 预测引擎，在device为```GPU```时生效。   |
| use_mkldnn | 是否启用```MKL-DNN```加速库，注意```use_mkldnn```，在device为```CPU```时生效。  |
| cpu_threads | CPU预测时，使用CPU线程数量，默认10  |
| precision | 预测时精度，可选：`fp32`, `fp16`, `int8`。 |


准备好预测模型，并且修改dataset_config中数据集路径为正确的路径后，启动测试：

##### 4.1.1 使用测试脚本进行批量测试：

我们提供两个脚本文件用于测试模型自动化压缩的效果，分别是[test_ocr_det.sh](./test_ocr_det.sh)和[test_ocr_rec.sh](./test_ocr_rec.sh)，这两个脚本都接收一个`model_type`参数，用于区分是测试mobile模型还是server模型，可选参数为`mobile`和`server`，使用示例：

  ```shell
  # 测试mobile模型
  bash test_ocr_det.sh mobile
  bash test_ocr_rec.sh mobile
  # 测试server模型
  bash test_ocr_det.sh server
  bash test_ocr_rec.sh server
  ```

##### 4.1.2 基于压缩模型进行基于GPU的批量测试：

```shell
cd deploy/slim/auto_compression
python test_ocr.py \
      --model_path save_quant_ppocrv4_det \
      --config_path configs/ppocrv4/ppocrv4_det_qat_dist.yaml \
      --device GPU  \
      --use_trt True  \
      --precision int8
```


##### 4.1.3 基于压缩前模型进行基于GPU的批量测试：

```shell
cd deploy/slim/auto_compression
python test_ocr.py \
      --model_path ch_PP-OCRv4_det_infer \
      --config_path configs/ppocrv4/ppocrv4_rec_det_dist.yaml \
      --device GPU  \
      --use_trt True  \
      --precision int8
```


##### 4.1.4 基于压缩模型进行基于CPU的批量测试：

- MKLDNN预测：

```shell
cd deploy/slim/auto_compression
python test_ocr.py \
      --model_path save_quant_ppocrv4_det \
      --config_path configs/ppocrv4/ppocrv4_det_qat_dist.yaml \
      --device GPU  \
      --use_trt True  \
      --use_mkldnn=True \
      --precision=int8 \
      --cpu_threads=10
```

### 4.2 PaddleLite端侧部署
PaddleLite端侧部署可参考：
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleOCR/tree/9cdab61d909eb595af849db885c257ca8c74cb57/deploy/lite)

## 5.FAQ

### 5.1 报错找不到模型文件或者数据集文件

如果在推理或者跑ACT时报错找不到模型文件或者数据集文件，可以检查一下配置文件中的路径是否正确，以det_mobile为例，配置文件中的指定模型路径的配置信息如下：

```yaml
Global:
  model_dir: ./models/ch_PP-OCRv4_det_infer
  model_filename: inference.pdmodel
  params_filename: inference.pdiparams
```
指定训练集验证集路径的配置信息如下：

```yaml
Train:
  dataset:
    name: SimpleDataSet
    data_dir: datasets/chinese
    label_file_list:
        - datasets/chinese/zhongce_training_fix_1.6k.txt
        - datasets/chinese/label_train_all_f4_part2.txt
        - datasets/chinese/label_train_all_f4_part3.txt
        - datasets/chinese/label_train_all_f4_part4.txt
        - datasets/chinese/label_train_all_f4_part5.txt
        - datasets/chinese/synth_en_my_clip.txt
        - datasets/chinese/synth_ch_my_clip.txt
        - datasets/chinese/synth_en_my_largeword_clip.txt
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: datasets/v4_4_test_dataset
    label_file_list:
      - datasets/v4_4_test_dataset/label.txt
```

### 5.2 软件环境一致，硬件不同导致精度差异很大？

这种情况是正常的，TensorRT针对不同的硬件设备有着不同的优化方法，同一种优化策略在不同硬件上可能有着截然不同的表现，以本实验的ppocrv4_det_server为举例。截取[test_ocr.py](./test_ocr.py)中的一部分代码如下所示：
```python
if args.precision == 'int8' and "ppocrv4_det_server_qat_dist.yaml" in args.config_path:
    # Use the following settings only when the hardware is a Tesla V100. If you are using
    # a RTX 3090, use the settings in the else branch.
    pred_cfg.enable_tensorrt_engine(
        workspace_size=1 << 30,
        max_batch_size=1,
        min_subgraph_size=30,
        precision_mode=precision_map[args.precision],
        use_static=True,
        use_calib_mode=False, )
    pred_cfg.exp_disable_tensorrt_ops(["elementwise_add"])
else:
    pred_cfg.enable_tensorrt_engine(
    workspace_size=1 << 30,
    max_batch_size=1,
    min_subgraph_size=4,
    precision_mode=precision_map[args.precision],
    use_static=True,
    use_calib_mode=False, )
```
当硬件为RTX 3090的时候，使用else分支中的策略即可得到正常的结果，但是当硬件是Tesla V100的时候，必须使用if分支中的策略才能保证量化后精度不下降，具体结果参考[benchmark](#2-benchmark)。
