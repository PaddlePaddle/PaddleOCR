# 视觉问答（VQA）

VQA主要特性如下：

- 集成[LayoutXLM](https://arxiv.org/pdf/2104.08836.pdf)模型以及PP-OCR预测引擎。
- 支持基于多模态方法的语义实体识别 (Semantic Entity Recognition, SER) 以及关系抽取 (Relation Extraction, RE) 任务。基于 SER 任务，可以完成对图像中的文本识别与分类；基于 RE 任务，可以完成对图象中的文本内容的关系提取（比如判断问题对）
- 支持SER任务与OCR引擎联合的端到端系统预测与评估。
- 支持SER任务和RE任务的自定义训练


本项目是 [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/pdf/2104.08836.pdf) 在 Paddle 2.2上的开源实现，
包含了在 [XFUND数据集](https://github.com/doc-analysis/XFUND) 上的微调代码。

## 1. 效果演示

**注意：** 测试图片来源于XFUN数据集。

### 1.1 SER

<div align="center">
<img src="./images/result_ser/zh_val_0_ser.jpg"  width = "600" />
</div>

<div align="center">
<img src="./images/result_ser/zh_val_42_ser.jpg"  width = "600" />
</div>

其中不同颜色的框表示不同的类别，对于XFUN数据集，有`QUESTION`, `ANSWER`, `HEADER` 3种类别，在OCR检测框的左上方也标出了对应的类别和OCR识别结果。


### 1.2 RE

* Coming soon!



## 2. 安装

### 2.1 安装依赖

- **（1) 安装PaddlePaddle**

```bash
pip3 install --upgrade pip

# GPU安装
python3 -m pip install paddlepaddle-gpu==2.2 -i https://mirror.baidu.com/pypi/simple

# CPU安装
python3 -m pip install paddlepaddle==2.2 -i https://mirror.baidu.com/pypi/simple

```
更多需求，请参照[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。


### 2.2 安装PaddleOCR（包含 PP-OCR 和 VQA ）

- **（1）pip快速安装PaddleOCR whl包（仅预测）**

```bash
pip install "paddleocr>=2.2" # 推荐使用2.2+版本
```

- **（2）下载VQA源码（预测+训练）**

```bash
【推荐】git clone https://github.com/PaddlePaddle/PaddleOCR

# 如果因为网络问题无法pull成功，也可选择使用码云上的托管：
git clone https://gitee.com/paddlepaddle/PaddleOCR

# 注：码云托管代码可能无法实时同步本github项目更新，存在3~5天延时，请优先使用推荐方式。
```

- **（3）安装PaddleNLP**

```bash
# 需要使用PaddleNLP最新的代码版本进行安装
git clone https://github.com/PaddlePaddle/PaddleNLP -b develop
cd PaddleNLP
pip install -e .
```


- **（4）安装VQA的`requirements`**

```bash
pip install -r requirements.txt
```

## 3. 使用


### 3.1 数据和预训练模型准备

处理好的XFUN中文数据集下载地址：[https://paddleocr.bj.bcebos.com/dataset/XFUND.tar](https://paddleocr.bj.bcebos.com/dataset/XFUND.tar)。


下载并解压该数据集，解压后将数据集放置在当前目录下。

```shell
wget https://paddleocr.bj.bcebos.com/dataset/XFUND.tar
```

如果希望转换XFUN中其他语言的数据集，可以参考[XFUN数据转换脚本](helper/trans_xfun_data.py)。

如果希望直接体验预测过程，可以下载我们提供的SER预训练模型，跳过训练过程，直接预测即可。

* SER任务预训练模型下载链接：[链接](https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_ser_pretrained.tar)
* RE任务预训练模型下载链接：coming soon!


### 3.2 SER任务

* 启动训练

```shell
python train_ser.py \
    --model_name_or_path "layoutxlm-base-uncased" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --num_train_epochs 200 \
    --eval_steps 10 \
    --save_steps 500 \
    --output_dir "./output/ser/" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --evaluate_during_training \
    --seed 2048
```

最终会打印出`precision`, `recall`, `f1`等指标，如下所示。

```
best metrics: {'loss': 1.066644651549203, 'precision': 0.8770182068017863, 'recall': 0.9361936193619362, 'f1': 0.9056402979780063}
```

模型和训练日志会保存在`./output/ser/`文件夹中。

* 使用评估集合中提供的OCR识别结果进行预测

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 infer_ser.py \
    --model_name_or_path "./PP-Layout_v1.0_ser_pretrained/" \
    --output_dir "output_res/" \
    --infer_imgs "XFUND/zh_val/image/" \
    --ocr_json_path "XFUND/zh_val/xfun_normalize_val.json"
```

最终会在`output_res`目录下保存预测结果可视化图像以及预测结果文本文件，文件名为`infer_results.txt`。

* 使用`OCR引擎 + SER`串联结果

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 infer_ser_e2e.py \
    --model_name_or_path "./output/PP-Layout_v1.0_ser_pretrained/" \
    --max_seq_length 512 \
    --output_dir "output_res_e2e/"
```

* 对`OCR引擎 + SER`预测系统进行端到端评估

```shell
export CUDA_VISIBLE_DEVICES=0
python helper/eval_with_label_end2end.py --gt_json_path XFUND/zh_val/xfun_normalize_val.json  --pred_json_path output_res/infer_results.txt
```


3.3 RE任务

coming soon!


## 参考链接

- LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding, https://arxiv.org/pdf/2104.08836.pdf
- microsoft/unilm/layoutxlm, https://github.com/microsoft/unilm/tree/master/layoutxlm
- XFUND dataset, https://github.com/doc-analysis/XFUND
