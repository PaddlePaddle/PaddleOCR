# 文档视觉问答（DOC-VQA）

VQA指视觉问答，主要针对图像内容进行提问和回答,DOC-VQA是VQA任务中的一种，DOC-VQA主要针对文本图像的文字内容提出问题。

PP-Structure 里的 DOC-VQA算法基于PaddleNLP自然语言处理算法库进行开发。

主要特性如下：

- 集成[LayoutXLM](https://arxiv.org/pdf/2104.08836.pdf)模型以及PP-OCR预测引擎。
- 支持基于多模态方法的语义实体识别 (Semantic Entity Recognition, SER) 以及关系抽取 (Relation Extraction, RE) 任务。基于 SER 任务，可以完成对图像中的文本识别与分类；基于 RE 任务，可以完成对图象中的文本内容的关系提取，如判断问题对(pair)。
- 支持SER任务和RE任务的自定义训练。
- 支持OCR+SER的端到端系统预测与评估。
- 支持OCR+SER+RE的端到端系统预测。


本项目是 [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/pdf/2104.08836.pdf) 在 Paddle 2.2上的开源实现，
包含了在 [XFUND数据集](https://github.com/doc-analysis/XFUND) 上的微调代码。

## 1 性能

我们在 [XFUN](https://github.com/doc-analysis/XFUND) 的中文数据集上对算法进行了评估，性能如下

| 模型 | 任务 | f1 | 模型下载地址 |
|:---:|:---:|:---:| :---:|
| LayoutXLM | RE | 0.7113 | [链接](https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_re_pretrained.tar) |
| LayoutXLM | SER | 0.9056 | [链接](https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_ser_pretrained.tar) |
| LayoutLM | SER | 0.78 | [链接](https://paddleocr.bj.bcebos.com/pplayout/LayoutLM_ser_pretrained.tar) |



## 2. 效果演示

**注意：** 测试图片来源于XFUN数据集。

### 2.1 SER

![](./images/result_ser/zh_val_0_ser.jpg) | ![](./images/result_ser/zh_val_42_ser.jpg)
---|---

图中不同颜色的框表示不同的类别，对于XFUN数据集，有`QUESTION`, `ANSWER`, `HEADER` 3种类别

* 深紫色：HEADER
* 浅紫色：QUESTION
* 军绿色：ANSWER

在OCR检测框的左上方也标出了对应的类别和OCR识别结果。


### 2.2 RE

![](./images/result_re/zh_val_21_re.jpg) | ![](./images/result_re/zh_val_40_re.jpg)
---|---


图中红色框表示问题，蓝色框表示答案，问题和答案之间使用绿色线连接。在OCR检测框的左上方也标出了对应的类别和OCR识别结果。


## 3. 安装

### 3.1 安装依赖

- **（1) 安装PaddlePaddle**

```bash
pip3 install --upgrade pip

# GPU安装
python3 -m pip install paddlepaddle-gpu==2.2 -i https://mirror.baidu.com/pypi/simple

# CPU安装
python3 -m pip install paddlepaddle==2.2 -i https://mirror.baidu.com/pypi/simple

```
更多需求，请参照[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。


### 3.2 安装PaddleOCR（包含 PP-OCR 和 VQA ）

- **（1）pip快速安装PaddleOCR whl包（仅预测）**

```bash
pip install paddleocr
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
pip3 install -e .
```


- **（4）安装VQA的`requirements`**

```bash
cd ppstructure/vqa
pip install -r requirements.txt
```

## 4. 使用


### 4.1 数据和预训练模型准备

处理好的XFUN中文数据集下载地址：[https://paddleocr.bj.bcebos.com/dataset/XFUND.tar](https://paddleocr.bj.bcebos.com/dataset/XFUND.tar)。


下载并解压该数据集，解压后将数据集放置在当前目录下。

```shell
wget https://paddleocr.bj.bcebos.com/dataset/XFUND.tar
```

如果希望转换XFUN中其他语言的数据集，可以参考[XFUN数据转换脚本](helper/trans_xfun_data.py)。

如果希望直接体验预测过程，可以下载我们提供的预训练模型，跳过训练过程，直接预测即可。


### 4.2 SER任务

* 启动训练

```shell
python3.7 train_ser.py \
    --model_name_or_path "layoutxlm-base-uncased" \
    --ser_model_type "LayoutXLM" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --num_train_epochs 200 \
    --eval_steps 10 \
    --output_dir "./output/ser/" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --evaluate_during_training \
    --seed 2048
```

最终会打印出`precision`, `recall`, `f1`等指标，模型和训练日志会保存在`./output/ser/`文件夹中。

* 恢复训练

```shell
python3.7 train_ser.py \
    --model_name_or_path "model_path" \
    --ser_model_type "LayoutXLM" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --num_train_epochs 200 \
    --eval_steps 10 \
    --output_dir "./output/ser/" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --evaluate_during_training \
    --num_workers 8 \
    --seed 2048 \
    --resume
```

* 评估
```shell
export CUDA_VISIBLE_DEVICES=0
python3 eval_ser.py \
    --model_name_or_path "PP-Layout_v1.0_ser_pretrained/" \
    --ser_model_type "LayoutXLM" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \
    --output_dir "output/ser/"  \
    --seed 2048
```
最终会打印出`precision`, `recall`, `f1`等指标

* 使用评估集合中提供的OCR识别结果进行预测

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 infer_ser.py \
    --model_name_or_path "PP-Layout_v1.0_ser_pretrained/" \
    --ser_model_type "LayoutXLM" \
    --output_dir "output/ser/" \
    --infer_imgs "XFUND/zh_val/image/" \
    --ocr_json_path "XFUND/zh_val/xfun_normalize_val.json"
```

最终会在`output_res`目录下保存预测结果可视化图像以及预测结果文本文件，文件名为`infer_results.txt`。

* 使用`OCR引擎 + SER`串联结果

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 infer_ser_e2e.py \
    --model_name_or_path "PP-Layout_v1.0_ser_pretrained/" \
    --ser_model_type "LayoutXLM" \
    --max_seq_length 512 \
    --output_dir "output/ser_e2e/" \
    --infer_imgs "images/input/zh_val_0.jpg"
```

* 对`OCR引擎 + SER`预测系统进行端到端评估

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 helper/eval_with_label_end2end.py --gt_json_path XFUND/zh_val/xfun_normalize_val.json  --pred_json_path output_res/infer_results.txt
```


### 3.3 RE任务

* 启动训练

```shell
export CUDA_VISIBLE_DEVICES=0
python3 train_re.py \
    --model_name_or_path "layoutxlm-base-uncased" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --label_map_path 'labels/labels_ser.txt' \
    --num_train_epochs 200 \
    --eval_steps 10 \
    --output_dir "output/re/"  \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \
    --evaluate_during_training \
    --seed 2048

```

* 恢复训练

```shell
export CUDA_VISIBLE_DEVICES=0
python3 train_re.py \
    --model_name_or_path "model_path" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --label_map_path 'labels/labels_ser.txt' \
    --num_train_epochs 2 \
    --eval_steps 10 \
    --output_dir "output/re/"  \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \
    --evaluate_during_training \
    --seed 2048 \
    --resume

```

最终会打印出`precision`, `recall`, `f1`等指标，模型和训练日志会保存在`./output/re/`文件夹中。

* 评估
```shell
export CUDA_VISIBLE_DEVICES=0
python3 eval_re.py \
    --model_name_or_path "PP-Layout_v1.0_re_pretrained/" \
    --max_seq_length 512 \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --label_map_path 'labels/labels_ser.txt' \
    --output_dir "output/re/"  \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \
    --seed 2048
```
最终会打印出`precision`, `recall`, `f1`等指标


* 使用评估集合中提供的OCR识别结果进行预测

```shell
export CUDA_VISIBLE_DEVICES=0
python3 infer_re.py \
    --model_name_or_path "PP-Layout_v1.0_re_pretrained/" \
    --max_seq_length 512 \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --label_map_path 'labels/labels_ser.txt' \
    --output_dir "output/re/"  \
    --per_gpu_eval_batch_size 1 \
    --seed 2048
```

最终会在`output_res`目录下保存预测结果可视化图像以及预测结果文本文件，文件名为`infer_results.txt`。

* 使用`OCR引擎 + SER + RE`串联结果

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 infer_ser_re_e2e.py \
    --model_name_or_path "PP-Layout_v1.0_ser_pretrained/" \
    --re_model_name_or_path "PP-Layout_v1.0_re_pretrained/" \
    --ser_model_type "LayoutXLM" \
    --max_seq_length 512 \
    --output_dir "output/ser_re_e2e/" \
    --infer_imgs "images/input/zh_val_21.jpg"
```

## 参考链接

- LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding, https://arxiv.org/pdf/2104.08836.pdf
- microsoft/unilm/layoutxlm, https://github.com/microsoft/unilm/tree/master/layoutxlm
- XFUND dataset, https://github.com/doc-analysis/XFUND
