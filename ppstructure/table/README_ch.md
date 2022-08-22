[English](README.md) | 简体中文

# 表格识别

- [1. 表格识别 pipeline](#1-表格识别-pipeline)
- [2. 性能](#2-性能)
- [3. 效果演示](#3-效果演示)
- [4. 使用](#4-使用)
  - [4.1 快速开始](#41-快速开始)
  - [4.2 训练](#42-训练)
  - [4.3 计算TEDS](#43-计算teds)
- [5. Reference](#5-reference)


## 1. 表格识别 pipeline

表格识别主要包含三个模型
1. 单行文本检测-DB
2. 单行文本识别-CRNN
3. 表格结构和cell坐标预测-SLANet

具体流程图如下

![tableocr_pipeline](../docs/table/tableocr_pipeline.jpg)

流程说明:

1. 图片由单行文字检测模型检测到单行文字的坐标，然后送入识别模型拿到识别结果。
2. 图片由SLANet模型拿到表格的结构信息和单元格的坐标信息。
3. 由单行文字的坐标、识别结果和单元格的坐标一起组合出单元格的识别结果。
4. 单元格的识别结果和表格结构一起构造表格的html字符串。


## 2. 性能

我们在 PubTabNet<sup>[1]</sup> 评估数据集上对算法进行了评估，性能如下


|算法|Acc|[TEDS(Tree-Edit-Distance-based Similarity)](https://github.com/ibm-aur-nlp/PubTabNet/tree/master/src)|Speed|
| --- | --- | --- | ---|
| EDD<sup>[2]</sup> |x| 88.3% |x|
| TableRec-RARE(ours) | 71.73%| 93.88% |779ms|
| SLANet(ours) |76.31%|	95.89%|766ms|

性能指标解释如下：
- Acc: 模型对每张图像里表格结构的识别准确率，错一个token就算错误。
- TEDS: 模型对表格信息还原的准确度，此指标评价内容不仅包含表格结构，还包含表格内的文字内容。
- Speed: 模型在CPU机器上，开启MKL的情况下，单张图片的推理速度。

## 3. 效果演示

![](../docs/imgs/table_ch_result1.jpg)
![](../docs/imgs/table_ch_result2.jpg)
![](../docs/imgs/table_ch_result3.jpg)

## 4. 使用

### 4.1 快速开始

使用如下命令即可快速完成一张表格的识别。
```python
cd PaddleOCR/ppstructure

# 下载模型
mkdir inference && cd inference
# 下载PP-OCRv3文本检测模型并解压
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_det_infer.tar
# 下载PP-OCRv3文本识别模型并解压
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar
# 下载PP-Structurev2表格识别模型并解压
wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar && tar xf ch_ppstructure_mobile_v2.0_SLANet_infer.tar
cd ..
# 执行表格识别
python table/predict_table.py \
    --det_model_dir=inference/ch_PP-OCRv3_det_infer \
    --rec_model_dir=inference/ch_PP-OCRv3_rec_infer  \
    --table_model_dir=inference/ch_ppstructure_mobile_v2.0_SLANet_infer \
    --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
    --table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt \
    --image_dir=docs/table/table.jpg \
    --output=../output/table
```
运行完成后，每张图片的excel表格会保存到output字段指定的目录下，同时在该目录下回生产一个html文件，用于可视化查看单元格坐标和识别的表格。

### 4.2 训练

文本检测模型的训练、评估和推理流程可参考 [detection](../../doc/doc_ch/detection.md)

文本识别模型的训练、评估和推理流程可参考 [recognition](../../doc/doc_ch/recognition.md)

表格识别模型的训练、评估和推理流程可参考 [table_recognition](../../doc/doc_ch/table_recognition.md)

### 4.3 计算TEDS

表格使用 [TEDS(Tree-Edit-Distance-based Similarity)](https://github.com/ibm-aur-nlp/PubTabNet/tree/master/src) 作为模型的评估指标。在进行模型评估之前，需要将pipeline中的三个模型分别导出为inference模型(我们已经提供好)，还需要准备评估的gt， gt示例如下:
```txt
PMC5755158_010_01.png    <html><body><table><thead><tr><td></td><td><b>Weaning</b></td><td><b>Week 15</b></td><td><b>Off-test</b></td></tr></thead><tbody><tr><td>Weaning</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Week 15</td><td>–</td><td>0.17 ± 0.08</td><td>0.16 ± 0.03</td></tr><tr><td>Off-test</td><td>–</td><td>0.80 ± 0.24</td><td>0.19 ± 0.09</td></tr></tbody></table></body></html>
```
gt每一行都由文件名和表格的html字符串组成，文件名和表格的html字符串之间使用`\t`分隔。

也可使用如下命令，由标注文件生成评估的gt文件：
```python
python3 ppstructure/table/convert_label2html.py --ori_gt_path /path/to/your_label_file --save_path /path/to/save_file
```

准备完成后使用如下命令进行评估，评估完成后会输出teds指标。
```python
cd PaddleOCR/ppstructure
python3 table/eval_table.py \
    --det_model_dir=path/to/det_model_dir \
    --rec_model_dir=path/to/rec_model_dir \
    --table_model_dir=path/to/table_model_dir \
    --image_dir=../doc/table/1.png \
    --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt \
    --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt \
    --det_limit_side_len=736 \
    --det_limit_type=min \
    --gt_path=path/to/gt.txt
```
如使用PubLatNet评估数据集，将会输出
```bash
teds: 94.98
```

## 5. Reference
1. https://github.com/ibm-aur-nlp/PubTabNet
2. https://arxiv.org/pdf/1911.10683
