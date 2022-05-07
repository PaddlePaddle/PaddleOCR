[English](README.md) | 简体中文

# 版面恢复使用说明

- [1. 简介](#1)
- [2. 使用](#2)


<a name="1"></a>

## 1.  简介

版面恢复就是在OCR识别后，内容仍然像原文档图片那样排列着，段落不变、顺序不变的输出到word文档中等。

版面恢复结合了[版面分析](../layout/README_ch.md)、[表格识别](../table/README_ch.md)技术，从而更好地恢复图片、表格、标题等内容，下图展示了版面恢复的结果：

<div align="center">
<img src="../docs/table/recovery.jpg"  width = "700" />
</div>

<a name="2"></a>

## 2. 使用

恢复给定文档的版面：

```python
cd PaddleOCR/ppstructure

# 下载模型
mkdir inference && cd inference
# 下载超英文轻量级PP-OCRv3模型的检测模型并解压
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_det_infer.tar
# 下载英文轻量级PP-OCRv3模型的识别模型并解压
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar && tar xf  ch_PP-OCRv3_rec_infer.tar
# 下载超轻量级英文表格英寸模型并解压
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar
cd ..
# 执行预测
python3 predict_system.py --det_model_dir=inference/en_PP-OCRv3_det_infer --rec_model_dir=inference/en_PP-OCRv3_rec_infer --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer --rec_char_dict_path=../ppocr/utils/en_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --output ./output/table --rec_image_shape=3,48,320 --vis_font_path=../doc/fonts/simfang.ttf --recovery=True --image_dir=./docs/table/1.png
```

运行完成后，每张图片的docx文档会保存到output字段指定的目录下

