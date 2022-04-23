# 基于Python预测引擎推理

- [版面分析+表格识别](#1)
- [DocVQA](#2)

<a name="1"></a>
## 1. 版面分析+表格识别

```bash
cd ppstructure

# 下载模型
mkdir inference && cd inference
# 下载PP-OCRv2文本检测模型并解压
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar && tar xf ch_PP-OCRv2_det_slim_quant_infer.tar
# 下载PP-OCRv2文本识别模型并解压
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar && tar xf ch_PP-OCRv2_rec_slim_quant_infer.tar
# 下载超轻量级英文表格预测模型并解压
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar
cd ..

python3 predict_system.py --det_model_dir=inference/ch_PP-OCRv2_det_slim_quant_infer \
                          --rec_model_dir=inference/ch_PP-OCRv2_rec_slim_quant_infer \
                          --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer \
                          --image_dir=../doc/table/1.png \
                          --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
                          --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt \
                          --output=../output/table \
                          --vis_font_path=../doc/fonts/simfang.ttf
```
运行完成后，每张图片会在`output`字段指定的目录下的`talbe`目录下有一个同名目录，图片里的每个表格会存储为一个excel，图片区域会被裁剪之后保存下来，excel文件和图片名名为表格在图片里的坐标。

<a name="2"></a>
## 2. DocVQA

```bash
cd ppstructure

# 下载模型
mkdir inference && cd inference
# 下载SER xfun 模型并解压
wget https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_ser_pretrained.tar && tar xf PP-Layout_v1.0_ser_pretrained.tar
cd ..

python3 predict_system.py --model_name_or_path=vqa/PP-Layout_v1.0_ser_pretrained/ \
                          --mode=vqa \
                          --image_dir=vqa/images/input/zh_val_0.jpg  \
                          --vis_font_path=../doc/fonts/simfang.ttf
```
运行完成后，每张图片会在`output`字段指定的目录下的`vqa`目录下存放可视化之后的图片，图片名和输入图片名一致。