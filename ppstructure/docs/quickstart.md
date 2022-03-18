# PP-Structure 快速开始

- [PP-Structure 快速开始](#pp-structure-快速开始)
  - [1. 安装依赖包](#1-安装依赖包)
  - [2. 便捷使用](#2-便捷使用)
    - [2.1 命令行使用](#21-命令行使用)
    - [2.2 Python脚本使用](#22-python脚本使用)
    - [2.3 返回结果说明](#23-返回结果说明)
    - [2.4 参数说明](#24-参数说明)
  - [3. Python脚本使用](#3-python脚本使用)

## 1. 安装依赖包

```bash
pip install "paddleocr>=2.3.0.2" # 推荐使用2.3.0.2+版本
pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl

# 安装 PaddleNLP
git clone https://github.com/PaddlePaddle/PaddleNLP -b develop
cd PaddleNLP
pip3 install -e .

```

## 2. 便捷使用

### 2.1 命令行使用

* 版面分析+表格识别
```bash
paddleocr --image_dir=../doc/table/1.png --type=structure
```

* VQA

请参考：[文档视觉问答](../vqa/README.md)。

### 2.2 Python脚本使用

* 版面分析+表格识别
```python
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True)

save_folder = './output/table'
img_path = '../doc/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = '../doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

* VQA

请参考：[文档视觉问答](../vqa/README.md)。

### 2.3 返回结果说明
PP-Structure的返回结果为一个dict组成的list，示例如下

* 版面分析+表格识别
```shell
[
  {   'type': 'Text',
      'bbox': [34, 432, 345, 462],
      'res': ([[36.0, 437.0, 341.0, 437.0, 341.0, 446.0, 36.0, 447.0], [41.0, 454.0, 125.0, 453.0, 125.0, 459.0, 41.0, 460.0]],
                [('Tigure-6. The performance of CNN and IPT models using difforen', 0.90060663), ('Tent  ', 0.465441)])
  }
]
```
dict 里各个字段说明如下

| 字段            | 说明           |
| --------------- | -------------|
|type|图片区域的类型|
|bbox|图片区域的在原图的坐标，分别[左上角x，左上角y，右下角x，右下角y]|
|res|图片区域的OCR或表格识别结果。<br> 表格: 表格的HTML字符串; <br> OCR: 一个包含各个单行文字的检测坐标和识别结果的元组|

* VQA

请参考：[文档视觉问答](../vqa/README.md)。

### 2.4 参数说明

| 字段            | 说明                                     | 默认值                                      |
| --------------- | ---------------------------------------- | ------------------------------------------- |
| output          | excel和识别结果保存的地址                | ./output/table                              |
| table_max_len   | 表格结构模型预测时，图像的长边resize尺度 | 488                                         |
| table_model_dir | 表格结构模型 inference 模型地址          | None                                        |
| table_char_dict_path | 表格结构模型所用字典地址                 | ../ppocr/utils/dict/table_structure_dict.txt |
| layout_path_model | 版面分析模型模型地址，可以为在线地址或者本地地址，当为本地地址时，需要指定 layout_label_map, 命令行模式下可通过--layout_label_map='{0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}' 指定              | lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config |
| layout_label_map | 版面分析模型模型label映射字典                 | None |
| model_name_or_path | VQA SER模型地址                | None |
| max_seq_length | VQA SER模型最大支持token长度              | 512 |
| label_map_path | VQA SER 标签文件地址              | ./vqa/labels/labels_ser.txt |
| mode | pipeline预测模式，structure: 版面分析+表格识别; VQA: SER文档信息抽取              | structure |

大部分参数和PaddleOCR whl包保持一致，见 [whl包文档](../../doc/doc_ch/whl.md)

运行完成后，每张图片会在`output`字段指定的目录下有一个同名目录，图片里的每个表格会存储为一个excel，图片区域会被裁剪之后保存下来，excel文件和图片名名为表格在图片里的坐标。

## 3. Python脚本使用

* 版面分析+表格识别

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

* VQA

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
