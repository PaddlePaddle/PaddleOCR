---
comments: true
---

# PP-Structure 快速开始

## 1. 准备环境

### 1.1 安装PaddlePaddle
>
> 如果您没有基础的Python运行环境，请参考[运行环境准备](../ppocr/environment.md)。

- 您的机器安装的是CUDA9或CUDA10，请运行以下命令安装

  ```bash linenums="1"
  python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```

- 您的机器是CPU，请运行以下命令安装

  ```bash linenums="1"
  python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

### 1.2 安装PaddleOCR whl包

```bash linenums="1"
# 安装 paddleocr，推荐使用2.6版本
pip3 install "paddleocr>=2.6.0.3"

# 安装 图像方向分类依赖包paddleclas（如不需要图像方向分类功能，可跳过）
pip3 install paddleclas>=2.4.3
```

## 2. 便捷使用

### 2.1 命令行使用

#### 2.1.1 图像方向分类+版面分析+表格识别

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --image_orientation=true
```

#### 2.1.2 版面分析+表格识别

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure
```

#### 2.1.3 版面分析

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --table=false --ocr=false
```

#### 2.1.4 表格识别

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/table.jpg --type=structure --layout=false
```

#### 2.1.5 关键信息抽取

关键信息抽取暂不支持通过whl包调用，详细使用教程请参考：[关键信息抽取教程](../ppocr/model_train/kie.md)。

#### 2.1.6 版面恢复

版面恢复分为2种方法，详细介绍请参考：[版面恢复教程](./model_train/recovery_to_doc.md)：

- PDF解析
- OCR技术

通过PDF解析(只支持pdf格式的输入)：

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/recovery/UnrealText.pdf --type=structure --recovery=true --use_pdf2docx_api=true
```

通过OCR技术：

版面恢复分为2种方法，详细介绍请参考：[版面恢复教程](./model_train/recovery_to_doc.md)：

- PDF解析
- OCR技术

通过PDF解析(只支持pdf格式的输入)：

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/recovery/UnrealText.pdf --type=structure --recovery=true --use_pdf2docx_api=true
```

通过OCR技术：

```bash linenums="1"
# 中文测试图
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --recovery=true
# 英文测试图
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --recovery=true --lang='en'
# pdf测试文件
paddleocr --image_dir=ppstructure/docs/recovery/UnrealText.pdf --type=structure --recovery=true --lang='en'
```

#### 2.1.7 版面恢复+转换为markdown文件

不使用LaTeXOCR模型进行公式识别：

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/recovery/UnrealText.pdf --type=structure --recovery=true --recovery_to_markdown=true --lang='en'
```

使用LaTeXOCR模型进行公式识别，其中必须使用中文layout模型：

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/recovery/UnrealText.pdf --type=structure --recovery=true --formula=true --recovery_to_markdown=true --lang='ch'
```

### 2.2 Python脚本使用

#### 2.2.1 图像方向分类+版面分析+表格识别

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True, image_orientation=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

#### 2.2.2 版面分析+表格识别

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

#### 2.2.3 版面分析

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(table=False, ocr=False, show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
```

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,save_structure_res

ocr_engine = PPStructure(table=False, ocr=True, show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/recovery/UnrealText.pdf'
result = ocr_engine(img_path)
for index, res in enumerate(result):
    save_structure_res(res, save_folder, os.path.basename(img_path).split('.')[0], index)

for res in result:
    for line in res:
        line.pop('img')
        print(line)
```

```python linenums="1"
import os
import cv2
import numpy as np
from paddleocr import PPStructure,save_structure_res
from paddle.utils import try_import
from PIL import Image

ocr_engine = PPStructure(table=False, ocr=True, show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/recovery/UnrealText.pdf'

fitz = try_import("fitz")
imgs = []
with fitz.open(img_path) as pdf:
    for pg in range(0, pdf.page_count):
        page = pdf[pg]
        mat = fitz.Matrix(2, 2)
        pm = page.get_pixmap(matrix=mat, alpha=False)

        # if width or height > 2000 pixels, don't enlarge the image
        if pm.width > 2000 or pm.height > 2000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        imgs.append(img)

for index, img in enumerate(imgs):
    result = ocr_engine(img)
    save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0], index)
    for line in result:
        line.pop('img')
        print(line)
```

#### 2.2.4 表格识别

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(layout=False, show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/table.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
```

#### 2.2.5 关键信息抽取

关键信息抽取暂不支持通过whl包调用，详细使用教程请参考：[inference文档](./infer_deploy/python_infer.md)。

#### 2.2.6 版面恢复

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

# 中文测试图
table_engine = PPStructure(recovery=True)
# 英文测试图
# table_engine = PPStructure(recovery=True, lang='en')

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

h, w, _ = img.shape
res = sorted_layout_boxes(result, w)
convert_info_docx(img, res, save_folder, os.path.basename(img_path).split('.')[0])
```

#### 2.2.7 版面恢复+转换为markdown文件

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from paddleocr.ppstructure.recovery.recovery_to_markdown import convert_info_markdown

# 中文测试图
table_engine = PPStructure(recovery=True)
# 英文测试图
# table_engine = PPStructure(recovery=True, lang='en')

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

h, w, _ = img.shape
res = sorted_layout_boxes(result, w)
convert_info_markdown(res, save_folder, os.path.basename(img_path).split('.')[0])
```

### 2.3 返回结果说明

PP-Structure的返回结果为一个dict组成的list，示例如下：

#### 2.3.1 版面分析+表格识别

```bash linenums="1"
[
  {   'type': 'Text',
      'bbox': [34, 432, 345, 462],
      'res': ([[36.0, 437.0, 341.0, 437.0, 341.0, 446.0, 36.0, 447.0], [41.0, 454.0, 125.0, 453.0, 125.0, 459.0, 41.0, 460.0]],
                [('Tigure-6. The performance of CNN and IPT models using difforen', 0.90060663), ('Tent  ', 0.465441)])
  }
]
```

dict 里各个字段说明如下：

| 字段 | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| type | 图片区域的类型                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| bbox | 图片区域的在原图的坐标，分别[左上角x，左上角y，右下角x，右下角y]                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| res  | 图片区域的OCR或表格识别结果。<br> 表格: 一个dict，字段说明如下<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `html`: 表格的HTML字符串<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 在代码使用模式下，前向传入return_ocr_result_in_table=True可以拿到表格中每个文本的检测识别结果，对应为如下字段: <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `boxes`: 文本检测坐标<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `rec_res`: 文本识别结果。<br> OCR: 一个包含各个单行文字的检测坐标和识别结果的元组 |

运行完成后，每张图片会在`output`字段指定的目录下有一个同名目录，图片里的每个表格会存储为一个excel，图片区域会被裁剪之后保存下来，excel文件和图片名为表格在图片里的坐标。

  ```
  /output/table/1/
    └─ res.txt
    └─ [454, 360, 824, 658].xlsx  表格识别结果
    └─ [16, 2, 828, 305].jpg            被裁剪出的图片区域
    └─ [17, 361, 404, 711].xlsx        表格识别结果
  ```

#### 2.3.2 关键信息抽取

请参考：[关键信息抽取教程](../ppocr/model_train/kie.md)。

### 2.4 参数说明

| 字段                      | 说明                                              | 默认值                                        |
|-------------------------|-------------------------------------------------| ------ |
| output                  | 结果保存地址                                          | ./output/table                                |
| table_max_len           | 表格结构模型预测时，图像的长边resize尺度                         | 488                                           |
| table_model_dir         | 表格结构模型 inference 模型地址                           | None                                          |
| table_char_dict_path    | 表格结构模型所用字典地址                                    | ../ppocr/utils/dict/table_structure_dict.txt  |
| merge_no_span_structure | 表格识别模型中，是否对'\<td>'和'\</td>' 进行合并                | False                                         |
| formula_model_dir       | 公式识别模型 inference 模型地址                           | None                                          |
| formula_char_dict_path  | 公式识别模型所用字典地址                                    | ../ppocr/utils/dict/latex_ocr_tokenizer.json |
| layout_model_dir        | 版面分析模型 inference 模型地址                           | None                                          |
| layout_dict_path        | 版面分析模型字典                                        | ../ppocr/utils/dict/layout_publaynet_dict.txt |
| layout_score_threshold  | 版面分析模型检测框阈值                                     | 0.5                                           |
| layout_nms_threshold    | 版面分析模型nms阈值                                     | 0.5                                           |
| kie_algorithm           | kie模型算法                                         | LayoutXLM                                     |
| ser_model_dir           | ser模型  inference 模型地址                           | None                                          |
| ser_dict_path           | ser模型字典                                         | ../train_data/XFUND/class_list_xfun.txt       |
| mode                    | structure or kie                                | structure                                     |
| image_orientation       | 前向中是否执行图像方向分类                                   | False                                         |
| layout                  | 前向中是否执行版面分析                                     | True                                          |
| table                   | 前向中是否执行表格识别                                     | True                                          |
| formula                 | 前向中是否执行公式识别                                     | False                                         |
| ocr                     | 对于版面分析中的非表格区域，是否执行ocr。当layout为False时会被自动设置为False | True                                          |
| recovery                | 前向中是否执行版面恢复                                     | False                                         |
| recovery_to_markdown    | 是否将版面恢复结果转换为markdown文件                        | False                                         |
| save_pdf                | 版面恢复导出docx文件的同时，是否导出pdf文件                       | False                                         |
| structure_version       | 模型版本，可选 PP-structure和PP-structurev2             | PP-structure                                  |

大部分参数和PaddleOCR whl包保持一致，见 [whl包文档](../ppocr/blog/whl.md)

## 3. 小结

通过本节内容，相信您已经熟练掌握通过PaddleOCR whl包调用PP-Structure相关功能的使用方法，您可以参考[文档教程](../index.md)，获取包括模型训练、推理部署等更详细的使用教程。
