---
comments: true
---

# PP-Structure Quick Start

## 1. Environment Preparation

### 1.1 Install PaddlePaddle

> If you do not have a Python environment, please refer to [Environment Preparation](../ppocr/environment.en.md).

- If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install

  ```bash linenums="1"
  python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```

- If you have no available GPU on your machine, please run the following command to install the CPU version

  ```bash linenums="1"
  python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```

For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.

### 1.2 Install PaddleOCR Whl Package

```bash linenums="1"
# Install paddleocr, version 2.6 is recommended
pip3 install "paddleocr>=2.6.0.3"

# Install the image direction classification dependency package paddleclas (if you do not use the image direction classification, you can skip it)
pip3 install paddleclas>=2.4.3
```

## 2. Quick Use

### 2.1 Use by command line

#### 2.1.1 image orientation + layout analysis + table recognition

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --image_orientation=true
```

#### 2.1.2 layout analysis + table recognition

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure
```

#### 2.1.3 layout analysis

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --table=false --ocr=false
```

#### 2.1.4 table recognition

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/table.jpg --type=structure --layout=false
```

#### 2.1.5 Key Information Extraction

Key information extraction does not currently support use by the whl package. For detailed usage tutorials, please refer to: [inference document](./infer_deploy/python_infer.en.md).

#### 2.1.6 layout recovery(PDF to Word)

Two layout recovery methods are provided, For detailed usage tutorials, please refer to: [Layout Recovery](./model_train/recovery_to_doc.en.md).

- PDF parse
- OCR

Recovery by using PDF parse (only support pdf as input):

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/recovery/UnrealText.pdf --type=structure --recovery=true --use_pdf2docx_api=true
```

Recovery by using OCR：

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --recovery=true --lang='en'
```

#### 2.1.7 layout recovery(PDF to Markdown)

Do not use LaTeXCOR model for formula recognition：

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/recovery/UnrealText.pdf --type=structure --recovery=true --recovery_to_markdown=true --lang='en'
```

Use LaTeXCOR model for formula recognition, where Chinese layout model must be used：

```bash linenums="1"
paddleocr --image_dir=ppstructure/docs/recovery/UnrealText.pdf --type=structure --recovery=true --formula=true --recovery_to_markdown=true --lang='ch'
```

### 2.2 Use by python script

#### 2.2.1 image orientation + layout analysis + table recognition

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

#### 2.2.2 layout analysis + table recognition

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

font_path = 'doc/fonts/simfang.ttf' # font provided in PaddleOCR
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

#### 2.2.3 layout analysis

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

#### 2.2.4 table recognition

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

#### 2.2.5 Key Information Extraction

Key information extraction does not currently support use by the whl package. For detailed usage tutorials, please refer to: [Inference](../infer_deploy/python_infer.en.md).

#### 2.2.6 layout recovery

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

# Chinese image
table_engine = PPStructure(recovery=True)
# English image
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

#### 2.2.7 layout recovery(PDF to Markdown)

```python linenums="1"
import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from paddleocr.ppstructure.recovery.recovery_to_markdown import convert_info_markdown

# Chinese image
table_engine = PPStructure(recovery=True)
# English image
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

### 2.3 Result description

The return of PP-Structure is a list of dicts, the example is as follows:

#### 2.3.1 layout analysis + table recognition

```bash linenums="1"
[
  {   'type': 'Text',
      'bbox': [34, 432, 345, 462],
      'res': ([[36.0, 437.0, 341.0, 437.0, 341.0, 446.0, 36.0, 447.0], [41.0, 454.0, 125.0, 453.0, 125.0, 459.0, 41.0, 460.0]],
                [('Tigure-6. The performance of CNN and IPT models using difforen', 0.90060663), ('Tent  ', 0.465441)])
  }
]
```

Each field in dict is described as follows:

| field | description  |
| --- |---|
|type| Type of image area. |
|bbox| The coordinates of the image area in the original image, respectively [upper left corner x, upper left corner y, lower right corner x, lower right corner y]. |
|res| OCR or table recognition result of the image area. <br> table: a dict with field descriptions as follows: <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `html`: html str of table.<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; In the code usage mode, set return_ocr_result_in_table=True whrn call can get the detection and recognition results of each text in the table area, corresponding to the following fields: <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `boxes`: text detection boxes.<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `rec_res`: text recognition results.<br> OCR: A tuple containing the detection boxes and recognition results of each single text. |

After the recognition is completed, each image will have a directory with the same name under the directory specified by the `output` field. Each table in the image will be stored as an excel, and the picture area will be cropped and saved. The filename of  excel and picture is their coordinates in the image.

  ```text linenums="1"
  /output/table/1/
    └─ res.txt
    └─ [454, 360, 824, 658].xlsx        table recognition result
    └─ [16, 2, 828, 305].jpg            picture in Image
    └─ [17, 361, 404, 711].xlsx        table recognition result
  ```

#### 2.3.2 Key Information Extraction

Please refer to: [Key Information Extraction](../ppocr/model_train/kie.en.md) .

### 2.4 Parameter Description

| field                   | description                                                                                                                | default |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------|---|
| output                  | result save path                                                                                                           | ./output/table |
| table_max_len           | long side of the image resize in table structure model                                                                     | 488 |
| table_model_dir         | Table structure model inference model path                                                                                 | None |
| table_char_dict_path    | The dictionary path of table structure model                                                                               | ../ppocr/utils/dict/table_structure_dict.txt  |
| merge_no_span_structure | In the table recognition model, whether to merge '\<td>' and '\</td>'                                                      | False |
| formula_model_dir       | Formula recognition model inference model path                                                                             | None                                          |
| formula_char_dict_path  | The dictionary path of formula recognition model                                                                           | ../ppocr/utils/dict/latex_ocr_tokenizer.json |
| layout_model_dir        | Layout analysis model inference model path                                                                                 | None |
| layout_dict_path        | The dictionary path of layout analysis model                                                                               | ../ppocr/utils/dict/layout_publaynet_dict.txt |
| layout_score_threshold  | The box threshold path of layout analysis model                                                                            | 0.5|
| layout_nms_threshold    | The nms threshold path of layout analysis model                                                                            | 0.5|
| kie_algorithm           | kie model algorithm                                                                                                        | LayoutXLM|
| ser_model_dir           | Ser model inference model path                                                                                             | None|
| ser_dict_path           | The dictionary path of Ser model                                                                                           | ../train_data/XFUND/class_list_xfun.txt|
| mode                    | structure or kie                                                                                                           | structure   |
| image_orientation       | Whether to perform image orientation classification in forward                                                             | False   |
| layout                  | Whether to perform layout analysis in forward                                                                              | True   |
| table                   | Whether to perform table recognition in forward                                                                            | True   | 
| formula                 | Whether to perform formula recognition in forward                                                                          | False |
| ocr                     | Whether to perform ocr for non-table areas in layout analysis. When layout is False, it will be automatically set to False | True |
| recovery                | Whether to perform layout recovery in forward                                                                              | False |
| recovery_to_markdown    | Whether to convert the layout recovery results into a markdown file                                                        | False |
| save_pdf                | Whether to convert docx to pdf when recovery                                                                               | False |
| structure_version       | Structure version, optional PP-structure and PP-structurev2                                                                | PP-structure |

Most of the parameters are consistent with the PaddleOCR whl package, see [whl package documentation](../ppocr/blog/whl.en.md)

## 3. Summary

Through the content in this section, you can master the use of PP-Structure related functions through PaddleOCR whl package. Please refer to [documentation tutorial](../index.en.md) for more detailed usage tutorials including model training, inference and deployment, etc.
