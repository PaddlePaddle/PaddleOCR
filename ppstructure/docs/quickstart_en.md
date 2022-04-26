# PP-Structure Quick Start

- [1. Install package](#1)
- [2. Use](#2)
    - [2.1 Use by command line](#21)
        - [2.1.1 layout analysis + table recognition](#211)
        - [2.1.2 layout analysis](#212)
        - [2.1.3 table recognition](#213)
        - [2.1.4 DocVQA](#214)
    - [2.2 Use by code](#22)
        - [2.2.1 layout analysis + table recognition](#221)
        - [2.2.2 layout analysis](#222)
        - [2.2.3 table recognition](#223)
        - [2.2.4 DocVQA](#224)
    - [2.3 Result description](#23)
        - [2.3.1 layout analysis + table recognition](#231)
        - [2.3.2 DocVQA](#232)
    - [2.4 Parameter Description](#24)


<a name="1"></a>
## 1. Install package

```bash
# Install paddleocr, version 2.5+ is recommended
pip3 install "paddleocr>=2.5"
# Install layoutparser (if you do not use the layout analysis, you can skip it)
pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
# Install the DocVQA dependency package paddlenlp (if you do not use the DocVQA, you can skip it)
pip install paddlenlp

```

<a name="2"></a>
## 2. Use

<a name="21"></a>
### 2.1 Use by command line

<a name="211"></a>
#### 2.1.1 layout analysis + table recognition
```bash
paddleocr --image_dir=PaddleOCR/ppstructure/docs/table/1.png --type=structure
```

<a name="212"></a>
#### 2.1.2 layout analysis
```bash
paddleocr --image_dir=PaddleOCR/ppstructure/docs/table/1.png --type=structure --table=false --ocr=false
```

<a name="213"></a>
#### 2.1.3 table recognition
```bash
paddleocr --image_dir=PaddleOCR/ppstructure/docs/table/table.jpg --type=structure --layout=false
```

<a name="214"></a>
#### 2.1.4 DocVQA

Please refer to: [Documentation Visual Q&A](../vqa/README.md) .

<a name="22"></a>
### 2.2 Use by code

<a name="221"></a>
#### 2.2.1 layout analysis + table recognition

```python
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True)

save_folder = './output'
img_path = 'PaddleOCR/ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'PaddleOCR/doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

<a name="222"></a>
#### 2.2.2 layout analysis

```python
import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(table=False, ocr=False, show_log=True)

save_folder = './output'
img_path = 'PaddleOCR/ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
```

<a name="223"></a>
#### 2.2.3 table recognition

```python
import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(layout=False, show_log=True)

save_folder = './output'
img_path = 'PaddleOCR/ppstructure/docs/table/table.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
```

<a name="224"></a>
#### 2.2.4 DocVQA

Please refer to: [Documentation Visual Q&A](../vqa/README.md) .

<a name="23"></a>
### 2.3 Result description

The return of PP-Structure is a list of dicts, the example is as follows:

<a name="231"></a>
#### 2.3.1 layout analysis + table recognition
```shell
[
  {   'type': 'Text',
      'bbox': [34, 432, 345, 462],
      'res': ([[36.0, 437.0, 341.0, 437.0, 341.0, 446.0, 36.0, 447.0], [41.0, 454.0, 125.0, 453.0, 125.0, 459.0, 41.0, 460.0]],
                [('Tigure-6. The performance of CNN and IPT models using difforen', 0.90060663), ('Tent  ', 0.465441)])
  }
]
```
Each field in dict is described as follows:

| field            | description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|type| Type of image area.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|bbox| The coordinates of the image area in the original image, respectively [upper left corner x, upper left corner y, lower right corner x, lower right corner y].                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|res| OCR or table recognition result of the image area. <br> table: a dict with field descriptions as follows: <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `html`: html str of table.<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; In the code usage mode, set return_ocr_result_in_table=True whrn call can get the detection and recognition results of each text in the table area, corresponding to the following fields: <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `boxes`: text detection boxes.<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `rec_res`: text recognition results.<br> OCR: A tuple containing the detection boxes and recognition results of each single text. |

After the recognition is completed, each image will have a directory with the same name under the directory specified by the `output` field. Each table in the image will be stored as an excel, and the picture area will be cropped and saved. The filename of  excel and picture is their coordinates in the image.
  ```
  /output/table/1/
    └─ res.txt
    └─ [454, 360, 824, 658].xlsx        table recognition result
    └─ [16, 2, 828, 305].jpg            picture in Image
    └─ [17, 361, 404, 711].xlsx        table recognition result
  ```

<a name="232"></a>
#### 2.3.2 DocVQA

Please refer to: [Documentation Visual Q&A](../vqa/README.md) .

<a name="24"></a>
### 2.4 Parameter Description

| field                | description                                                                                                                                                                                                                                                      | default                                                 |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| output               | The save path of result                                                                                                                                                                                                                                          | ./output/table                                          |
| table_max_len        | When the table structure model predicts, the long side of the image                                                                                                                                                                                              | 488                                                     |
| table_model_dir      | the path of table structure model                                                                                                                                                                                                                                | None                                                    |
| table_char_dict_path | the dict path of table structure model                                                                                                                                                                                                                           | ../ppocr/utils/dict/table_structure_dict.txt            |
| layout_path_model    | The model path of the layout analysis model, which can be an online address or a local path. When it is a local path, layout_label_map needs to be set. In command line mode, use --layout_label_map='{0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}' | lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config |
| layout_label_map     | Layout analysis model model label mapping dictionary path                                                                                                                                                                                                        | None                                                    |
| model_name_or_path   | the model path of VQA SER model                                                                                                                                                                                                                                  | None                                                    |
| max_seq_length       | the max token length of VQA SER model                                                                                                                                                                                                                            | 512                                                     |
| label_map_path       | the label path of VQA SER model                                                                                                                                                                                                                                  | ./vqa/labels/labels_ser.txt                             |
| layout               | Whether to perform layout analysis in forward                                                                                                                                                                                                                    | True                                                    |
| table                | Whether to perform table recognition in forward                                                                                                                                                                                                                  | True                                                    |
| ocr                  | Whether to perform ocr for non-table areas in layout analysis. When layout is False, it will be automatically set to False                                                                                                                                                                                                                 | True                                                    |

Most of the parameters are consistent with the PaddleOCR whl package, see [whl package documentation](../../doc/doc_en/whl.md)
