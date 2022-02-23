<!---
# Copyright (c) 2016-2022 BigoneLab Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may
# not use this file except in compliance with the License. A copy of the
# License is located at
#   
#   https://github.com/XinyiXiang/PaddleOCR/blob/release/2.4/LICENSE
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
--->
# 版面分析
- **版面分析 `.png` 格式的文件**

    其中 `image_dir` 为图片路径

```
paddleocr --image_dir=./imgs/ --type=structure > text_output_with_layout_pg1.txt
```
- **返回结果说明**

  PP-Structure的返回结果为一个dict组成的list，示例如下

  ```shell
  [{   'type': 'Text',
        'bbox': [34, 432, 345, 462],
        'res': ([[36.0, 437.0, 341.0, 437.0, 341.0, 446.0, 36.0, 447.0], [41.0, 454.0, 125.0, 453.0, 125.0, 459.0, 41.0, 460.0]],
                  [('Tigure-6. The performance of CNN and IPT models using difforen', 0.90060663), ('Tent  ', 0.465441)])
    }
  ]
  ```

  其中各个字段说明如下

  | 字段 | 说明                                                         |
  | ---- | ------------------------------------------------------------ |
  | type | 图片区域的类型                                               |
  | bbox | 图片区域的在原图的坐标，分别[左上角x，左上角y，右下角x，右下角y] |
  | res  | 图片区域的OCR或表格识别结果。<br>表格: 表格的HTML字符串; <br>OCR: 一个包含各个单行文字的检测坐标和识别结果的元组 |


- **版面分析测试文件类型与测试数据**
    #### Text only `BERN-18.png` 
    #### Text and table combined `BERN-06.png`

    #### Non-text only `BERN-15.png`


# 无版面分析
#### 双语模型导出未处理文件
``` 
paddleocr --image_dir $1 --lang=en > raw_output.txt 
```
#### 过滤未处理结果，提取仅文字识别输出 
```echo "Filtering text output"
grep -oi "'.*'" raw_output.txt > text_output.txt  
```

