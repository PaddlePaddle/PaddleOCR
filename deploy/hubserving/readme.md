[English](readme_en.md) | 简体中文

- [基于PaddleHub Serving的服务部署](#基于paddlehub-serving的服务部署)
  - [1. 近期更新](#1-近期更新)
  - [2. 快速启动服务](#2-快速启动服务)
    - [2.1 安装PaddleHub](#21-安装PaddleHub)
    - [2.2 下载推理模型](#22-下载推理模型)
    - [2.3 安装服务模块](#23-安装服务模块)
    - [2.4 启动服务](#24-启动服务)
      - [2.4.1. 命令行命令启动（仅支持CPU）](#241-命令行命令启动仅支持cpu)
      - [2.4.2 配置文件启动（支持CPU、GPU）](#242-配置文件启动支持cpugpu)
  - [3. 发送预测请求](#3-发送预测请求)
  - [4. 返回结果格式说明](#4-返回结果格式说明)
  - [5. 自定义修改服务模块](#5-自定义修改服务模块)


PaddleOCR提供2种服务部署方式：
- 基于PaddleHub Serving的部署：代码路径为`./deploy/hubserving`，按照本教程使用；
- 基于PaddleServing的部署：代码路径为`./deploy/pdserving`，使用方法参考[文档](../../deploy/pdserving/README_CN.md)。

# 基于PaddleHub Serving的服务部署

hubserving服务部署目录下包括文本检测、文本方向分类，文本识别、文本检测+文本方向分类+文本识别3阶段串联，版面分析、表格识别和PP-Structure七种服务包，请根据需求选择相应的服务包进行安装和启动。目录结构如下：
```
deploy/hubserving/
  └─  ocr_cls     文本方向分类模块服务包
  └─  ocr_det     文本检测模块服务包
  └─  ocr_rec     文本识别模块服务包
  └─  ocr_system  文本检测+文本方向分类+文本识别串联服务包
  └─  structure_layout  版面分析服务包
  └─  structure_table  表格识别服务包
  └─  structure_system  PP-Structure服务包
  └─  kie_ser  关键信息抽取-SER服务包
  └─  kie_ser_re  关键信息抽取-SER+RE服务包
```

每个服务包下包含3个文件。以2阶段串联服务包为例，目录如下：
```
deploy/hubserving/ocr_system/
  └─  __init__.py    空文件，必选
  └─  config.json    配置文件，可选，使用配置启动服务时作为参数传入
  └─  module.py      主模块，必选，包含服务的完整逻辑
  └─  params.py      参数文件，必选，包含模型路径、前后处理参数等参数
```
## 1. 近期更新

* 2022.10.09 新增关键信息抽取服务。
* 2022.08.23 新增版面分析服务。
* 2022.05.05 新增PP-OCRv3检测和识别模型。
* 2022.03.30 新增PP-Structure和表格识别两种服务。

## 2. 快速启动服务
以下步骤以检测+识别2阶段串联服务为例，如果只需要检测服务或识别服务，替换相应文件路径即可。
### 2.1 安装PaddleHub
paddlehub 需要 python>3.6.2
```bash
pip3 install paddlehub==2.1.0 --upgrade -i https://mirror.baidu.com/pypi/simple
```

### 2.2 下载推理模型
安装服务模块前，需要准备推理模型并放到正确路径。默认使用的是PP-OCRv3模型，默认模型路径为：
| 模型 | 路径 |
| ------- | - |
| 检测模型 | `./inference/ch_PP-OCRv3_det_infer/` |
| 识别模型 | `./inference/ch_PP-OCRv3_rec_infer/` |
| 方向分类器 | `./inference/ch_ppocr_mobile_v2.0_cls_infer/` |
| 版面分析模型 | `./inference/picodet_lcnet_x1_0_fgd_layout_infer/` |
| 表格结构识别模型 | `./inference/ch_ppstructure_mobile_v2.0_SLANet_infer/` |
| 关键信息抽取SER模型 | `./inference/ser_vi_layoutxlm_xfund_infer/` |
| 关键信息抽取RE模型 | `./inference/re_vi_layoutxlm_xfund_infer/` |

**模型路径可在`params.py`中查看和修改。**

更多模型可以从PaddleOCR提供的模型库[PP-OCR](../../doc/doc_ch/models_list.md)和[PP-Structure](../../ppstructure/docs/models_list.md)下载，也可以替换成自己训练转换好的模型。

### 2.3 安装服务模块
PaddleOCR提供5种服务模块，根据需要安装所需模块。

在Linux环境（Windows环境请将`/`替换为`\`）下，安装模块命令如下表：
| 服务模块 | 命令 |
| ------- | - |
| 检测 | `hub install deploy/hubserving/ocr_det` |
| 分类 | `hub install deploy/hubserving/ocr_cls` |
| 识别 | `hub install deploy/hubserving/ocr_rec` |
| 检测+识别串联 | `hub install deploy/hubserving/ocr_system` |
| 表格识别 | `hub install deploy/hubserving/structure_table` |
| PP-Structure | `hub install deploy/hubserving/structure_system` |
| 版面分析 | `hub install deploy/hubserving/structure_layout` |
| 关键信息抽取SER | `hub install deploy/hubserving/kie_ser` |
| 关键信息抽取SER+RE | `hub install deploy/hubserving/kie_ser_re` |

### 2.4 启动服务
#### 2.4.1. 命令行命令启动（仅支持CPU）
**启动命令：**
```bash
hub serving start --modules Module1==Version1, Module2==Version2, ... \
                  --port 8866 \
                  --use_multiprocess \
                  --workers \
```

**参数：**
|参数|用途|
|---|---|
|`--modules`/`-m`|PaddleHub Serving预安装模型，以多个Module==Version键值对的形式列出<br>**当不指定Version时，默认选择最新版本**|
|`--port`/`-p`|服务端口，默认为8866|
|`--use_multiprocess`|是否启用并发方式，默认为单进程方式，推荐多核CPU机器使用此方式<br>**Windows操作系统只支持单进程方式**|
|`--workers`|在并发方式下指定的并发任务数，默认为`2*cpu_count-1`，其中`cpu_count`为CPU核数|

如启动串联服务：
```bash
hub serving start -m ocr_system
```

这样就完成了一个服务化API的部署，使用默认端口号8866。

#### 2.4.2 配置文件启动（支持CPU、GPU）
**启动命令：**
```bash
hub serving start -c config.json
```

其中，`config.json`格式如下：
```json
{
    "modules_info": {
        "ocr_system": {
            "init_args": {
                "version": "1.0.0",
                "use_gpu": true
            },
            "predict_args": {
            }
        }
    },
    "port": 8868,
    "use_multiprocess": false,
    "workers": 2
}
```

- `init_args`中的可配参数与`module.py`中的`_initialize`函数接口一致。

   **当`use_gpu`为`true`时，表示使用GPU启动服务。**
- `predict_args`中的可配参数与`module.py`中的`predict`函数接口一致。

**注意：**
- 使用配置文件启动服务时，其他参数会被忽略。
- 如果使用GPU预测(即，`use_gpu`置为`true`)，则需要在启动服务之前，设置CUDA_VISIBLE_DEVICES环境变量，如：
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  ```
- **`use_gpu`不可与`use_multiprocess`同时为`true`**。

如，使用GPU 3号卡启动串联服务：
```bash
export CUDA_VISIBLE_DEVICES=3
hub serving start -c deploy/hubserving/ocr_system/config.json
```

## 3. 发送预测请求
配置好服务端，可使用以下命令发送预测请求，获取预测结果：
```bash
python tools/test_hubserving.py --server_url=server_url --image_dir=image_path
```

需要给脚本传递2个参数：
- `server_url`：服务地址，格式为`http://[ip_address]:[port]/predict/[module_name]`

   例如，如果使用配置文件启动分类，检测、识别，检测+分类+识别3阶段，表格识别和PP-Structure服务

   并为每个服务修改了port，那么发送请求的url将分别是：
   ```
   http://127.0.0.1:8865/predict/ocr_det
   http://127.0.0.1:8866/predict/ocr_cls
   http://127.0.0.1:8867/predict/ocr_rec
   http://127.0.0.1:8868/predict/ocr_system
   http://127.0.0.1:8869/predict/structure_table
   http://127.0.0.1:8870/predict/structure_system
   http://127.0.0.1:8870/predict/structure_layout
   http://127.0.0.1:8871/predict/kie_ser
   http://127.0.0.1:8872/predict/kie_ser_re
   ```
- `image_dir`：测试图像路径，可以是单张图片路径，也可以是图像集合目录路径
- `visualize`：是否可视化结果，默认为False
- `output`：可视化结果保存路径，默认为`./hubserving_result`

访问示例：
```bash
python tools/test_hubserving.py --server_url=http://127.0.0.1:8868/predict/ocr_system --image_dir=./doc/imgs/ --visualize=false
```

## 4. 返回结果格式说明
返回结果为列表（list），列表中的每一项为词典（dict），词典一共可能包含3种字段，信息如下：
|字段名称|数据类型|意义|
|---|---|---|
|angle|str|文本角度|
|text|str|文本内容|
|confidence|float| 文本识别置信度或文本角度分类置信度|
|text_region|list|文本位置坐标|
|html|str|表格的html字符串|
|regions|list|版面分析+表格识别+OCR的结果，每一项为一个list<br>包含表示区域坐标的`bbox`，区域类型的`type`和区域结果的`res`三个字段|
|layout|list|版面分析的结果，每一项一个dict，包含版面区域坐标的`bbox`，区域类型的`label`|

不同模块返回的字段不同，如，文本识别服务模块返回结果不含`text_region`字段，具体信息如下：
|字段名/模块名     |ocr_det |ocr_cls |ocr_rec |ocr_system |structure_table |structure_system |structure_layout |kie_ser |kie_re |
|---             |---     |---     |---     |---        |---             |---              |---              |---     |---    |
|angle           |        |✔       |        |✔          |                |                 |                 |
|text            |        |        |✔       |✔          |                |✔                |                 |✔       |✔      |
|confidence      |        |✔       |✔       |✔          |                |✔                |                 |✔       |✔      |
|text_region     |✔       |        |        |✔          |                |✔                |                 |✔       |✔      |
|html            |        |        |        |           |✔               |✔                |                 |        |       |
|regions         |        |        |        |           |✔               |✔                |                 |        |       |
|layout          |        |        |        |           |                |                 |✔                |        |       |
|ser_res         |        |        |        |           |                |                 |                 |✔       |       |
|re_res          |        |        |        |           |                |                 |                 |        |✔      |

**说明：** 如果需要增加、删除、修改返回字段，可在相应模块的`module.py`文件中进行修改，完整流程参考下一节自定义修改服务模块。

## 5. 自定义修改服务模块
如果需要修改服务逻辑，一般需要操作以下步骤（以修改`deploy/hubserving/ocr_system`为例）：

1. 停止服务：
   ```bash
   hub serving stop --port/-p XXXX
   ```
2. 到`deploy/hubserving/ocr_system`下的`module.py`和`params.py`等文件中根据实际需求修改代码。

   例如，如果需要替换部署服务所用模型，则需要到`params.py`中修改模型路径参数`det_model_dir`和`rec_model_dir`，如果需要关闭文本方向分类器，则将参数`use_angle_cls`置为`False`

   当然，同时可能还需要修改其他相关参数，请根据实际情况修改调试。

   **强烈建议修改后先直接运行`module.py`调试，能正确运行预测后再启动服务测试。**

   **注意：** PPOCR-v3识别模型使用的图片输入shape为`3,48,320`,因此需要修改`params.py`中的`cfg.rec_image_shape = "3, 48, 320"`，如果不使用PPOCR-v3识别模型，则无需修改该参数。
3. （可选）如果想要重命名模块需要更改`module.py`文件中的以下行：
   - [`from deploy.hubserving.ocr_system.params import read_params`中的`ocr_system`](https://github.com/PaddlePaddle/PaddleOCR/blob/a923f35de57b5e378f8dd16e54d0a3e4f51267fd/deploy/hubserving/ocr_system/module.py#L35)
   - [`name="ocr_system",`中的`ocr_system`](https://github.com/PaddlePaddle/PaddleOCR/blob/a923f35de57b5e378f8dd16e54d0a3e4f51267fd/deploy/hubserving/ocr_system/module.py#L39)
4. （可选）可能需要删除`__pycache__`目录以强制刷新CPython缓存：
   ```bash
   find deploy/hubserving/ocr_system -name '__pycache__' -exec rm -r {} \;
   ```
5. 安装修改后的新服务包：
   ```bash
   hub install deploy/hubserving/ocr_system
   ```
6. 重新启动服务：
   ```bash
   hub serving start -m ocr_system
   ```
