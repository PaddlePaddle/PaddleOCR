English | [简体中文](readme.md)

- [Service deployment based on PaddleHub Serving](#service-deployment-based-on-paddlehub-serving)
  - [1. Update](#1-update)
  - [2. Quick start service](#2-quick-start-service)
    - [2.1 Prepare the environment](#21-prepare-the-environment)
    - [2.2 Download inference model](#22-download-inference-model)
    - [2.3 Install Service Module](#23-install-service-module)
    - [2.4 Start service](#24-start-service)
      - [2.4.1 Start with command line parameters (CPU only)](#241-start-with-command-line-parameters-cpu-only)
      - [2.4.2 Start with configuration file（CPU、GPU）](#242-start-with-configuration-filecpugpu)
  - [3. Send prediction requests](#3-send-prediction-requests)
  - [4. Returned result format](#4-returned-result-format)
  - [5. User defined service module modification](#5-user-defined-service-module-modification)


PaddleOCR provides 2 service deployment methods:
- Based on **PaddleHub Serving**: Code path is "`./deploy/hubserving`". Please follow this tutorial.
- Based on **PaddleServing**: Code path is "`./deploy/pdserving`". Please refer to the [tutorial](../../deploy/pdserving/README.md) for usage.

# Service deployment based on PaddleHub Serving  

The hubserving service deployment directory includes six service packages: text detection, text angle class, text recognition, text detection+text angle class+text recognition three-stage series connection, table recognition and PP-Structure. Please select the corresponding service package to install and start service according to your needs. The directory is as follows:  
```
deploy/hubserving/
  └─  ocr_det     text detection module service package
  └─  ocr_cls     text angle class module service package
  └─  ocr_rec     text recognition module service package
  └─  ocr_system  text detection+text angle class+text recognition three-stage series connection service package
  └─  structure_table  table recognition service package
  └─  structure_system  PP-Structure service package
```

Each service pack contains 3 files. Take the 2-stage series connection service package as an example, the directory is as follows:  
```
deploy/hubserving/ocr_system/
  └─  __init__.py    Empty file, required
  └─  config.json    Configuration file, optional, passed in as a parameter when using configuration to start the service
  └─  module.py      Main module file, required, contains the complete logic of the service
  └─  params.py      Parameter file, required, including parameters such as model path, pre- and post-processing parameters
```
## 1. Update

* 2022.03.30 add PP-Structure and table recognition services。


## 2. Quick start service
The following steps take the 2-stage series service as an example. If only the detection service or recognition service is needed, replace the corresponding file path.

### 2.1 Prepare the environment
```shell
# Install paddlehub  
# python>3.6.2 is required bt paddlehub
pip3 install paddlehub==2.1.0 --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.2 Download inference model
Before installing the service module, you need to prepare the inference model and put it in the correct path. By default, the PP-OCRv2 models are used, and the default model path is:  
```
text detection model: ./inference/ch_PP-OCRv2_det_infer/
text recognition model: ./inference/ch_PP-OCRv2_rec_infer/
text angle classifier: ./inference/ch_ppocr_mobile_v2.0_cls_infer/
tanle recognition: ./inference/en_ppocr_mobile_v2.0_table_structure_infer/
```  

**The model path can be found and modified in `params.py`.** More models provided by PaddleOCR can be obtained from the [model library](../../doc/doc_en/models_list_en.md). You can also use models trained by yourself.

### 2.3 Install Service Module
PaddleOCR provides 5 kinds of service modules, install the required modules according to your needs.

* On Linux platform, the examples are as follows.
```shell
# Install the text detection service module:
hub install deploy/hubserving/ocr_det/

# Or, install the text angle class service module:
hub install deploy/hubserving/ocr_cls/

# Or, install the text recognition service module:
hub install deploy/hubserving/ocr_rec/

# Or, install the 2-stage series service module:
hub install deploy/hubserving/ocr_system/

# Or install table recognition service module
hub install deploy/hubserving/structure_table/

# Or install PP-Structure service module
hub install deploy/hubserving/structure_system/
```

* On Windows platform, the examples are as follows.
```shell
# Install the detection service module:
hub install deploy\hubserving\ocr_det\

# Or, install the angle class service module:
hub install deploy\hubserving\ocr_cls\

# Or, install the recognition service module:
hub install deploy\hubserving\ocr_rec\

# Or, install the 2-stage series service module:
hub install deploy\hubserving\ocr_system\

# Or install table recognition service module
hub install deploy/hubserving/structure_table/

# Or install PP-Structure service module
hub install deploy\hubserving\structure_system\
```

### 2.4 Start service
#### 2.4.1 Start with command line parameters (CPU only)

**start command：**  
```shell
$ hub serving start --modules [Module1==Version1, Module2==Version2, ...] \
                    --port XXXX \
                    --use_multiprocess \
                    --workers \
```  
**parameters：**  

|parameters|usage|  
|---|---|  
|--modules/-m|PaddleHub Serving pre-installed model, listed in the form of multiple Module==Version key-value pairs<br>*`When Version is not specified, the latest version is selected by default`*|
|--port/-p|Service port, default is 8866|  
|--use_multiprocess|Enable concurrent mode, the default is single-process mode, this mode is recommended for multi-core CPU machines<br>*`Windows operating system only supports single-process mode`*|
|--workers|The number of concurrent tasks specified in concurrent mode, the default is `2*cpu_count-1`, where `cpu_count` is the number of CPU cores|  

For example, start the 2-stage series service:  
```shell
hub serving start -m ocr_system
```  

This completes the deployment of a service API, using the default port number 8866.  

#### 2.4.2 Start with configuration file（CPU、GPU）
**start command：**  
```shell
hub serving start --config/-c config.json
```  
Wherein, the format of `config.json` is as follows:
```python
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
- The configurable parameters in `init_args` are consistent with the `_initialize` function interface in `module.py`. Among them, **when `use_gpu` is `true`, it means that the GPU is used to start the service**.
- The configurable parameters in `predict_args` are consistent with the `predict` function interface in `module.py`.

**Note:**  
- When using the configuration file to start the service, other parameters will be ignored.
- If you use GPU prediction (that is, `use_gpu` is set to `true`), you need to set the environment variable CUDA_VISIBLE_DEVICES before starting the service, such as: ```export CUDA_VISIBLE_DEVICES=0```, otherwise you do not need to set it.
- **`use_gpu` and `use_multiprocess` cannot be `true` at the same time.**  

For example, use GPU card No. 3 to start the 2-stage series service:
```shell
export CUDA_VISIBLE_DEVICES=3
hub serving start -c deploy/hubserving/ocr_system/config.json
```  

## 3. Send prediction requests
After the service starts, you can use the following command to send a prediction request to obtain the prediction result:  
```shell
python tools/test_hubserving.py server_url image_path
```  

Two parameters need to be passed to the script:
- **server_url**：service address，format of which is
`http://[ip_address]:[port]/predict/[module_name]`  
For example, if using the configuration file to start the text angle classification, text detection, text recognition, detection+classification+recognition 3 stages, table recognition and PP-Structure service, then the `server_url` to send the request will be:

`http://127.0.0.1:8865/predict/ocr_det`  
`http://127.0.0.1:8866/predict/ocr_cls`  
`http://127.0.0.1:8867/predict/ocr_rec`  
`http://127.0.0.1:8868/predict/ocr_system`  
`http://127.0.0.1:8869/predict/structure_table`
`http://127.0.0.1:8870/predict/structure_system`  
- **image_dir**：Test image path, can be a single image path or an image directory path
- **visualize**：Whether to visualize the results, the default value is False
- **output**：The floder to save Visualization result, default value is `./hubserving_result`

**Eg.**
```shell
python tools/test_hubserving.py --server_url=http://127.0.0.1:8868/predict/ocr_system --image_dir./doc/imgs/ --visualize=false`
```

## 4. Returned result format
The returned result is a list. Each item in the list is a dict. The dict may contain three fields. The information is as follows:

|field name|data type|description|
|----|----|----|
|angle|str|angle|
|text|str|text content|
|confidence|float|text recognition confidence|
|text_region|list|text location coordinates|
|html|str|table html str|
|regions|list|The result of layout analysis + table recognition + OCR, each item is a list, including `bbox` indicating area coordinates, `type` of area type and `res` of area results|

The fields returned by different modules are different. For example, the results returned by the text recognition service module do not contain `text_region`. The details are as follows:

| field name/module name | ocr_det | ocr_cls | ocr_rec | ocr_system | structure_table | structure_system |
|  ---  |  ---  |  ---  |  ---  |  ---  | ---  |---  |
|angle| | ✔ | | ✔ | ||
|text| | |✔|✔| | ✔ |
|confidence| |✔ |✔| | | ✔|
|text_region| ✔| | |✔ | | ✔|
|html| | | | |✔ |✔|
|regions| | | | |✔ |✔ |

**Note：** If you need to add, delete or modify the returned fields, you can modify the file `module.py` of the corresponding module. For the complete process, refer to the user-defined modification service module in the next section.

## 5. User defined service module modification
If you need to modify the service logic, the following steps are generally required (take the modification of `ocr_system` for example):

- 1. Stop service
```shell
hub serving stop --port/-p XXXX
```
- 2. Modify the code in the corresponding files, like `module.py` and `params.py`, according to the actual needs.  
For example, if you need to replace the model used by the deployed service, you need to modify model path parameters `det_model_dir` and `rec_model_dir` in `params.py`. If you want to turn off the text direction classifier, set the parameter `use_angle_cls` to `False`. Of course, other related parameters may need to be modified at the same time. Please modify and debug according to the actual situation. It is suggested to run `module.py` directly for debugging after modification before starting the service test.  
- 3. Uninstall old service module
```shell
hub uninstall ocr_system
```
- 4. Install modified service module
```shell
hub install deploy/hubserving/ocr_system/
```
- 5. Restart service
```shell
hub serving start -m ocr_system
```
