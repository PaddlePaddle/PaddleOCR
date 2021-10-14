# Python功能测试

Python功能测试的主程序为`test_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

## 测试结论汇总

训练相关：方式包括：
【单机单卡、单机多卡、多机多卡】*【正常训练、混合精度训练】*【裁剪、在线量化、蒸馏】

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩 |
|  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
|  DB  | ch_ppocr_mobile_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 裁剪、在线/离线量化、蒸馏 |
|  DB  | ch_ppocr_server_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 裁剪、在线/离线量化、蒸馏 |
| CRNN | ch_ppocr_mobile_v2.0_rec| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 裁剪、在线/离线量化、蒸馏 |
| CRNN | ch_ppocr_server_v2.0_rec| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 裁剪、在线/离线量化、蒸馏 |
|PP-OCR| ch_ppocr_server_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 裁剪、在线/离线量化、蒸馏 |
|PP-OCR| ch_ppocr_server_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 裁剪、在线/离线量化、蒸馏 |

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 |
|  :----  |   :----  |    :----  |  :----   |  :----   |
|  DB  | ch_ppocr_mobile_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 
|  DB  | ch_ppocr_server_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 
|  DB  | ch_ppocr_mobile_v2.0_det_pact| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 
|  DB  | ch_ppocr_mobile_v2.0_det_fpgm| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 
| CRNN | ch_ppocr_mobile_v2.0_rec| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 |
| CRNN | ch_ppocr_server_v2.0_rec| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 |
| CRNN | ch_ppocr_mobile_v2.0_rec_pact| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 |
| CRNN | ch_ppocr_mobile_v2.0_rec_fpgm| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 |
|PP-OCR| ch_ppocr_server_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 |
|PP-OCR| ch_ppocr_server_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 |


预测相关：

| 模型名称 | 算法名称 | 模型类型 |device | batchsize=1/6 | mkldnn | tensorrt | cpu多线程 | 
|  ----  |   ----  |  ----  |  ---- |  ---- |  ----  |  ----| --- | 
|ch_ppocr_mobile_v2.0_det| DB     | 检测   | CPU/GPU | 支持 | 支持 | fp32/fp16/int8 | 支持 |  
|ch_ppocr_mobile_v2.0_rec| CRNN   | 识别   | CPU/GPU | 支持 | 支持 | fp32/fp16/int8 | 支持 |  
|ch_ppocr_server_v2.0_det| DB     | 检测   | CPU/GPU | 支持 | 支持 | fp32/fp16/int8 | 支持 |  
|ch_ppocr_server_v2.0_rec| CRNN   | 识别   | CPU/GPU | 支持 | 支持 | fp32/fp16/int8 | 支持 |  



## 1. 安装依赖
- 安装PaddlePaddle >= 2.0
- 安装PaddleOCR依赖
    ```
    pip3 install  -r ../requirements.txt
    ```
- 安装autolog（规范化日志输出工具）
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip3 install -r requirements.txt
    python3 setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```


## 2. 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_python.sh`进行测试，最终在```tests/output```目录下生成.log后缀的日志文件。

test_python.sh包含四种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
bash tests/prepare.sh ./tests/configs/ppocr_det_mobile_params.txt 'lite_train_infer'
bash tests/test_python.sh ./tests/configs/ppocr_det_mobile_params.txt 'lite_train_infer'
```  

- 模式2：whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
```shell
bash tests/prepare.sh ./tests/configs/ppocr_det_mobile_params.txt 'whole_infer'
bash tests/test_python.sh ./tests/configs/ppocr_det_mobile_params.txt 'whole_infer'
```  

- 模式3：infer 不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash tests/prepare.sh ./tests/configs/ppocr_det_mobile_params.txt 'infer'
# 用法1:
bash tests/test_python.sh ./tests/configs/ppocr_det_mobile_params.txt 'infer'
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash tests/test_python.sh ./tests/configs/ppocr_det_mobile_params.txt 'infer' '1'
```  

- 模式4：whole_train_infer , CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；
```shell
bash tests/prepare.sh ./tests/configs/ppocr_det_mobile_params.txt 'whole_train_infer'
bash tests/test.sh ./tests/configs/ppocr_det_mobile_params.txt 'whole_train_infer'
```  


## 3. 精度测试