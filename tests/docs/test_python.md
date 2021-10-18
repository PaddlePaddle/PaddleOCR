# Python功能测试

Python功能测试的主程序为`test_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

## 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
|  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
|  DB  | ch_ppocr_mobile_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练: FPGM裁剪、PACT量化 |
|  DB  | ch_ppocr_server_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练: FPGM裁剪、PACT量化 |
| CRNN | ch_ppocr_mobile_v2.0_rec| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练: PACT量化 |
| CRNN | ch_ppocr_server_v2.0_rec| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练: PACT量化 |
|PP-OCR| ch_ppocr_mobile_v2.0| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |
|PP-OCR| ch_ppocr_server_v2.0| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |


- 预测相关：

| 算法名称 | 模型名称 |device | batchsize | mkldnn | cpu多线程 | tensorrt | 离线量化 |
|  ----  |   ----  |  ----  |  ---- |  ---- |  ----  |  ----| --- |
| DB   |ch_ppocr_mobile_v2.0_det| CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
| DB   |ch_ppocr_server_v2.0_det| CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
| CRNN |ch_ppocr_mobile_v2.0_rec| CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
| CRNN |ch_ppocr_server_v2.0_rec| CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
|PP-OCR|ch_ppocr_mobile_v2.0    | CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
|PP-OCR|ch_ppocr_server_v2.0    | CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |



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
先运行`prepare.sh`准备数据和模型，然后运行`test_python.sh`进行测试，最终在```tests/output```目录下生成`python_infer_*.log`格式的日志文件。

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

- 模式5：klquant_infer , 测试离线量化；
```shell
bash tests/test_python.sh tests/configs/ppocr_det_mobile_params.txt  'klquant_infer'
```


## 3. 精度测试

使用compare_results.py脚本比较模型预测的结果是否符合预期，主要步骤包括：
- 提取日志中的预测坐标；
- 从本地文件中提取保存好的坐标结果；
- 比较上述两个结果是否符合精度预期，误差大于设置阈值时会报错。

### 使用方式
运行命令：
```shell
python3.7 tests/compare_results.py --gt_file=./tests/results/python_*.txt  --log_file=./tests/output/python_*.log --atol=1e-3 --rtol=1e-3
```

参数介绍：  
- gt_file： 指向事先保存好的预测结果路径，支持*.txt 结尾，会自动索引*.txt格式的文件，文件默认保存在tests/result/ 文件夹下
- log_file: 指向运行tests/test.sh 脚本的infer模式保存的预测日志，预测日志中打印的有预测结果，比如：文本框，预测文本，类别等等，同样支持infer_*.log格式传入
- atol: 设置的绝对误差
- rtol: 设置的相对误差

### 运行结果

正常运行效果如下图：
<img src="compare_right.png" width="1000">

出现不一致结果时的运行输出：
<img src="compare_wrong.png" width="1000">
