# Mac端基础训练预测功能测试

Mac端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型CPU训练，包括裁剪、PACT在线量化、蒸馏训练，以及评估、CPU推理等基本功能。

注：Mac端测试用法同linux端测试方法类似，但是无需测试需要在GPU上运行的测试。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡（CPU） | 单机多卡 | 多机多卡 | 模型压缩（CPU） |
|  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
|  DB  | ch_ppocr_mobile_v2.0_det| 正常训练 | - | - | 正常训练：FPGM裁剪、PACT量化 |


- 预测相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的预测功能汇总如下，

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| 正常模型 | CPU | 1/6 | - | fp32 | 支持 |
| 量化模型 | CPU | 1/6 | - | int8 | 支持 |


## 2. 测试流程

Mac端无GPU，环境准备只需要Python环境即可，安装PaddlePaddle等依赖参考下述文档。

### 2.1 安装依赖
- 安装PaddlePaddle >= 2.3
- 安装PaddleOCR依赖
    ```
    pip install  -r ../requirements.txt
    ```
- 安装autolog（规范化日志输出工具）
    ```
    pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```
- 安装PaddleSlim (可选)
   ```
   # 如果要测试量化、裁剪等功能，需要安装PaddleSlim
   pip install paddleslim
   ```


### 2.2 功能测试

先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`,model_name/lite_train_lite_infer/*.log`格式的日志文件。

`test_train_inference_python.sh`包含基础链条的4种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
# 同linux端运行不同的是，Mac端测试使用新的配置文件mac_ppocr_det_mobile_params.txt，
# 配置文件中默认去掉了GPU和mkldnn相关的测试链条
bash test_tipc/prepare.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'lite_train_lite_infer'
```

- 模式2：lite_train_whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'lite_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt  'lite_train_whole_infer'
```

- 模式3：whole_infer，不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_infer'
# 用法1:
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_infer'
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_infer' '1'
```

- 模式4：whole_train_whole_infer，CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；（Mac端不建议运行此模式）
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ch_ppocr_mobile_v2_0_det/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_train_whole_infer'
```

运行相应指令后，在`test_tipc/output`文件夹下自动会保存运行日志。如`lite_train_lite_infer`模式下，会运行训练+inference的链条，因此，在`test_tipc/output`文件夹有以下文件：
```
test_tipc/output/model_name/lite_train_lite_infer/
|- results_python.log    # 运行指令状态的日志
|- norm_train_gpus_-1_autocast_null/  # CPU上正常训练的训练日志和模型保存文件夹
......
|- python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log  # CPU上关闭Mkldnn线程数设置为1，测试batch_size=1条件下的fp32精度预测运行日志
......
```

其中`results_python.log`中包含了每条指令的运行状态，如果运行成功会输出：
```
Run successfully with command - python3.7 tools/train.py -c tests/configs/det_mv3_db.yml -o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained Global.use_gpu=False  Global.save_model_dir=./tests/output/norm_train_gpus_-1_autocast_null Global.epoch_num=1     Train.loader.batch_size_per_card=2   !
Run successfully with command - python3.7 tools/export_model.py -c tests/configs/det_mv3_db.yml -o  Global.pretrained_model=./tests/output/norm_train_gpus_-1_autocast_null/latest Global.save_inference_dir=./tests/output/norm_train_gpus_-1_autocast_null!
......
```
如果运行失败，会输出：
```
Run failed with command - python3.7 tools/train.py -c tests/configs/det_mv3_db.yml -o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained Global.use_gpu=Faslse  Global.save_model_dir=./tests/output/norm_train_gpus_-1_autocast_null Global.epoch_num=1     Train.loader.batch_size_per_card=2   !
Run failed with command - python3.7 tools/export_model.py -c tests/configs/det_mv3_db.yml -o  Global.pretrained_model=./tests/output/norm_train_gpus_0_autocast_null/latest Global.save_inference_dir=./tests/output/norm_train_gpus_-1_autocast_null!
......
```
可以很方便的根据`results_python.log`中的内容判定哪一个指令运行错误。

### 2.3 精度测试

使用compare_results.py脚本比较模型预测的结果是否符合预期，主要步骤包括：
- 提取日志中的预测坐标；
- 从本地文件中提取保存好的坐标结果；
- 比较上述两个结果是否符合精度预期，误差大于设置阈值时会报错。

#### 使用方式
运行命令：
```shell
python test_tipc/compare_results.py --gt_file=./test_tipc/results/python_*.txt  --log_file=./test_tipc/output/python_*.log --atol=1e-3 --rtol=1e-3
```

参数介绍：
- gt_file： 指向事先保存好的预测结果路径，支持*.txt 结尾，会自动索引*.txt格式的文件，文件默认保存在test_tipc/result/ 文件夹下
- log_file: 指向运行test_tipc/test_train_inference_python.sh 脚本的infer模式保存的预测日志，预测日志中打印的有预测结果，比如：文本框，预测文本，类别等等，同样支持python_infer_*.log格式传入
- atol: 设置的绝对误差
- rtol: 设置的相对误差

#### 运行结果

正常运行效果如下：
```
Assert allclose passed! The results of python_infer_cpu_usemkldnn_False_threads_1_batchsize_1.log and ./test_tipc/results/python_ppocr_det_mobile_results_fp32.txt are consistent!
```

出现不一致结果时的运行输出：
```
......
Traceback (most recent call last):
  File "test_tipc/compare_results.py", line 140, in <module>
    format(filename, gt_filename))
ValueError: The results of python_infer_cpu_usemkldnn_False_threads_1_batchsize_1.log and the results of ./test_tipc/results/python_ppocr_det_mobile_results_fp32.txt are inconsistent!
```


## 3. 更多教程
本文档为功能测试用，更丰富的训练预测使用教程请参考：
[模型训练](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/training.md)
[基于Python预测引擎推理](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/inference_ppocr.md)
