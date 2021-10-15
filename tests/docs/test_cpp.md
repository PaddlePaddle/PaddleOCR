# C++预测功能测试

C++预测功能测试的主程序为`test_cpp.sh`，可以测试基于C++预测库的模型推理功能。

## 测试结论汇总

| 算法名称 | 模型名称 |device | batchsize | mkldnn | cpu多线程 | tensorrt | 离线量化 |
|  ----  |   ----  |  ----  |  ---- |  ---- |  ----  |  ----| --- | 
| DB   |ch_ppocr_mobile_v2.0_det| CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
| DB   |ch_ppocr_server_v2.0_det| CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
| CRNN |ch_ppocr_mobile_v2.0_rec| CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
| CRNN |ch_ppocr_server_v2.0_rec| CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
|PP-OCR|ch_ppocr_server_v2.0    | CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |
|PP-OCR|ch_ppocr_server_v2.0    | CPU/GPU | 1/6 | 支持 | 支持 | fp32/fp16/int8 | 支持 |



## 1. 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_cpp.sh`进行测试，最终在```tests/output```目录下生成`cpp_infer_*.log`后缀的日志文件。

```shell
bash tests/prepare.sh ./tests/configs/ppocr_det_mobile_params.txt

# 用法1:
bash tests/test_cpp.sh ./tests/configs/ppocr_det_mobile_params.txt
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash tests/test_cpp.sh ./tests/configs/ppocr_det_mobile_params.txt '1'
```  
 

## 2. 精度测试

使用compare_results.py脚本比较模型预测的结果是否符合预期，主要步骤包括：
- 提取日志中的预测坐标；
- 从本地文件中提取保存好的坐标结果；
- 比较上述两个结果是否符合精度预期，误差大于设置阈值时会报错。

### 使用方式
运行命令：
```shell
python3.7 tests/compare_results.py --gt_file=./tests/results/cpp_*.txt  --log_file=./tests/output/cpp_*.log --atol=1e-3 --rtol=1e-3
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
