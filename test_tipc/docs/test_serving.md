# PaddleServing预测功能测试

PaddleServing预测功能测试的主程序为`test_serving_infer_python.sh`和`test_serving_infer_cpp.sh`，可以测试基于PaddleServing的部署功能。

## 1. 测试结论汇总

基于训练是否使用量化，进行本测试的模型可以分为`正常模型`和`量化模型`，这两类模型对应的Serving预测功能汇总如下：

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1/6 | fp32/fp16 | - | - |
| 正常模型 | CPU | 1/6 | - | fp32 | 支持 |
| 量化模型 | GPU | 1/6 | int8 | - | - |
| 量化模型 | CPU | 1/6 | - | int8 | 支持 |

## 2. 测试流程
运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.1 功能测试
**python serving**
先运行`prepare.sh`准备数据和模型，然后运行`test_serving_infer_python.sh`进行测试，最终在```test_tipc/output/{model_name}/serving_infer/python```目录下生成`python_*.log`后缀的日志文件。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ch_PP-OCRv2/model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt "serving_infer"

# 用法:
bash test_tipc/test_serving_infer_python.sh ./test_tipc/configs/ch_PP-OCRv2/model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt "serving_infer"
```  
**cpp serving**
先运行`prepare.sh`准备数据和模型，然后运行`test_serving_infer_cpp.sh`进行测试，最终在```test_tipc/output/{model_name}/serving_infer/cpp```目录下生成`cpp_*.log`后缀的日志文件。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ch_PP-OCRv2/model_linux_gpu_normal_normal_serving_cpp_linux_gpu_cpu.txt "serving_infer"

# 用法:
bash test_tipc/test_serving_infer_cpp.sh ./test_tipc/configs/ch_PP-OCRv2/model_linux_gpu_normal_normal_serving_cpp_linux_gpu_cpu.txt "serving_infer"
```  

#### 运行结果

各测试的运行情况会打印在 `test_tipc/output/{model_name}/serving_infer/python(cpp)/results_python(cpp)_serving.log` 中：
运行成功时会输出：

```
Run successfully with command - ch_PP-OCRv2_rec - nohup python3.7 web_service_rec.py --config=config.yml --opt op.rec.concurrency="1" op.det.local_service_conf.devices= op.det.local_service_conf.use_mkldnn=False op.det.local_service_conf.thread_num=6 op.rec.local_service_conf.model_config=ppocr_rec_v2_serving > ./test_tipc/output/ch_PP-OCRv2_rec/serving_infer/python/python_server_cpu_usemkldnn_False_threads_6.log 2>&1 &!
Run successfully with command - ch_PP-OCRv2_rec - python3.7 pipeline_http_client.py --det=False --image_dir=../../inference/rec_inference > ./test_tipc/output/ch_PP-OCRv2_rec/serving_infer/python/python_client_cpu_pipeline_http_usemkldnn_False_threads_6_batchsize_1.log 2>&1 !
...
```

运行失败时会输出：

```
Run failed with command - ch_PP-OCRv2_rec - nohup python3.7 web_service_rec.py --config=config.yml --opt op.rec.concurrency="1" op.det.local_service_conf.devices= op.det.local_service_conf.use_mkldnn=False op.det.local_service_conf.thread_num=6 op.rec.local_service_conf.model_config=ppocr_rec_v2_serving > ./test_tipc/output/ch_PP-OCRv2_rec/serving_infer/python/python_server_cpu_usemkldnn_False_threads_6.log 2>&1 &!
Run failed with command - ch_PP-OCRv2_rec - python3.7 pipeline_http_client.py --det=False --image_dir=../../inference/rec_inference > ./test_tipc/output/ch_PP-OCRv2_rec/serving_infer/python/python_client_cpu_pipeline_http_usemkldnn_False_threads_6_batchsize_1.log 2>&1 !
...
```

详细的预测结果会存在 test_tipc/output/{model_name}/serving_infer/python(cpp)/ 文件夹下


## 3. 更多教程

本文档为功能测试用，更详细的Serving预测使用教程请参考：[PPOCR 服务化部署](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/deploy/pdserving/README_CN.md)  
