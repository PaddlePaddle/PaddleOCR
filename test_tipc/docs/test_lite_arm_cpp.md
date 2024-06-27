# Lite\_arm\_cpp预测功能测试

Lite\_arm\_cpp预测功能测试的主程序为`test_lite_arm_cpp.sh`，可以在ARM上基于Lite预测库测试模型的C++推理功能。

## 1. 测试结论汇总

目前Lite端的样本间支持以方式的组合：

**字段说明：**
- 模型类型：包括正常模型（FP32）和量化模型（INT8）
- batch-size：包括1和4
- threads：包括1和4
- predictor数量：包括单predictor预测和多predictor预测
- 预测库来源：包括下载方式和编译方式
- 测试硬件：ARM\_CPU/ARM\_GPU_OPENCL

| 模型类型 | batch-size | threads | predictor数量 | 预测库来源 | 测试硬件 |
|  :----:   |  :----:  | :----:  |  :----:  |  :----:  |  :----:  |
| 正常模型/量化模型 | 1 | 1/4 |  单/多 | 下载方式/编译方式 | ARM\_CPU/ARM\_GPU_OPENCL |


## 2. 测试流程
运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.1 功能测试

先运行`prepare_lite_cpp.sh`，运行后会在当前路径下生成`test_lite.tar`，其中包含了测试数据、测试模型和用于预测的可执行文件。将`test_lite.tar`上传到被测试的手机上，在手机的终端解压该文件，进入`test_lite`目录中，然后运行`test_lite_arm_cpp.sh`进行测试，最终在`test_lite/output`目录下生成`lite_*.log`后缀的日志文件。

#### 2.1.1 基于ARM\_CPU测试

```shell

# 数据、模型、Paddle-Lite预测库准备
#预测库为下载方式
bash test_tipc/prepare_lite_cpp.sh ./test_tipc/configs/ch_PP-OCRv2_det/model_linux_gpu_normal_normal_lite_cpp_arm_cpu.txt download
#预测库为编译方式
bash test_tipc/prepare_lite_cpp.sh ./test_tipc/configs/ch_PP-OCRv2_det/model_linux_gpu_normal_normal_lite_cpp_arm_cpu.txt compile

# 手机端测试:
bash test_lite_arm_cpp.sh model_linux_gpu_normal_normal_lite_cpp_arm_cpu.txt

```

#### 2.1.2 基于ARM\_GPU\_OPENCL测试

```shell

# 数据、模型、Paddle-Lite预测库准备
#预测库下载方式
bash test_tipc/prepare_lite_cpp.sh ./test_tipc/configs/ch_PP-OCRv2_det/model_linux_gpu_normal_normal_lite_cpp_arm_gpu_opencl.txt download
#预测库编译方式
bash test_tipc/prepare_lite_cpp.sh ./test_tipc/configs/ch_PP-OCRv2_det/model_linux_gpu_normal_normal_lite_cpp_arm_gpu_opencl.txt compile

# 手机端测试:
bash test_lite_arm_cpp.sh model_linux_gpu_normal_normal_lite_cpp_arm_gpu_opencl.txt

```


**注意**：

由于运行该项目需要bash等命令，传统的adb方式不能很好的安装。所以此处推荐通在手机上开启虚拟终端的方式连接电脑，连接方式可以参考[安卓手机termux连接电脑](./termux_for_android.md)。

### 2.2 运行结果

各测试的运行情况会打印在 `./output/` 中：
运行成功时会输出：

```
Run successfully with command - ./ocr_db_crnn det ch_PP-OCRv2_det_infer_opt.nb ARM_CPU FP32 1 1  ./test_data/icdar2015_lite/text_localization/ch4_test_images/ ./config.txt True > ./output/lite_ch_PP-OCRv2_det_infer_opt.nb_runtime_device_ARM_CPU_precision_FP32_batchsize_1_threads_1.log 2>&1!
Run successfully with command xxx
...
```

运行失败时会输出：

```
Run failed with command - ./ocr_db_crnn det ch_PP-OCRv2_det_infer_opt.nb ARM_CPU FP32 1 1  ./test_data/icdar2015_lite/text_localization/ch4_test_images/ ./config.txt True > ./output/lite_ch_PP-OCRv2_det_infer_opt.nb_runtime_device_ARM_CPU_precision_FP32_batchsize_1_threads_1.log 2>&1!
Run failed with command xxx
...
```

在./output/文件夹下，会存在如下日志，每一个日志都是不同配置下的log结果：

<img src="lite_log.png" width="1000">

在每一个log中，都会调用autolog打印如下信息：

<img src="lite_auto_log.png" width="1000">



## 3. 更多教程

本文档为功能测试用，更详细的Lite端预测使用教程请参考：[Lite端部署](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/lite/readme.md)。
