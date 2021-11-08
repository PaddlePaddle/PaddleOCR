# PaddleServing预测功能测试

PaddleServing预测功能测试的主程序为`test_serving.sh`，可以测试基于PaddleServing的部署功能。

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
先运行`prepare.sh`准备数据和模型，然后运行`test_serving.sh`进行测试，最终在```test_tipc/output```目录下生成`serving_infer_*.log`后缀的日志文件。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile_params.txt "serving_infer"

# 用法:
bash test_tipc/test_serving.sh ./test_tipc/configs/ppocr_det_mobile_params.txt
```  

#### 运行结果

各测试的运行情况会打印在 `test_tipc/output/results_serving.log` 中：
运行成功时会输出：

```
Run successfully  with command - python3.7 pipeline_http_client.py --image_dir=../../doc/imgs > ../../tests/output/server_infer_cpu_usemkldnn_True_threads_1_batchsize_1.log 2>&1 !
Run successfully  with command - xxxxx
...
```

运行失败时会输出：

```
Run failed with command - python3.7 pipeline_http_client.py --image_dir=../../doc/imgs > ../../tests/output/server_infer_cpu_usemkldnn_True_threads_1_batchsize_1.log 2>&1 !
Run failed with command - python3.7 pipeline_http_client.py --image_dir=../../doc/imgs > ../../tests/output/server_infer_cpu_usemkldnn_True_threads_6_batchsize_1.log 2>&1 !
Run failed with command - xxxxx
...
```

详细的预测结果会存在 test_tipc/output/ 文件夹下，例如`server_infer_gpu_usetrt_True_precision_fp16_batchsize_1.log`中会返回检测框的坐标:

```
{'err_no': 0, 'err_msg': '', 'key': ['dt_boxes'], 'value': ['[[[ 78. 642.]\n  [409. 640.]\n  [409. 657.]\n  
[ 78. 659.]]\n\n [[ 75. 614.]\n  [211. 614.]\n      [211. 635.]\n  [ 75. 635.]]\n\n
[[103. 554.]\n  [135. 554.]\n  [135. 575.]\n  [103. 575.]]\n\n [[ 75. 531.]\n  
[347. 531.]\n  [347. 549.]\n  [ 75. 549.]    ]\n\n [[ 76. 503.]\n  [309. 498.]\n  
[309. 521.]\n  [ 76. 526.]]\n\n [[163. 462.]\n  [317. 462.]\n  [317. 493.]\n  
[163. 493.]]\n\n [[324. 431.]\n  [414.     431.]\n  [414. 452.]\n  [324. 452.]]\n\n
[[ 76. 412.]\n  [208. 408.]\n  [209. 424.]\n  [ 76. 428.]]\n\n [[307. 409.]\n  
[428. 409.]\n  [428. 426.]\n  [307    . 426.]]\n\n [[ 74. 385.]\n  [217. 382.]\n  
[217. 400.]\n  [ 74. 403.]]\n\n [[308. 381.]\n  [427. 380.]\n  [427. 400.]\n  
[308. 401.]]\n\n [[ 74. 363.]\n      [195. 362.]\n  [195. 378.]\n  [ 74. 379.]]\n\n
[[303. 359.]\n  [423. 357.]\n  [423. 375.]\n  [303. 377.]]\n\n [[ 70. 336.]\n  
[239. 334.]\n  [239. 354.]\    n  [ 70. 356.]]\n\n [[ 70. 312.]\n  [204. 310.]\n  
[204. 327.]\n  [ 70. 330.]]\n\n [[303. 308.]\n  [419. 306.]\n  [419. 326.]\n  
[303. 328.]]\n\n [[113. 2    72.]\n  [246. 270.]\n  [247. 299.]\n  [113. 301.]]\n\n
 [[361. 269.]\n  [384. 269.]\n  [384. 296.]\n  [361. 296.]]\n\n [[ 70. 250.]\n
 [243. 246.]\n  [243.     265.]\n  [ 70. 269.]]\n\n [[ 65. 221.]\n  [187. 220.]\n  
[187. 240.]\n  [ 65. 241.]]\n\n [[337. 216.]\n  [382. 216.]\n  [382. 240.]\n  
[337. 240.]]\n\n [    [ 65. 196.]\n  [247. 193.]\n  [247. 213.]\n  [ 65. 216.]]\n\n
[[296. 197.]\n  [423. 191.]\n  [424. 209.]\n  [296. 215.]]\n\n [[ 65. 167.]\n  [244. 167.]\n  
[244. 186.]\n  [ 65. 186.]]\n\n [[ 67. 139.]\n  [290. 139.]\n  [290. 159.]\n  [ 67. 159.]]\n\n
[[ 68. 113.]\n  [410. 113.]\n  [410. 128.]\n  [ 68. 129.]    ]\n\n [[277.  87.]\n  [416.  87.]\n  
[416. 108.]\n  [277. 108.]]\n\n [[ 79.  28.]\n  [132.  28.]\n  [132.  62.]\n  [ 79.  62.]]\n\n
[[163.  17.]\n  [410.      14.]\n  [410.  50.]\n  [163.  53.]]]']}
```


## 3. 更多教程

本文档为功能测试用，更详细的Serving预测使用教程请参考：[PPOCR 服务化部署](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/deploy/pdserving/README_CN.md)  
