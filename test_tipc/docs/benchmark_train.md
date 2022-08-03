
# TIPC Linux端Benchmark测试文档

该文档为Benchmark测试说明，Benchmark预测功能测试的主程序为`benchmark_train.sh`，用于验证监控模型训练的性能。

# 1. 测试流程
## 1.1 准备数据和环境安装
运行`test_tipc/prepare.sh`，完成训练数据准备和安装环境流程。

```shell
# 运行格式：bash test_tipc/prepare.sh  train_benchmark.txt  mode
bash test_tipc/prepare.sh test_tipc/configs/det_mv3_db_v2_0/train_infer_python.txt benchmark_train
```

## 1.2 功能测试
执行`test_tipc/benchmark_train.sh`，完成模型训练和日志解析

```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_benchmark.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/det_mv3_db_v2_0/train_infer_python.txt benchmark_train

```

`test_tipc/benchmark_train.sh`支持根据传入的第三个参数实现只运行某一个训练配置，如下：
```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_benchmark.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/det_mv3_db_v2_0/train_infer_python.txt benchmark_train  dynamic_bs8_fp32_DP_N1C1
```
dynamic_bs8_fp32_DP_N1C1为test_tipc/benchmark_train.sh传入的参数，格式如下：
`${modeltype}_${batch_size}_${fp_item}_${run_mode}_${device_num}`
包含的信息有：模型类型、batchsize大小、训练精度如fp32,fp16等、分布式运行模式以及分布式训练使用的机器信息如单机单卡（N1C1）。


## 2. 日志输出

运行后将保存模型的训练日志和解析日志，使用 `test_tipc/configs/det_mv3_db_v2_0/train_infer_python.txt` 参数文件的训练日志解析结果是：

```
{"model_branch": "dygaph", "model_commit": "7c39a1996b19087737c05d883fd346d2f39dbcc0", "model_name": "det_mv3_db_v2_0_bs8_fp32_SingleP_DP", "batch_size": 8, "fp_item": "fp32", "run_process_type": "SingleP", "run_mode": "DP", "convergence_value": "5.413110", "convergence_key": "loss:", "ips": 19.333, "speed_unit": "samples/s", "device_num": "N1C1", "model_run_time": "0", "frame_commit": "8cc09552473b842c651ead3b9848d41827a3dbab", "frame_version": "0.0.0"}
```

训练日志和日志解析结果保存在benchmark_log目录下，文件组织格式如下：
```
train_log/
├── index
│   ├── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C1_speed
│   └── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C4_speed
├── profiling_log
│   └── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C1_profiling
└── train_log
    ├── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C1_log
    └── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C4_log
```
## 3. 各模型单卡性能数据一览

*注：本节中的速度指标均使用单卡（1块Nvidia V100 16G GPU）测得。通常情况下。


|模型名称|配置文件|大数据集 float32 fps |小数据集 float32 fps |diff |大数据集 float16 fps|小数据集 float16 fps| diff | 大数据集大小 | 小数据集大小 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ch_ppocr_mobile_v2.0_det |[config](../configs/ch_ppocr_mobile_v2.0_det/train_infer_python.txt) | 53.836 | 53.343 / 53.914 / 52.785 |0.020940758 | 45.574 | 45.57 / 46.292 / 46.213 | 0.015596647 | 10,000| 2,000|
| ch_ppocr_mobile_v2.0_rec |[config](../configs/ch_ppocr_mobile_v2.0_rec/train_infer_python.txt) | 2083.311 | 2043.194	/ 2066.372 / 2093.317 |0.023944295 | 2153.261 | 2167.561 /	2165.726 /	2155.614| 0.005511725 | 600,000| 160,000|
| ch_ppocr_server_v2.0_det |[config](../configs/ch_ppocr_server_v2.0_det/train_infer_python.txt) | 20.716 | 20.739 /	20.807 /	20.755 |0.003268131 | 20.592 | 20.498 /	20.993 /	20.75| 0.023579288 | 10,000| 2,000|
| ch_ppocr_server_v2.0_rec |[config](../configs/ch_ppocr_server_v2.0_rec/train_infer_python.txt) | 528.56 | 528.386 /	528.991 /	528.391 |0.001143687 | 1189.788 | 1190.007 /	1176.332 /	1192.084| 0.013213834 |  600,000| 160,000|
| ch_PP-OCRv2_det	 |[config](../configs/ch_PP-OCRv2_det/train_infer_python.txt) | 13.87 | 13.386 /	13.529 /	13.428 |0.010569887 | 17.847 | 17.746 /	17.908 /	17.96| 0.011915367 | 10,000| 2,000|
| ch_PP-OCRv2_rec	 |[config](../configs/ch_PP-OCRv2_rec/train_infer_python.txt) | 109.248 | 106.32 /	106.318 /	108.587 |0.020895687 | 117.491 | 117.62 /	117.757 /	117.726| 0.001163413 | 140,000| 40,000|
| det_mv3_db_v2.0	 |[config](../configs/det_mv3_db_v2_0/train_infer_python.txt) | 61.802 | 62.078 /	61.802 /	62.008 |0.00444602 | 82.947 | 84.294 /	84.457 /	84.005| 0.005351836 | 10,000| 2,000|
| det_r50_vd_db_v2.0	 |[config](../configs/det_r50_vd_db_v2.0/train_infer_python.txt) | 29.955 | 29.092 /	29.31 /	28.844 |0.015899011 | 51.097 |50.367 /	50.879 /	50.227| 0.012814717 | 10,000| 2,000|
| det_r50_vd_east_v2.0	 |[config](../configs/det_r50_vd_east_v2.0/train_infer_python.txt) | 42.485 | 42.624 /	42.663 /	42.561 |0.00239083 | 67.61 |67.825/ 	68.299/ 	68.51| 0.00999854 | 10,000| 2,000|
| det_r50_vd_pse_v2.0	 |[config](../configs/det_r50_vd_pse_v2.0/train_infer_python.txt) | 16.455 | 16.517 / 16.555 /	16.353 |0.012201752 | 27.02 |27.288 /	27.152 /	27.408| 0.009340339 | 10,000| 2,000|
| rec_mv3_none_bilstm_ctc_v2.0	 |[config](../configs/rec_mv3_none_bilstm_ctc_v2.0/train_infer_python.txt) | 2288.358 | 2291.906 /	2293.725 /	2290.05 |0.001602197 | 2336.17 |2327.042 /	2328.093 /	2344.915| 0.007622025 | 600,000| 160,000|
| layoutxlm_ser	 |[config](../configs/layoutxlm/train_infer_python.txt) | 18.001 | 18.114 /	18.107 /	18.307 |0.010924783 | 21.982 | 21.507 /	21.116 /	21.406| 0.018180127 | 1490 | 1490|
| PP-Structure-table	 |[config](../configs/en_table_structure/train_infer_python.txt) | 14.151 | 14.077 /	14.23 /	14.25 |0.012140351 | 16.285 | 16.595 /	16.878 /	16.531 | 0.020559308 | 20,000| 5,000|
| det_r50_dcn_fce_ctw_v2.0	 |[config](../configs/det_r50_dcn_fce_ctw_v2.0/train_infer_python.txt) | 14.057 | 14.029 /	14.02 /	14.014 |0.001069214 | 18.298 |18.411 /	18.376 /	18.331| 0.004345228 | 10,000| 2,000|
| ch_PP-OCRv3_det	 |[config](../configs/ch_PP-OCRv3_det/train_infer_python.txt) | 8.622 | 8.431 /	8.423 /	8.479|0.006604552 | 14.203 |14.346	14.468	14.23| 0.016450097 | 10,000| 2,000|
| ch_PP-OCRv3_rec	 |[config](../configs/ch_PP-OCRv3_rec/train_infer_python.txt) | 90.239 | 90.077 /	91.513 /	91.325|0.01569176 | | |  | 160,000| 40,000|
