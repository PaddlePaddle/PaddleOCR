
# 推理部署导航

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleOCR中所有模型的推理部署导航，方便用户查阅每种模型的推理部署打通情况，并可以进行一键测试。

<div align="center">
    <img src="docs/guide.png" width="1000">
</div>

打通情况分为以下四种情况：
- **支持**：可以一键测试
- **未接入**：PaddleOCR已支持该功能，但还未接入一键测试
- **未覆盖**：PaddleOCR未进行打通测试，也没有接入一键测试
- **不支持**：由于飞桨框架限制，暂时无法支持该功能

| 算法论文 | 模型名称 | 模型类型 | python训练预测 |   其他  |
| --- | :--- |  ----  | :-------- |  :----  |
| DB     |ch_ppocr_mobile_v2.0_det_infer | 检测  | 已支持 | 已支持：C++预测 / Paddle Serving / Paddle-Lite <br> 待支持：Paddle.js |
| DB     |ch_ppocr_server_v2.0_det_infer | 检测  | 支持 | |
| DB     |ch_PP-OCRv2_det_infer          | 检测  |
| CRNN   |ch_ppocr_mobile_v2.0_rec_infer | 识别  | 支持 | |
| CRNN   |ch_ppocr_server_v2.0_rec_infer | 识别  | 支持 | |
| CRNN   |ch_PP-OCRv2_rec_infer          | 识别  |
| DB     |det_mv3_db_v2.0                | 检测  |
| DB     |det_r50_vd_db_v2.0             | 检测  |
| EAST   |det_mv3_east_v2.0              | 检测  |
| EAST   |det_r50_vd_east_v2.0           | 检测  |
| PSENet |det_mv3_pse_v2.0               | 检测  |
| PSENet |det_r50_vd_pse_v2.0            | 检测  |
| SAST   |det_r50_vd_sast_totaltext_v2.0 | 检测  |
| Rosetta|rec_mv3_none_none_ctc_v2.0     | 识别  |
| Rosetta|rec_r34_vd_none_none_ctc_v2.0  | 识别  |
| CRNN   |rec_mv3_none_bilstm_ctc_v2.0   | 识别  |
| CRNN   |rec_r34_vd_none_bilstm_ctc_v2.0| 识别  |
| StarNet|rec_mv3_tps_bilstm_ctc_v2.0    | 识别  |
| StarNet|rec_r34_vd_tps_bilstm_ctc_v2.0 | 识别  |
| RARE   |rec_mv3_tps_bilstm_att_v2.0    | 识别  |
| RARE   |rec_r34_vd_tps_bilstm_att_v2.0 | 识别  |
| SRN    |rec_r50fpn_vd_none_srn         | 识别  |
| NRTR   |rec_mtb_nrtr                   | 识别  |
| SAR    |rec_r31_sar                    | 识别  |
| PGNet  |rec_r34_vd_none_none_ctc_v2.0  | 端到端|



## 一键测试工具使用
### 目录介绍

```shell
tests/
├── configs/  # 配置文件目录
	├── det_mv3_db.yml               # 测试mobile版ppocr检测模型训练的yml文件
	├── det_r50_vd_db.yml            # 测试server版ppocr检测模型训练的yml文件
	├── rec_icdar15_r34_train.yml    # 测试server版ppocr识别模型训练的yml文件
	├── ppocr_sys_mobile_params.txt     # 测试mobile版ppocr检测+识别模型串联的参数配置文件
	├── ppocr_det_mobile_params.txt     # 测试mobile版ppocr检测模型的参数配置文件
	├── ppocr_rec_mobile_params.txt     # 测试mobile版ppocr识别模型的参数配置文件
	├── ppocr_sys_server_params.txt     # 测试server版ppocr检测+识别模型串联的参数配置文件
	├── ppocr_det_server_params.txt     # 测试server版ppocr检测模型的参数配置文件
	├── ppocr_rec_server_params.txt     # 测试server版ppocr识别模型的参数配置文件
	├── ...                                
├── results/   # 预先保存的预测结果，用于和实际预测结果进行精读比对
	├── ppocr_det_mobile_results_fp32.txt           # 预存的mobile版ppocr检测模型fp32精度的结果
	├── ppocr_det_mobile_results_fp16.txt           # 预存的mobile版ppocr检测模型fp16精度的结果
	├── ppocr_det_mobile_results_fp32_cpp.txt       # 预存的mobile版ppocr检测模型c++预测的fp32精度的结果
	├── ppocr_det_mobile_results_fp16_cpp.txt       # 预存的mobile版ppocr检测模型c++预测的fp16精度的结果
	├── ...
├── prepare.sh                # 完成test_*.sh运行所需要的数据和模型下载
├── test_python.sh            # 测试python训练预测的主程序
├── test_cpp.sh               # 测试c++预测的主程序
├── test_serving.sh           # 测试serving部署预测的主程序
├── test_lite.sh              # 测试lite部署预测的主程序
├── compare_results.py        # 用于对比log中的预测结果与results中的预存结果精度误差是否在限定范围内
└── readme.md                 # 使用文档
```

### 测试流程
使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程如下：
<div align="center">
    <img src="docs/test.png" width="800">
</div>

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_*.sh`，产出log，由log可以看到不同配置是否运行成功；
3. 【可选】用`compare_results.py`对比log中的预测结果和预存在results目录下的结果，判断预测精度是否符合预期（在误差范围内）。

其中，有4个测试主程序，功能如下：
- `test_python.sh`：测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。
- `test_cpp.sh`：测试基于C++的模型推理。
- `test_serving.sh`：测试基于Paddle Serving的服务化部署功能。
- `test_lite.sh`：测试基于Paddle-Lite的端侧预测部署功能。

各功能测试中涉及GPU/CPU、mkldnn、Tensorrt等多种参数配置，点击相应链接了解更多细节和使用教程：  
[test_python使用](docs/test_python.md)  
[test_cpp使用](docs/test_cpp.md)  
[test_serving使用](docs/test_serving.md)  
[test_lite使用](docs/test_lite.md)  
