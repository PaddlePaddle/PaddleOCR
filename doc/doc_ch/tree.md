# 整体目录结构

PaddleOCR 的整体目录结构介绍如下：

```
PaddleOCR
├── configs                                 // 配置文件，可通过 yml 文件选择模型结构并修改超参
│   ├── cls                                 // 方向分类器相关配置文件
│   │   ├── cls_mv3.yml                     // 训练配置相关，包括骨干网络、head、loss、优化器和数据
│   ├── det                                 // 检测相关配置文件
│   │   ├── det_mv3_db.yml                  // 训练配置
│   │   ...  
│   └── rec                                 // 识别相关配置文件
│       ├── rec_mv3_none_bilstm_ctc.yml     // crnn 训练配置
│       ...  
├── deploy                                  // 部署相关
│   ├── android_demo                        // android_demo
│   │   ...
│   ├── cpp_infer                           // C++ infer
│   │   ├── CMakeLists.txt                  // Cmake 文件
│   │   ├── docs                            // 说明文档
│   │   │   └── windows_vs2019_build.md
│   │   ├── include                         // 头文件
│   │   │   ├── clipper.h                   // clipper 库
│   │   │   ├── config.h                    // 预测配置
│   │   │   ├── ocr_cls.h                   // 方向分类器
│   │   │   ├── ocr_det.h                   // 文字检测
│   │   │   ├── ocr_rec.h                   // 文字识别
│   │   │   ├── postprocess_op.h            // 检测后处理
│   │   │   ├── preprocess_op.h             // 检测预处理
│   │   │   └── utility.h                   // 工具
│   │   ├── readme.md                       // 说明文档
│   │   ├── ...
│   │   ├── src                             // 源文件
│   │   │   ├── clipper.cpp  
│   │   │   ├── config.cpp
│   │   │   ├── main.cpp
│   │   │   ├── ocr_cls.cpp
│   │   │   ├── ocr_det.cpp
│   │   │   ├── ocr_rec.cpp
│   │   │   ├── postprocess_op.cpp
│   │   │   ├── preprocess_op.cpp
│   │   │   └── utility.cpp
│   │   └── tools                           // 编译、执行脚本
│   │       ├── build.sh                    // 编译脚本
│   │       ├── config.txt                  // 配置文件
│   │       └── run.sh                      // 测试启动脚本
│   ├── docker
│   │   └── hubserving
│   │       ├── cpu
│   │       │   └── Dockerfile
│   │       ├── gpu
│   │       │   └── Dockerfile
│   │       ├── README_cn.md
│   │       ├── README.md
│   │       └── sample_request.txt
│   ├── hubserving                          // hubserving
│   │   ├── ocr_cls                         // 方向分类器
│   │   │   ├── config.json                 // serving 配置
│   │   │   ├── __init__.py  
│   │   │   ├── module.py                   // 预测模型
│   │   │   └── params.py                   // 预测参数
│   │   ├── ocr_det                         // 文字检测
│   │   │   ├── config.json                 // serving 配置
│   │   │   ├── __init__.py  
│   │   │   ├── module.py                   // 预测模型
│   │   │   └── params.py                   // 预测参数
│   │   ├── ocr_rec                         // 文字识别
│   │   │   ├── config.json
│   │   │   ├── __init__.py
│   │   │   ├── module.py
│   │   │   └── params.py
│   │   └── ocr_system                      // 系统预测
│   │       ├── config.json
│   │       ├── __init__.py
│   │       ├── module.py
│   │       └── params.py
│   ├── imgs                                // 预测图片
│   │   ├── cpp_infer_pred_12.png
│   │   └── demo.png
│   ├── ios_demo                            // ios demo
│   │   ...
│   ├── lite                                // lite 部署
│   │   ├── cls_process.cc                  // 方向分类器数据处理
│   │   ├── cls_process.h
│   │   ├── config.txt                      // 检测配置参数
│   │   ├── crnn_process.cc                 // crnn 数据处理
│   │   ├── crnn_process.h
│   │   ├── db_post_process.cc              // db 数据处理
│   │   ├── db_post_process.h
│   │   ├── Makefile                        // 编译文件
│   │   ├── ocr_db_crnn.cc                  // 串联预测
│   │   ├── prepare.sh                      // 数据准备
│   │   ├── readme.md                       // 说明文档
│   │   ...
│   ├── pdserving                           // pdserving 部署
│   │   ├── det_local_server.py             // 检测 快速版，部署方便预测速度快
│   │   ├── det_web_server.py               // 检测 完整版，稳定性高分布式部署
│   │   ├── ocr_local_server.py             // 检测+识别 快速版
│   │   ├── ocr_web_client.py               // 客户端
│   │   ├── ocr_web_server.py               // 检测+识别 完整版
│   │   ├── readme.md                       // 说明文档
│   │   ├── rec_local_server.py             // 识别 快速版
│   │   └── rec_web_server.py               // 识别 完整版
│   └── slim  
│       └── quantization                    // 量化相关
│           ├── export_model.py             // 导出模型
│           ├── quant.py                    // 量化
│           └── README.md                   // 说明文档
├── doc                                     // 文档教程
│   ...
├── ppocr                                   // 网络核心代码
│   ├── data                                // 数据处理
│   │   ├── imaug                           // 图片和 label 处理代码
│   │   │   ├── text_image_aug              // 文本识别的 tia 数据扩充
│   │   │   │   ├── __init__.py
│   │   │   │   ├── augment.py              // tia_distort,tia_stretch 和 tia_perspective 的代码
│   │   │   │   ├── warp_mls.py
│   │   │   ├── __init__.py
│   │   │   ├── east_process.py             // EAST 算法的数据处理步骤
│   │   │   ├── make_border_map.py          // 生成边界图
│   │   │   ├── make_shrink_map.py          // 生成收缩图
│   │   │   ├── operators.py                // 图像基本操作，如读取和归一化
│   │   │   ├── randaugment.py              // 随机数据增广操作
│   │   │   ├── random_crop_data.py         // 随机裁剪
│   │   │   ├── rec_img_aug.py              // 文本识别的数据扩充
│   │   │   └── sast_process.py             // SAST 算法的数据处理步骤
│   │   ├── __init__.py                     // 构造 dataloader 相关代码
│   │   ├── lmdb_dataset.py                 // 读取lmdb数据集的 dataset
│   │   ├── simple_dataset.py               // 读取文本格式存储数据集的 dataset
│   ├── losses                              // 损失函数
│   │   ├── __init__.py                     // 构造 loss 相关代码
│   │   ├── cls_loss.py                     // 方向分类器 loss
│   │   ├── det_basic_loss.py               // 检测基础 loss
│   │   ├── det_db_loss.py                  // DB loss
│   │   ├── det_east_loss.py                // EAST loss
│   │   ├── det_sast_loss.py                // SAST loss
│   │   ├── rec_ctc_loss.py                 // CTC loss
│   │   ├── rec_att_loss.py                 // Attention loss
│   ├── metrics                             // 评估指标
│   │   ├── __init__.py                     // 构造 metric 相关代码
│   │   ├── cls_metric.py                   // 方向分类器 metric
│   │   ├── det_metric.py                   // 检测 metric
    │   ├── eval_det_iou.py                 // 检测 iou 相关
│   │   ├── rec_metric.py                   // 识别 metric
│   ├── modeling                            // 组网相关
│   │   ├── architectures                   // 网络
│   │   │   ├── __init__.py                 // 构造 model 相关代码
│   │   │   ├── base_model.py               // 组网代码
│   │   ├── backbones                       // 骨干网络
│   │   │   ├── __init__.py                 // 构造 backbone 相关代码
│   │   │   ├── det_mobilenet_v3.py         // 检测 mobilenet_v3
│   │   │   ├── det_resnet_vd.py            // 检测 resnet
│   │   │   ├── det_resnet_vd_sast.py       // 检测 SAST算法的resnet backbone
│   │   │   ├── rec_mobilenet_v3.py         // 识别 mobilenet_v3
│   │   │   └── rec_resnet_vd.py            // 识别 resnet
│   │   ├── necks                           // 颈函数
│   │   │   ├── __init__.py                 // 构造 neck 相关代码
│   │   │   ├── db_fpn.py                   // 标准 fpn 网络
│   │   │   ├── east_fpn.py                 // EAST 算法的 fpn 网络
│   │   │   ├── sast_fpn.py                 // SAST 算法的 fpn 网络
│   │   │   ├── rnn.py                      // 识别 序列编码
│   │   ├── heads                           // 头函数
│   │   │   ├── __init__.py                 // 构造 head 相关代码
│   │   │   ├── cls_head.py                 // 方向分类器 分类头
│   │   │   ├── det_db_head.py              // DB 检测头
│   │   │   ├── det_east_head.py            // EAST 检测头
│   │   │   ├── det_sast_head.py            // SAST 检测头
│   │   │   ├── rec_ctc_head.py             // 识别 ctc
│   │   │   ├── rec_att_head.py             // 识别 attention
│   │   ├── transforms                      // 图像变换
│   │   │   ├── __init__.py                 // 构造 transform 相关代码
│   │   │   └── tps.py                      // TPS 变换
│   ├── optimizer                           // 优化器
│   │   ├── __init__.py                     // 构造 optimizer 相关代码
│   │   └── learning_rate.py                // 学习率衰减
│   │   └── optimizer.py                    // 优化器
│   │   └── regularizer.py                  // 网络正则化
│   ├── postprocess                         // 后处理
│   │   ├── cls_postprocess.py              // 方向分类器 后处理
│   │   ├── db_postprocess.py               // DB 后处理
│   │   ├── east_postprocess.py             // EAST 后处理
│   │   ├── locality_aware_nms.py           // NMS
│   │   ├── rec_postprocess.py              // 识别网络 后处理
│   │   └── sast_postprocess.py             // SAST 后处理
│   └── utils                               // 工具
│       ├── dict                            // 小语种字典
│            ....  
│       ├── ic15_dict.txt                   // 英文数字字典，区分大小写
│       ├── ppocr_keys_v1.txt               // 中文字典，用于训练中文模型
│       ├── logging.py                      // logger
│       ├── save_load.py                    // 模型保存和加载函数
│       ├── stats.py                        // 统计
│       └── utility.py                      // 工具函数
├── tools
│   ├── eval.py                             // 评估函数
│   ├── export_model.py                     // 导出 inference 模型
│   ├── infer                               // 基于预测引擎预测
│   │   ├── predict_cls.py
│   │   ├── predict_det.py
│   │   ├── predict_rec.py
│   │   ├── predict_system.py
│   │   └── utility.py
│   ├── infer_cls.py                        // 基于训练引擎 预测分类
│   ├── infer_det.py                        // 基于训练引擎 预测检测
│   ├── infer_rec.py                        // 基于训练引擎 预测识别
│   ├── program.py                          // 整体流程
│   ├── test_hubserving.py
│   └── train.py                            // 启动训练
├── paddleocr.py
├── README_ch.md                            // 中文说明文档
├── README_en.md                            // 英文说明文档
├── README.md                               // 主页说明文档
├── requirements.txt                        // 安装依赖
├── setup.py                                // whl包打包脚本
├── train.sh                                // 启动训练脚本
