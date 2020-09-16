# 整体目录结构

PaddleOCR 的整理目录结构介绍如下：

```
PaddleOCR
├── configs   // 配置文件
│   ├── cls   // 方向分类器相关配置文件
│   │   ├── cls_mv3.yml               // 训练配置
│   │   └── cls_reader.yml            // 数据读取
│   ├── det   // 检测相关配置文件
│   │   ├── det_db_icdar15_reader.yml  // 数据读取
│   │   ├── det_mv3_db.yml             // 训练配置
│   │   ...                            // 略
│   └── rec   // 识别相关配置文件
│       ├── rec_benchmark_reader.yml      // LMDB 数据读取
│       ├── rec_chinese_common_train.yml  // 通用中文训练配置
│       ├── rec_icdar15_reader.yml        // 普通数据读取
│       ...                               // 略
├── deploy   // 部署相关
│   ├── android_demo // android_demo 
│   │   ...
│   ├── cpp_infer    // C++ infer
│   │   ├── CMakeLists.txt    // Cmake 文件
│   │   ├── docs              // 说明文档
│   │   │   └── windows_vs2019_build.md
│   │   ├── include           // 头文件
│   │   │   ├── clipper.h     // clipper 库
│   │   │   ├── config.h      // 预测配置
│   │   │   ├── ocr_cls.h     // 方向分类器
│   │   │   ├── ocr_det.h     // 文字检测
│   │   │   ├── ocr_rec.h     // 文字识别
│   │   │   ├── postprocess_op.h  // 检测后处理
│   │   │   ├── preprocess_op.h   // 检测预处理
│   │   │   └── utility.h         // 工具
│   │   ├── readme.md     // 说明文档
│   │   ├── ...
│   │   ├── src              // 源文件
│   │   │   ├── clipper.cpp  
│   │   │   ├── config.cpp
│   │   │   ├── main.cpp
│   │   │   ├── ocr_cls.cpp
│   │   │   ├── ocr_det.cpp
│   │   │   ├── ocr_rec.cpp
│   │   │   ├── postprocess_op.cpp
│   │   │   ├── preprocess_op.cpp
│   │   │   └── utility.cpp
│   │   └── tools      // 工具
│   │       ├── build.sh   // 编译脚本
│   │       ├── config.txt // 配置文件
│   │       └── run.sh     // 测试启动脚本
│   ├── docker
│   │   └── hubserving
│   │       ├── cpu
│   │       │   └── Dockerfile
│   │       ├── gpu
│   │       │   └── Dockerfile
│   │       ├── README_cn.md
│   │       ├── README.md
│   │       └── sample_request.txt
│   ├── hubserving  // hubserving
│   │   ├── ocr_det   // 文字检测
│   │   │   ├── config.json  // serving 配置
│   │   │   ├── __init__.py  
│   │   │   ├── module.py    // 预测模型
│   │   │   └── params.py    // 预测参数
│   │   ├── ocr_rec   // 文字识别
│   │   │   ├── config.json
│   │   │   ├── __init__.py
│   │   │   ├── module.py
│   │   │   └── params.py
│   │   └── ocr_system  // 系统预测
│   │       ├── config.json
│   │       ├── __init__.py
│   │       ├── module.py
│   │       └── params.py
│   ├── imgs  // 预测图片
│   │   ├── cpp_infer_pred_12.png
│   │   └── demo.png
│   ├── ios_demo  // ios demo
│   │   ...
│   ├── lite      // lite 部署
│   │   ├── cls_process.cc  // 方向分类器数据处理
│   │   ├── cls_process.h
│   │   ├── config.txt      // 检测配置参数
│   │   ├── crnn_process.cc  // crnn数据处理
│   │   ├── crnn_process.h
│   │   ├── db_post_process.cc  // db数据处理
│   │   ├── db_post_process.h
│   │   ├── Makefile            // 编译文件
│   │   ├── ocr_db_crnn.cc      // 串联预测
│   │   ├── prepare.sh          // 数据准备
│   │   ├── readme.md        // 说明文档
│   │   ...
│   ├── pdserving  // pdserving 部署
│   │   ├── det_local_server.py
│   │   ├── det_web_server.py
│   │   ├── ocr_local_server.py
│   │   ├── ocr_web_client.py
│   │   ├── ocr_web_server.py
│   │   ├── readme.md
│   │   ├── rec_local_server.py
│   │   └── rec_web_server.py
│   └── slim     
│       └── quantization         // 量化相关
│           ├── export_model.py  // 导出模型
│           ├── quant.py         // 量化
│           └── README.md        // 说明文档
├── doc  // 文档说明
│   ├── datasets  // 数据集
│   │   ...
│   ├── doc_ch    // 中文文档
│   │   ...
│   ├── doc_en
│   │   ...       // 英文文档
│   ├── imgs      // 中文测试图片
│   │   ...
│   ├── imgs_en   // 英文测试图片
│   │   ...
│   ├── imgs_results  // 预测结果可视化
│   │   ...
│   ├── imgs_results_vis2  // 预测结果可视化效果2
│   │   ...
│   ├── imgs_words    // 行文本测试图片
│   │   ├── ch        // 中文文本
│   │   │   ...
│   │   └── en        // 英文文本
│   │       ...
│   ├── simfang.ttf   // 可视化字体
│   └── tricks        // 训练技巧
│       ...
├── LICENSE
├── MANIFEST.in
├── paddleocr.py
├── ppocr            // 网络核心代码
│   ├── data         // 数据处理
│   │   ├── cls   // 方向分类器
│   │   │   ├── dataset_traversal.py  // 数据传输
│   │   │   └── randaugment.py        // 增广
│   │   ├── det   // 检测
│   │   │   ├── data_augment.py       // 增广
│   │   │   ├── dataset_traversal.py  // 数据传输
│   │   │   ├── db_process.py         // db 数据处理
│   │   │   ├── east_process.py       // east 数据处理
│   │   │   ├── make_border_map.py    // 生成边界图
│   │   │   ├── make_shrink_map.py    // 生成收缩图
│   │   │   ├── random_crop_data.py   // 随机切割
│   │   │   └── sast_process.py       // sast 数据处理
│   │   ├── reader_main.py   // 数据读取器主函数
│   │   └── rec  // 识别
│   │       ├── dataset_traversal.py  // 数据传输
│   │       └── img_tools.py          // 数据处理
│   ├── __init__.py
│   ├── modeling       // 组网相关
│   │   ├── architectures  // 模型架构
│   │   │   ├── cls_model.py  // 方向分类器
│   │   │   ├── det_model.py  // 检测
│   │   │   └── rec_model.py  // 识别
│   │   ├── backbones  // 骨干网络
│   │   │   ├── det_mobilenet_v3.py  // 检测 mobilenet_v3
│   │   │   ├── det_resnet_vd.py   
│   │   │   ├── det_resnet_vd_sast.py
│   │   │   ├── rec_mobilenet_v3.py  // 识别 mobilenet_v3
│   │   │   ├── rec_resnet_fpn.py
│   │   │   └── rec_resnet_vd.py
│   │   ├── common_functions.py      // 公共函数
│   │   ├── heads      // 头函数
│   │   │   ├── cls_head.py          // 分类头
│   │   │   ├── det_db_head.py       // db 检测头
│   │   │   ├── det_east_head.py     // east 检测头
│   │   │   ├── det_sast_head.py     // sast 检测头
│   │   │   ├── rec_attention_head.py  // 识别 attention
│   │   │   ├── rec_ctc_head.py        // 识别 ctc
│   │   │   ├── rec_seq_encoder.py     // 识别 序列编码
│   │   │   ├── rec_srn_all_head.py    // 识别 srn 相关
│   │   │   └── self_attention         // srn attention
│   │   │       └── model.py
│   │   ├── losses    // 损失函数
│   │   │   ├── cls_loss.py            // 方向分类器损失函数
│   │   │   ├── det_basic_loss.py      // 检测基础loss
│   │   │   ├── det_db_loss.py         // DB loss
│   │   │   ├── det_east_loss.py       // EAST loss
│   │   │   ├── det_sast_loss.py       // SAST loss
│   │   │   ├── rec_attention_loss.py  // attention loss
│   │   │   ├── rec_ctc_loss.py        // ctc loss
│   │   │   └── rec_srn_loss.py        // srn loss
│   │   └── stns     // 空间变换网络
│   │       └── tps.py   // TPS 变换
│   ├── optimizer.py  // 优化器
│   ├── postprocess   // 后处理
│   │   ├── db_postprocess.py    // DB 后处理
│   │   ├── east_postprocess.py  // East 后处理
│   │   ├── lanms
│   │   │   ├── adaptor.cpp
│   │   │   ├── include
│   │   │   │   ├── clipper
│   │   │   │   │   ├── clipper.cpp
│   │   │   │   │   └── clipper.hpp
│   │   │   │   └── pybind11
│   │   │   │       ├── attr.h
│   │   │   │       ├── buffer_info.h
│   │   │   │       ├── cast.h
│   │   │   │       ├── chrono.h
│   │   │   │       ├── class_support.h
│   │   │   │       ├── common.h
│   │   │   │       ├── complex.h
│   │   │   │       ├── descr.h
│   │   │   │       ├── eigen.h
│   │   │   │       ├── embed.h
│   │   │   │       ├── eval.h
│   │   │   │       ├── functional.h
│   │   │   │       ├── numpy.h
│   │   │   │       ├── operators.h
│   │   │   │       ├── options.h
│   │   │   │       ├── pybind11.h
│   │   │   │       ├── pytypes.h
│   │   │   │       ├── stl_bind.h
│   │   │   │       ├── stl.h
│   │   │   │       └── typeid.h
│   │   │   ├── lanms.h
│   │   │   ├── __main__.py
│   │   │   └── Makefile
│   │   ├── locality_aware_nms.py
│   │   └── sast_postprocess.py
│   └── utils  // 工具
│       ├── character.py       // 字符处理
│       ├── check.py           // 输入参数检查
│       ├── ic15_dict.txt      // 英文数字字典
│       ├── ppocr_keys_v1.txt  // 中文字典
│       ├── save_load.py       // 模型保存和加载
│       ├── stats.py           // 统计
│       └── utility.py         // 工具函数
├── README_en.md    // 说明文档
├── README.md
├── requirments.txt // 安装依赖
├── setup.py        // 
└── tools           // 启动工具
    ├── eval.py                 // 评估函数
    ├── eval_utils              // 评估工具
    │   ├── eval_cls_utils.py   // 分类相关
    │   ├── eval_det_iou.py     // 检测 iou 相关
    │   ├── eval_det_utils.py   // 检测相关
    │   ├── eval_rec_utils.py   // 识别相关
    │   └── __init__.py
    ├── export_model.py         // 导出 infer 模型
    ├── infer                   // 基于预测引擎预测
    │   ├── predict_cls.py      
    │   ├── predict_det.py
    │   ├── predict_rec.py
    │   ├── predict_system.py
    │   └── utility.py
    ├── infer_cls.py            // 基于训练引擎 预测分类
    ├── infer_det.py            // 基于训练引擎 预测检测
    ├── infer_rec.py            // 基于训练引擎 预测识别
    ├── program.py              // 整体流程
    ├── test_hubserving.py      
    └── train.py                // 启动训练

```
