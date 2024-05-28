# Overall directory structure

The overall directory structure of PaddleOCR is introduced as follows:


```
PaddleOCR
├── configs                                 // Configuration file, you can config the model structure and modify the hyperparameters through the yml file
│   ├── cls                                 // Angle classifier config files
│   │   ├── cls_mv3.yml                     // Training config, including backbone network, head, loss, optimizer and data
│   ├── det                                 // Text detection config files
│   │   ├── det_mv3_db.yml                  // Training config
│   │   ...
│   └── rec                                 // Text recognition config files
│       ├── rec_mv3_none_bilstm_ctc.yml     // CRNN config
│       ...
├── deploy                                  // Depoly
│   ├── android_demo                        // Android demo
│   │   ...
│   ├── cpp_infer                           // C++ infer
│   │   ├── CMakeLists.txt                  // Cmake file
│   │   ├── docs                            // Docs
│   │   │   └── windows_vs2019_build.md
│   │   ├── include                         // Head Files
│   │   │   ├── clipper.h                   // clipper
│   │   │   ├── config.h                    // Inference config
│   │   │   ├── ocr_cls.h                   // Angle class
│   │   │   ├── ocr_det.h                   // Text detection
│   │   │   ├── ocr_rec.h                   // Text recognition
│   │   │   ├── postprocess_op.h            // Post-processing
│   │   │   ├── preprocess_op.h             // Pre-processing
│   │   │   └── utility.h                   // tools
│   │   ├── readme.md                       // Documentation
│   │   ├── ...
│   │   ├── src                             // Source code files
│   │   │   ├── clipper.cpp
│   │   │   ├── config.cpp
│   │   │   ├── main.cpp
│   │   │   ├── ocr_cls.cpp
│   │   │   ├── ocr_det.cpp
│   │   │   ├── ocr_rec.cpp
│   │   │   ├── postprocess_op.cpp
│   │   │   ├── preprocess_op.cpp
│   │   │   └── utility.cpp
│   │   └── tools                           // Compile and execute script
│   │       ├── build.sh                    // Compile script
│   │       ├── config.txt                  // Config file
│   │       └── run.sh                      // Execute script
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
│   │   ├── ocr_cls                         // Angle class
│   │   │   ├── config.json                 // Serving config
│   │   │   ├── __init__.py
│   │   │   ├── module.py                   // Model
│   │   │   └── params.py                   // Parameters
│   │   ├── ocr_det                         // Text detection
│   │   │   ├── config.json                 // serving config
│   │   │   ├── __init__.py
│   │   │   ├── module.py                   // Model
│   │   │   └── params.py                   // Parameters
│   │   ├── ocr_rec                         // Text recognition
│   │   │   ├── config.json
│   │   │   ├── __init__.py
│   │   │   ├── module.py
│   │   │   └── params.py
│   │   └── ocr_system                      // Inference System
│   │       ├── config.json
│   │       ├── __init__.py
│   │       ├── module.py
│   │       └── params.py
│   ├── imgs                                // Inference images
│   │   ├── cpp_infer_pred_12.png
│   │   └── demo.png
│   ├── ios_demo                            // IOS demo
│   │   ...
│   ├── lite                                // Lite depoly
│   │   ├── cls_process.cc                  // Pre-process for angle class
│   │   ├── cls_process.h
│   │   ├── config.txt                      // Config file
│   │   ├── crnn_process.cc                 // Pre-process for CRNN
│   │   ├── crnn_process.h
│   │   ├── db_post_process.cc              // Pre-process for DB
│   │   ├── db_post_process.h
│   │   ├── Makefile                        // Compile file
│   │   ├── ocr_db_crnn.cc                  // Inference system
│   │   ├── prepare.sh                      // Prepare bash script
│   │   ├── readme.md                       // Documentation
│   │   ...
│   ├── pdserving                           // Pdserving depoly
│   │   ├── det_local_server.py             // Text detection fast version, easy to deploy and fast to predict
│   │   ├── det_web_server.py               // Text detection full version, high stability distributed deployment
│   │   ├── ocr_local_server.py             // Text detection + recognition fast version
│   │   ├── ocr_web_client.py               // client
│   │   ├── ocr_web_server.py               // Text detection + recognition full version
│   │   ├── readme.md                       // Documentation
│   │   ├── rec_local_server.py             // Text recognition fast version
│   │   └── rec_web_server.py               // Text recognition full version
│   └── slim
│       └── quantization                    // Quantization
│           ├── export_model.py             // Export model
│           ├── quant.py                    // Quantization script
│           └── README.md                   // Documentation
├── doc                                     // Documentation and Tutorials
│   ...
├── ppocr                                   // Core code
│   ├── data                                // Data processing
│   │   ├── imaug                           // Image and label processing code
│   │   │   ├── text_image_aug              // Tia data augment for text recognition
│   │   │   │   ├── __init__.py
│   │   │   │   ├── augment.py              // Tia_distort,tia_stretch and tia_perspective
│   │   │   │   ├── warp_mls.py
│   │   │   ├── __init__.py
│   │   │   ├── east_process.py             // Data processing steps of EAST algorithm
│   │   │   ├── iaa_augment.py              // Data augmentation operations
│   │   │   ├── label_ops.py                // label encode operations
│   │   │   ├── make_border_map.py          // Generate boundary map
│   │   │   ├── make_shrink_map.py          // Generate shrink graph
│   │   │   ├── operators.py                // Basic image operations, such as reading and normalization
│   │   │   ├── randaugment.py              // Random data augmentation operation
│   │   │   ├── random_crop_data.py         // Random crop
│   │   │   ├── rec_img_aug.py              // Data augmentation for text recognition
│   │   │   └── sast_process.py             // Data processing steps of SAST algorithm
│   │   ├── __init__.py                     // Construct dataloader code
│   │   ├── lmdb_dataset.py                 // Read lmdb dataset
│   │   ├── simple_dataset.py               // Read the dataset stored in text format
│   ├── losses                              // Loss function
│   │   ├── __init__.py                     // Construct loss code
│   │   ├── cls_loss.py                     // Angle class loss
│   │   ├── det_basic_loss.py               // Text detection basic loss
│   │   ├── det_db_loss.py                  // DB loss
│   │   ├── det_east_loss.py                // EAST loss
│   │   ├── det_sast_loss.py                // SAST loss
│   │   ├── rec_ctc_loss.py                 // CTC loss
│   │   ├── rec_att_loss.py                 // Attention loss
│   ├── metrics                             // Metrics
│   │   ├── __init__.py                     // Construct metric code
│   │   ├── cls_metric.py                   // Angle class metric
│   │   ├── det_metric.py                   // Text detection metric
    │   ├── eval_det_iou.py                 // Text detection iou code
│   │   ├── rec_metric.py                   // Text recognition metric
│   ├── modeling                            // Network
│   │   ├── architectures                   // Architecture
│   │   │   ├── __init__.py                 // Construct model code
│   │   │   ├── base_model.py               // Base model
│   │   ├── backbones                       // backbones
│   │   │   ├── __init__.py                 // Construct backbone code
│   │   │   ├── det_mobilenet_v3.py         // Text detection mobilenet_v3
│   │   │   ├── det_resnet_vd.py            // Text detection resnet
│   │   │   ├── det_resnet_vd_sast.py       // Text detection resnet backbone of the SAST algorithm
│   │   │   ├── rec_mobilenet_v3.py         // Text recognition mobilenet_v3
│   │   │   └── rec_resnet_vd.py            // Text recognition resnet
│   │   ├── necks                           // Necks
│   │   │   ├── __init__.py                 // Construct neck code
│   │   │   ├── db_fpn.py                   // Standard fpn
│   │   │   ├── east_fpn.py                 // EAST algorithm fpn network
│   │   │   ├── sast_fpn.py                 // SAST algorithm fpn network
│   │   │   ├── rnn.py                      // Character recognition sequence encoding
│   │   ├── heads                           // Heads
│   │   │   ├── __init__.py                 // Construct head code
│   │   │   ├── cls_head.py                 // Angle class head
│   │   │   ├── det_db_head.py              // DB head
│   │   │   ├── det_east_head.py            // EAST head
│   │   │   ├── det_sast_head.py            // SAST head
│   │   │   ├── rec_ctc_head.py             // CTC head
│   │   │   ├── rec_att_head.py             // Attention head
│   │   ├── transforms                      // Transforms
│   │   │   ├── __init__.py                 // Construct transform code
│   │   │   └── tps.py                      // TPS transform
│   ├── optimizer                           // Optimizer
│   │   ├── __init__.py                     // Construct optimizer code
│   │   └── learning_rate.py                // Learning rate decay
│   │   └── optimizer.py                    // Optimizer
│   │   └── regularizer.py                  // Network regularization
│   ├── postprocess                         // Post-processing
│   │   ├── cls_postprocess.py              // Angle class post-processing
│   │   ├── db_postprocess.py               // DB post-processing
│   │   ├── east_postprocess.py             // EAST post-processing
│   │   ├── locality_aware_nms.py           // NMS
│   │   ├── rec_postprocess.py              // Text recognition post-processing
│   │   └── sast_postprocess.py             // SAST post-processing
│   └── utils                               // utils
│       ├── dict                            // Minor language dictionary
│            ....
│       ├── ic15_dict.txt                   // English number dictionary, case sensitive
│       ├── ppocr_keys_v1.txt               // Chinese dictionary for training Chinese models
│       ├── logging.py                      // logger
│       ├── save_load.py                    // Model saving and loading functions
│       ├── stats.py                        // Training status statistics
│       └── utility.py                      // Utility function
├── tools
│   ├── eval.py                             // Evaluation function
│   ├── export_model.py                     // Export inference model
│   ├── infer                               // Inference based on Inference engine
│   │   ├── predict_cls.py
│   │   ├── predict_det.py
│   │   ├── predict_rec.py
│   │   ├── predict_system.py
│   │   └── utility.py
│   ├── infer_cls.py                        // Angle classification inference based on training engine
│   ├── infer_det.py                        // Text detection inference based on training engine
│   ├── infer_rec.py                        // Text recognition inference based on training engine
│   ├── program.py                          // Inference system
│   ├── test_hubserving.py
│   └── train.py                            // Start training script
├── paddleocr.py
├── README_ch.md                            // Chinese documentation
├── README_en.md                            // English documentation
├── README.md                               // Home page documentation
├── requirements.txt                         // Requirements
├── setup.py                                // Whl package packaging script
├── train.sh                                // Start training bash script
