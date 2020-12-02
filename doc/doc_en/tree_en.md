# Overall directory structure

The overall directory structure of PaddleOCR is introduced as follows:

```
PaddleOCR
├── configs   // configuration file, you can select model structure and modify hyperparameters through yml file
│   ├── cls   // Related configuration files of direction classifier
│   │   ├── cls_mv3.yml               // training configuration related, including backbone network, head, loss, optimizer
│   │   └── cls_reader.yml            // Data reading related, data reading method, data storage path
│   ├── det   // Detection related configuration files
│   │   ├── det_db_icdar15_reader.yml  // data read
│   │   ├── det_mv3_db.yml             // training configuration
│   │   ...  
│   └── rec   // Identify related configuration files
│       ├── rec_benchmark_reader.yml      // LMDB format data reading related
│       ├── rec_chinese_common_train.yml  // General Chinese training configuration
│       ├── rec_icdar15_reader.yml        // simple data reading related, including data reading function, data path, label file
│       ...  
├── deploy   // deployment related
│   ├── android_demo // android_demo
│   │   ...
│   ├── cpp_infer    // C++ infer
│   │   ├── CMakeLists.txt    // Cmake file
│   │   ├── docs              // documentation
│   │   │   └── windows_vs2019_build.md
│   │   ├── include  
│   │   │   ├── clipper.h     // clipper library
│   │   │   ├── config.h      // infer configuration
│   │   │   ├── ocr_cls.h     // direction classifier
│   │   │   ├── ocr_det.h     // text detection
│   │   │   ├── ocr_rec.h     // text recognition
│   │   │   ├── postprocess_op.h  // postprocess after detection
│   │   │   ├── preprocess_op.h   // preprocess detection
│   │   │   └── utility.h         // tools
│   │   ├── readme.md     // documentation
│   │   ├── ...
│   │   ├── src              // source file
│   │   │   ├── clipper.cpp  
│   │   │   ├── config.cpp
│   │   │   ├── main.cpp
│   │   │   ├── ocr_cls.cpp
│   │   │   ├── ocr_det.cpp
│   │   │   ├── ocr_rec.cpp
│   │   │   ├── postprocess_op.cpp
│   │   │   ├── preprocess_op.cpp
│   │   │   └── utility.cpp
│   │   └── tools      // compile and execute script
│   │       ├── build.sh   // compile script
│   │       ├── config.txt // configuration file
│   │       └── run.sh     // Test startup script
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
│   │   ├── ocr_det   // text detection
│   │   │   ├── config.json  // serving configuration
│   │   │   ├── __init__.py  
│   │   │   ├── module.py    // prediction model
│   │   │   └── params.py    // prediction parameters
│   │   ├── ocr_rec   // text recognition
│   │   │   ├── config.json
│   │   │   ├── __init__.py
│   │   │   ├── module.py
│   │   │   └── params.py
│   │   └── ocr_system  // system forecast
│   │       ├── config.json
│   │       ├── __init__.py
│   │       ├── module.py
│   │       └── params.py
│   ├── imgs  // prediction picture
│   │   ├── cpp_infer_pred_12.png
│   │   └── demo.png
│   ├── ios_demo  // ios demo
│   │   ...
│   ├── lite      // lite deployment
│   │   ├── cls_process.cc  // direction classifier data processing
│   │   ├── cls_process.h
│   │   ├── config.txt      // check configuration parameters
│   │   ├── crnn_process.cc  // crnn data processing
│   │   ├── crnn_process.h
│   │   ├── db_post_process.cc  // db data processing
│   │   ├── db_post_process.h
│   │   ├── Makefile            // compile file
│   │   ├── ocr_db_crnn.cc      // series prediction
│   │   ├── prepare.sh          // data preparation
│   │   ├── readme.md        // documentation
│   │   ...
│   ├── pdserving  // pdserving deployment
│   │   ├── det_local_server.py  // fast detection version, easy deployment and fast prediction
│   │   ├── det_web_server.py    // Full version of detection, high stability and distributed deployment
│   │   ├── ocr_local_server.py  // detection + identification quick version
│   │   ├── ocr_web_client.py    // client
│   │   ├── ocr_web_server.py    // detection + identification full version
│   │   ├── readme.md            // documentation
│   │   ├── rec_local_server.py  // recognize quick version
│   │   └── rec_web_server.py    // Identify the full version
│   └── slim  
│       └── quantization         // quantization related
│           ├── export_model.py  // export model
│           ├── quant.py         // quantization
│           └── README.md        // Documentation
├── doc  // Documentation tutorial
│   ...
├── paddleocr.py
├── ppocr            // network core code
│   ├── data         // data processing
│   │   ├── cls   // direction classifier
│   │   │   ├── dataset_traversal.py  // Data transmission, define data reader, read data and form batch
│   │   │   └── randaugment.py        // Random data augmentation operation
│   │   ├── det   // detection
│   │   │   ├── data_augment.py       // data augmentation operation
│   │   │   ├── dataset_traversal.py  // Data transmission, define data reader, read data and form batch
│   │   │   ├── db_process.py         // db data processing
│   │   │   ├── east_process.py       // east data processing
│   │   │   ├── make_border_map.py    // Generate boundary map
│   │   │   ├── make_shrink_map.py    // Generate shrink map
│   │   │   ├── random_crop_data.py   // random crop
│   │   │   └── sast_process.py       // sast data processing
│   │   ├── reader_main.py   // main function of data reader
│   │   └── rec  // recognation
│   │       ├── dataset_traversal.py  // Data transmission, define data reader, including LMDB_Reader and Simple_Reader
│   │       └── img_tools.py          // Data processing related, including data normalization and disturbance
│   ├── __init__.py
│   ├── modeling       // networking related
│   │   ├── architectures  // Model architecture, which defines the various modules required by the model
│   │   │   ├── cls_model.py  // direction classifier
│   │   │   ├── det_model.py  // detection
│   │   │   └── rec_model.py  // recognition
│   │   ├── backbones  // backbone network
│   │   │   ├── det_mobilenet_v3.py  // detect mobilenet_v3
│   │   │   ├── det_resnet_vd.py  
│   │   │   ├── det_resnet_vd_sast.py
│   │   │   ├── rec_mobilenet_v3.py  // recognize mobilenet_v3
│   │   │   ├── rec_resnet_fpn.py
│   │   │   └── rec_resnet_vd.py
│   │   ├── common_functions.py      // common functions
│   │   ├── heads  
│   │   │   ├── cls_head.py          // class header
│   │   │   ├── det_db_head.py       // db detection head
│   │   │   ├── det_east_head.py     // east detection head
│   │   │   ├── det_sast_head.py     // sast detection head
│   │   │   ├── rec_attention_head.py  // recognition attention
│   │   │   ├── rec_ctc_head.py        // recognition ctc
│   │   │   ├── rec_seq_encoder.py     // recognition sequence code
│   │   │   ├── rec_srn_all_head.py    // srn related
│   │   │   └── self_attention         // srn attention
│   │   │       └── model.py
│   │   ├── losses    // loss function
│   │   │   ├── cls_loss.py            // Directional classifier loss function
│   │   │   ├── det_basic_loss.py      // detect basic loss
│   │   │   ├── det_db_loss.py         // DB loss
│   │   │   ├── det_east_loss.py       // EAST loss
│   │   │   ├── det_sast_loss.py       // SAST loss
│   │   │   ├── rec_attention_loss.py  // attention loss
│   │   │   ├── rec_ctc_loss.py        // ctc loss
│   │   │   └── rec_srn_loss.py        // srn loss
│   │   └── stns     // Spatial transformation network
│   │       └── tps.py   // TPS conversion
│   ├── optimizer.py  // optimizer
│   ├── postprocess   // post-processing
│   │   ├── db_postprocess.py     // DB postprocess
│   │   ├── east_postprocess.py   // East postprocess
│   │   ├── lanms                 // lanms related
│   │   │   ...
│   │   ├── locality_aware_nms.py // nms
│   │   └── sast_postprocess.py   // sast post-processing
│   └── utils  // tools
│       ├── character.py       // Character processing, including text encoding and decoding, and calculation of prediction accuracy
│       ├── check.py           // parameter loading check
│       ├── ic15_dict.txt      // English number dictionary, case sensitive
│       ├── ppocr_keys_v1.txt  // Chinese dictionary, used to train Chinese models
│       ├── save_load.py       // model save and load function
│       ├── stats.py           // Statistics
│       └── utility.py         // Tool functions, including related check tools such as whether the input parameters are legal
├── README_en.md    // documentation
├── README.md
├── requirments.txt // installation dependencies
├── setup.py        // whl package packaging script
└── tools           // start tool
    ├── eval.py                 // evaluation function
    ├── eval_utils              // evaluation tools
    │   ├── eval_cls_utils.py   // category related
    │   ├── eval_det_iou.py     // detect iou related
    │   ├── eval_det_utils.py   // detection related
    │   ├── eval_rec_utils.py   // recognition related
    │   └── __init__.py
    ├── export_model.py         // export infer model
    ├── infer                   // Forecast based on prediction engine
    │   ├── predict_cls.py  
    │   ├── predict_det.py
    │   ├── predict_rec.py
    │   ├── predict_system.py
    │   └── utility.py
    ├── infer_cls.py            // Predict classification based on training engine
    ├── infer_det.py            // Predictive detection based on training engine
    ├── infer_rec.py            // Predictive recognition based on training engine
    ├── program.py              //  overall process
    ├── test_hubserving.py  
    └── train.py                // start training

```
