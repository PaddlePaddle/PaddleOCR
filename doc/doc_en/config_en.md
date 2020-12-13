## Optional parameter list

The following list can be viewed through `--help`

|         FLAG             |     Supported script    |        Use        |      Defaults       |         Note         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  Specify configuration file to use  |  None  |  **Please refer to the parameter introduction for configuration file usage** |
|          -o              |      ALL       |  set configuration options  |  None  |  Configuration using -o has higher priority than the configuration file selected with -c. E.g: -o Global.use_gpu=false |

## INTRODUCTION TO GLOBAL PARAMETERS OF CONFIGURATION FILE

Take rec_chinese_lite_train_v2.0.yml as an example
### Global

|         Parameter             |            Use                |      Defaults       |            Note            |
| :----------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      use_gpu             |    Set using GPU or not           |       true        |                \                 |
|      epoch_num           |    Maximum training epoch number             |       500        |                \                 |
|      log_smooth_window   |    Log queue length, the median value in the queue each time will be printed           |       20          |                \                 |
|      print_batch_step    |    Set print log interval         |       10          |                \                 |
|      save_model_dir      |    Set model save path        |  output/{算法名称}  |                \                 |
|      save_epoch_step     |    Set model save interval        |       3           |                \                 |
|      eval_batch_step     |    Set the model evaluation interval        | 2000 or [1000, 2000]        | runing evaluation every 2000 iters or evaluation is run every 2000 iterations after the 1000th iteration   |
|      cal_metric_during_train     |    Set whether to evaluate the metric during the training process. At this time, the metric of the model under the current batch is evaluated        |       true         |                \                 |
|      load_static_weights     |   Set whether the pre-training model is saved in static graph mode (currently only required by the detection algorithm)        |       true         |                \                 |
|      pretrained_model    |    Set the path of the pre-trained model      |  ./pretrain_models/CRNN/best_accuracy  |  \          |
|      checkpoints         |    set model parameter path            |       None        |   Used to load parameters after interruption to continue training|
|      use_visualdl  |    Set whether to enable visualdl for visual log display |          False        |    [Tutorial](https://www.paddlepaddle.org.cn/paddle/visualdl) |
|      infer_img            |    Set inference image path or folder path     |       ./infer_img | \|
|      character_dict_path |    Set dictionary path            |  ./ppocr/utils/ppocr_keys_v1.txt  |    \                 |
|      max_text_length     |    Set the maximum length of text        |       25          |                \                 |
|      character_type      |    Set character type            |       ch          |    en/ch, the default dict will be used for en, and the custom dict will be used for ch |
|      use_space_char     |    Set whether to recognize spaces             |        True      |          Only support in character_type=ch mode                 |
|      label_list          |    Set the angle supported by the direction classifier       |    ['0','180']    |     Only valid in angle classifier model |
|      save_res_path          |    Set the save address of the test model results       |    ./output/det_db/predicts_db.txt    |     Only valid in the text detection model |

### Optimizer ([ppocr/optimizer](../../ppocr/optimizer))

|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         Optimizer class name          |  Adam  |  Currently supports`Momentum`,`Adam`,`RMSProp`, see [ppocr/optimizer/optimizer.py](../../ppocr/optimizer/optimizer.py)  |
|      beta1           |    Set the exponential decay rate for the 1st moment estimates  |       0.9         |               \             |
|      beta2           |    Set the exponential decay rate for the 2nd moment estimates  |     0.999         |               \             |
|      **lr**                |         Set the learning rate decay method       |   -    |       \  |
|        name    |      Learning rate decay class name   |         Cosine       | Currently supports`Linear`,`Cosine`,`Step`,`Piecewise`, see[ppocr/optimizer/learning_rate.py](../../ppocr/optimizer/learning_rate.py) |
|        learning_rate      |    Set the base learning rate        |       0.001      |  \        |
|      **regularizer**      |  Set network regularization method        |       -      | \        |
|        name      |    Regularizer class name      |       L2     |  Currently support`L1`,`L2`, see[ppocr/optimizer/regularizer.py](../../ppocr/optimizer/regularizer.py)        |
|        factor      |    Learning rate decay coefficient       |       0.00004     |  \        |


### Architecture ([ppocr/modeling](../../ppocr/modeling))
In ppocr, the network is divided into four stages: Transform, Backbone, Neck and Head

|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      model_type        |         Network Type          |  rec  |  Currently support`rec`,`det`,`cls`  |
|      algorithm           |    Model name  |       CRNN         |               See [algorithm_overview](./algorithm_overview.md) for the support list             |
|      **Transform**           |    Set the transformation method  |       -       |               Currently only recognition algorithms are supported, see [ppocr/modeling/transform](../../ppocr/modeling/transform) for details            |
|        name    |      Transformation class name   |         TPS       | Currently supports `TPS` |
|        num_fiducial      |   Number of TPS control points        |       20      |  Ten on the top and bottom       |
|        loc_lr      |    Localization network learning rate        |       0.1      |  \      |
|        model_name      |    Localization network size        |       small      |  Currently support`small`,`large`       |
|      **Backbone**      |  Set the network backbone class name        |       -      | see [ppocr/modeling/backbones](../../ppocr/modeling/backbones)        |
|        name      |    backbone class name       |       ResNet     | Currently support`MobileNetV3`,`ResNet`        |
|        layers      |    resnet layers       |       34     |  Currently support18,34,50,101,152,200       |
|        model_name      |    MobileNetV3 network size       |       small     |  Currently support`small`,`large`       |
|      **Neck**      |  Set network neck        |       -      | see[ppocr/modeling/necks](../../ppocr/modeling/necks)        |
|        name      |    neck class name       |       SequenceEncoder     | Currently support`SequenceEncoder`,`DBFPN`        |
|        encoder_type      |    SequenceEncoder encoder type       |       rnn     |  Currently support`reshape`,`fc`,`rnn`       |
|        hidden_size      |   rnn number of internal units       |       48     |  \      |
|        out_channels      |   Number of DBFPN output channels       |       256     |  \      |
|      **Head**      |  Set the network head        |       -      | see[ppocr/modeling/heads](../../ppocr/modeling/heads)        |
|        name      |    head class name       |       CTCHead     | Currently support`CTCHead`,`DBHead`,`ClsHead`        |
|        fc_decay      |    CTCHead regularization coefficient       |       0.0004     |  \      |
|        k      |   DBHead binarization coefficient       |       50     |  \      |
|        class_dim      |   ClsHead output category number       |       2     |  \      |


### Loss ([ppocr/losses](../../ppocr/losses))

|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         loss class name          |  CTCLoss  |  Currently support`CTCLoss`,`DBLoss`,`ClsLoss`  |
|      balance_loss        |        Whether to balance the number of positive and negative samples in DBLossloss (using OHEM)         |  True  |  \  |
|      ohem_ratio        |        The negative and positive sample ratio of OHEM in DBLossloss         |  3  |  \  |
|      main_loss_type        |        The loss used by shrink_map in DBLossloss        |  DiceLoss  |  Currently support`DiceLoss`,`BCELoss`  |
|      alpha        |        The coefficient of shrink_map_loss in DBLossloss       |  5  |  \  |
|      beta        |        The coefficient of threshold_map_loss in DBLossloss       |  10  |  \  |

### PostProcess ([ppocr/postprocess](../../ppocr/postprocess))

|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         Post-processing class name          |  CTCLabelDecode  |  Currently support`CTCLoss`,`AttnLabelDecode`,`DBPostProcess`,`ClsPostProcess`  |
|      thresh        |        The threshold for binarization of the segmentation map in DBPostProcess         |  0.3  |  \  |
|      box_thresh        |        The threshold for filtering output boxes in DBPostProcess. Boxes below this threshold will not be output         |  0.7  |  \  |
|      max_candidates        |        The maximum number of text boxes output in DBPostProcess        |  1000  |   |
|      unclip_ratio        |        The unclip ratio of the text box in DBPostProcess       |  2.0  |  \  |

### Metric ([ppocr/metrics](../../ppocr/metrics))

|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         Metric method name          |  CTCLabelDecode  |  Currently support`DetMetric`,`RecMetric`,`ClsMetric`  |
|      main_indicator        |        Main indicators, used to select the best model        |  acc |  For the detection method is hmean, the recognition and classification method is acc  |

### Dataset  ([ppocr/data](../../ppocr/data))
|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      **dataset**        |         Return one sample per iteration          |  -  |  -  |
|      name        |        dataset class name         |  SimpleDataSet |   Currently support`SimpleDataSet`,`LMDBDateSet`  |
|      data_dir        |        Image folder path        |  ./train_data |  \  |
|      label_file_list        |        Groundtruth file path         |  ["./train_data/train_list.txt"] | This parameter is not required when dataset is LMDBDateSet   |
|      ratio_list        |        Ratio of data set         |  [1.0] | If there are two train_lists in label_file_list and ratio_list is [0.4,0.6], 40% will be sampled from train_list1, and 60% will be sampled from train_list2 to combine the entire dataset   |
|      transforms        |        List of methods to transform images and labels         |  [DecodeImage,CTCLabelEncode,RecResizeImg,KeepKeys] |   see[ppocr/data/imaug](../../ppocr/data/imaug)  |
|      **loader**        |        dataloader related         |  - |   |
|      shuffle        |        Does each epoch disrupt the order of the data set         |  True | \  |
|      batch_size_per_card        |        Single card batch size during training         |  256 | \  |
|      drop_last        |        Whether to discard the last incomplete mini-batch because the number of samples in the data set cannot be divisible by batch_size        |  True | \  |
|      num_workers        |        The number of sub-processes used to load data, if it is 0, the sub-process is not started, and the data is loaded in the main process       |  8 | \  |
