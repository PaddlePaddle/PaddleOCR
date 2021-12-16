# Configuration 

- [1. Optional Parameter List](#1-optional-parameter-list)
- [2. Intorduction to Global Parameters of Configuration File](#2-intorduction-to-global-parameters-of-configuration-file)
- [3. Multilingual Config File Generation](#3-multilingual-config-file-generation)

<a name="1-optional-parameter-list"></a>

## 1. Optional Parameter List

The following list can be viewed through `--help`

|         FLAG             |     Supported script    |        Use        |      Defaults       |         Note         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  Specify configuration file to use  |  None  |  **Please refer to the parameter introduction for configuration file usage** |
|          -o              |      ALL       |  set configuration options  |  None  |  Configuration using -o has higher priority than the configuration file selected with -c. E.g: -o Global.use_gpu=false |

<a name="2-intorduction-to-global-parameters-of-configuration-file"></a>

## 2. Intorduction to Global Parameters of Configuration File

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
|      infer_img            |    Set inference image path or folder path     |       ./infer_img | \||
|      character_dict_path |    Set dictionary path            |  ./ppocr/utils/ppocr_keys_v1.txt  | If the character_dict_path is None, model can only recognize number and lower letters |
|      max_text_length     |    Set the maximum length of text        |       25          |                \                 |
|      use_space_char     |    Set whether to recognize spaces             |        True      |          \|               |
|      label_list          |    Set the angle supported by the direction classifier       |    ['0','180']    |     Only valid in angle classifier model |
|      save_res_path          |    Set the save address of the test model results       |    ./output/det_db/predicts_db.txt    |     Only valid in the text detection model |

### Optimizer ([ppocr/optimizer](../../ppocr/optimizer))

|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         Optimizer class name          |  Adam  |  Currently supports`Momentum`,`Adam`,`RMSProp`, see [ppocr/optimizer/optimizer.py](../../ppocr/optimizer/optimizer.py)  |
|      beta1           |    Set the exponential decay rate for the 1st moment estimates  |       0.9         |               \             |
|      beta2           |    Set the exponential decay rate for the 2nd moment estimates  |     0.999         |               \             |
|      clip_norm           |    The maximum norm value  |    -         |               \             |
|      **lr**                |         Set the learning rate decay method       |   -    |       \  |
|        name    |      Learning rate decay class name   |         Cosine       | Currently supports`Linear`,`Cosine`,`Step`,`Piecewise`, see[ppocr/optimizer/learning_rate.py](../../ppocr/optimizer/learning_rate.py) |
|        learning_rate      |    Set the base learning rate        |       0.001      |  \        |
|      **regularizer**      |  Set network regularization method        |       -      | \        |
|        name      |    Regularizer class name      |       L2     |  Currently support`L1`,`L2`, see[ppocr/optimizer/regularizer.py](../../ppocr/optimizer/regularizer.py)        |
|        factor      |    Learning rate decay coefficient       |       0.00004     |  \        |


### Architecture ([ppocr/modeling](../../ppocr/modeling))
In PaddleOCR, the network is divided into four stages: Transform, Backbone, Neck and Head

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
|      name        |        dataset class name         |  SimpleDataSet |   Currently support`SimpleDataSet`,`LMDBDataSet`  |
|      data_dir        |        Image folder path        |  ./train_data |  \  |
|      label_file_list        |        Groundtruth file path         |  ["./train_data/train_list.txt"] | This parameter is not required when dataset is LMDBDataSet   |
|      ratio_list        |        Ratio of data set         |  [1.0] | If there are two train_lists in label_file_list and ratio_list is [0.4,0.6], 40% will be sampled from train_list1, and 60% will be sampled from train_list2 to combine the entire dataset   |
|      transforms        |        List of methods to transform images and labels         |  [DecodeImage,CTCLabelEncode,RecResizeImg,KeepKeys] |   see[ppocr/data/imaug](../../ppocr/data/imaug)  |
|      **loader**        |        dataloader related         |  - |   |
|      shuffle        |        Does each epoch disrupt the order of the data set         |  True | \  |
|      batch_size_per_card        |        Single card batch size during training         |  256 | \  |
|      drop_last        |        Whether to discard the last incomplete mini-batch because the number of samples in the data set cannot be divisible by batch_size        |  True | \  |
|      num_workers        |        The number of sub-processes used to load data, if it is 0, the sub-process is not started, and the data is loaded in the main process       |  8 | \  |

<a name="3-multilingual-config-file-generation"></a>

## 3. Multilingual Config File Generation

PaddleOCR currently supports 80 (except Chinese) language recognition. A multi-language configuration file template is
provided under the path `configs/rec/multi_languages`: [rec_multi_language_lite_train.yml](../../configs/rec/multi_language/rec_multi_language_lite_train.yml)。

There are two ways to create the required configuration file:：

1. Automatically generated by script

[generate_multi_language_configs.py](../../configs/rec/multi_language/generate_multi_language_configs.py) Can help you generate configuration files for multi-language models

- Take Italian as an example, if your data is prepared in the following format:
    ```
    |-train_data
        |- it_train.txt # train_set label
        |- it_val.txt # val_set label
        |- data
            |- word_001.jpg
            |- word_002.jpg
            |- word_003.jpg
            | ...
    ```

    You can use the default parameters to generate a configuration file:

    ```bash
    # The code needs to be run in the specified directory
    cd PaddleOCR/configs/rec/multi_language/
    # Set the configuration file of the language to be generated through the -l or --language parameter.
    # This command will write the default parameters into the configuration file
    python3 generate_multi_language_configs.py -l it
    ```

- If your data is placed in another location, or you want to use your own dictionary, you can generate the configuration file by specifying the relevant parameters:

    ```bash
    # -l or --language field is required
    # --train to modify the training set
    # --val to modify the validation set
    # --data_dir to modify the data set directory
    # --dict to modify the dict path
    # -o to modify the corresponding default parameters
    cd PaddleOCR/configs/rec/multi_language/
    python3 generate_multi_language_configs.py -l it \  # language
    --train {path/of/train_label.txt} \ # path of train_label
    --val {path/of/val_label.txt} \     # path of val_label
    --data_dir {train_data/path} \      # root directory of training data
    --dict {path/of/dict} \             # path of dict
    -o Global.use_gpu=False             # whether to use gpu
    ...

    ```
Italian is made up of Latin letters, so after executing the command, you will get the rec_latin_lite_train.yml.

2. Manually modify the configuration file

   You can also manually modify the following fields in the template:

   ```
    Global:
      use_gpu: True
      epoch_num: 500
      ...
      character_dict_path:  {path/of/dict} # path of dict
   
   Train:
      dataset:
        name: SimpleDataSet
        data_dir: train_data/ # root directory of training data
        label_file_list: ["./train_data/train_list.txt"] # train label path
      ...
   
   Eval:
      dataset:
        name: SimpleDataSet
        data_dir: train_data/ # root directory of val data
        label_file_list: ["./train_data/val_list.txt"] # val label path
      ...
   
   ```


Currently, the multi-language algorithms supported by PaddleOCR are:

| Configuration file |  Algorithm name |   backbone |   trans   |   seq      |     pred     |  language |
| :--------: |  :-------:   | :-------:  |   :-------:   |   :-----:   |  :-----:   | :-----:  |
| rec_chinese_cht_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | chinese traditional  |
| rec_en_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | English(Case sensitive)   |
| rec_french_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | French |
| rec_ger_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | German   |
| rec_japan_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Japanese |
| rec_korean_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Korean  |
| rec_latin_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Latin  |
| rec_arabic_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | arabic |
| rec_cyrillic_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | cyrillic   |
| rec_devanagari_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | devanagari  |

For more supported languages, please refer to : [Multi-language model](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/multi_languages_en.md#4-support-languages-and-abbreviations)

The multi-language model training method is the same as the Chinese model. The training data set is 100w synthetic data. A small amount of fonts and test data can be downloaded using the following two methods.
* [Baidu Netdisk](https://pan.baidu.com/s/1bS_u207Rm7YbY33wOECKDA),Extraction code:frgi.
* [Google drive](https://drive.google.com/file/d/18cSWX7wXSy4G0tbKJ0d9PuIaiwRLHpjA/view)
