# OPTIONAL PARAMETERS LIST

The following list can be viewed via `--help`

|         FLAG             |     Supported script    |        Use        |      Defaults       |         Note         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  Specify configuration file to use |  None  |  **Please refer to the parameter introduction for configuration file usage** |
|          -o              |      ALL       |  set configuration options  |  None  |  Configuration using -o has higher priority than the configuration file selected with -c. E.g: `-o Global.use_gpu=false`  |  


## INTRODUCTION TO GLOBAL PARAMETERS OF CONFIGURATION FILE

Take `rec_chinese_lite_train.yml` as an example


|         Parameter             |            Use                |      Default       |            Note            |
| :----------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      algorithm           |    Select algorithm to use                    |  Synchronize with configuration file   |     For selecting model, please refer to the supported model [list](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/README_en.md) |
|      use_gpu             |    Set using GPU or not            |       true        |                \                 |
|      epoch_num           |    Maximum training epoch number             |       3000        |                \                 |
|      log_smooth_window   |    Sliding window size            |       20          |                \                 |
|      print_batch_step    |    Set print log interval         |       10          |                \                 |
|      save_model_dir      |    Set model save path        |  output/{model_name}  |                \                 |
|      save_epoch_step     |    Set model save interval        |       3           |                \                 |
|      eval_batch_step     |    Set the model evaluation interval        |2000 or [1000, 2000] |runing evaluation every 2000 iters or evaluation is run every 2000 iterations after the 1000th iteration  |
|train_batch_size_per_card |  Set the batch size during training   |         256         |                \                 |
| test_batch_size_per_card |  Set the batch size during testing    |         256         |                \                 |
|      image_shape         |    Set input image size        |   [3, 32, 100]    |                \                 |
|      max_text_length     |    Set the maximum text length        |       25          |                \                 |
|      character_type      |    Set character type            |       ch          |    en/ch, the default dict will be used for en, and the custom dict will be used for ch|
|      character_dict_path |    Set dictionary path            |  ./ppocr/utils/ic15_dict.txt  |    \                 |
|      loss_type           |    Set loss type              |       ctc         |    Supports two types of loss: ctc / attention |
|       distort            |    Set use distort          |       false       |  Support distort type ,read [img_tools.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/rec/img_tools.py)                 |
|      use_space_char          |    Wether to recognize space             |        false      |         Only support in character_type=ch mode                 |
|      reader_yml          |    Set the reader configuration file          |  ./configs/rec/rec_icdar15_reader.yml  |  \          |
|      pretrain_weights    |    Load pre-trained model path      |  ./pretrain_models/CRNN/best_accuracy  |  \          |
|      checkpoints         |    Load saved model path            |       None        |    Used to load saved parameters to continue training after interruption |
|      save_inference_dir  |   path to save model for inference |          None        |   Use to save inference model |

## INTRODUCTION TO READER PARAMETERS OF CONFIGURATION FILE

Take `rec_chinese_reader.yml` as an example:

|         Parameter             |            Use                |      Default       |            Note            |
| :----------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      reader_function     |    Select data reading method        |  ppocr.data.rec.dataset_traversal,SimpleReader  | Support two data reading methods: SimpleReader / LMDBReader  |
|      num_workers             |    Set the number of data reading threads            |       8        |                \                 |
|      img_set_dir          |    Image folder path             |       ./train_data        |                \                 |
|      label_file_path      |    Groundtruth file path           |       ./train_data/rec_gt_train.txt| \    |
|      infer_img            |    Result folder path     |       ./infer_img | \|

## INTRODUCTION TO OPTIMIZER PARAMETERS OF CONFIGURATION FILE

Take `rec_icdar15_train.yml` as an example:

|         Parameter             |            Use          |      Default        |            None             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|         function        |         Select Optimizer function          |  pocr.optimizer,AdamDecay  |  Only support Adam  |
|         base_lr         |      Set the base lr          |       0.0005      |               \             |
|         beta1           |    Set the exponential decay rate for the 1st moment estimates  |       0.9         |               \             |
|         beta2           |    Set the exponential decay rate for the 2nd moment estimates  |     0.999         |               \             |
|         decay           |         Whether to use decay       |    \              |               \             |
|      function(decay)    |         Set the decay function       |   cosine_decay    |           Only support cosine_decay            |
|      step_each_epoch    |      The number of steps in an epoch.  |         20       | Calculation ：total_image_num / (batch_size_per_card * card_size) |
|        total_epoch      |    The number of epochs      |       1000      | Consistent with Global.epoch_num      |
