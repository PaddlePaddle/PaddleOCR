---
comments: true
---

# Fine-tune

## 1. Background and meaning

The PP-OCR series models provided by PaddleOCR have excellent performance in general scenarios and can solve detection and recognition problems in most cases. In vertical scenarios, if you want to obtain better model, you can further improve the accuracy of the PP-OCR series detection and recognition models through fine-tune.

This article mainly introduces some precautions when fine-tuning the text detection and recognition model. Finally, you can obtain a text detection and recognition model with higher accuracy through model fine-tuning in your own scenarios.

The core points of this article are as follows:

1. The pre-trained model provided by PP-OCR has better generalization ability
2. Adding a small amount of real data (detection:>=500, recognition:>=5000) will greatly improve the detection and recognition effect of vertical scenes
3. When fine-tuning the model, adding real general scene data can further improve the model accuracy and generalization performance
4. In the text detection task, increasing the prediction shape of the image can further improve the detection effect of the smaller text area
5. When fine-tuning the model, it is necessary to properly adjust the hyperparameters (learning rate and batch size are the most important) to obtain a better fine-tuning effect.

For more details, please refer to Chapter 2 and Chapter 3.

## 2. Text detection model fine-tuning

### 2.1 Dataset

* Dataset: It is recommended to prepare at least 500 text detection datasets for model fine-tuning.

* Dataset annotation: single-line text annotation format, it is recommended that the labeled detection frame be consistent with the actual semantic content. For example, in the train ticket scene, the surname and first name may be far apart, but they belong to the same detection field semantically. Here, the entire name also needs to be marked as a detection frame.

### 2.2 Model

It is recommended to choose the PP-OCRv3 model (configuration file: [ch_PP-OCRv3_det_student.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml)，pre-trained model: [ch_PP-OCRv3_det_distill_train.tar](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar), its accuracy and generalization performance is the best pre-training model currently available.

For more PP-OCR series models, please refer to [PP-OCR Series Model Library](../models_list.en.md).

Note: When using the above pre-trained model, you need to use the `student.pdparams` file in the folder as the pre-trained model, that is, only use the student model.

### 2.3 Training hyperparameter

When fine-tuning the model, the most important hyperparameter is the pre-training model path `pretrained_model`, `learning_rate` and `batch_size`，some hyperparameters are as follows:

```yaml linenums="1"
Global:
  pretrained_model: ./ch_PP-OCRv3_det_distill_train/student.pdparams # pre-training model path
Optimizer:
  lr:
    name: Cosine
    learning_rate: 0.001 # learning_rate
    warmup_epoch: 2
  regularizer:
    name: 'L2'
    factor: 0

Train:
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 8  # single gpu batch size
    num_workers: 4
```

In the above configuration file, you need to specify the `pretrained_model` field as the `student.pdparams` file path.

The configuration file provided by PaddleOCR is for 8-gpu training (equivalent to a total batch size of `8*8=64`) and no pre-trained model is loaded. Therefore, in your scenario, the learning rate is the same as the total The batch size needs to be adjusted linearly, for example

* If your scenario is single-gpu training, single gpu batch_size=8, then the total batch_size=8, it is recommended to adjust the learning rate to about `1e-4`.
* If your scenario is for single-gpu training, due to memory limitations, you can only set batch_size=4 for a single gpu, and the total batch_size=4. It is recommended to adjust the learning rate to about `5e-5`.

### 2.4 Prediction hyperparameter

When exporting and inferring the trained model, you can further adjust the predicted image scale to improve the detection effect of small-area text. The following are some hyperparameters during DBNet inference, which can be adjusted appropriately to improve the effect.

| hyperparameter | type | default | meaning |
| :--: | :--: | :--: | :--: |
|  det_db_thresh | float | 0.3 | In the probability map output by DB, pixels with a score greater than the threshold will be considered as text pixels |
|  det_db_box_thresh | float | 0.6 | When the average score of all pixels within the frame of the detection result is greater than the threshold, the result will be considered as a text area |
|  det_db_unclip_ratio | float | 1.5 | The expansion coefficient of `Vatti clipping`, using this method to expand the text area |
|  max_batch_size | int | 10 | batch size |
|  use_dilation | bool | False | Whether to expand the segmentation results to obtain better detection results |
|  det_db_score_mode | str | "fast" | DB's detection result score calculation method supports `fast` and `slow`. `fast` calculates the average score based on all pixels in the polygon’s circumscribed rectangle border, and `slow` calculates the average score based on all pixels in the original polygon. The calculation speed is relatively slower, but more accurate. |

For more information on inference methods, please refer to[Paddle Inference doc](../infer_deploy/python_infer.en.md).

## 3. Text recognition model fine-tuning

### 3.1 Dataset

* Dataset：If the dictionary is not changed, it is recommended to prepare at least 5,000 text recognition datasets for model fine-tuning; if the dictionary is changed (not recommended), more quantities are required.

* Data distribution: It is recommended that the distribution be as consistent as possible with the actual measurement scenario. If the actual scene contains a lot of short text, it is recommended to include more short text in the training data. If the actual scene has high requirements for the recognition effect of spaces, it is recommended to include more text content with spaces in the training data.

* Data synthesis: In the case of some character recognition errors, it is recommended to obtain a batch of specific character dataset, add it to the original dataset and use a small learning rate for fine-tuning. The ratio of original dataset to new dataset can be 10:1 to 5:1 to avoid overfitting of the model caused by too much data in a single scene. At the same time, try to balance the word frequency of the corpus to ensure that the frequency of common words will not be too low.

  Specific characters can be generated using the TextRenderer tool, for synthesis examples, please refer to [data synthesis](../../applications/光功率计数码管字符识别.md)
  . The synthetic data corpus should come from real usage scenarios as much as possible, and keep the richness of fonts and backgrounds on the basis of being close to the real scene, which will help improve the model effect.

* Common Chinese and English data: During training, common real data can be added to the training set (for example, in the fine-tuning scenario without changing the dictionary, it is recommended to add real data such as LSVT, RCTW, MTWI) to further improve the generalization performance of the model.

### 3.2 Model

It is recommended to choose the PP-OCRv3 model (configuration file: [ch_PP-OCRv3_rec_distillation.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml)，pre-trained model: [ch_PP-OCRv3_rec_train.tar](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar)，its accuracy and generalization performance is the best pre-training model currently available.

For more PP-OCR series models, please refer to [PP-OCR Series Model Library](../model_list.en.md).

The PP-OCRv3 model uses the GTC strategy. The SAR branch has a large number of parameters. When the training data is a simple scene, the model is easy to overfit, resulting in poor fine-tuning effect. It is recommended to remove the GTC strategy. The configuration file of the model structure is modified as follows:

```yaml linenums="1"
Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 64
    depth: 2
    hidden_dims: 120
    use_guide: False
  Head:
    name: CTCHead
    fc_decay: 0.00001
Loss:
  name: CTCLoss

Train:
  dataset:
  ......
    transforms:
    # remove RecConAug
    # - RecConAug:
    #     prob: 0.5
    #     ext_data_num: 2
    #     image_shape: [48, 320, 3]
    #     max_text_length: *max_text_length
    - RecAug:
    # modify Encode
    - CTCLabelEncode:
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
...

Eval:
  dataset:
  ...
    transforms:
    ...
    - CTCLabelEncode:
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
...


```

### 3.3 Training hyperparameter

Similar to text detection task fine-tuning, when fine-tuning the recognition model, the most important hyperparameters are the pre-trained model path `pretrained_model`, `learning_rate` and `batch_size`, some default configuration files are shown below.

```yaml linenums="1"
Global:
  pretrained_model:  # pre-training model path
Optimizer:
  lr:
    name: Piecewise
    decay_epochs : [700, 800]
    values : [0.001, 0.0001]  # learning_rate
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    factor: 0

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
    - ./train_data/train_list.txt
    ratio_list: [1.0] # Sampling ratio, the default value is [1.0]
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 128 # single gpu batch size
    num_workers: 8

```

In the above configuration file, you first need to specify the `pretrained_model` field as the `ch_PP-OCRv3_rec_train/best_accuracy.pdparams` file path decompressed in Chapter 3.2.

The configuration file provided by PaddleOCR is for 8-gpu training (equivalent to a total batch size of `8*128=1024`) and no pre-trained model is loaded. Therefore, in your scenario, the learning rate is the same as the total The batch size needs to be adjusted linearly, for example:

* If your scenario is single-gpu training, single gpu batch_size=128, then the total batch_size=128, in the case of loading the pre-trained model, it is recommended to adjust the learning rate to about `[1e-4, 2e-5]` (For the piecewise learning rate strategy, two values need to be set, the same below).
* If your scenario is for single-gpu training, due to memory limitations, you can only set batch_size=64 for a single gpu, and the total batch_size=64. When loading the pre-trained model, it is recommended to adjust the learning rate to `[5e-5 , 1e-5]`about.

If there is general real scene data added, it is recommended that in each epoch, the amount of vertical scene data and real scene data should be kept at about 1:1.

For example: your own vertical scene recognition data volume is 1W, the data label file is `vertical.txt`, the collected general scene recognition data volume is 10W, and the data label file is `general.txt`.

Then, the `label_file_list` and `ratio_list` parameters can be set as shown below. In each epoch, `vertical.txt` will be fully sampled (sampling ratio is 1.0), including 1W pieces of data; `general.txt` will be sampled according to a sampling ratio of 0.1, including `10W*0.1=1W` pieces of data, the final ratio of the two is `1:1`.

```yaml linenums="1"
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
    - vertical.txt
    - general.txt
    ratio_list: [1.0, 0.1]
```

### 3.4 Training optimization

The training process does not happen overnight. After completing a stage of training evaluation, it is recommended to collect and analyze the badcase of the current model in the real scene, adjust the proportion of training data in a targeted manner, or further add synthetic data. Through multiple iterations of training, the model effect is continuously optimized.

If you modify the custom dictionary during training, since the parameters of the last layer of FC cannot be loaded, it is normal for acc=0 at the beginning of the iteration. Don't worry, loading the pre-trained model can still speed up the model convergence.
