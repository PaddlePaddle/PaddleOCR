## TEXT RECOGNITION

- [DATA PREPARATION](#DATA_PREPARATION)
    - [Dataset Download](#Dataset_download)
    - [Costom Dataset](#Costom_Dataset)  
    - [Dictionary](#Dictionary)  
    - [Add Space Category](#Add_space_category)

- [TRAINING](#TRAINING)
    - [Data Augmentation](#Data_Augmentation)
    - [Training](#Training)
    - [Multi-language](#Multi_language)

- [EVALUATION](#EVALUATION)

- [PREDICTION](#PREDICTION)
    - [Training engine prediction](#Training_engine_prediction)

<a name="DATA_PREPARATION"></a>
### DATA PREPARATION


PaddleOCR supports two data formats: `LMDB` is used to train public data and evaluation algorithms; `general data` is used to train your own data:

Please organize the dataset as follows:

The default storage path for training data is `PaddleOCR/train_data`, if you already have a dataset on your disk, just create a soft link to the dataset directory:

```
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/dataset
```

<a name="Dataset_download"></a>
* Dataset download

If you do not have a dataset locally, you can download it on the official website [icdar2015](http://rrc.cvc.uab.es/?ch=4&com=downloads). Also refer to [DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)，download the lmdb format dataset required for benchmark

If you want to reproduce the paper indicators of SRN, you need to download offline [augmented data](https://pan.baidu.com/s/1-HSZ-ZVdqBF2HaBZ5pRAKA), extraction code: y3ry. The augmented data is obtained by rotation and perturbation of mjsynth and synthtext. Please unzip the data to {your_path}/PaddleOCR/train_data/data_lmdb_Release/training/path.

<a name="Costom_Dataset"></a>
* Use your own dataset:

If you want to use your own data for training, please refer to the following to organize your data.

- Training set

First put the training images in the same folder (train_images), and use a txt file (rec_gt_train.txt) to store the image path and label.

* Note: by default, the image path and image label are split with \t, if you use other methods to split, it will cause training error

```
" Image file name           Image annotation "

train_data/train_0001.jpg   简单可依赖
train_data/train_0002.jpg   用科技让复杂的世界更简单
```
PaddleOCR provides label files for training the icdar2015 dataset, which can be downloaded in the following ways:

```
# Training set label
wget -P ./train_data/ic15_data  https://paddleocr.bj.bcebos.com/dataset/rec_gt_train.txt
# Test Set Label
wget -P ./train_data/ic15_data  https://paddleocr.bj.bcebos.com/dataset/rec_gt_test.txt
```

The final training set should have the following file structure:

```
|-train_data
    |-ic15_data
        |- rec_gt_train.txt
        |- train
            |- word_001.png
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

- Test set

Similar to the training set, the test set also needs to be provided a folder containing all images (test) and a rec_gt_test.txt. The structure of the test set is as follows:

```
|-train_data
    |-ic15_data
        |- rec_gt_test.txt
        |- test
            |- word_001.jpg
            |- word_002.jpg
            |- word_003.jpg
            | ...
```
<a name="Dictionary"></a>
- Dictionary

Finally, a dictionary ({word_dict_name}.txt) needs to be provided so that when the model is trained, all the characters that appear can be mapped to the dictionary index.

Therefore, the dictionary needs to contain all the characters that you want to be recognized correctly. {word_dict_name}.txt needs to be written in the following format and saved in the `utf-8` encoding format:

```
l
d
a
d
r
n
```

In `word_dict.txt`, there is a single word in each line, which maps characters and numeric indexes together, e.g "and" will be mapped to [2 5 1]

`ppocr/utils/ppocr_keys_v1.txt` is a Chinese dictionary with 6623 characters.

`ppocr/utils/ic15_dict.txt` is an English dictionary with 63 characters

`ppocr/utils/dict/french_dict.txt` is a French dictionary with 118 characters

`ppocr/utils/dict/japan_dict.txt` is a Japanese dictionary with 4399 characters

`ppocr/utils/dict/korean_dict.txt` is a Korean dictionary with 3636 characters

`ppocr/utils/dict/german_dict.txt` is a German dictionary with 131 characters

`ppocr/utils/dict/en_dict.txt` is a English dictionary with 63 characters


You can use it on demand.

The current multi-language model is still in the demo stage and will continue to optimize the model and add languages. **You are very welcome to provide us with dictionaries and fonts in other languages**,
If you like, you can submit the dictionary file to [dict](../../ppocr/utils/dict) and we will thank you in the Repo.


To customize the dict file, please modify the `character_dict_path` field in `configs/rec/rec_icdar15_train.yml` and set `character_type` to `ch`.

- Custom dictionary

If you need to customize dic file, please add character_dict_path field in configs/rec/rec_icdar15_train.yml to point to your dictionary path. And set character_type to ch.

<a name="Add_space_category"></a>
- Add space category

If you want to support the recognition of the `space` category, please set the `use_space_char` field in the yml file to `True`.

**Note: use_space_char only takes effect when character_type=ch**

<a name="TRAINING"></a>
### TRAINING

PaddleOCR provides training scripts, evaluation scripts, and prediction scripts. In this section, the CRNN recognition model will be used as an example:

First download the pretrain model, you can download the trained model to finetune on the icdar2015 data:

```
cd PaddleOCR/
# Download the pre-trained model of MobileNetV3
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar
# Decompress model parameters
cd pretrain_models
tar -xf rec_mv3_none_bilstm_ctc_v2.0_train.tar && rm -rf rec_mv3_none_bilstm_ctc_v2.0_train.tar
```

Start training:

```
# GPU training Support single card and multi-card training, specify the card number through --gpus
# Training icdar15 English data and The training log will be automatically saved as train.log under "{save_model_dir}"
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_icdar15_train.yml
```
<a name="Data_Augmentation"></a>
- Data Augmentation

PaddleOCR provides a variety of data augmentation methods. If you want to add disturbance during training, please set `distort: true` in the configuration file.

The default perturbation methods are: cvtColor, blur, jitter, Gasuss noise, random crop, perspective, color reverse.

Each disturbance method is selected with a 50% probability during the training process. For specific code implementation, please refer to: [img_tools.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/rec/img_tools.py)

<a name="Training"></a>
- Training

PaddleOCR supports alternating training and evaluation. You can modify `eval_batch_step` in `configs/rec/rec_icdar15_train.yml` to set the evaluation frequency. By default, it is evaluated every 500 iter and the best acc model is saved under `output/rec_CRNN/best_accuracy` during the evaluation process.

If the evaluation set is large, the test will be time-consuming. It is recommended to reduce the number of evaluations, or evaluate after training.

* Tip: You can use the `-c` parameter to select multiple model configurations under the `configs/rec/` path for training. The recognition algorithms supported by PaddleOCR are:


| Configuration file |  Algorithm |   backbone |   trans   |   seq      |     pred     |
| :--------: |  :-------:   | :-------:  |   :-------:   |   :-----:   |  :-----:   |
| [rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml) |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  |
| [rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml) |  CRNN | ResNet34_vd |  None   |  BiLSTM |  ctc  |
| rec_chinese_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  |
| rec_chinese_common_train.yml |  CRNN |   ResNet34_vd |  None   |  BiLSTM |  ctc  |
| rec_icdar15_train.yml |  CRNN |   Mobilenet_v3 large 0.5 |  None   |  BiLSTM |  ctc  |
| rec_mv3_none_bilstm_ctc.yml |  CRNN |   Mobilenet_v3 large 0.5 |  None   |  BiLSTM |  ctc  |
| rec_mv3_none_none_ctc.yml |  Rosetta |   Mobilenet_v3 large 0.5 |  None   |  None |  ctc  |
| rec_r34_vd_none_bilstm_ctc.yml |  CRNN |   Resnet34_vd |  None   |  BiLSTM |  ctc  |
| rec_r34_vd_none_none_ctc.yml |  Rosetta |   Resnet34_vd |  None   |  None |  ctc  |
| rec_mv3_tps_bilstm_att.yml |  CRNN |   Mobilenet_v3 |  TPS   |  BiLSTM |  att  |
| rec_r34_vd_tps_bilstm_att.yml |  CRNN |   Resnet34_vd |  TPS   |  BiLSTM |  att  |
| rec_r50fpn_vd_none_srn.yml    | SRN | Resnet50_fpn_vd    | None    | rnn | srn |


For training Chinese data, it is recommended to use
[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml). If you want to try the result of other algorithms on the Chinese data set, please refer to the following instructions to modify the configuration file:
co
Take `rec_chinese_lite_train_v2.0.yml` as an example:
```
Global:
  ...
  # Add a custom dictionary, such as modify the dictionary, please point the path to the new dictionary
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt
  # Modify character type
  character_type: ch
  ...
  # Whether to recognize spaces
  use_space_char: True


Optimizer:
  ...
  # Add learning rate decay strategy
  lr:
    name: Cosine
    learning_rate: 0.001
  ...

...

Train:
  dataset:
    # Type of dataset，we support LMDBDateSet and SimpleDataSet
    name: SimpleDataSet
    # Path of dataset
    data_dir: ./train_data/
    # Path of train list
    label_file_list: ["./train_data/train_list.txt"]
    transforms:
      ...
      - RecResizeImg:
          # Modify image_shape to fit long text
          image_shape: [3, 32, 320]
      ...
  loader:
    ...
    # Train batch_size for Single card
    batch_size_per_card: 256
    ...

Eval:
  dataset:
    # Type of dataset，we support LMDBDateSet and SimpleDataSet
    name: SimpleDataSet
    # Path of dataset
    data_dir: ./train_data
    # Path of eval list
    label_file_list: ["./train_data/val_list.txt"]
    transforms:
      ...
      - RecResizeImg:
          # Modify image_shape to fit long text
          image_shape: [3, 32, 320]
      ...
  loader:
    # Eval batch_size for Single card
    batch_size_per_card: 256
    ...
```
**Note that the configuration file for prediction/evaluation must be consistent with the training.**

<a name="Multi_language"></a>
- Multi-language

PaddleOCR currently supports 26 (except Chinese) language recognition. A multi-language configuration file template is
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

2. Manually modify the configuration file

   You can also manually modify the following fields in the template:

   ```
    Global:
      use_gpu: True
      epoch_num: 500
      ...
      character_type: it  # language
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

| Configuration file |  Algorithm name |   backbone |   trans   |   seq      |     pred     |  language | character_type |
| :--------: |  :-------:   | :-------:  |   :-------:   |   :-----:   |  :-----:   | :-----:  | :-----:  |
| rec_chinese_cht_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | chinese traditional  | chinese_cht|
| rec_en_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | English(Case sensitive)   | EN |
| rec_french_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | French |  french |
| rec_ger_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | German   | german |
| rec_japan_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Japanese | japan |
| rec_korean_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Korean  | korean |
| rec_it_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Italian  | it |
| rec_xi_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Spanish |  xi |
| rec_pu_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Portuguese   | pu |
| rec_ru_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Russia  | ru |
| rec_ar_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Arabic  | ar |
| rec_hi_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Hindi |  hi |
| rec_ug_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Uyghur  | ug |
| rec_fa_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Persian(Farsi)  | fa |
| rec_ur_ite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Urdu  | ur |
| rec_rs_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Serbian(latin) | rs |
| rec_oc_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Occitan  | oc |
| rec_mr_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Marathi  | mr |
| rec_ne_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Nepali  | ne |
| rec_rsc_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Serbian(cyrillic) |  rsc |
| rec_bg_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Bulgarian  | bg |
| rec_uk_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Ukranian  | uk |
| rec_be_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Belarusian   | be |
| rec_te_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Telugu  | te |
| rec_ka_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Kannada  | ka |
| rec_ta_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Tamil |  ta |


The multi-language model training method is the same as the Chinese model. The training data set is 100w synthetic data. A small amount of fonts and test data can be downloaded on [Baidu Netdisk](https://pan.baidu.com/s/1bS_u207Rm7YbY33wOECKDA),Extraction code:frgi.

If you want to finetune on the basis of the existing model effect, please refer to the following instructions to modify the configuration file:

Take `rec_french_lite_train` as an example:

```
Global:
  ...
  # Add a custom dictionary, such as modify the dictionary, please point the path to the new dictionary
  character_dict_path: ./ppocr/utils/dict/french_dict.txt
  ...
  # Whether to recognize spaces
  use_space_char: True

...

Train:
  dataset:
    # Type of dataset，we support LMDBDateSet and SimpleDataSet
    name: SimpleDataSet
    # Path of dataset
    data_dir: ./train_data/
    # Path of train list
    label_file_list: ["./train_data/french_train.txt"]
    ...

Eval:
  dataset:
    # Type of dataset，we support LMDBDateSet and SimpleDataSet
    name: SimpleDataSet
    # Path of dataset
    data_dir: ./train_data
    # Path of eval list
    label_file_list: ["./train_data/french_val.txt"]
    ...
```

<a name="EVALUATION"></a>
### EVALUATION

The evaluation dataset can be set by modifying the `Eval.dataset.label_file_list` field in the `configs/rec/rec_icdar15_train.yml` file.

```
# GPU evaluation, Global.checkpoints is the weight to be tested
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

<a name="PREDICTION"></a>
### PREDICTION

<a name="Training_engine_prediction"></a>
* Training engine prediction

Using the model trained by paddleocr, you can quickly get prediction through the following script.

The default prediction picture is stored in `infer_img`, and the weight is specified via `-o Global.checkpoints`:

```
# Predict English results
python3 tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.load_static_weights=false Global.infer_img=doc/imgs_words/en/word_1.jpg
```

Input image:

![](../imgs_words/en/word_1.png)

Get the prediction result of the input image:

```
infer_img: doc/imgs_words/en/word_1.png
        result: ('joint', 0.9998967)
```

The configuration file used for prediction must be consistent with the training. For example, you completed the training of the Chinese model with `python3 tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml`, you can use the following command to predict the Chinese model:

```
# Predict Chinese results
python3 tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.load_static_weights=false Global.infer_img=doc/imgs_words/ch/word_1.jpg
```

Input image:

![](../imgs_words/ch/word_1.jpg)

Get the prediction result of the input image:

```
infer_img: doc/imgs_words/ch/word_1.jpg
        result: ('韩国小馆', 0.997218)
```
