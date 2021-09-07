# TEXT RECOGNITION

- [1 DATA PREPARATION](#DATA_PREPARATION)
    - [1.1 Costom Dataset](#Costom_Dataset)
    - [1.2 Dataset Download](#Dataset_download)
    - [1.3 Dictionary](#Dictionary)  
    - [1.4 Add Space Category](#Add_space_category)

- [2 TRAINING](#TRAINING)
    - [2.1 Data Augmentation](#Data_Augmentation)
    - [2.2 General Training](#Training)
    - [2.3 Multi-language Training](#Multi_language)

- [3 EVALUATION](#EVALUATION)

- [4 PREDICTION](#PREDICTION)
    - [4.1 Training engine prediction](#Training_engine_prediction)
- [5 CONVERT TO INFERENCE MODEL](#Inference)

<a name="DATA_PREPARATION"></a>
## 1 DATA PREPARATION


PaddleOCR supports two data formats:
- `LMDB` is used to train data sets stored in lmdb format（LMDBDataSet）;
- `general data` is used to train data sets stored in text files（SimpleDataSet）:

Please organize the dataset as follows:

The default storage path for training data is `PaddleOCR/train_data`, if you already have a dataset on your disk, just create a soft link to the dataset directory:

```
# linux and mac os
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/dataset
# windows
mklink /d <path/to/paddle_ocr>/train_data/dataset <path/to/dataset>
```

<a name="Costom_Dataset"></a>
### 1.1 Costom dataset

If you want to use your own data for training, please refer to the following to organize your data.

- Training set

It is recommended to put the training images in the same folder, and use a txt file (rec_gt_train.txt) to store the image path and label. The contents of the txt file are as follows:

* Note: by default, the image path and image label are split with \t, if you use other methods to split, it will cause training error

```
" Image file name           Image annotation "

train_data/rec/train/word_001.jpg   简单可依赖
train_data/rec/train/word_002.jpg   用科技让复杂的世界更简单
...
```

The final training set should have the following file structure:

```
|-train_data
  |-rec
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
  |-rec
    |-ic15_data
        |- rec_gt_test.txt
        |- test
            |- word_001.jpg
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

<a name="Dataset_download"></a>
### 1.2 Dataset download

- ICDAR2015

If you do not have a dataset locally, you can download it on the official website [icdar2015](http://rrc.cvc.uab.es/?ch=4&com=downloads).
Also refer to [DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) ，download the lmdb format dataset required for benchmark

PaddleOCR provides label files for training the icdar2015 dataset, which can be downloaded in the following ways:

```
# Training set label
wget -P ./train_data/ic15_data  https://paddleocr.bj.bcebos.com/dataset/rec_gt_train.txt
# Test Set Label
wget -P ./train_data/ic15_data  https://paddleocr.bj.bcebos.com/dataset/rec_gt_test.txt
```

PaddleOCR also provides a data format conversion script, which can convert ICDAR official website label to a data format
supported by PaddleOCR. The data conversion tool is in `ppocr/utils/gen_label.py`, here is the training set as an example:

```
# convert the official gt to rec_gt_label.txt
python gen_label.py --mode="rec" --input_path="{path/of/origin/label}" --output_label="rec_gt_label.txt"
```

The data format is as follows, (a) is the original picture, (b) is the Ground Truth text file corresponding to each picture:

![](../datasets/icdar_rec.png)


- Multilingual dataset

The multi-language model training method is the same as the Chinese model. The training data set is 100w synthetic data. A small amount of fonts and test data can be downloaded using the following two methods.
* [Baidu Netdisk](https://pan.baidu.com/s/1bS_u207Rm7YbY33wOECKDA) ,Extraction code:frgi.
* [Google drive](https://drive.google.com/file/d/18cSWX7wXSy4G0tbKJ0d9PuIaiwRLHpjA/view)


<a name="Dictionary"></a>
### 1.3 Dictionary

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

PaddleOCR has built-in dictionaries, which can be used on demand.

`ppocr/utils/ppocr_keys_v1.txt` is a Chinese dictionary with 6623 characters.

`ppocr/utils/ic15_dict.txt` is an English dictionary with 63 characters

`ppocr/utils/dict/french_dict.txt` is a French dictionary with 118 characters

`ppocr/utils/dict/japan_dict.txt` is a Japanese dictionary with 4399 characters

`ppocr/utils/dict/korean_dict.txt` is a Korean dictionary with 3636 characters

`ppocr/utils/dict/german_dict.txt` is a German dictionary with 131 characters

`ppocr/utils/en_dict.txt` is a English dictionary with 96 characters


The current multi-language model is still in the demo stage and will continue to optimize the model and add languages. **You are very welcome to provide us with dictionaries and fonts in other languages**,
If you like, you can submit the dictionary file to [dict](../../ppocr/utils/dict) and we will thank you in the Repo.


To customize the dict file, please modify the `character_dict_path` field in `configs/rec/rec_icdar15_train.yml` and set `character_type` to `ch`.

- Custom dictionary

If you need to customize dic file, please add character_dict_path field in configs/rec/rec_icdar15_train.yml to point to your dictionary path. And set character_type to ch.

<a name="Add_space_category"></a>
### 1.4 Add space category

If you want to support the recognition of the `space` category, please set the `use_space_char` field in the yml file to `True`.

**Note: use_space_char only takes effect when character_type=ch**

<a name="TRAINING"></a>
## 2 TRAINING

<a name="Data_Augmentation"></a>
### 2.1 Data Augmentation

PaddleOCR provides a variety of data augmentation methods. All the augmentation methods are enabled by default.

The default perturbation methods are: cvtColor, blur, jitter, Gasuss noise, random crop, perspective, color reverse, TIA augmentation.

Each disturbance method is selected with a 40% probability during the training process. For specific code implementation, please refer to: [rec_img_aug.py](../../ppocr/data/imaug/rec_img_aug.py)

<a name="Training"></a>
### 2.2 General Training

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
# GPU training Support single card and multi-card training
# Training icdar15 English data and The training log will be automatically saved as train.log under "{save_model_dir}"

#specify the single card training(Long training time, not recommended)
python3 tools/train.py -c configs/rec/rec_icdar15_train.yml
#specify the card number through --gpus
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_icdar15_train.yml
```


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
| rec_mtb_nrtr.yml    | NRTR | nrtr_mtb    | None    | transformer encoder | transformer decoder |

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
    # Type of dataset，we support LMDBDataSet and SimpleDataSet
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
    # Type of dataset，we support LMDBDataSet and SimpleDataSet
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
### 2.3 Multi-language Training

Currently, the multi-language algorithms supported by PaddleOCR are:

| Configuration file |  Algorithm name |   backbone |   trans   |   seq      |     pred     |  language | character_type |
| :--------: |  :-------:   | :-------:  |   :-------:   |   :-----:   |  :-----:   | :-----:  | :-----:  |
| rec_chinese_cht_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | chinese traditional  | chinese_cht|
| rec_en_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | English(Case sensitive)   | EN |
| rec_french_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | French |  french |
| rec_ger_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | German   | german |
| rec_japan_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Japanese | japan |
| rec_korean_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Korean  | korean |
| rec_latin_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | Latin  | latin |
| rec_arabic_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | arabic |  ar |
| rec_cyrillic_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | cyrillic   | cyrillic |
| rec_devanagari_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | devanagari  | devanagari |

For more supported languages, please refer to : [Multi-language model](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/multi_languages_en.md#4-support-languages-and-abbreviations)


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
    # Type of dataset，we support LMDBDataSet and SimpleDataSet
    name: SimpleDataSet
    # Path of dataset
    data_dir: ./train_data/
    # Path of train list
    label_file_list: ["./train_data/french_train.txt"]
    ...

Eval:
  dataset:
    # Type of dataset，we support LMDBDataSet and SimpleDataSet
    name: SimpleDataSet
    # Path of dataset
    data_dir: ./train_data
    # Path of eval list
    label_file_list: ["./train_data/french_val.txt"]
    ...
```

<a name="EVALUATION"></a>

## 3 EVALUATION

The evaluation dataset can be set by modifying the `Eval.dataset.label_file_list` field in the `configs/rec/rec_icdar15_train.yml` file.

```
# GPU evaluation, Global.checkpoints is the weight to be tested
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

<a name="PREDICTION"></a>
## 4 PREDICTION


Using the model trained by paddleocr, you can quickly get prediction through the following script.

The default prediction picture is stored in `infer_img`, and the trained weight is specified via `-o Global.checkpoints`:


According to the `save_model_dir` and `save_epoch_step` fields set in the configuration file, the following parameters will be saved:

```
output/rec/
├── best_accuracy.pdopt  
├── best_accuracy.pdparams  
├── best_accuracy.states  
├── config.yml  
├── iter_epoch_3.pdopt  
├── iter_epoch_3.pdparams  
├── iter_epoch_3.states  
├── latest.pdopt  
├── latest.pdparams  
├── latest.states  
└── train.log
```

Among them, best_accuracy.* is the best model on the evaluation set; iter_epoch_x.* is the model saved at intervals of `save_epoch_step`; latest.* is the model of the last epoch.

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

<a name="Inference"></a>

## 5 CONVERT TO INFERENCE MODEL

The recognition model is converted to the inference model in the same way as the detection, as follows:

```
# -c Set the training algorithm yml configuration file
# -o Set optional parameters
# Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model=./ch_lite/ch_ppocr_mobile_v2.0_rec_train/best_accuracy  Global.save_inference_dir=./inference/rec_crnn/
```

If you have a model trained on your own dataset with a different dictionary file, please make sure that you modify the `character_dict_path` in the configuration file to your dictionary file path.

After the conversion is successful, there are three files in the model save directory:

```
inference/det_db/
    ├── inference.pdiparams         # The parameter file of recognition inference model
    ├── inference.pdiparams.info    # The parameter information of recognition inference model, which can be ignored
    └── inference.pdmodel           # The program file of recognition model
```

- Text recognition model Inference using custom characters dictionary

  If the text dictionary is modified during training, when using the inference model to predict, you need to specify the dictionary path used by `--rec_char_dict_path`, and set `rec_char_type=ch`

  ```
  python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./your inference model" --rec_image_shape="3, 32, 100" --rec_char_type="ch" --rec_char_dict_path="your text dict path"
  ```

  

