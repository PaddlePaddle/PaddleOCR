---
comments: true
typora-copy-images-to: images
---

# Text Recognition

This article provides a comprehensive guide for the PaddleOCR text recognition task, covering the entire workflow including data preparation, model training, fine-tuning, evaluation, and prediction, with detailed explanations for each phase.

## 1. Data Preparation

### 1.1. Prepare the Dataset

PaddleOCR supports two data formats:

- `lmdb`: Used for training with datasets stored in LMDB format (LMDBDataSet);
- `General Data`: Used for training with datasets stored in text files (SimpleDataSet);

The default storage path for training data is `PaddleOCR/train_data`. If you already have a dataset on your disk, simply create a symbolic link to the dataset directory:

```bash
# Linux and macOS
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/dataset
# Windows
mklink /d <path/to/paddle_ocr>/train_data/dataset <path/to/dataset>
```

### 1.2. Custom Dataset

Here, we will use a general dataset as an example to explain how to prepare the dataset:

- Training Dataset

It is recommended to place the training images in the same folder and record the image paths and labels in a txt file (`rec_gt_train.txt`). The content of the txt file should be as follows:

**Note:** In the txt file, please use `\t` to separate the image path and the label. Using any other separator will cause errors during training.

```text
" Image Filename                   Image Label "

train_data/rec/train/word_001.jpg   Simple and reliable
train_data/rec/train/word_002.jpg   Making the complex world simpler with technology
...
```

The final structure of the training dataset should look like this:

```text
|-train_data
  |-rec
    |- rec_gt_train.txt
    |- train
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

In addition to the single-image-per-line format described above, PaddleOCR also supports training on data augmented offline. To avoid sampling the same sample multiple times in the same batch, we can list image paths with the same label on one line. During training, PaddleOCR will randomly select one image from the list. The corresponding format of the label file is as follows:

```text
["11.jpg", "12.jpg"]   Simple and reliable
["21.jpg", "22.jpg", "23.jpg"]   Making the complex world simpler with technology
3.jpg   ocr
```

In the above example, both "11.jpg" and "12.jpg" have the same label `Simple and reliable`. During training, one of these images will be randomly chosen for training.

- Validation Dataset

Similarly to the training dataset, the validation dataset should also provide a folder containing all the images (test) and a `rec_gt_test.txt` file. The structure of the validation dataset is as follows:

```text
|-train_data
  |-rec
    |- rec_gt_test.txt
    |- test
        |- word_001.jpg
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

### 1.3. Data Download

- ICDAR2015

If you don't have a dataset locally, you can download the [ICDAR2015](http://rrc.cvc.uab.es/?ch=4&com=downloads) dataset from the official website for quick testing. You can also refer to [DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) to download the LMDB formatted dataset needed for benchmarking.

If you're using the public ICDAR2015 dataset, PaddleOCR provides a label file for training the ICDAR2015 dataset. You can download it as follows:

```bash linenums="1"
# Training set label
wget -P ./train_data/ic15_data  https://paddleocr.bj.bcebos.com/dataset/rec_gt_train.txt
# Test Set Label
wget -P ./train_data/ic15_data  https://paddleocr.bj.bcebos.com/dataset/rec_gt_test.txt
```

PaddleOCR also provides a data format conversion script, which can convert ICDAR official website label to a data format
supported by PaddleOCR. The data conversion tool is in `ppocr/utils/gen_label.py`, here is the training set as an example:

```bash linenums="1"
# convert the official gt to rec_gt_label.txt
python gen_label.py --mode="rec" --input_path="{path/of/origin/label}" --output_label="rec_gt_label.txt"
```

The data format is as follows, (a) is the original picture, (b) is the Ground Truth text file corresponding to each picture:

![img](./images/icdar_rec.png)

- Multilingual Datasets

The multi-language model training method is the same as the Chinese model. The training data set is 100w synthetic data. A small amount of fonts and test data can be downloaded using the following two methods.

- [Baidu Netdisk](https://pan.baidu.com/s/1bS_u207Rm7YbY33wOECKDA) ,Extraction code:frgi.
- [Google drive](https://drive.google.com/file/d/18cSWX7wXSy4G0tbKJ0d9PuIaiwRLHpjA/view)

### 1.4. Dictionary

Finally, a dictionary ({word_dict_name}.txt) needs to be provided so that when the model is trained, all the characters that appear can be mapped to the dictionary index.

Therefore, the dictionary needs to contain all the characters that you want to be recognized correctly. {word_dict_name}.txt needs to be written in the following format and saved in the `utf-8` encoding format:

```text linenums="1"
l
d
a
d
r
n
```

In `word_dict.txt`, there is a single word in each line, which maps characters and numeric indexes together, e.g "and" will be mapped to [2 5 1]

PaddleOCR includes several built-in dictionaries that can be used as needed:

- `ppocr/utils/ppocr_keys_v1.txt`: A Chinese dictionary containing 6623 characters.
- `ppocr/utils/ic15_dict.txt`: An English dictionary containing 36 characters.
- `ppocr/utils/dict/french_dict.txt`: A French dictionary containing 118 characters.
- `ppocr/utils/dict/japan_dict.txt`: A Japanese dictionary containing 4399 characters.
- `ppocr/utils/dict/korean_dict.txt`: A Korean dictionary containing 3636 characters.
- `ppocr/utils/dict/german_dict.txt`: A German dictionary containing 131 characters.
- `ppocr/utils/en_dict.txt`: An English dictionary containing 96 characters.

Currently, the multilingual models are still in the demo stage, and we are continuously improving the models and adding new languages. **We highly welcome you to provide dictionaries and fonts for other languages**. If you are willing, you can submit your dictionary files to the [dict](../../ppocr/utils/dict) directory, and we will credit you in the repo.
To customize the dict file, please modify the `character_dict_path` field in `configs/rec/rec_icdar15_train.yml`.

- Custom Dictionary

If you need to customize dic file, please add character_dict_path field in configs/rec/rec_icdar15_train.yml to point to your dictionary path. And set character_type to ch.

### 1.5. Add Space Category

To support recognition of the "space" category, set the `use_space_char` field in the YAML file to `True`.

### 1.6. Data Augmentation

PaddleOCR provides a variety of data augmentation methods. All the augmentation methods are enabled by default.

The default perturbation methods are: cvtColor, blur, jitter, Gasuss noise, random crop, perspective, color reverse, TIA augmentation.

Each disturbance method is selected with a 40% probability during the training process. For specific code implementation, please refer to: [rec_img_aug.py](../../ppocr/data/imaug/rec_img_aug.py)

## 2. Training

PaddleOCR provides training scripts, evaluation scripts, and prediction scripts. This section will take the PP-OCRv3 English recognition model as an example:

### 2.1. Start Training

First download the pretrain model, you can download the trained model to finetune on the icdar2015 data:

```bash linenums="1"
cd PaddleOCR/
# Download the pre-trained model of en_PP-OCRv3
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar
# Decompress model parameters
cd pretrain_models
tar -xf en_PP-OCRv3_rec_train.tar && rm -rf en_PP-OCRv3_rec_train.tar
```

Start training:

```bash linenums="1"
# GPU training Support single card and multi-card training
# Training icdar15 English data and The training log will be automatically saved as train.log under "{save_model_dir}"

#specify the single card training(Long training time, not recommended)
python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy

#specify the card number through --gpus
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy
```

PaddleOCR supports alternating training and evaluation. You can modify `eval_batch_step` in `configs/rec/rec_icdar15_train.yml` to set the evaluation frequency. By default, it is evaluated every 500 iter and the best acc model is saved under `output/rec_CRNN/best_accuracy` during the evaluation process.

If the evaluation set is large, the test will be time-consuming. It is recommended to reduce the number of evaluations, or evaluate after training.

- Tip: You can use the `-c` parameter to select multiple model configurations under the `configs/rec/` path for training. The recognition algorithms supported at [rec_algorithm](../../algorithm/overview.en.md):

For training Chinese data, it is recommended to use
[ch_PP-OCRv3_rec_distillation.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml). If you want to try the result of other algorithms on the Chinese data set, please refer to the following instructions to modify the configuration file:

Take `ch_PP-OCRv3_rec_distillation.yml` as an example:

```yaml linenums="1"
Global:
  ...
  # Add a custom dictionary, such as modify the dictionary, please point the path to the new dictionary
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt
  # Modify character type
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
          image_shape: [3, 48, 320]
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
          image_shape: [3, 48, 320]
      ...
  loader:
    # Eval batch_size for Single card
    batch_size_per_card: 256
    ...
```

**Note that the configuration file for prediction/evaluation must be consistent with the training.**

### 2.2 Load Trained Model and Continue Training

If you expect to load trained model and continue the training again, you can specify the parameter `Global.checkpoints` as the model path to be loaded.

For example:

```bash linenums="1"
python3 tools/train.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints=./your/trained/model
```

**Note**: The priority of `Global.checkpoints` is higher than that of `Global.pretrained_model`, that is, when two parameters are specified at the same time, the model specified by `Global.checkpoints` will be loaded first. If the model path specified by `Global.checkpoints` is wrong, the one specified by `Global.pretrained_model` will be loaded.

### 2.3 Training with New Backbone

The network part completes the construction of the network, and PaddleOCR divides the network into four parts, which are under [ppocr/modeling](../../ppocr/modeling). The data entering the network will pass through these four parts in sequence(transforms->backbones->
necks->heads).

```bash linenums="1"
├── architectures # Code for building network
├── transforms    # Image Transformation Module
├── backbones     # Feature extraction module
├── necks         # Feature enhancement module
└── heads         # Output module
```

If the Backbone to be replaced has a corresponding implementation in PaddleOCR, you can directly modify the parameters in the `Backbone` part of the configuration yml file.

However, if you want to use a new Backbone, an example of replacing the backbones is as follows:

1. Create a new file under the [ppocr/modeling/backbones](../../ppocr/modeling/backbones) folder, such as my_backbone.py.
2. Add code in the my_backbone.py file, the sample code is as follows:

```python linenums="1"
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MyBackbone(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(MyBackbone, self).__init__()
        # your init code
        self.conv = nn.xxxx

    def forward(self, inputs):
        # your network forward
        y = self.conv(inputs)
        return y
```

3. Import the added module in the [ppocr/modeling/backbones/\__init\__.py](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/__init__.py) file.

After adding the four-part modules of the network, you only need to configure them in the configuration file to use, such as:

```yaml linenums="1"
  Backbone:
    name: MyBackbone
    args1: args1
```

**NOTE**: More details about replace Backbone and other module can be found in [doc](../../algorithm/add_new_algorithm.en.md).

### 2.4. Mixed Precision Training

If you want to speed up your training further, you can use [Auto Mixed Precision Training](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/performance_improving/amp_en.html), taking a single machine and a single gpu as an example, the commands are as follows:

```bash linenums="1"
python3 tools/train.py -c configs/rec/rec_icdar15_train.yml \
     -o Global.pretrained_model=./pretrain_models/rec_mv3_none_bilstm_ctc_v2.0_train \
     Global.use_amp=True Global.scale_loss=1024.0 Global.use_dynamic_loss_scaling=True
```

### 2.5. Distributed Training

During multi-machine multi-gpu training, use the `--ips` parameter to set the used machine IP address, and the `--gpus` parameter to set the used GPU ID:

```bash linenums="1"
python3 -m paddle.distributed.launch --ips="xx.xx.xx.xx,xx.xx.xx.xx" --gpus '0,1,2,3' tools/train.py -c configs/rec/rec_icdar15_train.yml \
     -o Global.pretrained_model=./pretrain_models/rec_mv3_none_bilstm_ctc_v2.0_train
```

**Note:**
1. When using multi-machine and multi-gpu training, you need to replace the ips value in the above command with the address of your machine, and the machines need to be able to ping each other.
2. Training needs to be launched separately on multiple machines. The command to view the ip address of the machine is `ifconfig`. 
3. For more details about the distributed training speedup ratio, please refer to [Distributed Training Tutorial](../blog/distributed_training.en.md).

### 2.6. Training with Knowledge Distillation

Knowledge distillation is supported in PaddleOCR for text recognition training process. For more details, please refer to [doc](../model_compress/knowledge_distillation.en.md).

### 2.7. Multi-Language Model Training

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

For more supported languages, please refer to : [Multi-language model](../blog/multi_languages.en.md)

If you want to finetune on the basis of the existing model effect, please refer to the following instructions to modify the configuration file:

Take `rec_french_lite_train` as an example:

```yaml linenums="1"
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

### 2.8 Training on other platform(Windows/macOS/Linux DCU)

- Windows GPU/CPU
The Windows platform is slightly different from the Linux platform:
Windows platform only supports `single gpu` training and inference, specify GPU for training `set CUDA_VISIBLE_DEVICES=0`
On the Windows platform, DataLoader only supports single-process mode, so you need to set `num_workers` to 0;

- macOS
GPU mode is not supported, you need to set `use_gpu` to False in the configuration file, and the rest of the training evaluation prediction commands are exactly the same as Linux GPU.

- Linux DCU
Running on a DCU device requires setting the environment variable `export HIP_VISIBLE_DEVICES=0,1,2,3`, and the rest of the training and evaluation prediction commands are exactly the same as the Linux GPU.

## 2.9 Fine-tuning

In actual use, it is recommended to load the official pre-trained model and fine-tune it in your own data set. For the fine-tuning method of the recognition model, please refer to: [Model Fine-tuning Tutorial](./finetune.en.md).

## 3. Evaluation and Test

### 3.1. Evaluation

The model parameters during training are saved in the `Global.save_model_dir` directory by default. When evaluating indicators, you need to set `Global.checkpoints` to point to the saved parameter file. The evaluation dataset can be set by modifying the `Eval.dataset.label_file_list` field in the `configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml` file.

```bash linenums="1"
# GPU evaluation, Global.checkpoints is the weight to be tested
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

### 3.2 Test

Using the model trained by paddleocr, you can quickly get prediction through the following script.

The default prediction picture is stored in `infer_img`, and the trained weight is specified via `-o Global.checkpoints`:

According to the `save_model_dir` and `save_epoch_step` fields set in the configuration file, the following parameters will be saved:

```text linenums="1"
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

Among them, best_accuracy._is the best model on the evaluation set; iter_epoch_x._ is the model saved at intervals of `save_epoch_step`; latest.* is the model of the last epoch.

```bash linenums="1"
# Predict English results
python3 tools/infer_rec.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model={path/to/weights}/best_accuracy  Global.infer_img=doc/imgs_words/en/word_1.png
```

Input image:

![img](./images/word_1-20240704092705543.png)

Get the prediction result of the input image:

```bash linenums="1"
infer_img: doc/imgs_words/en/word_1.png
        result: ('joint', 0.9998967)
```

The configuration file used for prediction must be consistent with the training. For example, you completed the training of the Chinese model with `python3 tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml`, you can use the following command to predict the Chinese model:

```bash linenums="1"
# Predict Chinese results
python3 tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/ch/word_1.jpg
```

Input image:

![img](./images/word_1-20240704092713071.jpg)

Get the prediction result of the input image:

```bash linenums="1"
infer_img: doc/imgs_words/ch/word_1.jpg
        result: ('韩国小馆', 0.997218)
```

### 4. Model Export and Prediction

**Inference Model** (saved using `paddle.jit.save`)

The inference model is a "frozen" version of the model, where both the model structure and model parameters are saved in a file. It is typically used for prediction and deployment scenarios.  
In contrast, the **checkpoint model** only saves the model's parameters and is mostly used for training resumption, etc. Compared to the checkpoint model, the inference model also includes the model structure information, which makes it more efficient for deployment, inference acceleration, and flexible integration with systems.

The process of converting a recognition model to an inference model is similar to the detection model conversion, as shown below:

```bash linenums="1"
# Enable old IR mode
export FLAGS_enable_pir_api=0

# -c Set the training algorithm yml configuration file
# -o Set optional parameters
# Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy  Global.save_inference_dir=./inference/en_PP-OCRv3_rec/
```

If you have a model trained on your own dataset with a different dictionary file, please make sure that you modify the `character_dict_path` in the configuration file to your dictionary file path.

After the conversion is successful, there are three files in the model save directory:

```text linenums="1"
inference/en_PP-OCRv3_rec/
    ├── inference.pdiparams         # The parameter file of recognition inference model
    ├── inference.pdiparams.info    # The parameter information of recognition inference model, which can be ignored
    └── inference.pdmodel           # The program file of recognition model
```

**Note**: If you need to store the model in the new IR mode (i.e., `.json` format), use the following command to switch to the new IR mode:

```bash
export FLAGS_enable_pir_api=1
python3 tools/export_model.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy Global.save_inference_dir=./inference/en_PP-OCRv3_rec/
```

Once successful, you will have two files in the directory:

```text
inference/en_PP-OCRv3_rec/
    ├── inference.pdiparams         # Model parameter file for the inference model
    └── inference.json              # Program file for the inference model
```

### Custom Model Inference

If you modified the text dictionary during training, you must specify the path to the custom dictionary when using the inference model for prediction. For more information about configuring and explaining inference hyperparameters, refer to the [Inference Hyperparameters Explanation Tutorial](../blog/inference_args.md).

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./your_inference_model" --rec_image_shape="3, 48, 320" --rec_char_dict_path="your_text_dict_path"
```

### 5. FAQ

**Q1:** Why is the prediction result inconsistent after converting a trained model to an inference model?

**A**: This is a common issue. It typically arises due to differences in the preprocessing and postprocessing parameters used during training and inference. To troubleshoot, check whether the preprocessing, postprocessing, and prediction settings in the configuration file used for training match those used during inference.
