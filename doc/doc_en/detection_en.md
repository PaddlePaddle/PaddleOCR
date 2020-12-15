# TEXT DETECTION

This section uses the icdar2015 dataset as an example to introduce the training, evaluation, and testing of the detection model in PaddleOCR.

## DATA PREPARATION
The icdar2015 dataset can be obtained from [official website](https://rrc.cvc.uab.es/?ch=4&com=downloads). Registration is required for downloading.

Decompress the downloaded dataset to the working directory, assuming it is decompressed under PaddleOCR/train_data/. In addition, PaddleOCR organizes many scattered annotation files into two separate annotation files for train and test respectively, which can be downloaded by wget:
```shell
# Under the PaddleOCR path
cd PaddleOCR/
wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt
wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt
```

After decompressing the data set and downloading the annotation file, PaddleOCR/train_data/ has two folders and two files, which are:
```
/PaddleOCR/train_data/icdar2015/text_localization/
  └─ icdar_c4_train_imgs/         Training data of icdar dataset
  └─ ch4_test_images/             Testing data of icdar dataset
  └─ train_icdar2015_label.txt    Training annotation of icdar dataset
  └─ test_icdar2015_label.txt     Test annotation of icdar dataset
```

The provided annotation file format is as follow, seperated by "\t":
```
" Image file name             Image annotation information encoded by json.dumps"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```
The image annotation after **json.dumps()** encoding is a list containing multiple dictionaries.

The `points` in the dictionary represent the coordinates (x, y) of the four points of the text box, arranged clockwise from the point at the upper left corner.

`transcription` represents the text of the current text box. **When its content is "###" it means that the text box is invalid and will be skipped during training.**

If you want to train PaddleOCR on other datasets, please build the annotation file according to the above format.


## TRAINING

First download the pretrained model. The detection model of PaddleOCR currently supports 3 backbones, namely MobileNetV3, ResNet18_vd and ResNet50_vd. You can use the model in [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/master/ppcls/modeling/architectures) to replace backbone according to your needs.
```shell
cd PaddleOCR/
# Download the pre-trained model of MobileNetV3
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x0_5_pretrained.tar
# or, download the pre-trained model of ResNet18_vd
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_vd_pretrained.tar
# or, download the pre-trained model of ResNet50_vd
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar

# decompressing the pre-training model file, take MobileNetV3 as an example
tar -xf ./pretrain_models/MobileNetV3_large_x0_5_pretrained.tar ./pretrain_models/

# Note: After decompressing the backbone pre-training weight file correctly, the file list in the folder is as follows:
./pretrain_models/MobileNetV3_large_x0_5_pretrained/
  └─ conv_last_bn_mean
  └─ conv_last_bn_offset
  └─ conv_last_bn_scale
  └─ conv_last_bn_variance
  └─ ......

```

#### START TRAINING
*If CPU version installed, please set the parameter `use_gpu` to `false` in the configuration.*
```shell
python3 tools/train.py -c configs/det/det_mv3_db.yml
```

In the above instruction, use `-c` to select the training to use the `configs/det/det_db_mv3.yml` configuration file.
For a detailed explanation of the configuration file, please refer to [config](./config_en.md).

You can also use `-o` to change the training parameters without modifying the yml file. For example, adjust the training learning rate to 0.0001
```shell
# single GPU training
python3 tools/train.py -c configs/det/det_mv3_db.yml -o Optimizer.base_lr=0.0001

# multi-GPU training
# Set the GPU ID used by the '--gpus' parameter; If your paddle version has less than 2.0rc1, please use '--selected_gpus'
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/det/det_mv3_db.yml -o Optimizer.base_lr=0.0001


```

#### load trained model and continue training
If you expect to load trained model and continue the training again, you can specify the parameter `Global.checkpoints` as the model path to be loaded.

For example:
```shell
python3 tools/train.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=./your/trained/model
```

**Note**: The priority of `Global.checkpoints` is higher than that of `Global.pretrain_weights`, that is, when two parameters are specified at the same time, the model specified by `Global.checkpoints` will be loaded first. If the model path specified by `Global.checkpoints` is wrong, the one specified by `Global.pretrain_weights` will be loaded.


## EVALUATION

PaddleOCR calculates three indicators for evaluating performance of OCR detection task: Precision, Recall, and Hmean.

Run the following code to calculate the evaluation indicators. The result will be saved in the test result file specified by `save_res_path` in the configuration file `det_db_mv3.yml`

When evaluating, set post-processing parameters `box_thresh=0.6`, `unclip_ratio=1.5`. If you use different datasets, different models for training, these two parameters should be adjusted for better result.

```shell
python3 tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints="{path/to/weights}/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```
The model parameters during training are saved in the `Global.save_model_dir` directory by default. When evaluating indicators, you need to set `Global.checkpoints` to point to the saved parameter file.

Such as:
```shell
python3 tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints="./output/det_db/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```

* Note: `box_thresh` and `unclip_ratio` are parameters required for DB post-processing, and not need to be set when evaluating the EAST model.

## TEST

Test the detection result on a single image:
```shell
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o Global.infer_img="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy"
```

When testing the DB model, adjust the post-processing threshold:
```shell
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o Global.infer_img="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```


Test the detection result on all images in the folder:
```shell
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o Global.infer_img="./doc/imgs_en/" Global.checkpoints="./output/det_db/best_accuracy"
```
