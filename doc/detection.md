# Text detection

This section uses the icdar15 dataset as an example to introduce the training, evaluation, and testing of the detection model in PaddleOCR.

## Data preparation
The icdar2015 dataset can be obtained from [official website](https://rrc.cvc.uab.es/?ch=4&com=downloads). Registration is required for downloading.

Decompress the downloaded dataset to the working directory, assuming it is decompressed under PaddleOCR/train_data/. In addition, PaddleOCR organizes scattered annotation files into separate annotation files. You can download by wget:
```
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

The label file format provided is:
```
" Image file name                    Image annotation information encoded by json.dumps"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]], ...}]
```
The image annotation information before json.dumps encoding is a list containing multiple dictionaries. The `points` in the dictionary represent the coordinates (x, y) of the four points of the text box, arranged clockwise from the point in the upper left corner.

`transcription` represents the text of the current text box, and this information is not needed in the text detection task.
If you want to train PaddleOCR on other datasets, you can build the annotation file according to the above format。


## Quickstart training

First download the pretrained model. The detection model of PaddleOCR currently supports two backbones, namely MobileNetV3 and ResNet50_vd. You can use the model in [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/master/ppcls/modeling/architectures) to replace backbone according to your needs.
```
cd PaddleOCR/
# Download the pre-trained model of MobileNetV3
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x0_5_pretrained.tar
# Download the pre-trained model of ResNet50
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
```

**Start training**
```
python3 tools/train.py -c configs/det/det_mv3_db.yml
```

In the above instruction, use -c to select the training to use the configs/det/det_db_mv3.yml configuration file.
For a detailed explanation of the configuration file, please refer to [link](./doc/config.md).

You can also use the -o parameter to change the training parameters without modifying the yml file. For example, adjust the training learning rate to 0.0001
```
python3 tools/train.py -c configs/det/det_mv3_db.yml -o Optimizer.base_lr=0.0001
```

## Index evaluation

PaddleOCR calculates three indicators related to OCR detection: Precision, Recall, and Hmean.

Run the following code to calculate the evaluation index based on the test result file specified by save_res_path in the configuration file det_db_mv3.yml

When evaluating, set post-processing parameters box_thresh=0.6, unclip_ratio=1.5, use different data sets, different models for training, these two parameters can be adjusted for optimization.

```
python3 tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints="{path/to/weights}/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```
The model parameters during training are saved in the Global.save_model_dir directory by default. When evaluating indicators, you need to set Global.checkpoints to point to the saved parameter file.

Such as:
```
python3 tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints="./output/det_db/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```

* Note: box_thresh and unclip_ratio are parameters required for DB post-processing, and do not need to be set when evaluating the EAST model.

## Test detection result

Test the detection result on a single image:
```
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o TestReader.infer_img="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy"
```

When testing the DB model, adjust the post-processing threshold:
```
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o TestReader.infer_img="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```


Test the detection effect of all images in the folder:
```
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o TestReader.infer_img="./doc/imgs_en/" Global.checkpoints="./output/det_db/best_accuracy"
```
