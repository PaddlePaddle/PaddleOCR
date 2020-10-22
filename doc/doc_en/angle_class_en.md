## TEXT ANGLE CLASSIFICATION

### DATA PREPARATION

Please organize the dataset as follows:

The default storage path for training data is `PaddleOCR/train_data/cls`, if you already have a dataset on your disk, just create a soft link to the dataset directory:

```
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/cls/dataset
```

please refer to the following to organize your data.

- Training set

First put the training images in the same folder (train_images), and use a txt file (cls_gt_train.txt) to store the image path and label.

* Note: by default, the image path and image label are split with `\t`, if you use other methods to split, it will cause training error

0 and 180 indicate that the angle of the image is 0 degrees and 180 degrees, respectively.

```
" Image file name           Image annotation "

train_data/word_001.jpg   0
train_data/word_002.jpg   180
```

The final training set should have the following file structure:

```
|-train_data
    |-cls
        |- cls_gt_train.txt
        |- train
            |- word_001.png
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

- Test set

Similar to the training set, the test set also needs to be provided a folder
containing all images (test) and a cls_gt_test.txt. The structure of the test set is as follows:

```
|-train_data
    |-cls
        |- cls_gt_test.txt
        |- test
            |- word_001.jpg
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

### TRAINING

PaddleOCR provides training scripts, evaluation scripts, and prediction scripts.

Start training:

```
# Set PYTHONPATH path
export PYTHONPATH=$PYTHONPATH:.
# GPU training Support single card and multi-card training, specify the card number through CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Training icdar15 English data
python3 tools/train.py -c configs/cls/cls_mv3.yml
```

- Data Augmentation

PaddleOCR provides a variety of data augmentation methods. If you want to add disturbance during training, please set `distort: true` in the configuration file.

The default perturbation methods are: cvtColor, blur, jitter, Gasuss noise, random crop, perspective, color reverse, RandAugment.

Except for RandAugment, each disturbance method is selected with a 50% probability during the training process. For specific code implementation, please refer to:
[randaugment.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/cls/randaugment.py)
[img_tools.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/rec/img_tools.py)


- Training

PaddleOCR supports alternating training and evaluation. You can modify `eval_batch_step` in `configs/cls/cls_mv3.yml` to set the evaluation frequency. By default, it is evaluated every 500 iter and the best acc model is saved under `output/cls_mv3/best_accuracy` during the evaluation process.

If the evaluation set is large, the test will be time-consuming. It is recommended to reduce the number of evaluations, or evaluate after training.

**Note that the configuration file for prediction/evaluation must be consistent with the training.**

### EVALUATION

The evaluation data set can be modified via `configs/cls/cls_reader.yml` setting of `label_file_path` in EvalReader.

```
export CUDA_VISIBLE_DEVICES=0
# GPU evaluation, Global.checkpoints is the weight to be tested
python3 tools/eval.py -c configs/cls/cls_mv3.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

### PREDICTION

* Training engine prediction

Using the model trained by paddleocr, you can quickly get prediction through the following script.

The default prediction picture is stored in `infer_img`, and the weight is specified via `-o Global.checkpoints`:

```
# Predict English results
python3 tools/infer_rec.py -c configs/cls/cls_mv3.yml -o Global.checkpoints={path/to/weights}/best_accuracy TestReader.infer_img=doc/imgs_words/en/word_1.jpg
```

Input image:

![](../imgs_words/en/word_1.png)

Get the prediction result of the input image:

```
infer_img: doc/imgs_words/en/word_1.png
    scores: [[0.93161047 0.06838956]]
    label: [0]
```
