# How to make your own custom ultra-lightweight models?

The process of making a customized ultra-lightweight models can be divided into three steps: training text detection model, training text recognition model, and make prediction with trained models.

## step1: Train text detection model

PaddleOCR provides two text detection algorithms: EAST and DB. Both support MobileNetV3 and ResNet50_vd backbone networks. Select the corresponding configuration file as needed and start training. For example, to train with MobileNetV3 as the backbone network for DB detection model :
```
python3 tools/train.py -c configs/det/det_mv3_db.yml
```
For more details about data preparation and training tutorials, refer to the documentation [Text detection model training/evaluation/prediction](./detection.md)

## step2: Train text recognition model

PaddleOCR provides four text recognition algorithms: CRNN, Rosetta, STAR-Net, and RARE. They all support two backbone networks, MobileNetV3 and ResNet34_vd, and select the corresponding configuration files as needed to start training. For example, to train a CRNN recognition model that uses MobileNetV3 as the backbone network:
```
python3 tools/train.py -c configs/rec/rec_chinese_lite_train.yml
```
For more details about data preparation and training tutorials, refer to the documentation [Text recognition model training/evaluation/prediction](./recognition.md)

## step3: Make prediction

PaddleOCR provides a concatenation tool for detection and recognition models, which can connect any trained detection model and any recognition model into a two-stage text recognition system. The input image goes through four main stages of text detection, detection frame correction, text recognition, and score filtering to output the text position and recognition results, and at the same time, the results can be selected for visualization.

When performing prediction, you need to specify the path of a single image or a image folder through the parameter `image_dir`, the parameter `det_model_dir` specifies the path to detect the inference model, and the parameter `rec_model_dir` specifies the path to identify the inference model. The visual recognition results are saved to the `./inference_results` folder by default.

```
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/det/"  --rec_model_dir="./inference/rec/"
```
For more text detection and recognition concatenation, please refer to the document [Inference](./inference.md)
