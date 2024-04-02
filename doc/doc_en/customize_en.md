# HOW TO MAKE YOUR OWN LIGHTWEIGHT OCR MODEL?

The process of making a customized ultra-lightweight OCR models can be divided into three steps: training text detection model, training text recognition model, and concatenate the predictions from previous steps.

## STEP1: TRAIN TEXT DETECTION MODEL

PaddleOCR provides two text detection algorithms: EAST and DB. Both support MobileNetV3 and ResNet50_vd backbone networks, select the corresponding configuration file as needed and start training. For example, to train with MobileNetV3 as the backbone network for DB detection model :
```
python3 tools/train.py -c configs/det/det_mv3_db.yml 2>&1 | tee det_db.log
```
For more details about data preparation and training tutorials, refer to the documentation [Text detection model training/evaluation/prediction](./detection_en.md)

## STEP2: TRAIN TEXT RECOGNITION MODEL

PaddleOCR provides four text recognition algorithms: CRNN, Rosetta, STAR-Net, and RARE. They all support two backbone networks: MobileNetV3 and ResNet34_vd, select the corresponding configuration files as needed to start training. For example, to train a CRNN recognition model that uses MobileNetV3 as the backbone network:
```
python3 tools/train.py -c configs/rec/rec_chinese_lite_train.yml 2>&1 | tee rec_ch_lite.log
```
For more details about data preparation and training tutorials, refer to the documentation [Text recognition model training/evaluation/prediction](./recognition_en.md)

## STEP3: CONCATENATE PREDICTIONS

PaddleOCR provides a concatenation tool for detection and recognition models, which can connect any trained detection model and any recognition model into a two-stage text recognition system. The input image goes through four main stages: text detection, text rectification, text recognition, and score filtering to output the text position and recognition results, and at the same time, you can choose to visualize the results.

When performing prediction, you need to specify the path of a single image or a image folder through the parameter `image_dir`, the parameter `det_model_dir` specifies the path of detection model, and the parameter `rec_model_dir` specifies the path of recognition model. The visualized results are saved to the `./inference_results` folder by default.

```
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/det/"  --rec_model_dir="./inference/rec/"
```
For more details about text detection and recognition concatenation, please refer to the document [Inference](./inference_en.md)
