# PP-OCR Model Zoo
The PP-OCR model zoo section explains some basic concepts of the OCR model and how to quickly use the models in the PP-OCR model library.

This section contains two parts. Firstly, [PP-OCR Model Download](./models_list_en.md) explains the concept of PP-OCR model types and provides links to download all models.  The next [Python Inference for PP-OCR Model Zoo](./inference_ppocr_en.md) is an introduction to the use of the PP-OCR model library, which can quickly utilize the rich model library models to obtain test results through the Python inference engine.

------

Let's first understand some basic concepts.

- [Introduction to OCR](#introduction-to-ocr)
  * [Basic Concepts of OCR Detection Model](#basic-concepts-of-ocr-detection-model)
  * [Basic Concepts of OCR Recognition Model](#basic-concepts-of-ocr-recognition-model)
  * [PP-OCR Model](#pp-ocr-model)


## 1. Introduction to OCR

This section briefly introduces the basic concepts of OCR detection model and recognition model, and introduces PaddleOCR's PP-OCR model.

OCR (Optical Character Recognition, Optical Character Recognition) is currently the general term for text recognition. It is not limited to document or book text recognition, but also includes recognizing text in natural scenes. It can also be called STR (Scene Text Recognition).

OCR text recognition generally includes two parts, text detection and text recognition. The text detection module first uses detection algorithms to detect text lines in the image. And then the recognition algorithm to identify the specific text in the text line.


### 1.1 Basic Concepts of OCR Detection Model

Text detection can locate the text area in the image, and then usually mark the word or text line in the form of a bounding box. Traditional text detection algorithms mostly extract features manually, which are characterized by fast speed and good effect in simple scenes, but the effect will be greatly reduced when faced with natural scenes. Currently, deep learning methods are mostly used.

Text detection algorithms based on deep learning can be roughly divided into the following categories:
1. Method based on target detection. Generally, after the text box is predicted, the final text box is filtered through NMS, which is mostly four-point text box, which is not ideal for curved text scenes. Typical algorithms are methods such as EAST and Text Box.
2. Method based on text segmentation. The text line is regarded as the segmentation target, and then the external text box is constructed through the segmentation result, which can handle curved text, and the effect is not ideal for the text cross scene problem. Typical algorithms are DB, PSENet and other methods.
3. Hybrid target detection and segmentation method.


### 1.2 Basic Concepts of OCR Recognition Model

The input of the OCR recognition algorithm is generally text lines images which has less background information, and the text information occupies the main part. The recognition algorithm can be divided into two types of algorithms:
1. CTC-based method. The text prediction module of the recognition algorithm is based on CTC, and the commonly used algorithm combination is CNN+RNN+CTC. There are also some algorithms that try to add transformer modules to the network and so on.
2. Attention-based method. The text prediction module of the recognition algorithm is based on Attention, and the commonly used algorithm combination is CNN+RNN+Attention.


### 1.3 PP-OCR Model

PaddleOCR integrates many OCR algorithms, text detection algorithms include DB, EAST, SAST, etc., text recognition algorithms include CRNN, RARE, StarNet, Rosetta, SRN and other algorithms.

Among them, PaddleOCR has released the PP-OCR series model for the general OCR in Chinese and English natural scenes. The PP-OCR model is composed of the DB+CRNN algorithm. It uses massive Chinese data training and model tuning methods to have high text detection and recognition capabilities in Chinese scenes. And PaddleOCR has launched a high-precision and ultra-lightweight PP-OCRv2 model. The detection model is only 3M, and the recognition model is only 8.5M. Using [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)'s model quantification method, the detection model can be compressed to 0.8M without reducing the accuracy. The recognition is compressed to 3M, which is more suitable for mobile deployment scenarios.
