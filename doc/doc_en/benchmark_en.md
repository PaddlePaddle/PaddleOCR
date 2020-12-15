# BENCHMARK

This document gives the performance of the series models for Chinese and English recognition.

## TEST DATA

We collected 300 images for different real application scenarios to evaluate the overall OCR system, including contract samples, license plates, nameplates, train tickets, test sheets, forms, certificates, street view images, business cards, digital meter, etc. The following figure shows some images of the test set.

<div align="center">
<img src="../datasets/doc.jpg"  width = "1000" height = "500" />
</div>

## MEASUREMENT

Explanation:
- v1.0 indicates DB+CRNN models without the strategies. v1.1 indicates the PP-OCR models with the strategies and the direction classify. slim_v1.1 indicates the PP-OCR models with prunner or quantization.

- The long size of the input for the text detector is 960.

- The evaluation time-consuming stage is the complete stage from image input to result output, including image pre-processing and post-processing.

- ```Intel Xeon 6148``` is the server-side CPU model. Intel MKL-DNN is used in the test to accelerate the CPU prediction speed.

- ```Snapdragon 855``` is a mobile processing platform model.

Compares the model size and F-score:

| Model Name                    | Model Size <br> of the <br> Whole System\(M\) | Model Size <br>of the Text <br> Detector\(M\) | Model Size <br> of the Direction <br> Classifier\(M\) | Model Size<br>of the Text <br> Recognizer \(M\) | F\-score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| ch\_ppocr\_mobile\_v1\.1 | 8\.1        | 2\.6        | 0\.9           | 4\.6        | 0\.5193      |
| ch\_ppocr\_server\_v1\.1 | 155\.1      | 47\.2       | 0\.9           | 107         | 0\.5414      |
| ch\_ppocr\_mobile\_v1\.0 | 8\.6        | 4\.1        | \-             | 4\.5        | 0\.393       |
| ch\_ppocr\_server\_v1\.0 | 203\.8      | 98\.5       | \-             | 105\.3      | 0\.4436      |

Compares the time-consuming on T4 GPU (ms):

| Model Name                     | Overall  | Text Detector  | Direction Classifier  | Text Recognizer |
|:-:|:-:|:-:|:-:|:-:|
| ch\_ppocr\_mobile\_v1\.1 | 137 | 35 | 24    | 78  |
| ch\_ppocr\_server\_v1\.1 | 204 | 39 | 25    | 140 |
| ch\_ppocr\_mobile\_v1\.0 | 117 | 41 | \-    | 76  |
| ch\_ppocr\_server\_v1\.0 | 199 | 52 | \-    | 147 |

Compares the time-consuming on CPU (ms):

| Model Name                     | Overall  | Text Detector  | Direction Classifier  | Text Recognizer |
|:-:|:-:|:-:|:-:|:-:|
| ch\_ppocr\_mobile\_v1\.1 | 421  | 164 | 51    | 206 |
| ch\_ppocr\_mobile\_v1\.0 | 398  | 219 | \-    | 179 |

Compares the model size, F-score, the time-consuming on SD 855 of between the slim models and the original models:

| Model Name                          | Model Size <br> of the <br> Whole System\(M\) | Model Size <br>of the Text <br> Detector\(M\) | Model Size <br> of the Direction <br> Classifier\(M\) | Model Size<br>of the Text <br> Recognizer \(M\) | F\-score | SD 855<br>\(ms\) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ch\_ppocr\_mobile\_v1\.1       | 8\.1        | 2\.6        | 0\.9           | 4\.6        | 0\.5193      | 306          |
| ch\_ppocr\_mobile\_slim\_v1\.1 | 3\.5        | 1\.4        | 0\.5           | 1\.6        | 0\.521       | 268          |
