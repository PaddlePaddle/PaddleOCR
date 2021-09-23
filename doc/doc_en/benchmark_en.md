# BENCHMARK

This document gives the performance of the series models for Chinese and English recognition.

## TEST DATA

We collected 300 images for different real application scenarios to evaluate the overall OCR system, including contract samples, license plates, nameplates, train tickets, test sheets, forms, certificates, street view images, business cards, digital meter, etc. The following figure shows some images of the test set.

<div align="center">
<img src="../datasets/doc.jpg"  width = "1000" height = "500" />
</div>

## MEASUREMENT

Explanation:

- The long size of the input for the text detector is 960.

- The evaluation time-consuming stage is the complete stage from image input to result output, including image pre-processing and post-processing.

- ```Intel Xeon 6148``` is the server-side CPU model. Intel MKL-DNN is used in the test to accelerate the CPU prediction speed.

- ```Snapdragon 855``` is a mobile processing platform model.

Compares the model size and F-score:

| Model Name                    | Model Size <br> of the <br> Whole System\(M\) | Model Size <br>of the Text <br> Detector\(M\) | Model Size <br> of the Direction <br> Classifier\(M\) | Model Size<br>of the Text <br> Recognizer \(M\) | F\-score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| PP-OCRv2                 | 11\.6        | 3\.0        | 0\.9           | 8\.6        | 0\.5224      |
| PP-OCR mobile            |   8\.1       | 2\.6        | 0\.9           | 4\.6        | 0\.503       |
| PP-OCR server            | 155\.1       | 47\.2       | 0\.9           | 107         | 0\.570       |

Compares the time-consuming on CPU and T4 GPU (ms):

| Model Name    | CPU  | T4 GPU |
|:-:|:-:|:-:|
| PP-OCRv2      | 330  | 111 |
| PP-OCR mobile | 356  | 116|
| PP-OCR server | 1056 | 200 |

More indicators of PP-OCR series models can be referred to [PP-OCR Benchamrk](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/doc/doc_en/benchmark_en.md)
