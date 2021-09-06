# PaddleOCR Overview and Project Clone

## 1. PaddleOCR Overview

PaddleOCR contains rich text detection, text recognition and end-to-end algorithms. Combining actual testing and industrial experience, PaddleOCR chooses DB and CRNN as the basic detection and recognition models, and proposes a series of models, named PP-OCR, for industrial applications after a series of optimization strategies. The PP-OCR model is aimed at general scenarios and forms a model library according to different languages. Based on the capabilities of PP-OCR, PaddleOCR releases the PP-Structure tool library for document scene tasks, including two major tasks: layout analysis and table recognition. In order to get through the entire process of industrial landing, PaddleOCR provides large-scale data production tools and a variety of prediction deployment tools to help developers quickly turn ideas into reality.

<div align="center">
    <img src="../overview_en.png">
</div>



## 2. Project Clone

### **2.1 Clone PaddleOCR repo**

```
# Recommend
git clone https://github.com/PaddlePaddle/PaddleOCR

# If you cannot pull successfully due to network problems, you can also choose to use the code hosting on the cloud:

git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The cloud-hosting code may not be able to synchronize the update with this GitHub project in real time. There might be a delay of 3-5 days. Please give priority to the recommended method.
```

### **2.2 Install third-party libraries**

```
cd PaddleOCR
pip3 install -r requirements.txt
```

If you getting this error `OSError: [WinError 126] The specified module could not be found` when you install shapely on windows.

Please try to download Shapely whl file using [http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely).

Reference: [Solve shapely installation on windows](