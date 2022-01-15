#!/usr/bin/env bash

set -xe
# install PaddleOCR whl if not yet
# pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl

# run program for English texts OCR from given directory
paddleocr --image_dir $1 --lang=en
# returns a list, item containing text box, text, and confidence

# this recognition mode is WITHOOT including structural recognition

paddleocr --image_dir $1 --lang=en

# grep recognized text and pipe into output file
egrep "" output.txt

  

# uncomment the last line in this block to use structural recognition
# details at (https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/quickstart.md#213-%E7%89%88%E9%9D%A2%E5%88%86%E6%9E%90)
# pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl