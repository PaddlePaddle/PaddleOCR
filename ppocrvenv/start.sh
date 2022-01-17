#!/usr/bin/env bash
# Copyright (c) 2016-2022 BigoneLab Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may
# not use this file except in compliance with the License. A copy of the
# License is located at
#   
#   https://github.com/XinyiXiang/PaddleOCR/blob/release/2.4/LICENSE
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

chmod +x $0
# install PaddleOCR whl
# pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl

# install grep
# dpkg -s $grep &> /dev/null  

#     if [ $? -ne 0 ]

#         then
#             echo "not installed"  
#             sudo apt-get update
#             sudo apt-get install $grep

#         else
#             echo    "installed"
#     fi

# Run program for English texts OCR from given directory
# returns a list, item containing text box, text, and confidence

# Recognize texts WITHOUT structural recognition
# and pipe unprocessed into an output file
echo "Processing images..."  
paddleocr --image_dir $1 --lang=en > raw_output.txt 

# Filter text recognition results and re-direct to destined output file 
echo "Filtering text output"
grep -oi "'.*'" raw_output.txt > text_output.txt  

# Recognize texts WITH structural recognition
# Read more at (https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/quickstart.md#213-%E7%89%88%E9%9D%A2%E5%88%86%E6%9E%90)
# To use structural recognition, uncomment the following lines
# pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl #layout-parser
# echo "Processing images with structural recognition..."
# paddleocr --image_dir $1 --lang=en --use_layout_parser=true > raw_output_with_layout.txt

# Enable output evaluation
# python3 bern_eval.py