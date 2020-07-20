#!/bin/bash

mkdir -p  $1/demo/cxx/ocr/debug/
cp  ../../ppocr/utils/ppocr_keys_v1.txt  $1/demo/cxx/ocr/debug/
cp -r  ./*   $1/demo/cxx/ocr/
cp ./config.txt  $1/demo/cxx/ocr/debug/
cp ../../doc/imgs/11.jpg  $1/demo/cxx/ocr/debug/

echo "Prepare Done"
