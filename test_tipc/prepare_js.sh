#!/bin/bash

set -o errexit
set -o nounset
shopt -s extglob

# paddlejs prepare 主要流程
# 1. 判断 node, npm 是否安装
# 2. 下载测试模型，当前检测模型是 ch_PP-OCRv2_det_infer ，识别模型是 ch_PP-OCRv2_rec_infer [1, 3, 32, 320]。如果需要替换模型，可直接将模型文件放在test_tipc/web/models/目录下。
#  - 文本检测模型：https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
#  - 文本识别模型：https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
#  - 文本识别模型[1, 3, 32, 320]：https://paddlejs.bj.bcebos.com/models/ch_PP-OCRv2_rec_infer.tar
#  - 保证较为准确的识别效果，需要将文本识别模型导出为输入shape[1, 3, 32, 320]的静态模型
# 3. 转换模型， model.pdmodel model.pdiparams 转换为 model.json chunk.dat（检测模型保存地址：test_tipc/web/models/ch_PP-OCRv2/det，识别模型保存地址：test_tipc/web/models/ch_PP-OCRv2/rec）
# 4. 安装最新版本 ocr sdk  @paddlejs-models/ocr@latest
# 5. 安装测试环境依赖 puppeteer、jest、jest-puppeteer，如果检查到已经安装，则不会进行二次安装

# 判断是否安装了node
if ! type node >/dev/null 2>&1; then
    echo -e "\033[31m node 未安装 \033[0m"
    exit
fi

# 判断是否安装了npm
if ! type npm >/dev/null 2>&1; then
    echo -e "\033[31m npm 未安装 \033[0m"
    exit
fi

# MODE be 'js_infer'
MODE=$1
# js_infer MODE , load model file and convert model to js_infer
if [ ${MODE} != "js_infer" ];then
    echo "Please change mode to 'js_infer'"
    exit
fi


# saved_model_name
det_saved_model_name=ch_PP-OCRv2_det_infer
rec_saved_model_name=ch_PP-OCRv2_rec_infer

# model_path
model_path=test_tipc/web/models/

rm -rf $model_path

echo ${model_path}${det_saved_model_name}
echo ${model_path}${rec_saved_model_name}

# download ocr_det inference model
wget -nc -P $model_path https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
cd $model_path && tar xf ch_PP-OCRv2_det_infer.tar && cd ../../../

# download ocr_rec inference model
wget -nc -P $model_path https://paddlejs.bj.bcebos.com/models/ch_PP-OCRv2_rec_infer.tar
cd $model_path && tar xf ch_PP-OCRv2_rec_infer.tar && cd ../../../

MYDIR=`pwd`
echo $MYDIR

pip3 install paddlejsconverter

# convert inference model to web model: model.json、chunk.dat
paddlejsconverter \
   --modelPath=$model_path$det_saved_model_name/inference.pdmodel \
   --paramPath=$model_path$det_saved_model_name/inference.pdiparams \
   --outputDir=$model_path$det_saved_model_name/ \

paddlejsconverter \
   --modelPath=$model_path$rec_saved_model_name/inference.pdmodel \
   --paramPath=$model_path$rec_saved_model_name/inference.pdiparams \
   --outputDir=$model_path$rec_saved_model_name/ \

# always install latest ocr sdk
cd test_tipc/web
echo -e "\033[33m Installing the latest ocr sdk... \033[0m"
npm install @paddlejs-models/ocr@latest
npm info @paddlejs-models/ocr
echo -e "\033[32m The latest ocr sdk installed completely.!~ \033[0m"

# install dependencies
if [ `npm list --dept 0 | grep puppeteer | wc -l` -ne 0 ] && [ `npm list --dept 0 | grep jest | wc -l` -ne 0 ];then
   echo -e "\033[32m Dependencies have installed \033[0m"
else
   echo -e "\033[33m Installing dependencies ... \033[0m"
   npm install jest jest-puppeteer puppeteer
   echo -e "\033[32m Dependencies installed completely.!~ \033[0m"
fi

# del package-lock.json
rm package-lock.json
