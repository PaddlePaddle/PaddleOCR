#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer',  'lite_infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2

if [ ${MODE} = "lite_train_lite_infer" ];then
    # pretrain lite train data
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams  --no-check-certificate
    wget -nc -P ./pretrain_models/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar  --no-check-certificate
    if [ ${model_name} == "PPOCRv2_ocr_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
    cd ./pretrain_models/ && tar xf det_mv3_db_v2.0_train.tar && cd ../
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar --no-check-certificate
    wget -nc -P ./deploy/slim/prune https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/sen.pickle --no-check-certificate
    
    cd ./train_data/ && tar xf icdar2015_lite.tar && tar xf ic15_data.tar
    ln -s ./icdar2015_lite ./icdar2015
    cd ../
    cd ./inference && tar xf rec_inference.tar && cd ../
elif [ ${MODE} = "whole_train_whole_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams --no-check-certificate
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    cd ./train_data/ && tar xf icdar2015.tar && tar xf ic15_data.tar && cd ../
    if [ ${model_name} == "PPOCRv2_ocr_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
elif [ ${MODE} = "lite_train_whole_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams --no-check-certificate
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_infer.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    cd ./train_data/ && tar xf icdar2015_infer.tar && tar xf ic15_data.tar
    ln -s ./icdar2015_infer ./icdar2015
    cd ../
    if [ ${model_name} == "PPOCRv2_ocr_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
elif [ ${MODE} = "whole_infer" ];then
    if [ ${model_name} = "ocr_det" ]; then
        eval_model_name="ch_ppocr_mobile_v2.0_det_train"
        rm -rf ./train_data/icdar2015
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ocr_server_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_train.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ocr_system_mobile" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ocr_system_server" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ocr_rec" ]; then
        rm -rf ./train_data/ic15_data
        eval_model_name="ch_ppocr_mobile_v2.0_rec_infer"
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ocr_server_rec" ]; then
        rm -rf ./train_data/ic15_data
        eval_model_name="ch_ppocr_server_v2.0_rec_infer"
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf rec_inference.tar && cd ../
    fi 

    elif [ ${model_name} = "PPOCRv2_ocr_det" ]; then
        eval_model_name="ch_PP-OCRv2_det_infer"
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    fi

if [ ${MODE} = "klquant_whole_infer" ]; then
    if [ ${model_name} = "ocr_det" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && cd ../ 
    fi
    if [ ${model_name} = "PPOCRv2_ocr_det" ]; then
        eval_model_name="ch_PP-OCRv2_det_infer"
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    fi 
fi

if [ ${MODE} = "cpp_infer" ];then
    if [ ${model_name} = "ocr_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ocr_rec" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf rec_inference.tar && cd ../
    elif  [ ${model_name} = "ocr_system" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    fi 
fi

if [ ${MODE} = "serving_infer" ];then
    # prepare serving env
    python_name=$(func_parser_value "${lines[2]}")
    wget https://paddle-serving.bj.bcebos.com/chain/paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    ${python_name} -m pip install install paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    ${python_name} -m pip install paddle_serving_client==0.6.1
    ${python_name} -m pip install paddle-serving-app==0.6.3
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar
    cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_det_infer.tar && cd ../
fi


if [ ${MODE} = "lite_infer" ];then    
    # prepare lite nb model and test data
    current_dir=${PWD}
    wget -nc  -P ./models https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_det_opt.nb
    wget -nc  -P ./models https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_det_slim_opt.nb
    wget -nc  -P ./test_data https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar
    cd ./test_data && tar -xf icdar2015_lite.tar && rm icdar2015_lite.tar && cd ../
    # prepare lite env
    export http_proxy=http://172.19.57.45:3128
    export https_proxy=http://172.19.57.45:3128
    paddlelite_url=https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.with_cv.tar.gz
    paddlelite_zipfile=$(echo $paddlelite_url | awk -F "/" '{print $NF}')
    paddlelite_file=inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.with_cv
    wget ${paddlelite_url}
    tar -xf ${paddlelite_zipfile}
    mkdir -p  ${paddlelite_file}/demo/cxx/ocr/test_lite
    mv models test_data ${paddlelite_file}/demo/cxx/ocr/test_lite
    cp ppocr/utils/ppocr_keys_v1.txt deploy/lite/config.txt ${paddlelite_file}/demo/cxx/ocr/test_lite
    cp ./deploy/lite/* ${paddlelite_file}/demo/cxx/ocr/
    cp ${paddlelite_file}/cxx/lib/libpaddle_light_api_shared.so ${paddlelite_file}/demo/cxx/ocr/test_lite
    cp PTDN/configs/ppocr_det_mobile_params.txt PTDN/test_lite.sh PTDN/common_func.sh ${paddlelite_file}/demo/cxx/ocr/test_lite
    cd ${paddlelite_file}/demo/cxx/ocr/
    git clone https://github.com/LDOUBLEV/AutoLog.git
    unset http_proxy
    unset https_proxy
    make -j
    sleep 1
    make -j
    cp ocr_db_crnn test_lite && cp test_lite/libpaddle_light_api_shared.so test_lite/libc++_shared.so
    tar -cf test_lite.tar ./test_lite && cp test_lite.tar ${current_dir} && cd ${current_dir}
fi

