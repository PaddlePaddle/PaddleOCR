#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
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

if [ ${MODE} = "lite_train_infer" ];then
    # pretrain lite train data
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar
    cd ./train_data/ && tar xf icdar2015_lite.tar
    ln -s ./icdar2015_lite ./icdar2015
    cd ../
elif [ ${MODE} = "whole_train_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar
    cd ./train_data/ && tar xf icdar2015.tar && cd ../
elif [ ${MODE} = "whole_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_infer.tar
    cd ./train_data/ && tar xf icdar2015_infer.tar
    ln -s ./icdar2015_infer ./icdar2015
    cd ../
    epoch=10
    eval_batch_step=10
else
    rm -rf ./train_data/icdar2015
    if [[ ${model_name} = "ocr_det" ]]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar
        eval_model_name="ch_ppocr_mobile_v2.0_det_infer"
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    else 
        eval_model_name="ch_ppocr_mobile_v2.0_rec_train"
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar
        cd ./inference && tar xf ${eval_model_name}.tar && cd ../
    fi 
fi


