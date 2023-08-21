#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

if [ ${MODE} = "lite_train_lite_infer" ];then
    python_name_list=$(func_parser_value "${lines[2]}")
    array=(${python_name_list}) 
    python_name=${array[0]}
    ${python_name} -m pip install -r requirement.txt
    if [[ ${model_name} =~ "det_res50_db" ]];then
        wget -nc https://paddle-wheel.bj.bcebos.com/benchmark/resnet50-19c8e357.pth -O /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth

        # 下载数据集并解压
        rm -rf datasets
        wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/benchmark_train/datasets.tar
        tar xf datasets.tar
    fi
elif [ ${MODE} = "benchmark_train" ];then
    python_name_list=$(func_parser_value "${lines[2]}")
    array=(${python_name_list}) 
    python_name=${array[0]}
    ${python_name} -m pip install -r requirement.txt
    if [[ ${model_name} =~ "det_res50_db" ]];then
        wget -nc https://paddle-wheel.bj.bcebos.com/benchmark/resnet50-19c8e357.pth -O /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth

        # 下载数据集并解压
        rm -rf datasets
        wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/benchmark_train/datasets.tar
        tar xf datasets.tar
        # expand gt.txt 2 times
        # cd ./train_data/icdar2015/text_localization
        # for i in `seq 2`;do cp train_icdar2015_label.txt dup$i.txt;done
        # cat dup* > train_icdar2015_label.txt && rm -rf dup*
        # cd ../../../
    fi
fi