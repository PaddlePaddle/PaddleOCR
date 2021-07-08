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
model_name=$(func_parser_value "${lines[0]}")
train_model_list=$(func_parser_value "${lines[0]}")
trainer_list=$(func_parser_value "${lines[10]}")

echo $train_model_list
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2
# prepare pretrained weights and dataset
if [ ${train_model_list[*]} = "ocr_det" ]; then
  wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
  wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar
  cd pretrain_models && tar xf det_mv3_db_v2.0_train.tar && cd ../
  fi
if [ ${MODE} = "lite_train_infer" ];then
    # pretrain lite train data
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar # todo change to bcebos

    cd ./train_data/ && tar xf icdar2015_lite.tar && tar xf ic15_data.tar
    ln -s ./icdar2015_lite ./icdar2015
    cd ../
    epoch=10
    eval_batch_step=10
elif [ ${MODE} = "whole_train_infer" ];then
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar
    cd ./train_data/ && tar xf icdar2015.tar && tar xf ic15_data.tar && cd ../
    epoch=500
    eval_batch_step=200
elif [ ${MODE} = "whole_infer" ];then
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_infer.tar
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar
    cd ./train_data/ && tar xf icdar2015_infer.tar && tar xf ic15_data.tar
    ln -s ./icdar2015_infer ./icdar2015
    cd ../
    epoch=10
    eval_batch_step=10
else
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar
    if [ ${model_name} = "ocr_det" ]; then
        eval_model_name="ch_ppocr_mobile_v2.0_det_train"
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
        cd ./inference && tar xf ${eval_model_name}.tar && cd ../
    else 
        eval_model_name="ch_ppocr_mobile_v2.0_rec_train"
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar
        cd ./inference && tar xf ${eval_model_name}.tar && cd ../
    fi 
fi


IFS='|'
for train_model in ${train_model_list[*]}; do 
    if [ ${train_model} = "ocr_det" ];then
        model_name="ocr_det"
        yml_file="configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml"
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar
        cd ./inference && tar xf ch_det_data_50.tar && cd ../
        img_dir="./inference/ch_det_data_50/all-sum-510"
        data_dir=./inference/ch_det_data_50/
        data_label_file=[./inference/ch_det_data_50/test_gt_50.txt]
    elif [ ${train_model} = "ocr_rec" ];then
        model_name="ocr_rec"
        yml_file="configs/rec/rec_mv3_none_bilstm_ctc.yml"
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar
        cd ./inference && tar xf rec_inference.tar  && cd ../
        img_dir="./inference/rec_inference/"
        data_dir=./inference/rec_inference
        data_label_file=[./inference/rec_inference/rec_gt_test.txt]
    fi

    # eval 
    for slim_trainer in ${trainer_list[*]}; do 
        if [ ${slim_trainer} = "norm" ]; then
            if [ ${model_name} = "ocr_det" ]; then
                eval_model_name="ch_ppocr_mobile_v2.0_det_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            else 
                eval_model_name="ch_ppocr_mobile_v2.0_rec_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            fi 
        elif [ ${slim_trainer} = "pact" ]; then
            if [ ${model_name} = "ocr_det" ]; then
                eval_model_name="ch_ppocr_mobile_v2.0_det_quant_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_quant_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            else
                eval_model_name="ch_ppocr_mobile_v2.0_rec_quant_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_quant_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            fi
        elif [ ${slim_trainer} = "distill" ]; then
            if [ ${model_name} = "ocr_det" ]; then
                eval_model_name="ch_ppocr_mobile_v2.0_det_distill_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_distill_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            else
                eval_model_name="ch_ppocr_mobile_v2.0_rec_distill_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_distill_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            fi 
        elif [ ${slim_trainer} = "fpgm" ]; then
            if [ ${model_name} = "ocr_det" ]; then
                eval_model_name="ch_ppocr_mobile_v2.0_det_prune_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            else
                eval_model_name="ch_ppocr_mobile_v2.0_rec_prune_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_prune_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            fi
        fi
    done
done
