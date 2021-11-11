#!/bin/bash
source test_tipc/common_func.sh
source test_tipc/test_train_inference_python.sh

FILENAME=$1
# MODE be one of ['whole_infer']
MODE=$2

dataline=$(awk 'NR==1, NR==17{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")

infer_model_dir_list=$(func_parser_value "${lines[3]}")
infer_export_list=$(func_parser_value "${lines[4]}")
infer_is_quant=$(func_parser_value "${lines[5]}")
# parser inference 
inference_py=$(func_parser_value "${lines[6]}")
use_gpu_key=$(func_parser_key "${lines[7]}")
use_gpu_list=$(func_parser_value "${lines[7]}")
use_mkldnn_key=$(func_parser_key "${lines[8]}")
use_mkldnn_list=$(func_parser_value "${lines[8]}")
cpu_threads_key=$(func_parser_key "${lines[9]}")
cpu_threads_list=$(func_parser_value "${lines[9]}")
batch_size_key=$(func_parser_key "${lines[10]}")
batch_size_list=$(func_parser_value "${lines[10]}")
use_trt_key=$(func_parser_key "${lines[11]}")
use_trt_list=$(func_parser_value "${lines[11]}")
precision_key=$(func_parser_key "${lines[12]}")
precision_list=$(func_parser_value "${lines[12]}")
infer_model_key=$(func_parser_key "${lines[13]}")
image_dir_key=$(func_parser_key "${lines[14]}")
infer_img_dir=$(func_parser_value "${lines[14]}")
save_log_key=$(func_parser_key "${lines[15]}")
benchmark_key=$(func_parser_key "${lines[16]}")
benchmark_value=$(func_parser_value "${lines[16]}")
infer_key1=$(func_parser_key "${lines[17]}")
infer_value1=$(func_parser_value "${lines[17]}")


LOG_PATH="./test_tipc/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"


if [ ${MODE} = "whole_infer" ]; then
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    # set CUDA_VISIBLE_DEVICES
    eval $env
    export Count=0
    IFS="|"
    infer_run_exports=(${infer_export_list})
    infer_quant_flag=(${infer_is_quant})
    for infer_model in ${infer_model_dir_list[*]}; do
        # run export
        if [ ${infer_run_exports[Count]} != "null" ];then
            save_infer_dir=$(dirname $infer_model)
            set_export_weight=$(func_set_params "${export_weight}" "${infer_model}")
            set_save_infer_key=$(func_set_params "${save_infer_key}" "${save_infer_dir}")
            export_cmd="${python} ${infer_run_exports[Count]} ${set_export_weight} ${set_save_infer_key}"
            echo ${infer_run_exports[Count]} 
            echo  $export_cmd
            eval $export_cmd
            status_export=$?
            status_check $status_export "${export_cmd}" "${status_log}"
        else
            save_infer_dir=${infer_model}
        fi
        #run inference
        is_quant=${infer_quant_flag[Count]}
        if [ ${MODE} = "klquant_infer" ]; then
            is_quant="True"
        fi
        func_inference "${python}" "${inference_py}" "${save_infer_dir}" "${LOG_PATH}" "${infer_img_dir}" ${is_quant}
        Count=$(($Count + 1))
    done
fi

