#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer']
MODE=$2

dataline=$(awk 'NR==1, NR==51{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
gpu_list=$(func_parser_value "${lines[3]}")
train_use_gpu_key=$(func_parser_key "${lines[4]}")
train_use_gpu_value=$(func_parser_value "${lines[4]}")
autocast_list=$(func_parser_value "${lines[5]}")
autocast_key=$(func_parser_key "${lines[5]}")
epoch_key=$(func_parser_key "${lines[6]}")
epoch_num=$(func_parser_params "${lines[6]}" "${MODE}")
save_model_key=$(func_parser_key "${lines[7]}")
train_batch_key=$(func_parser_key "${lines[8]}")
train_batch_value=$(func_parser_params "${lines[8]}" "${MODE}")
pretrain_model_key=$(func_parser_key "${lines[9]}")
pretrain_model_value=$(func_parser_value "${lines[9]}")
checkpoints_key=$(func_parser_key "${lines[10]}")
checkpoints_value=$(func_parser_value "${lines[10]}")
use_custom_key=$(func_parser_key "${lines[11]}")
use_custom_list=$(func_parser_value "${lines[11]}")
model_type_key=$(func_parser_key "${lines[12]}")
model_type_list=$(func_parser_value "${lines[12]}")
use_share_conv_key=$(func_parser_key "${lines[13]}")
use_share_conv_list=$(func_parser_value "${lines[13]}")
run_train_py=$(func_parser_value "${lines[14]}")

LOG_PATH="./test_tipc/extra_output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"

if [ ${MODE} = "lite_train_lite_infer" ] || [ ${MODE} = "whole_train_whole_infer" ]; then
    IFS="|"
    export Count=0
    USE_GPU_KEY=(${train_use_gpu_value})
    # select cpu\gpu\distribute training
    for gpu in ${gpu_list[*]}; do
        train_use_gpu=${USE_GPU_KEY[Count]}
        Count=$(($Count + 1))
        ips=""
        if [ ${gpu} = "-1" ];then
            env=""
        elif [ ${#gpu} -le 1 ];then
            env="export CUDA_VISIBLE_DEVICES=${gpu}"
            eval ${env}
        elif [ ${#gpu} -le 15 ];then
            IFS=","
            array=(${gpu})
            env="export CUDA_VISIBLE_DEVICES=${array[0]}"
            IFS="|"
        else
            IFS=";"
            array=(${gpu})
            ips=${array[0]}
            gpu=${array[1]}
            IFS="|"
            env=" "
        fi
        for autocast in ${autocast_list[*]}; do 
            # set amp
            if [ ${autocast} = "amp" ]; then
                set_amp_config="AMP.use_amp=True"
            else
                set_amp_config=" "
            fi

            if [ ${run_train_py} = "null" ]; then
                continue
            fi

            set_autocast=$(func_set_params "${autocast_key}" "${autocast}")
            set_epoch=$(func_set_params "${epoch_key}" "${epoch_num}")
            set_pretrain=$(func_set_params "${pretrain_model_key}" "${pretrain_model_value}")
            set_checkpoints=$(func_set_params "${checkpoints_key}" "${checkpoints_value}")
            set_batchsize=$(func_set_params "${train_batch_key}" "${train_batch_value}")
            set_use_gpu=$(func_set_params "${train_use_gpu_key}" "${train_use_gpu}")

            for custom_op in ${use_custom_list[*]}; do 
                for model_type in ${model_type_list[*]}; do
                    for share_conv in ${use_share_conv_list[*]}; do
                        set_use_custom_op=$(func_set_params "${use_custom_key}" "${custom_op}")
                        set_model_type=$(func_set_params "${model_type_key}" "${model_type}")
                        set_use_share_conv=$(func_set_params "${use_share_conv_key}" "${share_conv}")

                        set_save_model=$(func_set_params "${save_model_key}" "${save_log}")
                        if [ ${#gpu} -le 2 ];then  # train with cpu or single gpu
                            cmd="${python} ${run_train_py} ${set_use_gpu}  ${set_save_model}  ${set_epoch}  ${set_pretrain} ${set_checkpoints}  ${set_autocast} ${set_batchsize}  ${set_use_custom_op} ${set_model_type} ${set_use_share_conv} ${set_amp_config}"
                        elif [ ${#ips} -le 26 ];then  # train with multi-gpu
                            cmd="${python} -m paddle.distributed.launch --gpus=${gpu} ${run_train_py} ${set_use_gpu}  ${set_save_model}  ${set_epoch}  ${set_pretrain} ${set_checkpoints}  ${set_autocast} ${set_batchsize}  ${set_use_custom_op} ${set_model_type} ${set_use_share_conv} ${set_amp_config}"
                        else
                            cmd="${python} -m paddle.distributed.launch --ips=${ips} --gpus=${gpu} ${run_train_py} ${set_use_gpu}  ${set_save_model}  ${set_epoch}  ${set_pretrain} ${set_checkpoints}  ${set_autocast} ${set_batchsize}  ${set_use_custom_op} ${set_model_type} ${set_use_share_conv} ${set_amp_config}"
                        fi

                        # run train
                        eval "unset CUDA_VISIBLE_DEVICES"
                        # echo $cmd
                        eval $cmd
                        status_check $? "${cmd}" "${status_log}"
                    done
                done
            done
        done
    done
fi
