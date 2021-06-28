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
function status_check(){
    last_status=$1   # the exit code
    run_command=$2
    run_log=$3
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m Run successfully with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m Run failed with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    fi
}

IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[0]}")
python=$(func_parser_value "${lines[1]}")
gpu_list=$(func_parser_value "${lines[2]}")
autocast_list=$(func_parser_value "${lines[3]}")
autocast_key=$(func_parser_key "${lines[3]}")
epoch_key=$(func_parser_key "${lines[4]}")
save_model_key=$(func_parser_key "${lines[5]}")
save_infer_key=$(func_parser_key "${lines[6]}")
train_batch_key=$(func_parser_key "${lines[7]}")
train_use_gpu_key=$(func_parser_key "${lines[8]}")
pretrain_model_key=$(func_parser_key "${lines[9]}")

trainer_list=$(func_parser_value "${lines[10]}")
norm_trainer=$(func_parser_value "${lines[11]}")
pact_trainer=$(func_parser_value "${lines[12]}")
fpgm_trainer=$(func_parser_value "${lines[13]}")
distill_trainer=$(func_parser_value "${lines[14]}")

eval_py=$(func_parser_value "${lines[15]}")
norm_export=$(func_parser_value "${lines[16]}")
pact_export=$(func_parser_value "${lines[17]}")
fpgm_export=$(func_parser_value "${lines[18]}")
distill_export=$(func_parser_value "${lines[19]}")

inference_py=$(func_parser_value "${lines[20]}")
use_gpu_key=$(func_parser_key "${lines[21]}")
use_gpu_list=$(func_parser_value "${lines[21]}")
use_mkldnn_key=$(func_parser_key "${lines[22]}")
use_mkldnn_list=$(func_parser_value "${lines[22]}")
cpu_threads_key=$(func_parser_key "${lines[23]}")
cpu_threads_list=$(func_parser_value "${lines[23]}")
batch_size_key=$(func_parser_key "${lines[24]}")
batch_size_list=$(func_parser_value "${lines[24]}")
use_trt_key=$(func_parser_key "${lines[25]}")
use_trt_list=$(func_parser_value "${lines[25]}")
precision_key=$(func_parser_key "${lines[26]}")
precision_list=$(func_parser_value "${lines[26]}")
model_dir_key=$(func_parser_key "${lines[27]}")
image_dir_key=$(func_parser_key "${lines[28]}")
save_log_key=$(func_parser_key "${lines[29]}")

LOG_PATH="./test/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"

if [ ${MODE} = "lite_train_infer" ]; then
    export infer_img_dir="./train_data/icdar2015/text_localization/ch4_test_images/"
    export epoch_num=10
elif [ ${MODE} = "whole_infer" ]; then
    export infer_img_dir="./train_data/icdar2015/text_localization/ch4_test_images/"
    export epoch_num=10
elif [ ${MODE} = "whole_train_infer" ]; then
    export infer_img_dir="./train_data/icdar2015/text_localization/ch4_test_images/"
    export epoch_num=300
else
    export infer_img_dir="./inference/ch_det_data_50/all-sum-510"
    export infer_model_dir="./inference/ch_ppocr_mobile_v2.0_det_train/best_accuracy"
fi


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    _log_path=$4
    _img_dir=$5
    
    # inference 
    for use_gpu in ${use_gpu_list[*]}; do 
        if [ ${use_gpu} = "False" ]; then
            for use_mkldnn in ${use_mkldnn_list[*]}; do
                for threads in ${cpu_threads_list[*]}; do
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_${batch_size}"
                        command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${use_mkldnn_key}=${use_mkldnn} ${cpu_threads_key}=${threads} ${model_dir_key}=${_model_dir} ${batch_size_key}=${batch_size} ${image_dir_key}=${_img_dir}  ${save_log_key}=${_save_log_path}"
                        eval $command
                        status_check $? "${command}" "${status_log}"
                    done
                done
            done
        else
            for use_trt in ${use_trt_list[*]}; do
                for precision in ${precision_list[*]}; do
                    if [ ${use_trt} = "False" ] && [ ${precision} != "fp32" ]; then
                        continue
                    fi
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}"
                        command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${use_trt_key}=${use_trt} ${precision_key}=${precision} ${model_dir_key}=${_model_dir} ${batch_size_key}=${batch_size} ${image_dir_key}=${_img_dir}  ${save_log_key}=${_save_log_path}"
                        eval $command
                        status_check $? "${command}" "${status_log}"
                    done
                done
            done
        fi
    done
}

if [ ${MODE} != "infer" ]; then

IFS="|"
for gpu in ${gpu_list[*]}; do
    use_gpu=True
    if [ ${gpu} = "-1" ];then
        use_gpu=False
        env=""
    elif [ ${#gpu} -le 1 ];then
        env="export CUDA_VISIBLE_DEVICES=${gpu}"
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
    fi
    for autocast in ${autocast_list[*]}; do 
        for trainer in ${trainer_list[*]}; do 
            if [ ${trainer} = "pact" ]; then
                run_train=${pact_trainer}
                run_export=${pact_export}
            elif [ ${trainer} = "fpgm" ]; then
                run_train=${fpgm_trainer}
                run_export=${fpgm_export}
            elif [ ${trainer} = "distill" ]; then
                run_train=${distill_trainer}
                run_export=${distill_export}
            else
                run_train=${norm_trainer}
                run_export=${norm_export}
            fi

            if [ ${run_train} = "null" ]; then
                continue
            fi
            if [ ${run_export} = "null" ]; then
                continue
            fi

            save_log="${LOG_PATH}/${trainer}_gpus_${gpu}_autocast_${autocast}"
            if [ ${#gpu} -le 2 ];then  # epoch_num #TODO
                cmd="${python} ${run_train} ${train_use_gpu_key}=${use_gpu} ${autocast_key}=${autocast} ${epoch_key}=${epoch_num} ${save_model_key}=${save_log} "
            elif [ ${#gpu} -le 15 ];then
                cmd="${python} -m paddle.distributed.launch --gpus=${gpu} ${run_train} ${autocast_key}=${autocast} ${epoch_key}=${epoch_num}  ${save_model_key}=${save_log}"
            else
                cmd="${python} -m paddle.distributed.launch --ips=${ips} --gpus=${gpu} ${run_train} ${autocast_key}=${autocast} ${epoch_key}=${epoch_num} ${save_model_key}=${save_log}"
            fi
            # run train
            eval $cmd
            status_check $? "${cmd}" "${status_log}"

            # run eval
            eval_cmd="${python} ${eval_py} ${save_model_key}=${save_log} ${pretrain_model_key}=${save_log}/latest" 
            eval $eval_cmd
            status_check $? "${eval_cmd}" "${status_log}"

            # run export model
            save_infer_path="${save_log}"
            export_cmd="${python} ${run_export} ${save_model_key}=${save_log} ${pretrain_model_key}=${save_log}/latest ${save_infer_key}=${save_infer_path}"
            eval $export_cmd
            status_check $? "${export_cmd}" "${status_log}"

            #run inference
            save_infer_path="${save_log}"
            func_inference "${python}" "${inference_py}" "${save_infer_path}" "${LOG_PATH}" "${infer_img_dir}"
        done
    done
done

else
    save_infer_path="${LOG_PATH}/${MODE}"
    run_export=${norm_export}
    export_cmd="${python} ${run_export} ${save_model_key}=${save_infer_path} ${pretrain_model_key}=${infer_model_dir} ${save_infer_key}=${save_infer_path}"
    eval $export_cmd
    status_check $? "${export_cmd}" "${status_log}"

    #run inference
    func_inference "${python}" "${inference_py}" "${save_infer_path}" "${LOG_PATH}" "${infer_img_dir}"
fi
