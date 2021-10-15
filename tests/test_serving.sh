#!/bin/bash
source tests/common_func.sh

FILENAME=$1
dataline=$(awk 'NR==67, NR==81{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# parser serving
trans_model_py=$(func_parser_value "${lines[1]}")
infer_model_dir_key=$(func_parser_key "${lines[2]}")
infer_model_dir_value=$(func_parser_value "${lines[2]}")
model_filename_key=$(func_parser_key "${lines[3]}")
model_filename_value=$(func_parser_value "${lines[3]}")
params_filename_key=$(func_parser_key "${lines[4]}")
params_filename_value=$(func_parser_value "${lines[4]}")
serving_server_key=$(func_parser_key "${lines[5]}")
serving_server_value=$(func_parser_value "${lines[5]}")
serving_client_key=$(func_parser_key "${lines[6]}")
serving_client_value=$(func_parser_value "${lines[6]}")
serving_dir_value=$(func_parser_value "${lines[7]}")
web_service_py=$(func_parser_value "${lines[8]}")
web_use_gpu_key=$(func_parser_key "${lines[9]}")
web_use_gpu_list=$(func_parser_value "${lines[9]}")
web_use_mkldnn_key=$(func_parser_key "${lines[10]}")
web_use_mkldnn_list=$(func_parser_value "${lines[10]}")
web_cpu_threads_key=$(func_parser_key "${lines[11]}")
web_cpu_threads_list=$(func_parser_value "${lines[11]}")
web_use_trt_key=$(func_parser_key "${lines[12]}")
web_use_trt_list=$(func_parser_value "${lines[12]}")
web_precision_key=$(func_parser_key "${lines[13]}")
web_precision_list=$(func_parser_value "${lines[13]}")
pipeline_py=$(func_parser_value "${lines[14]}")


LOG_PATH="./tests/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_serving.log"


function func_serving(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    # pdserving
    set_dirname=$(func_set_params "${infer_model_dir_key}" "${infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_serving_server=$(func_set_params "${serving_server_key}" "${serving_server_value}")
    set_serving_client=$(func_set_params "${serving_client_key}" "${serving_client_value}")
    trans_model_cmd="${python} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
    eval $trans_model_cmd
    cd ${serving_dir_value}
    echo $PWD
    unset https_proxy
    unset http_proxy
    for use_gpu in ${web_use_gpu_list[*]}; do
        echo ${ues_gpu}
        if [ ${use_gpu} = "null" ]; then
            for use_mkldnn in ${web_use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ]; then
                    continue
                fi
                for threads in ${web_cpu_threads_list[*]}; do
                      _save_log_path="${_log_path}/server_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_1.log"
                      set_cpu_threads=$(func_set_params "${web_cpu_threads_key}" "${threads}")
                      web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}=${use_gpu} ${web_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} &>${_save_log_path} &"
                      eval $web_service_cmd
                      sleep 2s
                      pipeline_cmd="${python} ${pipeline_py}"
                      eval $pipeline_cmd
                      last_status=${PIPESTATUS[0]}
                      eval "cat ${_save_log_path}"
                      status_check $last_status "${pipeline_cmd}" "${status_log}"
                      PID=$!
                      kill $PID
                      sleep 2s
                      ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
                done
            done
        elif [ ${use_gpu} = "0" ]; then
            for use_trt in ${web_use_trt_list[*]}; do
                for precision in ${web_precision_list[*]}; do
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                        continue
                    fi
                    if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                        continue
                    fi
                    if [[ ${use_trt} = "False" || ${precision} =~ "int8" ]] && [[ ${_flag_quant} = "True" ]]; then
                        continue
                    fi
                    _save_log_path="${_log_path}/server_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_1.log"
                    set_tensorrt=$(func_set_params "${web_use_trt_key}" "${use_trt}")
                    set_precision=$(func_set_params "${web_precision_key}" "${precision}")
                    web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}=${use_gpu} ${set_tensorrt} ${set_precision} &>${_save_log_path} & "
                    eval $web_service_cmd
                    sleep 2s
                    pipeline_cmd="${python} ${pipeline_py}"
                    eval $pipeline_cmd
                    last_status=${PIPESTATUS[0]}
                    eval "cat ${_save_log_path}"
                    status_check $last_status "${pipeline_cmd}" "${status_log}"
                    PID=$!
                    kill $PID
                    sleep 2s
                    ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}


# set cuda device
GPUID=$2
if [ ${#GPUID} -le 0 ];then
    env=" "
else
    env="export CUDA_VISIBLE_DEVICES=${GPUID}"
fi
set CUDA_VISIBLE_DEVICES
eval $env


echo "################### run test ###################"
