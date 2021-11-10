#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
dataline=$(awk 'NR==1, NR==18{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# parser serving
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
trans_model_py=$(func_parser_value "${lines[3]}")
infer_model_dir_key=$(func_parser_key "${lines[4]}")
infer_model_dir_value=$(func_parser_value "${lines[4]}")
model_filename_key=$(func_parser_key "${lines[5]}")
model_filename_value=$(func_parser_value "${lines[5]}")
params_filename_key=$(func_parser_key "${lines[6]}")
params_filename_value=$(func_parser_value "${lines[6]}")
serving_server_key=$(func_parser_key "${lines[7]}")
serving_server_value=$(func_parser_value "${lines[7]}")
serving_client_key=$(func_parser_key "${lines[8]}")
serving_client_value=$(func_parser_value "${lines[8]}")
serving_dir_value=$(func_parser_value "${lines[9]}")
web_service_py=$(func_parser_value "${lines[10]}")
web_use_gpu_key=$(func_parser_key "${lines[11]}")
web_use_gpu_list=$(func_parser_value "${lines[11]}")
web_use_mkldnn_key=$(func_parser_key "${lines[12]}")
web_use_mkldnn_list=$(func_parser_value "${lines[12]}")
web_cpu_threads_key=$(func_parser_key "${lines[13]}")
web_cpu_threads_list=$(func_parser_value "${lines[13]}")
web_use_trt_key=$(func_parser_key "${lines[14]}")
web_use_trt_list=$(func_parser_value "${lines[14]}")
web_precision_key=$(func_parser_key "${lines[15]}")
web_precision_list=$(func_parser_value "${lines[15]}")
pipeline_py=$(func_parser_value "${lines[16]}")
image_dir_key=$(func_parser_key "${lines[17]}")
image_dir_value=$(func_parser_value "${lines[17]}")

LOG_PATH="../../test_tipc/output"
mkdir -p ./test_tipc/output
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
    set_image_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
    trans_model_cmd="${python} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
    eval $trans_model_cmd
    cd ${serving_dir_value}
    echo $PWD
    unset https_proxy
    unset http_proxy
    for python in ${python[*]}; do
        if [ ${python} = "cpp"]; then
            for use_gpu in ${web_use_gpu_list[*]}; do
                if [ ${use_gpu} = "null" ]; then
                    web_service_cpp_cmd="${python} -m paddle_serving_server.serve --model ppocr_det_mobile_2.0_serving/ ppocr_rec_mobile_2.0_serving/ --port 9293"
                    eval $web_service_cmd
                    sleep 2s
                    _save_log_path="${LOG_PATH}/server_infer_cpp_cpu_pipeline_usemkldnn_False_threads_4_batchsize_1.log"
                    pipeline_cmd="${python} ocr_cpp_client.py ppocr_det_mobile_2.0_client/ ppocr_rec_mobile_2.0_client/"
                    eval $pipeline_cmd
                    status_check $last_status "${pipeline_cmd}" "${status_log}"
                    sleep 2s
                    ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
                else
                    web_service_cpp_cmd="${python} -m paddle_serving_server.serve --model ppocr_det_mobile_2.0_serving/ ppocr_rec_mobile_2.0_serving/ --port 9293 --gpu_id=0"
                    eval $web_service_cmd
                    sleep 2s
                    _save_log_path="${LOG_PATH}/server_infer_cpp_cpu_pipeline_usemkldnn_False_threads_4_batchsize_1.log"
                    pipeline_cmd="${python} ocr_cpp_client.py ppocr_det_mobile_2.0_client/ ppocr_rec_mobile_2.0_client/"
                    eval $pipeline_cmd
                    status_check $last_status "${pipeline_cmd}" "${status_log}"
                    sleep 2s
                    ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9                
                fi
            done
        else
            # python serving
            for use_gpu in ${web_use_gpu_list[*]}; do
                echo ${ues_gpu}
                if [ ${use_gpu} = "null" ]; then
                    for use_mkldnn in ${web_use_mkldnn_list[*]}; do
                        if [ ${use_mkldnn} = "False" ]; then
                            continue
                        fi
                        for threads in ${web_cpu_threads_list[*]}; do
                            set_cpu_threads=$(func_set_params "${web_cpu_threads_key}" "${threads}")
                            web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}=${use_gpu} ${web_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} &"
                            eval $web_service_cmd
                            sleep 2s
                            for pipeline in ${pipeline_py[*]}; do
                                _save_log_path="${LOG_PATH}/server_infer_cpu_${pipeline%_client*}_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_1.log"
                                pipeline_cmd="${python} ${pipeline} ${set_image_dir} > ${_save_log_path} 2>&1 "
                                eval $pipeline_cmd
                                last_status=${PIPESTATUS[0]}
                                eval "cat ${_save_log_path}"
                                status_check $last_status "${pipeline_cmd}" "${status_log}"
                                sleep 2s
                            done
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
                            set_tensorrt=$(func_set_params "${web_use_trt_key}" "${use_trt}")
                            set_precision=$(func_set_params "${web_precision_key}" "${precision}")
                            web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}=${use_gpu} ${set_tensorrt} ${set_precision} & "
                            eval $web_service_cmd
                        
                            sleep 2s
                            for pipeline in ${pipeline_py[*]}; do
                                _save_log_path="${LOG_PATH}/server_infer_gpu_${pipeline%_client*}_usetrt_${use_trt}_precision_${precision}_batchsize_1.log"
                                pipeline_cmd="${python} ${pipeline} ${set_image_dir}> ${_save_log_path} 2>&1"
                                eval $pipeline_cmd
                                last_status=${PIPESTATUS[0]}
                                eval "cat ${_save_log_path}"
                                status_check $last_status "${pipeline_cmd}" "${status_log}"
                                sleep 2s
                            done
                            ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
                        done
                    done
                else
                    echo "Does not support hardware other than CPU and GPU Currently!"
                fi
            done
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

export Count=0
IFS="|"
func_serving "${web_service_cmd}"
