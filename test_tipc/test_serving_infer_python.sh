#!/bin/bash
source test_tipc/common_func.sh

function func_parser_model_config(){
    strs=$1
    IFS="/"
    array=(${strs})
    tmp=${array[-1]}
    echo ${tmp}
}

FILENAME=$1
dataline=$(awk 'NR==1, NR==23{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# parser serving
model_name=$(func_parser_value "${lines[1]}")
python_list=$(func_parser_value "${lines[2]}")
trans_model_py=$(func_parser_value "${lines[3]}")
det_infer_model_dir_key=$(func_parser_key "${lines[4]}")
det_infer_model_dir_value=$(func_parser_value "${lines[4]}")
model_filename_key=$(func_parser_key "${lines[5]}")
model_filename_value=$(func_parser_value "${lines[5]}")
params_filename_key=$(func_parser_key "${lines[6]}")
params_filename_value=$(func_parser_value "${lines[6]}")
det_serving_server_key=$(func_parser_key "${lines[7]}")
det_serving_server_value=$(func_parser_value "${lines[7]}")
det_serving_client_key=$(func_parser_key "${lines[8]}")
det_serving_client_value=$(func_parser_value "${lines[8]}")
rec_infer_model_dir_key=$(func_parser_key "${lines[9]}")
rec_infer_model_dir_value=$(func_parser_value "${lines[9]}")
rec_serving_server_key=$(func_parser_key "${lines[10]}")
rec_serving_server_value=$(func_parser_value "${lines[10]}")
rec_serving_client_key=$(func_parser_key "${lines[11]}")
rec_serving_client_value=$(func_parser_value "${lines[11]}")
serving_dir_value=$(func_parser_value "${lines[12]}")
web_service_py=$(func_parser_value "${lines[13]}")
web_use_gpu_key=$(func_parser_key "${lines[14]}")
web_use_gpu_list=$(func_parser_value "${lines[14]}")
web_use_mkldnn_key=$(func_parser_key "${lines[15]}")
web_use_mkldnn_list=$(func_parser_value "${lines[15]}")
web_cpu_threads_key=$(func_parser_key "${lines[16]}")
web_cpu_threads_list=$(func_parser_value "${lines[16]}")
web_use_trt_key=$(func_parser_key "${lines[17]}")
web_use_trt_list=$(func_parser_value "${lines[17]}")
web_precision_key=$(func_parser_key "${lines[18]}")
web_precision_list=$(func_parser_value "${lines[18]}")
det_server_key=$(func_parser_key "${lines[19]}")
det_server_value=$(func_parser_model_config "${lines[7]}")
det_client_value=$(func_parser_model_config "${lines[8]}")
rec_server_key=$(func_parser_key "${lines[20]}")
rec_server_value=$(func_parser_model_config "${lines[10]}")
rec_client_value=$(func_parser_model_config "${lines[11]}")
pipeline_py=$(func_parser_value "${lines[21]}")
image_dir_key=$(func_parser_key "${lines[22]}")
image_dir_value=$(func_parser_value "${lines[22]}")

LOG_PATH="$(pwd)/test_tipc/output/${model_name}/python_serving"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python_serving.log"

function func_serving(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    # pdserving
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    if [ ${model_name} = "ch_PP-OCRv2" ] || [ ${model_name} = "ch_PP-OCRv3" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0" ] || [ ${model_name} = "ch_ppocr_server_v2.0" ]; then
        # trans det
        set_dirname=$(func_set_params "--dirname" "${det_infer_model_dir_value}")
        set_serving_server=$(func_set_params "--serving_server" "${det_serving_server_value}")
        set_serving_client=$(func_set_params "--serving_client" "${det_serving_client_value}")
        python_list=(${python_list})
        trans_model_cmd="${python_list[0]} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
        eval $trans_model_cmd
        # trans rec
        set_dirname=$(func_set_params "--dirname" "${rec_infer_model_dir_value}")
        set_serving_server=$(func_set_params "--serving_server" "${rec_serving_server_value}")
        set_serving_client=$(func_set_params "--serving_client" "${rec_serving_client_value}")
        python_list=(${python_list})
        trans_model_cmd="${python_list[0]} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
        eval $trans_model_cmd
    elif [ ${model_name} = "ch_PP-OCRv2_det" ] || [ ${model_name} = "ch_PP-OCRv3_det" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_det" ] || [ ${model_name} = "ch_ppocr_server_v2.0_det" ]; then
        # trans det
        set_dirname=$(func_set_params "--dirname" "${det_infer_model_dir_value}")
        set_serving_server=$(func_set_params "--serving_server" "${det_serving_server_value}")
        set_serving_client=$(func_set_params "--serving_client" "${det_serving_client_value}")
        python_list=(${python_list})
        trans_model_cmd="${python_list[0]} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
        eval $trans_model_cmd
    elif [ ${model_name} = "ch_PP-OCRv2_rec" ] || [ ${model_name} = "ch_PP-OCRv3_rec" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_rec" ] || [ ${model_name} = "ch_ppocr_server_v2.0_rec" ]; then
        # trans rec
        set_dirname=$(func_set_params "--dirname" "${rec_infer_model_dir_value}")
        set_serving_server=$(func_set_params "--serving_server" "${rec_serving_server_value}")
        set_serving_client=$(func_set_params "--serving_client" "${rec_serving_client_value}")
        python_list=(${python_list})
        trans_model_cmd="${python_list[0]} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
        eval $trans_model_cmd
    fi
    set_image_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
    python_list=(${python_list})
    
    cd ${serving_dir_value}
    unset https_proxy
    unset http_proxy
    python=${python_list[0]}
        
    # python serving
    for use_gpu in ${web_use_gpu_list[*]}; do
        if [ ${use_gpu} = "null" ]; then
            for use_mkldnn in ${web_use_mkldnn_list[*]}; do
                for threads in ${web_cpu_threads_list[*]}; do
                    set_cpu_threads=$(func_set_params "${web_cpu_threads_key}" "${threads}")
                    if [ ${model_name} = "ch_PP-OCRv2" ] || [ ${model_name} = "ch_PP-OCRv3" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0" ] || [ ${model_name} = "ch_ppocr_server_v2.0" ]; then
                        set_det_model_config=$(func_set_params "${det_server_key}" "${det_server_value}")
                        set_rec_model_config=$(func_set_params "${rec_server_key}" "${rec_server_value}")
                        web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}="" ${web_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_det_model_config} ${set_rec_model_config} &"
                        eval $web_service_cmd
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
                    elif [ ${model_name} = "ch_PP-OCRv2_det" ] || [ ${model_name} = "ch_PP-OCRv3_det" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_det" ] || [ ${model_name} = "ch_ppocr_server_v2.0_det" ]; then
                        set_det_model_config=$(func_set_params "${det_server_key}" "${det_server_value}")
                        web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}="" ${web_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_det_model_config} &"
                        eval $web_service_cmd
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
                    elif [ ${model_name} = "ch_PP-OCRv2_rec" ] || [ ${model_name} = "ch_PP-OCRv3_rec" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_rec" ] || [ ${model_name} = "ch_ppocr_server_v2.0_rec" ]; then
                        set_rec_model_config=$(func_set_params "${rec_server_key}" "${rec_server_value}")
                        web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}="" ${web_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_rec_model_config} &"
                        eval $web_service_cmd
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
                    fi
                    sleep 2s
                    for pipeline in ${pipeline_py[*]}; do
                        _save_log_path="${LOG_PATH}/server_infer_cpu_${pipeline%_client*}_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_1.log"
                        pipeline_cmd="${python} ${pipeline} ${set_image_dir} > ${_save_log_path} 2>&1 "
                        eval $pipeline_cmd
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}"
                        sleep 2s
                    done
                    ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
                done
            done
        elif [ ${use_gpu} = "gpu" ]; then
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
                    if [ ${use_trt} = True ]; then
                        device_type=2
                    fi
                    set_precision=$(func_set_params "${web_precision_key}" "${precision}")
                    if [ ${model_name} = "ch_PP-OCRv2" ] || [ ${model_name} = "ch_PP-OCRv3" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0" ] || [ ${model_name} = "ch_ppocr_server_v2.0" ]; then
                        set_det_model_config=$(func_set_params "${det_server_key}" "${det_server_value}")
                        set_rec_model_config=$(func_set_params "${rec_server_key}" "${rec_server_value}")
                        web_service_cmd="${python} ${web_service_py} ${set_tensorrt} ${set_precision} ${set_det_model_config} ${set_rec_model_config} &"
                        eval $web_service_cmd
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
                    elif [ ${model_name} = "ch_PP-OCRv2_det" ] || [ ${model_name} = "ch_PP-OCRv3_det" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_det" ] || [ ${model_name} = "ch_ppocr_server_v2.0_det" ]; then
                        set_det_model_config=$(func_set_params "${det_server_key}" "${det_server_value}")
                        web_service_cmd="${python} ${web_service_py} ${set_tensorrt} ${set_precision} ${set_det_model_config} &"
                        eval $web_service_cmd
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
                    elif [ ${model_name} = "ch_PP-OCRv2_rec" ] || [ ${model_name} = "ch_PP-OCRv3_rec" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_rec" ] || [ ${model_name} = "ch_ppocr_server_v2.0_rec" ]; then
                        set_rec_model_config=$(func_set_params "${rec_server_key}" "${rec_server_value}")
                        web_service_cmd="${python} ${web_service_py} ${set_tensorrt} ${set_precision} ${set_rec_model_config} &"
                        eval $web_service_cmd
                        last_status=${PIPESTATUS[0]}
                        status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
                    fi
                    sleep 2s
                    for pipeline in ${pipeline_py[*]}; do
                        _save_log_path="${LOG_PATH}/server_infer_gpu_${pipeline%_client*}_usetrt_${use_trt}_precision_${precision}_batchsize_1.log"
                        pipeline_cmd="${python} ${pipeline} ${set_image_dir}> ${_save_log_path} 2>&1"
                        eval $pipeline_cmd
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}"
                        sleep 2s
                    done
                    ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}


#set cuda device
GPUID=$2
if [ ${#GPUID} -le 0 ];then
    env="export CUDA_VISIBLE_DEVICES=0"
else
    env="export CUDA_VISIBLE_DEVICES=${GPUID}"
fi
eval $env
echo $env


echo "################### run test ###################"

export Count=0
IFS="|"
func_serving "${web_service_cmd}"
