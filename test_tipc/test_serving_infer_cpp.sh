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
dataline=$(awk 'NR==1, NR==19{print}'  $FILENAME)
MODE=$2

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
det_server_value=$(func_parser_model_config "${lines[7]}")
det_client_value=$(func_parser_model_config "${lines[8]}")
rec_server_value=$(func_parser_model_config "${lines[10]}")
rec_client_value=$(func_parser_model_config "${lines[11]}")
serving_dir_value=$(func_parser_value "${lines[12]}")
web_service_py=$(func_parser_value "${lines[13]}")
op_key=$(func_parser_key "${lines[14]}")
op_value=$(func_parser_value "${lines[14]}")
port_key=$(func_parser_key "${lines[15]}")
port_value=$(func_parser_value "${lines[15]}")
gpu_key=$(func_parser_key "${lines[16]}")
gpu_value=$(func_parser_value "${lines[16]}")
cpp_client_py=$(func_parser_value "${lines[17]}")
image_dir_key=$(func_parser_key "${lines[18]}")
image_dir_value=$(func_parser_value "${lines[18]}")

LOG_PATH="$(pwd)/test_tipc/output/${model_name}/${MODE}/cpp"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_cpp_serving.log"

function func_serving(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    # pdserving
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    # trans det
    set_dirname=$(func_set_params "--dirname" "${det_infer_model_dir_value}")
    set_serving_server=$(func_set_params "--serving_server" "${det_serving_server_value}")
    set_serving_client=$(func_set_params "--serving_client" "${det_serving_client_value}")
    python_list=(${python_list})
    trans_det_log="${LOG_PATH}/cpp_trans_model_det.log"
    trans_model_cmd="${python_list[0]} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client} > ${trans_det_log} 2>&1 "
    eval $trans_model_cmd
    cp "deploy/pdserving/serving_client_conf.prototxt" ${det_serving_client_value}
    # trans rec
    set_dirname=$(func_set_params "--dirname" "${rec_infer_model_dir_value}")
    set_serving_server=$(func_set_params "--serving_server" "${rec_serving_server_value}")
    set_serving_client=$(func_set_params "--serving_client" "${rec_serving_client_value}")
    python_list=(${python_list})
    trans_rec_log="${LOG_PATH}/cpp_trans_model_rec.log"
    trans_model_cmd="${python_list[0]} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client} > ${trans_rec_log} 2>&1 "
    eval $trans_model_cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${trans_model_cmd}" "${status_log}" "${model_name}" "${trans_rec_log}"
    set_image_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
    python_list=(${python_list})
    cd ${serving_dir_value}
    
    # cpp serving
    for gpu_id in ${gpu_value[*]}; do
        if [ ${gpu_id} = "null" ]; then
            server_log_path="${LOG_PATH}/cpp_server_cpu.log"
            web_service_cpp_cmd="nohup ${python_list[0]} ${web_service_py} --model ${det_server_value} ${rec_server_value} ${op_key} ${op_value} ${port_key} ${port_value} > ${server_log_path} 2>&1 &"
            eval $web_service_cpp_cmd
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cpp_cmd}" "${status_log}" "${model_name}" "${server_log_path}"
            sleep 5s
            _save_log_path="${LOG_PATH}/cpp_client_cpu.log"
            cpp_client_cmd="${python_list[0]} ${cpp_client_py} ${det_client_value} ${rec_client_value} > ${_save_log_path} 2>&1"
            eval $cpp_client_cmd
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}"
            status_check $last_status "${cpp_client_cmd}" "${status_log}" "${model_name}" "${_save_log_path}"
            #ps ux | grep -i ${port_value} | awk '{print $2}' | xargs kill -s 9
            ${python_list[0]} ${web_service_py} stop
            sleep 5s
        else
            server_log_path="${LOG_PATH}/cpp_server_gpu.log"
            web_service_cpp_cmd="nohup ${python_list[0]} ${web_service_py} --model ${det_server_value} ${rec_server_value} ${op_key} ${op_value} ${port_key} ${port_value} ${gpu_key} ${gpu_id} > ${server_log_path} 2>&1 &"
            eval $web_service_cpp_cmd
            sleep 5s
            _save_log_path="${LOG_PATH}/cpp_client_gpu.log"
            cpp_client_cmd="${python_list[0]} ${cpp_client_py} ${det_client_value} ${rec_client_value} > ${_save_log_path} 2>&1"
            eval $cpp_client_cmd
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}" 
            status_check $last_status "${cpp_client_cmd}" "${status_log}" "${model_name}" "${_save_log_path}"
            #ps ux | grep -i ${port_value} | awk '{print $2}' | xargs kill -s 9
            ${python_list[0]} ${web_service_py} stop
        fi
    done
}


#set cuda device
GPUID=$3
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
func_serving "${web_service_cpp_cmd}"
