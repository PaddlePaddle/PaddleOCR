#!/bin/bash
source test_tipc/common_func.sh 

FILENAME=$1

dataline=$(cat ${FILENAME})
lines=(${dataline})
# common params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")


# parser params
dataline=$(awk 'NR==1, NR==14{print}'  $FILENAME)
IFS=$'\n'
lines=(${dataline})

# parser paddle2onnx
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
padlle2onnx_cmd=$(func_parser_value "${lines[3]}")
infer_model_dir_key=$(func_parser_key "${lines[4]}")
infer_model_dir_value=$(func_parser_value "${lines[4]}")
model_filename_key=$(func_parser_key "${lines[5]}")
model_filename_value=$(func_parser_value "${lines[5]}")
params_filename_key=$(func_parser_key "${lines[6]}")
params_filename_value=$(func_parser_value "${lines[6]}")
save_file_key=$(func_parser_key "${lines[7]}")
save_file_value=$(func_parser_value "${lines[7]}")
opset_version_key=$(func_parser_key "${lines[8]}")
opset_version_value=$(func_parser_value "${lines[8]}")
enable_onnx_checker_key=$(func_parser_key "${lines[9]}")
enable_onnx_checker_value=$(func_parser_value "${lines[9]}")
# parser onnx inference 
inference_py=$(func_parser_value "${lines[10]}")
use_gpu_key=$(func_parser_key "${lines[11]}")
use_gpu_list=$(func_parser_value "${lines[11]}")
det_model_key=$(func_parser_key "${lines[12]}")
image_dir_key=$(func_parser_key "${lines[13]}")
image_dir_value=$(func_parser_value "${lines[13]}")

LOG_PATH="./test_tipc/output/paddle2onnx/${model_name}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_paddle2onnx.log"


function func_paddle2onnx(){
    IFS='|'
    _script=$1

    # paddle2onnx
    set_dirname=$(func_set_params "${infer_model_dir_key}" "${infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_save_model=$(func_set_params "${save_file_key}" "${save_file_value}")
    set_opset_version=$(func_set_params "${opset_version_key}" "${opset_version_value}")
    set_enable_onnx_checker=$(func_set_params "${enable_onnx_checker_key}" "${enable_onnx_checker_value}")
    trans_model_cmd="${padlle2onnx_cmd} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_save_model} ${set_opset_version} ${set_enable_onnx_checker}"
    eval $trans_model_cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${trans_model_cmd}" "${status_log}"
    # python inference
    for use_gpu in ${use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            _save_log_path="${LOG_PATH}/paddle2onnx_infer_cpu.log"
            set_gpu=$(func_set_params "${use_gpu_key}" "${use_gpu}")
            set_model_dir=$(func_set_params "${det_model_key}" "${save_file_value}")
            set_img_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
            infer_model_cmd="${python} ${inference_py} ${set_gpu} ${set_img_dir} ${set_model_dir} --use_onnx=True > ${_save_log_path} 2>&1 "
            eval $infer_model_cmd
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}"
            status_check $last_status "${infer_model_cmd}" "${status_log}"
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            _save_log_path="${LOG_PATH}/paddle2onnx_infer_gpu.log"
            set_gpu=$(func_set_params "${use_gpu_key}" "${use_gpu}")
            set_model_dir=$(func_set_params "${det_model_key}" "${save_file_value}")
            set_img_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
            infer_model_cmd="${python} ${inference_py} ${set_gpu} ${set_img_dir} ${set_model_dir} --use_onnx=True > ${_save_log_path} 2>&1 "
            eval $infer_model_cmd
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}"
            status_check $last_status "${infer_model_cmd}" "${status_log}"
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}


echo "################### run test ###################"

export Count=0
IFS="|"
func_paddle2onnx 