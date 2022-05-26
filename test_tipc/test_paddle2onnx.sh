#!/bin/bash
source test_tipc/common_func.sh 

FILENAME=$1

dataline=$(cat ${FILENAME})
lines=(${dataline})
# common params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")


# parser params
dataline=$(awk 'NR==1, NR==17{print}'  $FILENAME)
IFS=$'\n'
lines=(${dataline})

# parser paddle2onnx
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
padlle2onnx_cmd=$(func_parser_value "${lines[3]}")
det_infer_model_dir_key=$(func_parser_key "${lines[4]}")
det_infer_model_dir_value=$(func_parser_value "${lines[4]}")
model_filename_key=$(func_parser_key "${lines[5]}")
model_filename_value=$(func_parser_value "${lines[5]}")
params_filename_key=$(func_parser_key "${lines[6]}")
params_filename_value=$(func_parser_value "${lines[6]}")
det_save_file_key=$(func_parser_key "${lines[7]}")
det_save_file_value=$(func_parser_value "${lines[7]}")
rec_infer_model_dir_key=$(func_parser_key "${lines[8]}")
rec_infer_model_dir_value=$(func_parser_value "${lines[8]}")
rec_save_file_key=$(func_parser_key "${lines[9]}")
rec_save_file_value=$(func_parser_value "${lines[9]}")
opset_version_key=$(func_parser_key "${lines[10]}")
opset_version_value=$(func_parser_value "${lines[10]}")
enable_onnx_checker_key=$(func_parser_key "${lines[11]}")
enable_onnx_checker_value=$(func_parser_value "${lines[11]}")
# parser onnx inference 
inference_py=$(func_parser_value "${lines[12]}")
use_gpu_key=$(func_parser_key "${lines[13]}")
use_gpu_list=$(func_parser_value "${lines[13]}")
det_model_key=$(func_parser_key "${lines[14]}")
rec_model_key=$(func_parser_key "${lines[15]}")
image_dir_key=$(func_parser_key "${lines[16]}")
image_dir_value=$(func_parser_value "${lines[16]}")

LOG_PATH="./test_tipc/output/${model_name}/paddle2onnx"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_paddle2onnx.log"


function func_paddle2onnx(){
    IFS='|'
    _script=$1

    # paddle2onnx
    if [ ${model_name} = "ch_PP-OCRv2" ] || [ ${model_name} = "ch_PP-OCRv3" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0" ] || [ ${model_name} = "ch_ppocr_server_v2.0" ]; then
        # trans det
        set_dirname=$(func_set_params "--model_dir" "${det_infer_model_dir_value}")
        set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
        set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
        set_save_model=$(func_set_params "--save_file" "${det_save_file_value}")
        set_opset_version=$(func_set_params "${opset_version_key}" "${opset_version_value}")
        set_enable_onnx_checker=$(func_set_params "${enable_onnx_checker_key}" "${enable_onnx_checker_value}")
        trans_model_cmd="${padlle2onnx_cmd} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_save_model} ${set_opset_version} ${set_enable_onnx_checker}"
        eval $trans_model_cmd
        last_status=${PIPESTATUS[0]}
        status_check $last_status "${trans_model_cmd}" "${status_log}"
        # trans rec
        set_dirname=$(func_set_params "--model_dir" "${rec_infer_model_dir_value}")
        set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
        set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
        set_save_model=$(func_set_params "--save_file" "${rec_save_file_value}")
        set_opset_version=$(func_set_params "${opset_version_key}" "${opset_version_value}")
        set_enable_onnx_checker=$(func_set_params "${enable_onnx_checker_key}" "${enable_onnx_checker_value}")
        trans_model_cmd="${padlle2onnx_cmd} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_save_model} ${set_opset_version} ${set_enable_onnx_checker}"
        eval $trans_model_cmd
        last_status=${PIPESTATUS[0]}
        status_check $last_status "${trans_model_cmd}" "${status_log}" 
    elif [ ${model_name} = "ch_PP-OCRv2_det" ] || [ ${model_name} = "ch_PP-OCRv3_det" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_det" ] || [ ${model_name} = "ch_ppocr_server_v2.0_det" ]; then
        # trans det
        set_dirname=$(func_set_params "--model_dir" "${det_infer_model_dir_value}")
        set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
        set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
        set_save_model=$(func_set_params "--save_file" "${det_save_file_value}")
        set_opset_version=$(func_set_params "${opset_version_key}" "${opset_version_value}")
        set_enable_onnx_checker=$(func_set_params "${enable_onnx_checker_key}" "${enable_onnx_checker_value}")
        trans_model_cmd="${padlle2onnx_cmd} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_save_model} ${set_opset_version} ${set_enable_onnx_checker}"
        eval $trans_model_cmd
        last_status=${PIPESTATUS[0]}
        status_check $last_status "${trans_model_cmd}" "${status_log}"     
    elif [ ${model_name} = "ch_PP-OCRv2_rec" ] || [ ${model_name} = "ch_PP-OCRv3_rec" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_rec" ] || [ ${model_name} = "ch_ppocr_server_v2.0_rec" ]; then
        # trans rec
        set_dirname=$(func_set_params "--model_dir" "${rec_infer_model_dir_value}")
        set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
        set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
        set_save_model=$(func_set_params "--save_file" "${rec_save_file_value}")
        set_opset_version=$(func_set_params "${opset_version_key}" "${opset_version_value}")
        set_enable_onnx_checker=$(func_set_params "${enable_onnx_checker_key}" "${enable_onnx_checker_value}")
        trans_model_cmd="${padlle2onnx_cmd} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_save_model} ${set_opset_version} ${set_enable_onnx_checker}"
        eval $trans_model_cmd
        last_status=${PIPESTATUS[0]}
        status_check $last_status "${trans_model_cmd}" "${status_log}"
    fi

    # python inference
    for use_gpu in ${use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            _save_log_path="${LOG_PATH}/paddle2onnx_infer_cpu.log"
            set_gpu=$(func_set_params "${use_gpu_key}" "${use_gpu}")
            set_img_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
            if [ ${model_name} = "ch_PP-OCRv2" ] || [ ${model_name} = "ch_PP-OCRv3" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0" ] || [ ${model_name} = "ch_ppocr_server_v2.0" ]; then
                set_det_model_dir=$(func_set_params "${det_model_key}" "${det_save_file_value}")
                set_rec_model_dir=$(func_set_params "${rec_model_key}" "${rec_save_file_value}")
                infer_model_cmd="${python} ${inference_py} ${set_gpu} ${set_img_dir} ${set_det_model_dir} ${set_rec_model_dir} --use_onnx=True > ${_save_log_path} 2>&1 "
            elif [ ${model_name} = "ch_PP-OCRv2_det" ] || [ ${model_name} = "ch_PP-OCRv3_det" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_det" ] || [ ${model_name} = "ch_ppocr_server_v2.0_det" ]; then
                set_det_model_dir=$(func_set_params "${det_model_key}" "${det_save_file_value}")
                infer_model_cmd="${python} ${inference_py} ${set_gpu} ${set_img_dir} ${set_det_model_dir} --use_onnx=True > ${_save_log_path} 2>&1 "
            elif [ ${model_name} = "ch_PP-OCRv2_rec" ] || [ ${model_name} = "ch_PP-OCRv3_rec" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_rec" ] || [ ${model_name} = "ch_ppocr_server_v2.0_rec" ]; then
                set_rec_model_dir=$(func_set_params "${rec_model_key}" "${rec_save_file_value}")
                infer_model_cmd="${python} ${inference_py} ${set_gpu} ${set_img_dir} ${set_rec_model_dir} --use_onnx=True > ${_save_log_path} 2>&1 "
            fi
            eval $infer_model_cmd
            last_status=${PIPESTATUS[0]}
            eval "cat ${_save_log_path}"
            status_check $last_status "${infer_model_cmd}" "${status_log}"
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            _save_log_path="${LOG_PATH}/paddle2onnx_infer_gpu.log"
            set_gpu=$(func_set_params "${use_gpu_key}" "${use_gpu}")
            set_img_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
            if [ ${model_name} = "ch_PP-OCRv2" ] || [ ${model_name} = "ch_PP-OCRv3" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0" ] || [ ${model_name} = "ch_ppocr_server_v2.0" ]; then
                set_det_model_dir=$(func_set_params "${det_model_key}" "${det_save_file_value}")
                set_rec_model_dir=$(func_set_params "${rec_model_key}" "${rec_save_file_value}")
                infer_model_cmd="${python} ${inference_py} ${set_gpu} ${set_img_dir} ${set_det_model_dir} ${set_rec_model_dir} --use_onnx=True > ${_save_log_path} 2>&1 "
            elif [ ${model_name} = "ch_PP-OCRv2_det" ] || [ ${model_name} = "ch_PP-OCRv3_det" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_det" ] || [ ${model_name} = "ch_ppocr_server_v2.0_det" ]; then
                set_det_model_dir=$(func_set_params "${det_model_key}" "${det_save_file_value}")
                infer_model_cmd="${python} ${inference_py} ${set_gpu} ${set_img_dir} ${set_det_model_dir} --use_onnx=True > ${_save_log_path} 2>&1 "
            elif [ ${model_name} = "ch_PP-OCRv2_rec" ] || [ ${model_name} = "ch_PP-OCRv3_rec" ] || [ ${model_name} = "ch_ppocr_mobile_v2.0_rec" ] || [ ${model_name} = "ch_ppocr_server_v2.0_rec" ]; then
                set_rec_model_dir=$(func_set_params "${rec_model_key}" "${rec_save_file_value}")
                infer_model_cmd="${python} ${inference_py} ${set_gpu} ${set_img_dir} ${set_rec_model_dir} --use_onnx=True > ${_save_log_path} 2>&1 "
            fi
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