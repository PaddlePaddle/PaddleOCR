#!/bin/bash
source ./common_func.sh
export LD_LIBRARY_PATH=${PWD}:$LD_LIBRARY_PATH

FILENAME=$1
dataline=$(cat $FILENAME)
# parser params
IFS=$'\n'
lines=(${dataline})

# parser lite inference
inference_cmd=$(func_parser_value "${lines[1]}")
runtime_device=$(func_parser_value "${lines[2]}")
det_model_list=$(func_parser_value "${lines[3]}")
rec_model_list=$(func_parser_value "${lines[4]}")
cls_model_list=$(func_parser_value "${lines[5]}")
cpu_threads_list=$(func_parser_value "${lines[6]}")
det_batch_size_list=$(func_parser_value "${lines[7]}")
rec_batch_size_list=$(func_parser_value "${lines[8]}")
infer_img_dir_list=$(func_parser_value "${lines[9]}")
config_dir=$(func_parser_value "${lines[10]}")
rec_dict_dir=$(func_parser_value "${lines[11]}")
benchmark_value=$(func_parser_value "${lines[12]}")

if [[ $inference_cmd =~ "det" ]]; then
    lite_model_list=${det_lite_model_list}
elif [[ $inference_cmd =~ "rec" ]]; then
    lite_model_list=(${rec_lite_model_list[*]} ${cls_lite_model_list[*]})
elif [[ $inference_cmd =~ "system" ]]; then
    lite_model_list=(${det_lite_model_list[*]} ${rec_lite_model_list[*]} ${cls_lite_model_list[*]})
else
    echo "inference_cmd is wrong, please check."
    exit 1
fi

LOG_PATH="./output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"


function func_test_det(){
    IFS='|'
    _script=$1
    _det_model=$2
    _log_path=$3
    _img_dir=$4
    _config=$5
    if [[ $_det_model =~ "slim" ]]; then
        precision="INT8"
    else
        precision="FP32"
    fi

    # lite inference
    for num_threads in ${cpu_threads_list[*]}; do
	for det_batchsize in ${det_batch_size_list[*]}; do
            _save_log_path="${_log_path}/lite_${_det_model}_runtime_device_${runtime_device}_precision_${precision}_det_batchsize_${det_batchsize}_threads_${num_threads}.log"
            command="${_script} ${_det_model} ${runtime_device} ${precision} ${num_threads} ${det_batchsize}  ${_img_dir} ${_config} ${benchmark_value} > ${_save_log_path} 2>&1"
            eval ${command}
            status_check $? "${command}" "${status_log}"
        done
    done
}

function func_test_rec(){
    IFS='|'
    _script=$1
    _rec_model=$2
    _cls_model=$3
    _log_path=$4
    _img_dir=$5
    _config=$6
    _rec_dict_dir=$7

    if [[ $_det_model =~ "slim" ]]; then
        _precision="INT8"
    else
        _precision="FP32"
    fi

    # lite inference
    for num_threads in ${cpu_threads_list[*]}; do
	for rec_batchsize in ${rec_batch_size_list[*]}; do
            _save_log_path="${_log_path}/lite_${_rec_model}_${cls_model}_runtime_device_${runtime_device}_precision_${_precision}_rec_batchsize_${rec_batchsize}_threads_${num_threads}.log"
            command="${_script} ${_rec_model} ${_cls_model} ${runtime_device} ${_precision} ${num_threads} ${rec_batchsize}  ${_img_dir} ${_config} ${_rec_dict_dir} ${benchmark_value} > ${_save_log_path} 2>&1"
            eval ${command}
            status_check $? "${command}" "${status_log}"
        done
    done
}

function func_test_system(){
    IFS='|'
    _script=$1
    _det_model=$2
    _rec_model=$3
    _cls_model=$4
    _log_path=$5
    _img_dir=$6
    _config=$7
    _rec_dict_dir=$8
    if [[ $_det_model =~ "slim" ]]; then
        _precision="INT8"
    else
        _precision="FP32"
    fi

    # lite inference
    for num_threads in ${cpu_threads_list[*]}; do
	for det_batchsize in ${det_batch_size_list[*]}; do
	   for rec_batchsize in ${rec_batch_size_list[*]}; do
                _save_log_path="${_log_path}/lite_${_det_model}_${_rec_model}_${_cls_model}_runtime_device_${runtime_device}_precision_${_precision}_det_batchsize_${det_batchsize}_rec_batchsize_${rec_batchsize}_threads_${num_threads}.log"
                command="${_script} ${_det_model} ${_rec_model} ${_cls_model} ${runtime_device} ${_precision} ${num_threads} ${det_batchsize}  ${_img_dir} ${_config} ${_rec_dict_dir} ${benchmark_value} > ${_save_log_path} 2>&1"
               eval ${command}
               status_check $? "${command}" "${status_log}"
	    done
        done
    done
}


echo "################### run test ###################"

if [[ $inference_cmd =~ "det" ]]; then
    IFS="|"
    det_model_list=(${det_model_list[*]})

    for i in {0..1}; do
        #run lite inference
        for img_dir in ${infer_img_dir_list[*]}; do
            func_test_det "${inference_cmd}" "${det_model_list[i]}_opt.nb" "${LOG_PATH}" "${img_dir}" "${config_dir}"
        done
    done

elif [[ $inference_cmd =~ "rec" ]]; then
    IFS="|"
    rec_model_list=(${rec_model_list[*]})
    cls_model_list=(${cls_model_list[*]})

    for i in {0..1}; do
        #run lite inference
        for img_dir in ${infer_img_dir_list[*]}; do
            func_test_rec "${inference_cmd}" "${rec_model}_opt.nb" "${cls_model_list[i]}_opt.nb" "${LOG_PATH}" "${img_dir}" "${rec_dict_dir}" "${config_dir}"
        done
    done

elif [[ $inference_cmd =~ "system" ]]; then
    IFS="|"
    det_model_list=(${det_model_list[*]})
    rec_model_list=(${rec_model_list[*]})
    cls_model_list=(${cls_model_list[*]})

    for i in {0..1}; do
	#run lite inference
        for img_dir in ${infer_img_dir_list[*]}; do
            func_test_system "${inference_cmd}" "${det_model_list[i]}_opt.nb" "${rec_model_list[i]}_opt.nb" "${cls_model_list[i]}_opt.nb" "${LOG_PATH}" "${img_dir}" "${config_dir}" "${rec_dict_dir}"
        done
    done
fi
