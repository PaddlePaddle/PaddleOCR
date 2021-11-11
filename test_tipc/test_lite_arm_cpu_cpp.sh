#!/bin/bash
source ./common_func.sh
export LD_LIBRARY_PATH=${PWD}:$LD_LIBRARY_PATH

FILENAME=$1
dataline=$(cat $FILENAME)
# parser params
IFS=$'\n'
lines=(${dataline})

# parser lite inference
lite_inference_cmd=$(func_parser_value "${lines[1]}")
lite_model_dir_list=$(func_parser_value "${lines[2]}")
runtime_device=$(func_parser_value "${lines[3]}")
lite_cpu_threads_list=$(func_parser_value "${lines[4]}")
lite_batch_size_list=$(func_parser_value "${lines[5]}")
lite_infer_img_dir_list=$(func_parser_value "${lines[8]}")
lite_config_dir=$(func_parser_value "${lines[9]}")
lite_rec_dict_dir=$(func_parser_value "${lines[10]}")
lite_benchmark_value=$(func_parser_value "${lines[11]}")


LOG_PATH="./output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"


function func_lite(){
    IFS='|'
    _script=$1
    _lite_model=$2
    _log_path=$3
    _img_dir=$4
    _config=$5
    if [[ $lite_model =~ "slim" ]]; then
        precision="INT8"
    else
        precision="FP32"
    fi

    # lite inference
    for num_threads in ${lite_cpu_threads_list[*]}; do
	for batchsize in ${lite_batch_size_list[*]}; do
            _save_log_path="${_log_path}/lite_${_lite_model}_runtime_device_${runtime_device}_precision_${precision}_batchsize_${batchsize}_threads_${num_threads}.log"
            command="${_script} ${_lite_model} ${runtime_device} ${precision} ${num_threads} ${batchsize}  ${_img_dir} ${_config} ${lite_benchmark_value} > ${_save_log_path} 2>&1"
            eval ${command}
            status_check $? "${command}" "${status_log}"
        done
    done
}


echo "################### run test ###################"
IFS="|"
for lite_model in ${lite_model_dir_list[*]}; do
    #run lite inference
    for img_dir in ${lite_infer_img_dir_list[*]}; do
        func_lite "${lite_inference_cmd}" "${lite_model}_opt.nb" "${LOG_PATH}" "${img_dir}" "${lite_config_dir}"
    done
done
