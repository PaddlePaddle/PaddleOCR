#!/bin/bash
source test_tipc/common_func.sh

# run benchmark sh 
# Usage:
# bash run_benchmark_train.sh config.txt params

function func_parser_params(){
    strs=$1
    IFS="="
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function func_sed_params(){
    filename=$1
    line=$2
    param_value=$3

    # cmd="sed -n '${line}p' $filename"
    # params=`eval $cmd`
    params=`sed -n "${line}p" $filename`
    # params=`cmd`
    #params=$($cmd)
    IFS=":"
    array=(${params})
    key=${array[0]}
    value=${array[1]}
    if [[ $value =~ 'benchmark_train' ]];then
        IFS='='
        _val=(${value})
        param_value="${_val[0]}=${param_value}"
    fi
    new_params="${key}:${param_value}"
    IFS=";"
    cmd="sed -i '${line}s/.*/${new_params}/' '${filename}'"
    # echo $cmd
    eval $cmd
}

function set_gpu_id(){
    string=$1
    _str=${string:1:6}
    IFS="C"
    arr=(${_str})
    M=${arr[0]}
    P=${arr[1]}
    gn=`expr $P - 1`
    gpu_num=`expr $gn / $M`
    seq=`seq -s "," 0 $gpu_num`
    echo $seq
}

function get_repo_name(){
    IFS=";"
    cur_dir=$(pwd)
    IFS="/"
    arr=(${cur_dir})
    echo ${arr[-1]}
}


# FILENAME=$1
# # MODE be one of ['benchmark_train']
# MODE=$2
# params=$3
IFS="\n"

FILENAME="test_tipc/configs/det_mv3_db_v2.0/train_benchmark.txt"
MODE="benchmark_train"
params="dynamic_bs8_fp32_SingleP_DP_N1C4"

# parser params from input: modeltype_bs${bs_item}_${fp_item}_${run_process_type}_${run_mode}_${device_num}
IFS="_"
params_list=(${params})
model_type=${params_list[0]}
batch_size=${params_list[1]}
batch_size=`echo  ${batch_size} | tr -cd "[0-9]" `
precision=${params_list[2]}
run_process_type=${params_list[3]}
run_mode=${params_list[4]}
device_num=${params_list[5]}
device_num_copy=$device_num
IFS=";"

echo $precision
# sed batchsize and precision
func_sed_params "$FILENAME" "6" "$precision"
func_sed_params "$FILENAME" "9" "$batch_size"

# parser params from train_benchmark.txt
dataline=`cat $FILENAME`
# parser params
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")

# 获取benchmark_params所在的行数
line_num=`grep -n "benchmark_params" $FILENAME  | cut -d ":" -f 1`
echo $line_num # debug
echo ${lines[$line_num]} # debug
# for train log parser
line_num=`expr $line_num + 3`
ips_unit_value=$(func_parser_value "${lines[line_num]}")

line_num=`expr $line_num + 1`
skip_steps_value=$(func_parser_value "${lines[line_num]}")

echo $skip_steps_value "haha"

line_num=`expr $line_num + 1`
keyword_value=$(func_parser_value "${lines[line_num]}")

line_num=`expr $line_num + 1`
convergence_key_value=$(func_parser_value "${lines[line_num]}")

line_num=`expr $line_num + 1`
flags_value=$(func_parser_value "${lines[line_num]}")

# echo "device_num: $device_num"
# gpu_id=$(set_gpu_id "N4C32")
# echo "gpuid: $gpu_id"

gpu_id=$(set_gpu_id $device_num)
echo "gpu_id: $gpu_id"
repo_name=$(get_repo_name )
echo "repo_name: ${repo_name}"

# train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
# profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
SAVE_LOG="benchmark_log"

if [ ${#gpu_id} -le 1 ];then
    echo "single_gpu train"
    log_path="$SAVE_LOG/profiling_log"
    mkdir -p $log_path
    log_name="${repo_name}_${model_name}_bs${batch_size}_${precision}_${run_process_type}_${run_mode}_${device_num}_profiling"
    echo $log_name
    func_sed_params "$FILENAME" "4" "0"  # sed used gpu_id 
    cmd="bash test_tipc/test_train_inference_python.sh benchmark_train > ${log_path}/${log_name} 2>&1 "
    echo $cmd
    # without profile
    log_path="$SAVE_LOG/train_log"
    mkdir -p $log_path
    log_name="${repo_name}_${model_name}_bs${batch_size}_${precision}_${run_process_type}_${run_mode}_${device_num}_log"
    func_sed_params "$FILENAME" "13" "null"  # sed used gpu_id 
    cmd="bash test_tipc/test_train_inference_python.sh benchmark_train > ${log_path}/${log_name} 2>&1 "
    echo $cmd
else
    echo "multi_gpu training"
    log_path="$SAVE_LOG/train_log"
    mkdir -p $log_path
    log_name="${repo_name}_${model_name}_bs${batch_size}_${precision}_${run_process_type}_${run_mode}_${device_num}_log"
    func_sed_params "$FILENAME" "4" "$gpu_id"  # sed used gpu_id 
    func_sed_params "$FILENAME" "13" "$null"  # sed --profile_option as null
    cmd="bash test_tipc/test_train_inference_python.sh benchmark_train > ${log_path}/${log_name} 2>&1 "
    echo $cmd
fi


