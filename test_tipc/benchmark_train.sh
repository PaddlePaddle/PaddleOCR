#!/bin/bash
source test_tipc/common_func.sh

# run benchmark sh 
# params: batch
# Usage:
# bash run_benchmark_train.sh config.txt benchmark_train  batch_size=2 precision=3 profile_option=4

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
    arr=(${_str})
    P=${arr[1]}
    gpu_num=`expr $P - 1`
    seq=`seq -s "," 0 $gpu_num`
    echo $seq
}


FILENAME=$1
# MODE be one of ['benchmark_train']
MODE=$2




# 
# line_bs=9
# line_presion=6
# line_profile=13

# if [ $# -eq 4 ] ; then
#     echo "Usage: bash run_benchmark_train.sh config.txt benchmark_train  batch_size=2  precision=fp16"

# elif [ $# -eq 5 ] ; then
#     echo "Usage: bash run_benchmark_train.sh config.txt benchmark_train  batch_size=2  precision=fp16  profile_option=None"
#     profile_options=$(func_parser_params "$5")
# else
#     echo "None"
# fi


# l=`grep -n "benchmark_params" ${FILENAME}  | cut -d ":" -f 1`
# echo $l


# sed -i '[第n行]s/[正则通配符]/[替换内容]/' [指定文件] 


