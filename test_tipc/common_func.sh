#!/bin/bash

function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function func_set_params(){
    key=$1
    value=$2
    if [ ${key}x = "null"x ];then
        echo " "
    elif [[ ${value} = "null" ]] || [[ ${value} = " " ]] || [ ${#value} -le 0 ];then
        echo " "
    else 
        echo "${key}=${value}"
    fi
}

function func_parser_params(){
    strs=$1
    MODE=$2
    IFS=":"
    array=(${strs})
    key=${array[0]}
    tmp=${array[1]}
    IFS="|"
    res=""
    for _params in ${tmp[*]}; do
        IFS="="
        array=(${_params})
        mode=${array[0]}
        value=${array[1]}
        if [[ ${mode} = ${MODE} ]]; then
            IFS="|"
            #echo $(func_set_params "${mode}" "${value}")
            echo $value
            break
        fi
        IFS="|"
    done
    echo ${res}
}

function status_check(){
    last_status=$1   # the exit code
    run_command=$2
    run_log=$3
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m Run successfully with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m Run failed with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    fi
}

