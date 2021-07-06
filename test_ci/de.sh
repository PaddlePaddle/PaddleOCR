#!/bin/bash

FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

Params=$3

echo ${FILENAME}
echo ${MODE}
echo $Params

if [ ${#Params} -le 0 ];then
echo "le 1"
else
echo "gt 1"
fi

