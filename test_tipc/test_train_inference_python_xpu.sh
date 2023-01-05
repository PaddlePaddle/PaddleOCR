#!/bin/bash
source test_tipc/common_func.sh

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

function func_parser_config() {
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[2]}
    echo ${tmp}
}

BASEDIR=$(dirname "$0")
REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

FILENAME=$1

# disable mkldnn on non x86_64 env
arch=$(uname -i)
if [ $arch != 'x86_64' ]; then
    sed -i 's/--enable_mkldnn:True|False/--enable_mkldnn:False/g' $FILENAME
    sed -i 's/--enable_mkldnn:True/--enable_mkldnn:False/g' $FILENAME
fi

# change gpu to xpu in tipc txt configs
sed -i 's/use_gpu/use_xpu/g' $FILENAME
# disable benchmark as AutoLog required nvidia-smi command
sed -i 's/--benchmark:True/--benchmark:False/g' $FILENAME
dataline=`cat $FILENAME`

# parser params
IFS=$'\n'
lines=(${dataline})

# replace training config file
grep -n 'tools/.*yml' $FILENAME  | cut -d ":" -f 1 \
| while read line_num ; do
    train_cmd=$(func_parser_value "${lines[line_num-1]}")
    trainer_config=$(func_parser_config ${train_cmd})
    sed -i 's/use_gpu/use_xpu/g' "$REPO_ROOT_PATH/$trainer_config"
done

# change gpu to xpu in execution script
sed -i 's/\"gpu\"/\"xpu\"/g' test_tipc/test_train_inference_python.sh

# pass parameters to test_train_inference_python.sh
cmd='bash test_tipc/test_train_inference_python.sh ${FILENAME} $2'
echo -e '\033[1;32m Started to run command: ${cmd}!  \033[0m'
eval $cmd
