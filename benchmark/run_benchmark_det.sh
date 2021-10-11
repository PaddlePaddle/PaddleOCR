#!/usr/bin/env bash
set -xe
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    max_iter=${4:-"500"}       # 可选，如果需要修改代码提前中断
    model_name=${5:-"model_name"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR 后续QA设置该参数

#   以下不用修改   
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}
function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd="-c configs/det/${model_name}.yml -o Train.loader.batch_size_per_card=${batch_size} Global.epoch_num=${max_iter} "   
    case ${run_mode} in
      sp) 
        train_cmd="python3.7 tools/train.py "${train_cmd}""
        ;;
      mp)
        train_cmd="python3.7 -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES tools/train.py ${train_cmd}"
        ;;
      *) echo "choose run_mode(sp or mp)"; exit 1;
    esac
# 以下不用修改
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
            echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python3.7'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi

    # run log analysis
    analysis_cmd="python3.7 benchmark/analysis.py --filename ${log_file}  --mission_name ${model_name} --run_mode ${mode} --direction_id 0 --keyword 'ips:' --base_batch_size ${batch_szie} --skip_steps 1 --gpu_num ${num_gpu_devices}  --index 1  --model_mode=-1  --ips_unit=samples/sec"
    eval $analysis_cmd
}

_set_params $@
_train

