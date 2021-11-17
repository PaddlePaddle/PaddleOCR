#!/usr/bin/env bash
set -x
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}         # fp32|fp16
    max_epoch=${4:-"10"}         # 可选，如果需要修改代码提前中断
    model_item=${5:-"model_item"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR 后续QA设置该参数
#   日志解析所需参数
    base_batch_size=${batch_size}
    mission_name="OCR"
    direction_id="0"
    ips_unit="instance/sec"
    skip_steps=2                 # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="ips:"               # 解析日志，筛选出数据所在行的关键字                                             (必填)
    index="1"
    model_name=${model_item}_${run_mode}_bs${batch_size}_${fp_item}        # model_item 用于yml文件名匹配，model_name 用于数据入库前端展示
#   以下不用修改   
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}
function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd="-c configs/det/${model_item}.yml -o Train.loader.batch_size_per_card=${batch_size} Global.epoch_num=${max_epoch} Global.eval_batch_step=[0,20000] Global.print_batch_step=2"
    case ${run_mode} in
      sp) 
        train_cmd="python tools/train.py "${train_cmd}""
        ;;
      mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES tools/train.py ${train_cmd}"
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

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh      # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;该脚本在连调时可从benchmark repo中下载https://github.com/PaddlePaddle/benchmark/blob/master/scripts/run_model.sh;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train      # 如果只想产出训练log,不解析,可取消注释
_run         # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只想要产出训练log可以注掉本行,提交时需打开
