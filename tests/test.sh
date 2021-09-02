#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer', 'cpp_infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

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
    if [ ${key} = "null" ];then
        echo " "
    elif [[ ${value} = "null" ]] || [[ ${value} = " " ]] || [ ${#value} -le 0 ];then
        echo " "
    else 
        echo "${key}=${value}"
    fi
}
function func_parser_params(){
    strs=$1
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

IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
gpu_list=$(func_parser_value "${lines[3]}")
train_use_gpu_key=$(func_parser_key "${lines[4]}")
train_use_gpu_value=$(func_parser_value "${lines[4]}")
autocast_list=$(func_parser_value "${lines[5]}")
autocast_key=$(func_parser_key "${lines[5]}")
epoch_key=$(func_parser_key "${lines[6]}")
epoch_num=$(func_parser_params "${lines[6]}")
save_model_key=$(func_parser_key "${lines[7]}")
train_batch_key=$(func_parser_key "${lines[8]}")
train_batch_value=$(func_parser_params "${lines[8]}")
pretrain_model_key=$(func_parser_key "${lines[9]}")
pretrain_model_value=$(func_parser_value "${lines[9]}")
train_model_name=$(func_parser_value "${lines[10]}")
train_infer_img_dir=$(func_parser_value "${lines[11]}")
train_param_key1=$(func_parser_key "${lines[12]}")
train_param_value1=$(func_parser_value "${lines[12]}")

trainer_list=$(func_parser_value "${lines[14]}")
trainer_norm=$(func_parser_key "${lines[15]}")
norm_trainer=$(func_parser_value "${lines[15]}")
pact_key=$(func_parser_key "${lines[16]}")
pact_trainer=$(func_parser_value "${lines[16]}")
fpgm_key=$(func_parser_key "${lines[17]}")
fpgm_trainer=$(func_parser_value "${lines[17]}")
distill_key=$(func_parser_key "${lines[18]}")
distill_trainer=$(func_parser_value "${lines[18]}")
trainer_key1=$(func_parser_key "${lines[19]}")
trainer_value1=$(func_parser_value "${lines[19]}")
trainer_key2=$(func_parser_key "${lines[20]}")
trainer_value2=$(func_parser_value "${lines[20]}")

eval_py=$(func_parser_value "${lines[23]}")
eval_key1=$(func_parser_key "${lines[24]}")
eval_value1=$(func_parser_value "${lines[24]}")

save_infer_key=$(func_parser_key "${lines[27]}")
export_weight=$(func_parser_key "${lines[28]}")
norm_export=$(func_parser_value "${lines[29]}")
pact_export=$(func_parser_value "${lines[30]}")
fpgm_export=$(func_parser_value "${lines[31]}")
distill_export=$(func_parser_value "${lines[32]}")
export_key1=$(func_parser_key "${lines[33]}")
export_value1=$(func_parser_value "${lines[33]}")
export_key2=$(func_parser_key "${lines[34]}")
export_value2=$(func_parser_value "${lines[34]}")

# parser inference model 
infer_model_dir_list=$(func_parser_value "${lines[36]}")
infer_export_list=$(func_parser_value "${lines[37]}")
infer_is_quant=$(func_parser_value "${lines[38]}")
# parser inference 
inference_py=$(func_parser_value "${lines[39]}")
use_gpu_key=$(func_parser_key "${lines[40]}")
use_gpu_list=$(func_parser_value "${lines[40]}")
use_mkldnn_key=$(func_parser_key "${lines[41]}")
use_mkldnn_list=$(func_parser_value "${lines[41]}")
cpu_threads_key=$(func_parser_key "${lines[42]}")
cpu_threads_list=$(func_parser_value "${lines[42]}")
batch_size_key=$(func_parser_key "${lines[43]}")
batch_size_list=$(func_parser_value "${lines[43]}")
use_trt_key=$(func_parser_key "${lines[44]}")
use_trt_list=$(func_parser_value "${lines[44]}")
precision_key=$(func_parser_key "${lines[45]}")
precision_list=$(func_parser_value "${lines[45]}")
infer_model_key=$(func_parser_key "${lines[46]}")
image_dir_key=$(func_parser_key "${lines[47]}")
infer_img_dir=$(func_parser_value "${lines[47]}")
save_log_key=$(func_parser_key "${lines[48]}")
benchmark_key=$(func_parser_key "${lines[49]}")
benchmark_value=$(func_parser_value "${lines[49]}")
infer_key1=$(func_parser_key "${lines[50]}")
infer_value1=$(func_parser_value "${lines[50]}")
# parser serving
trans_model_py=$(func_parser_value "${lines[67]}")
infer_model_dir_key=$(func_parser_key "${lines[68]}")
infer_model_dir_value=$(func_parser_value "${lines[68]}")
model_filename_key=$(func_parser_key "${lines[69]}")
model_filename_value=$(func_parser_value "${lines[69]}")
params_filename_key=$(func_parser_key "${lines[70]}")
params_filename_value=$(func_parser_value "${lines[70]}")
serving_server_key=$(func_parser_key "${lines[71]}")
serving_server_value=$(func_parser_value "${lines[71]}")
serving_client_key=$(func_parser_key "${lines[72]}")
serving_client_value=$(func_parser_value "${lines[72]}")
serving_dir_value=$(func_parser_value "${lines[73]}")
web_service_py=$(func_parser_value "${lines[74]}")
web_use_gpu_key=$(func_parser_key "${lines[75]}")
web_use_gpu_list=$(func_parser_value "${lines[75]}")
web_use_mkldnn_key=$(func_parser_key "${lines[76]}")
web_use_mkldnn_list=$(func_parser_value "${lines[76]}")
web_cpu_threads_key=$(func_parser_key "${lines[77]}")
web_cpu_threads_list=$(func_parser_value "${lines[77]}")
web_use_trt_key=$(func_parser_key "${lines[78]}")
web_use_trt_list=$(func_parser_value "${lines[78]}")
web_precision_key=$(func_parser_key "${lines[79]}")
web_precision_list=$(func_parser_value "${lines[79]}")
pipeline_py=$(func_parser_value "${lines[80]}")


if [ ${MODE} = "cpp_infer" ]; then
    # parser cpp inference model 
    cpp_infer_model_dir_list=$(func_parser_value "${lines[53]}")
    cpp_infer_is_quant=$(func_parser_value "${lines[54]}")
    # parser cpp inference 
    inference_cmd=$(func_parser_value "${lines[55]}")
    cpp_use_gpu_key=$(func_parser_key "${lines[56]}")
    cpp_use_gpu_list=$(func_parser_value "${lines[56]}")
    cpp_use_mkldnn_key=$(func_parser_key "${lines[57]}")
    cpp_use_mkldnn_list=$(func_parser_value "${lines[57]}")
    cpp_cpu_threads_key=$(func_parser_key "${lines[58]}")
    cpp_cpu_threads_list=$(func_parser_value "${lines[58]}")
    cpp_batch_size_key=$(func_parser_key "${lines[59]}")
    cpp_batch_size_list=$(func_parser_value "${lines[59]}")
    cpp_use_trt_key=$(func_parser_key "${lines[60]}")
    cpp_use_trt_list=$(func_parser_value "${lines[60]}")
    cpp_precision_key=$(func_parser_key "${lines[61]}")
    cpp_precision_list=$(func_parser_value "${lines[61]}")
    cpp_infer_model_key=$(func_parser_key "${lines[62]}")
    cpp_image_dir_key=$(func_parser_key "${lines[63]}")
    cpp_infer_img_dir=$(func_parser_value "${lines[63]}")
    cpp_save_log_key=$(func_parser_key "${lines[64]}")
    cpp_benchmark_key=$(func_parser_key "${lines[65]}")
    cpp_benchmark_value=$(func_parser_value "${lines[65]}")
fi


LOG_PATH="./tests/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    _log_path=$4
    _img_dir=$5
    _flag_quant=$6
    # inference 
    for use_gpu in ${use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for use_mkldnn in ${use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                    continue
                fi
                for threads in ${cpu_threads_list[*]}; do
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_${batch_size}.log"
                        set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                        set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                        set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                        set_cpu_threads=$(func_set_params "${cpu_threads_key}" "${threads}")
                        set_model_dir=$(func_set_params "${infer_model_key}" "${_model_dir}")
                        set_infer_params1=$(func_set_params "${infer_key1}" "${infer_value1}")
                        command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}"
                    done
                done
            done
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for use_trt in ${use_trt_list[*]}; do
                for precision in ${precision_list[*]}; do
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                        continue
                    fi 
                    if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                        continue
                    fi
                    if [[ ${use_trt} = "False" || ${precision} =~ "int8" ]] && [ ${_flag_quant} = "True" ]; then
                        continue
                    fi
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}.log"
                        set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                        set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                        set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                        set_tensorrt=$(func_set_params "${use_trt_key}" "${use_trt}")
                        set_precision=$(func_set_params "${precision_key}" "${precision}")
                        set_model_dir=$(func_set_params "${infer_model_key}" "${_model_dir}")
                        set_infer_params1=$(func_set_params "${infer_key1}" "${infer_value1}")
                        command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_tensorrt} ${set_precision} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}"
                        
                    done
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}
function func_serving(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    # pdserving
    set_dirname=$(func_set_params "${infer_model_dir_key}" "${infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_serving_server=$(func_set_params "${serving_server_key}" "${serving_server_value}")
    set_serving_client=$(func_set_params "${serving_client_key}" "${serving_client_value}")
    trans_model_cmd="${python} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client}"
    eval $trans_model_cmd
    cd ${serving_dir_value}
    echo $PWD
    unset https_proxy
    unset http_proxy
    for use_gpu in ${web_use_gpu_list[*]}; do
        echo ${ues_gpu}
        if [ ${use_gpu} = "null" ]; then
            for use_mkldnn in ${web_use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ]; then
                    continue
                fi
                for threads in ${web_cpu_threads_list[*]}; do
                      _save_log_path="${_log_path}/server_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_1.log"
                      set_cpu_threads=$(func_set_params "${web_cpu_threads_key}" "${threads}")
                      web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}=${use_gpu} ${web_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} &>${_save_log_path} &"
                      eval $web_service_cmd
                      sleep 2s
                      pipeline_cmd="${python} ${pipeline_py}"
                      eval $pipeline_cmd
                      last_status=${PIPESTATUS[0]}
                      eval "cat ${_save_log_path}"
                      status_check $last_status "${pipeline_cmd}" "${status_log}"
                      PID=$!
                      kill $PID
                      sleep 2s
                      ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
                done
            done
        elif [ ${use_gpu} = "0" ]; then
            for use_trt in ${web_use_trt_list[*]}; do
                for precision in ${web_precision_list[*]}; do
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                        continue
                    fi
                    if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                        continue
                    fi
                    if [[ ${use_trt} = "Falg_quantse" || ${precision} =~ "int8" ]]; then
                        continue
                    fi
                    _save_log_path="${_log_path}/infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_1.log"
                    set_tensorrt=$(func_set_params "${web_use_trt_key}" "${use_trt}")
                    set_precision=$(func_set_params "${web_precision_key}" "${precision}")
                    web_service_cmd="${python} ${web_service_py} ${web_use_gpu_key}=${use_gpu} ${set_tensorrt} ${set_precision} &>${_save_log_path} & "
                    eval $web_service_cmd
                    sleep 2s
                    pipeline_cmd="${python} ${pipeline_py}"
                    eval $pipeline_cmd
                    last_status=${PIPESTATUS[0]}
                    eval "cat ${_save_log_path}"
                    status_check $last_status "${pipeline_cmd}" "${status_log}"
                    PID=$!
                    kill $PID
                    sleep 2s
                    ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

function func_cpp_inference(){
    IFS='|'
    _script=$1
    _model_dir=$2
    _log_path=$3
    _img_dir=$4
    _flag_quant=$5
    # inference 
    for use_gpu in ${cpp_use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for use_mkldnn in ${cpp_use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                    continue
                fi
                for threads in ${cpp_cpu_threads_list[*]}; do
                    for batch_size in ${cpp_batch_size_list[*]}; do
                        _save_log_path="${_log_path}/cpp_infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_${batch_size}.log"
                        set_infer_data=$(func_set_params "${cpp_image_dir_key}" "${_img_dir}")
                        set_benchmark=$(func_set_params "${cpp_benchmark_key}" "${cpp_benchmark_value}")
                        set_batchsize=$(func_set_params "${cpp_batch_size_key}" "${batch_size}")
                        set_cpu_threads=$(func_set_params "${cpp_cpu_threads_key}" "${threads}")
                        set_model_dir=$(func_set_params "${cpp_infer_model_key}" "${_model_dir}")
                        command="${_script} ${cpp_use_gpu_key}=${use_gpu} ${cpp_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}"
                    done
                done
            done
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for use_trt in ${cpp_use_trt_list[*]}; do
                for precision in ${cpp_precision_list[*]}; do
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                        continue
                    fi 
                    if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                        continue
                    fi
                    if [[ ${use_trt} = "False" || ${precision} =~ "int8" ]] && [ ${_flag_quant} = "True" ]; then
                        continue
                    fi
                    for batch_size in ${cpp_batch_size_list[*]}; do
                        _save_log_path="${_log_path}/cpp_infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}.log"
                        set_infer_data=$(func_set_params "${cpp_image_dir_key}" "${_img_dir}")
                        set_benchmark=$(func_set_params "${cpp_benchmark_key}" "${cpp_benchmark_value}")
                        set_batchsize=$(func_set_params "${cpp_batch_size_key}" "${batch_size}")
                        set_tensorrt=$(func_set_params "${cpp_use_trt_key}" "${use_trt}")
                        set_precision=$(func_set_params "${cpp_precision_key}" "${precision}")
                        set_model_dir=$(func_set_params "${cpp_infer_model_key}" "${_model_dir}")
                        command="${_script} ${cpp_use_gpu_key}=${use_gpu} ${set_tensorrt} ${set_precision} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}"
                        
                    done
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

if [ ${MODE} = "infer" ]; then
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    # set CUDA_VISIBLE_DEVICES
    eval $env
    export Count=0
    IFS="|"
    infer_run_exports=(${infer_export_list})
    infer_quant_flag=(${infer_is_quant})
    for infer_model in ${infer_model_dir_list[*]}; do
        # run export
        if [ ${infer_run_exports[Count]} != "null" ];then
            save_infer_dir=$(dirname $infer_model)
            set_export_weight=$(func_set_params "${export_weight}" "${infer_model}")
            set_save_infer_key=$(func_set_params "${save_infer_key}" "${save_infer_dir}")
            export_cmd="${python} ${norm_export} ${set_export_weight} ${set_save_infer_key}"
            eval $export_cmd
            status_export=$?
            if [ ${status_export} = 0 ];then
                status_check $status_export "${export_cmd}" "${status_log}"
            fi
        else
            save_infer_dir=${infer_model}
        fi
        #run inference
        is_quant=${infer_quant_flag[Count]}
        func_inference "${python}" "${inference_py}" "${save_infer_dir}" "${LOG_PATH}" "${infer_img_dir}" ${is_quant}
        Count=$(($Count + 1))
    done

elif [ ${MODE} = "cpp_infer" ]; then
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    # set CUDA_VISIBLE_DEVICES
    eval $env
    export Count=0
    IFS="|"
    infer_quant_flag=(${cpp_infer_is_quant})
    for infer_model in ${cpp_infer_model_dir_list[*]}; do
        #run inference
        is_quant=${infer_quant_flag[Count]}
        func_cpp_inference "${inference_cmd}" "${infer_model}" "${LOG_PATH}" "${cpp_infer_img_dir}" ${is_quant}
        Count=$(($Count + 1))
    done
    
elif [ ${MODE} = "serving_infer" ]; then
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    # set CUDA_VISIBLE_DEVICES
    eval $env
    export Count=0
    IFS="|"
    #run serving
    func_serving "${web_service_cmd}"

else
    IFS="|"
    export Count=0
    USE_GPU_KEY=(${train_use_gpu_value})
    for gpu in ${gpu_list[*]}; do
        use_gpu=${USE_GPU_KEY[Count]}
        Count=$(($Count + 1))
        if [ ${gpu} = "-1" ];then
            env=""
        elif [ ${#gpu} -le 1 ];then
            env="export CUDA_VISIBLE_DEVICES=${gpu}"
            eval ${env}
        elif [ ${#gpu} -le 15 ];then
            IFS=","
            array=(${gpu})
            env="export CUDA_VISIBLE_DEVICES=${array[0]}"
            IFS="|"
        else
            IFS=";"
            array=(${gpu})
            ips=${array[0]}
            gpu=${array[1]}
            IFS="|"
            env=" "
        fi
        for autocast in ${autocast_list[*]}; do 
            for trainer in ${trainer_list[*]}; do 
                flag_quant=False
                if [ ${trainer} = ${pact_key} ]; then
                    run_train=${pact_trainer}
                    run_export=${pact_export}
                    flag_quant=True
                elif [ ${trainer} = "${fpgm_key}" ]; then
                    run_train=${fpgm_trainer}
                    run_export=${fpgm_export}
                elif [ ${trainer} = "${distill_key}" ]; then
                    run_train=${distill_trainer}
                    run_export=${distill_export}
                elif [ ${trainer} = ${trainer_key1} ]; then
                    run_train=${trainer_value1}
                    run_export=${export_value1}
                elif [[ ${trainer} = ${trainer_key2} ]]; then
                    run_train=${trainer_value2}
                    run_export=${export_value2}
                else
                    run_train=${norm_trainer}
                    run_export=${norm_export}
                fi

                if [ ${run_train} = "null" ]; then
                    continue
                fi
                
                set_autocast=$(func_set_params "${autocast_key}" "${autocast}")
                set_epoch=$(func_set_params "${epoch_key}" "${epoch_num}")
                set_pretrain=$(func_set_params "${pretrain_model_key}" "${pretrain_model_value}")
                set_batchsize=$(func_set_params "${train_batch_key}" "${train_batch_value}")
                set_train_params1=$(func_set_params "${train_param_key1}" "${train_param_value1}")
                set_use_gpu=$(func_set_params "${train_use_gpu_key}" "${use_gpu}")
                save_log="${LOG_PATH}/${trainer}_gpus_${gpu}_autocast_${autocast}"
                
                # load pretrain from norm training if current trainer is pact or fpgm trainer
                if [ ${trainer} = ${pact_key} ] || [ ${trainer} = ${fpgm_key} ]; then
                    set_pretrain="${load_norm_train_model}"
                fi

                set_save_model=$(func_set_params "${save_model_key}" "${save_log}")
                if [ ${#gpu} -le 2 ];then  # train with cpu or single gpu
                    cmd="${python} ${run_train} ${set_use_gpu}  ${set_save_model} ${set_epoch} ${set_pretrain} ${set_autocast} ${set_batchsize} ${set_train_params1} "
                elif [ ${#gpu} -le 15 ];then  # train with multi-gpu
                    cmd="${python} -m paddle.distributed.launch --gpus=${gpu} ${run_train} ${set_save_model} ${set_epoch} ${set_pretrain} ${set_autocast} ${set_batchsize} ${set_train_params1}"
                else     # train with multi-machine
                    cmd="${python} -m paddle.distributed.launch --ips=${ips} --gpus=${gpu} ${run_train} ${set_save_model} ${set_pretrain} ${set_epoch} ${set_autocast} ${set_batchsize} ${set_train_params1}"
                fi
                # run train
                eval "unset CUDA_VISIBLE_DEVICES"
                eval $cmd
                status_check $? "${cmd}" "${status_log}"

                set_eval_pretrain=$(func_set_params "${pretrain_model_key}" "${save_log}/${train_model_name}")
                # save norm trained models to set pretrain for pact training and fpgm training 
                if [ ${trainer} = ${trainer_norm} ]; then
                    load_norm_train_model=${set_eval_pretrain}
                fi
                # run eval 
                if [ ${eval_py} != "null" ]; then
                    set_eval_params1=$(func_set_params "${eval_key1}" "${eval_value1}")
                    eval_cmd="${python} ${eval_py} ${set_eval_pretrain} ${set_use_gpu} ${set_eval_params1}" 
                    eval $eval_cmd
                    status_check $? "${eval_cmd}" "${status_log}"
                fi
                # run export model
                if [ ${run_export} != "null" ]; then 
                    # run export model
                    save_infer_path="${save_log}"
                    set_export_weight=$(func_set_params "${export_weight}" "${save_log}/${train_model_name}")
                    set_save_infer_key=$(func_set_params "${save_infer_key}" "${save_infer_path}")
                    export_cmd="${python} ${run_export} ${set_export_weight} ${set_save_infer_key}"
                    eval $export_cmd
                    status_check $? "${export_cmd}" "${status_log}"

                    #run inference
                    eval $env
                    save_infer_path="${save_log}"
                    func_inference "${python}" "${inference_py}" "${save_infer_path}" "${LOG_PATH}" "${train_infer_img_dir}" "${flag_quant}"
                    eval "unset CUDA_VISIBLE_DEVICES"
                fi
            done  # done with:    for trainer in ${trainer_list[*]}; do 
        done      # done with:    for autocast in ${autocast_list[*]}; do 
    done          # done with:    for gpu in ${gpu_list[*]}; do
fi  # end if [ ${MODE} = "infer" ]; then
