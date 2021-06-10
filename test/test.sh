#!/bin/bash 
# Usage:
# bash test/test.sh ./test/paddleocr_ci_params.txt 'lite_train_infer'

FILENAME=$1

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2
# prepare pretrained weights and dataset 
wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar
cd pretrain_models && tar xf det_mv3_db_v2.0_train.tar && cd ../

if [ ${MODE} = "lite_train_infer" ];then
    # pretrain lite train data
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar
    cd ./train_data/ && tar xf icdar2015_lite.tar
    ln -s ./icdar2015_lite ./icdar2015
    cd ../
    epoch=10
    eval_batch_step=10
elif [ ${MODE} = "whole_train_infer" ];then
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar
    cd ./train_data/ && tar xf icdar2015.tar && cd ../
    epoch=500
    eval_batch_step=200
else
    rm -rf ./train_data/icdar2015
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_infer.tar
    cd ./train_data/ && tar xf icdar2015_infer.tar
    ln -s ./icdar2015_infer ./icdar2015
    cd ../
    epoch=10
    eval_batch_step=10
fi

img_dir="./train_data/icdar2015/text_localization/ch4_test_images/"


dataline=$(cat ${FILENAME})
# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser(){
    strs=$1
    IFS=": "
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
train_model_list=$(func_parser "${lines[0]}")
gpu_list=$(func_parser "${lines[1]}")
auto_cast_list=$(func_parser "${lines[2]}")
slim_trainer_list=$(func_parser "${lines[3]}")
python=$(func_parser "${lines[4]}")
# inference params
inference=$(func_parser "${lines[5]}")
devices=$(func_parser "${lines[6]}")
use_mkldnn_list=$(func_parser "${lines[7]}")
cpu_threads_list=$(func_parser "${lines[8]}")
rec_batch_size_list=$(func_parser "${lines[9]}")
gpu_trt_list=$(func_parser "${lines[10]}")
gpu_precision_list=$(func_parser "${lines[11]}")

log_path=$(func_parser "${lines[13]}")
status_log="${log_path}/result.log"

# install requirments
${python} -m pip install pynvml;
${python} -m pip install psutil;
${python} -m pip install GPUtil;
${python} -m pip install paddlesim==2.0.0

function status_check(){
    last_status=$1   # the exit code
    run_model=$2
    run_command=$3
    run_log=$4
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m $run_model successfully with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m $case failed with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    fi
}

IFS="|"
for train_model in ${train_model_list[*]}; do 
    if [ ${train_model} = "ocr_det" ];then
        model_name="det"
        yml_file="configs/det/det_mv3_db.yml"
    elif [ ${train_model} = "ocr_rec" ];then
        model_name="rec"
        yml_file="configs/rec/rec_mv3_none_bilstm_ctc.yml"
    else
        model_name="det"
        yml_file="configs/det/det_mv3_db.yml"
    fi
    IFS="|"
    for gpu in ${gpu_list[*]}; do
        use_gpu=True
        if [ ${gpu} = "-1" ];then
            use_gpu=False
            env=""
        elif [ ${#gpu} -le 1 ];then
            env="CUDA_VISIBLE_DEVICES=${gpu}"
        else
            IFS=","
            array=(${gpu})
            env="CUDA_VISIBLE_DEVICES=${array[0]}"
            IFS="|"
        fi
        for auto_cast in ${auto_cast_list[*]}; do 
            for slim_trainer in ${slim_trainer_list[*]}; do 
                if [ ${slim_trainer} = "norm" ]; then
                    trainer="tools/train.py"
                    export_model="tools/export_model.py"
                    pretrain="./pretrain_models/MobileNetV3_large_x0_5_pretrained"
                elif [ ${slim_trainer} = "quant" ]; then
                    trainer="deploy/slim/quantization/quant.py"
                    export_model="deploy/slim/quantization/export_model.py"
                    pretrain="./pretrain_models/det_mv3_db_v2.0_train/best_accuracy"
                elif [ ${slim_trainer} = "prune" ]; then
                    trainer="deploy/slim/prune/sensitivity_anal.py"
                    export_model="deploy/slim/prune/export_prune_model.py"
                    pretrain="./pretrain_models/det_mv3_db_v2.0_train/best_accuracy"
                    wget -nc -P https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/sen.pickle
                elif [ ${slim_trainer} = "distill" ]; then
                    trainer="deploy/slim/distill/train_dml.py"
                    export_model="deploy/slim/distill/export_distill_model.py"
                    pretrain=""
                else
                    trainer="tools/train.py"
                    export_model="tools/export_model.py"
                    pretrain="./pretrain_models/MobileNetV3_large_x0_5_pretrained"
                fi
                save_log="${log_path}/${model_name}_${slim_trainer}_autocast_${auto_cast}_gpuid_${gpu}"
                if [ ${#gpu} -le 2 ];then
                    command="${python} ${trainer}  -c ${yml_file} -o Global.epoch_num=${epoch} Global.eval_batch_step=${eval_batch_step} Global.auto_cast=${auto_cast} Global.pretrained_model=${pretrain}  Global.save_model_dir=${save_log} Global.use_gpu=${use_gpu} Train.loader.batch_size_per_card=2"
                    ${python} ${trainer}  -c ${yml_file} -o Global.epoch_num=${epoch} Global.eval_batch_step=${eval_batch_step} Global.auto_cast=${auto_cast} Global.pretrained_model=${pretrain}  Global.save_model_dir=${save_log} Global.use_gpu=${use_gpu}  Train.loader.batch_size_per_card=2
                else 
                    command="${python} -m paddle.distributed.launch --log_dir=./debug/ --gpus ${gpu} ${trainer}  -c ${yml_file} -o Global.epoch_num=${epoch} Global.eval_batch_step=${eval_batch_step} Global.auto_cast=${auto_cast} Global.pretrained_model=${pretrain}  Global.save_model_dir=${save_log} Global.use_gpu=${use_gpu} Train.loader.batch_size_per_card=2"
                    ${python} -m paddle.distributed.launch --log_dir=./debug/ --gpus ${gpu} ${trainer}  -c ${yml_file} -o Global.epoch_num=${epoch} Global.eval_batch_step=${eval_batch_step} Global.auto_cast=${auto_cast} Global.pretrained_model=${pretrain}  Global.save_model_dir=${save_log} Global.use_gpu=${use_gpu}  Train.loader.batch_size_per_card=2
                fi
                status_check $? "${trainer}" "${command}" "${status_log}"

                command="${python} ${export_model} -c ${yml_file} -o Global.pretrained_model=${save_log}/latest Global.save_inference_dir=${save_log}_infer/ Global.save_model_dir=${save_log}"
                ${python} ${export_model} -c ${yml_file} -o Global.pretrained_model=${save_log}/latest Global.save_inference_dir=${save_log}_infer/ Global.save_model_dir=${save_log} 
                status_check $? "${trainer}" "${command}" "${status_log}"
               
                if [ "${model_name}" = "det" ]; then 
                    export rec_batch_size_list=( "1" )
                    inference="tools/infer/predict_det.py"
                    det_model_dir=${save_log}_infer
                    rec_model_dir=""
                elif [ "${model_name}" = "rec" ]; then
                    inference="tools/infer/predict_rec.py"
                    rec_model_dir=${save_log}_infer
                    det_model_dir=""
                fi
                # inference 
                for device in ${devices[*]}; do 
                    if [ ${device} = "cpu" ]; then
                        for use_mkldnn in ${use_mkldnn_list[*]}; do
                            for threads in ${cpu_threads_list[*]}; do
                                for rec_batch_size in ${rec_batch_size_list[*]}; do    
                                    save_log_path="${log_path}/${model_name}_${slim_trainer}_cpu_usemkldnn_${use_mkldnn}_cputhreads_${threads}_recbatchnum_${rec_batch_size}_infer.log"
                                    command="${python} ${inference} --enable_mkldnn=${use_mkldnn} --use_gpu=False --cpu_threads=${threads} --benchmark=True --det_model_dir=${det_model_dir} --rec_batch_num=${rec_batch_size} --rec_model_dir=${rec_model_dir}  --image_dir=${img_dir}  --save_log_path=${save_log_path}"
                                    ${python} ${inference} --enable_mkldnn=${use_mkldnn} --use_gpu=False --cpu_threads=${threads} --benchmark=True --det_model_dir=${det_model_dir} --rec_batch_num=${rec_batch_size} --rec_model_dir=${rec_model_dir}  --image_dir=${img_dir}  --save_log_path=${save_log_path}
                                    status_check $? "${inference}" "${command}" "${status_log}"
                                done
                            done
                        done
                    else 
                        for use_trt in ${gpu_trt_list[*]}; do
                            for precision in ${gpu_precision_list[*]}; do
                                if [ ${use_trt} = "False" ] && [ ${precision} != "fp32" ]; then
                                    continue
                                fi
                                for rec_batch_size in ${rec_batch_size_list[*]}; do
                                    save_log_path="${log_path}/${model_name}_${slim_trainer}_gpu_usetensorrt_${use_trt}_usefp16_${precision}_recbatchnum_${rec_batch_size}_infer.log"
                                    command="${python} ${inference} --use_gpu=True --use_tensorrt=${use_trt}  --precision=${precision} --benchmark=True --det_model_dir=${det_model_dir} --rec_batch_num=${rec_batch_size} --rec_model_dir=${rec_model_dir} --image_dir=${img_dir} --save_log_path=${save_log_path}"
                                    ${python} ${inference} --use_gpu=True --use_tensorrt=${use_trt}  --precision=${precision} --benchmark=True --det_model_dir=${det_model_dir} --rec_batch_num=${rec_batch_size} --rec_model_dir=${rec_model_dir} --image_dir=${img_dir} --save_log_path=${save_log_path}
                                    status_check $? "${inference}" "${command}" "${status_log}"
                                done
                            done
                        done
                    fi
                done
            done
        done
    done
done
