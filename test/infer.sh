#!/bin/bash
FILENAME=$1
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
slim_trainer_list=$(func_parser "${lines[3]}")
python=$(func_parser "${lines[4]}")
# inference params
# inference=$(func_parser "${lines[5]}")
devices=$(func_parser "${lines[6]}")
use_mkldnn_list=$(func_parser "${lines[7]}")
cpu_threads_list=$(func_parser "${lines[8]}")
rec_batch_size_list=$(func_parser "${lines[9]}")
gpu_trt_list=$(func_parser "${lines[10]}")
gpu_precision_list=$(func_parser "${lines[11]}")

infer_gpu_id=$(func_parser "${lines[12]}")
log_path=$(func_parser "${lines[13]}")
status_log="${log_path}/result.log"


function status_check(){
    last_status=$1   # the exit code
    run_model=$2
    run_command=$3
    save_log=$4
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m $run_model successfully with command - ${run_command}!  \033[0m" | tee -a ${save_log}
    else
        echo -e "\033[33m $case failed with command - ${run_command}!  \033[0m" | tee -a ${save_log}
    fi
}
IFS='|'
for train_model in ${train_model_list[*]}; do 
    if [ ${train_model} = "ocr_det" ];then
        model_name="det"
        yml_file="configs/det/det_mv3_db.yml"
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar
        cd ./inference && tar xf ch_det_data_50.tar && cd ../
        img_dir="./inference/ch_det_data_50/"
    elif [ ${train_model} = "ocr_rec" ];then
        model_name="rec"
        yml_file="configs/rec/rec_mv3_none_bilstm_ctc.yml"
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_rec_data_200.tar 
        cd ./inference && tar xf ch_rec_data_200.tar  && cd ../
        img_dir="./inference/ch_rec_data_200/"
    fi

    # eval 
    for slim_trainer in ${slim_trainer_list[*]}; do 
        if [ ${slim_trainer} = "norm" ]; then
            if [ ${model_name} = "det" ]; then
                eval_model_name="ch_ppocr_mobile_v2.0_det_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            else 
                eval_model_name="ch_ppocr_mobile_v2.0_rec_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            fi 
        elif [ ${slim_trainer} = "quant" ]; then
            if [ ${model_name} = "det" ]; then
                eval_model_name="ch_ppocr_mobile_v2.0_det_quant_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_quant_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            else
                eval_model_name="ch_ppocr_mobile_v2.0_rec_quant_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_quant_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            fi
        elif [ ${slim_trainer} = "distill" ]; then
            if [ ${model_name} = "det" ]; then
                eval_model_name="ch_ppocr_mobile_v2.0_det_distill_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_distill_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            else
                eval_model_name="ch_ppocr_mobile_v2.0_rec_distill_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_distill_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            fi 
        elif [ ${slim_trainer} = "prune" ]; then
            if [ ${model_name} = "det" ]; then
                eval_model_name="ch_ppocr_mobile_v2.0_det_prune_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            else
                eval_model_name="ch_ppocr_mobile_v2.0_rec_prune_train"
                wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_prune_train.tar
                cd ./inference && tar xf ${eval_model_name}.tar && cd ../
            fi
        fi

        save_log_path="${log_path}/${eval_model_name}"
        command="${python} tools/eval.py -c ${yml_file} -o Global.pretrained_model="${eval_model_name}/best_accuracy" Global.save_model_dir=${save_log_path}"
        ${python} tools/eval.py -c ${yml_file} -o Global.pretrained_model="${eval_model_name}/best_accuracy" Global.save_model_dir=${save_log_path}
        status_check $? "${trainer}" "${command}" "${status_log}"

        command="${python} tools/export_model.py -c ${yml_file} -o Global.pretrained_model="${eval_model_name}/best_accuracy" Global.save_inference_dir=${log_path}/${eval_model_name}_infer Global.save_model_dir=${save_log_path}"
        ${python} tools/export_model.py -c ${yml_file} -o Global.pretrained_model="${eval_model_name}/best_accuracy" Global.save_inference_dir="${log_path}/${eval_model_name}_infer" Global.save_model_dir=${save_log_path}
        status_check $? "${trainer}" "${command}" "${status_log}"

        if [ $? -eq 0 ]; then
            echo -e "\033[33m training of $model_name successfully!\033[0m" | tee -a ${save_log}/train.log
        else
            cat ${save_log}/train.log
            echo -e "\033[33m training of $model_name failed!\033[0m" | tee -a ${save_log}/train.log
        fi
        if [ "${model_name}" = "det" ]; then 
            export rec_batch_size_list=( "1" )
            inference="tools/infer/predict_det.py"
            det_model_dir=${log_path}/${eval_model_name}_infer
            rec_model_dir=""
        elif [ "${model_name}" = "rec" ]; then
            inference="tools/infer/predict_rec.py"
            rec_model_dir=${log_path}/${eval_model_name}_infer
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
                            status_check $? "${trainer}" "${command}" "${status_log}"
                        done
                    done
                done
            else 
                env="CUDA_VISIBLE_DEVICES=${infer_gpu_id}"
                for use_trt in ${gpu_trt_list[*]}; do
                    for precision in ${gpu_precision_list[*]}; do
                        if [ ${use_trt} = "False" ] && [ ${precision} != "fp32" ]; then
                            continue
                        fi
                        for rec_batch_size in ${rec_batch_size_list[*]}; do
                            save_log_path="${log_path}/${model_name}_${slim_trainer}_gpu_usetensorrt_${use_trt}_usefp16_${precision}_recbatchnum_${rec_batch_size}_infer.log"
                            command="${env} ${python} ${inference} --use_gpu=True --use_tensorrt=${use_trt}  --precision=${precision} --benchmark=True --det_model_dir=${log_path}/${eval_model_name}_infer --rec_batch_num=${rec_batch_size} --rec_model_dir=${rec_model_dir} --image_dir=${img_dir} --save_log_path=${save_log_path}"
                            ${env} ${python} ${inference} --use_gpu=True --use_tensorrt=${use_trt}  --precision=${precision} --benchmark=True --det_model_dir=${log_path}/${eval_model_name}_infer --rec_batch_num=${rec_batch_size} --rec_model_dir=${rec_model_dir} --image_dir=${img_dir} --save_log_path=${save_log_path}
                            status_check $? "${trainer}" "${command}" "${status_log}"
                        done
                    done
                done
            fi
        done
    done
done
