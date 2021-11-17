#!/bin/bash
# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37
# 执行目录: ./PaddleOCR
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
python -m pip install -r requirements.txt
# 2 拷贝该模型需要数据、预训练模型
wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar && cd train_data  && tar xf icdar2015.tar && cd ../
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_vd_pretrained.pdparams
# 3 批量运行（如不方便批量，1，2需放到单个模型中）

model_mode_list=(det_res18_db_v2.0 det_r50_vd_east det_r50_vd_pse)
fp_item_list=(fp32)
bs_list=(8 16)
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]}; do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            log_name=ocr_${model_mode}_${run_mode}_bs${bs_item}_${fp_item}
            CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark_det.sh ${run_mode} ${bs_item} ${fp_item} 1 ${model_mode} | tee ${log_path}/${log_name}_speed_1gpus 2>&1    #  (5min)
            sleep 60
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            log_name=ocr_${model_mode}_${run_mode}_bs${bs_item}_${fp_item}
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark_det.sh ${run_mode} ${bs_item} ${fp_item} 2 ${model_mode} | tee ${log_path}/${log_name}_speed_8gpus8p 2>&1
            sleep 60
            done
      done
done


