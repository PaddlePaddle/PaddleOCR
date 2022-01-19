#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

if [ ${MODE} = "benchmark_train" ];then
    pip install -r requirements.txt
    if [[ ${model_name} =~ "det_mv3_db_v2.0_benchmark" ]];then
        rm -rf ./train_data/icdar2015
        wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams  --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015.tar && cd ../
    fi
fi

if [ ${MODE} = "lite_train_lite_infer" ];then
    # pretrain lite train data
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams  --no-check-certificate
    wget -nc -P ./pretrain_models/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar  --no-check-certificate
    if [[ ${model_name} =~ "PPOCRv2_det" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
    cd ./pretrain_models/ && tar xf det_mv3_db_v2.0_train.tar && cd ../
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar --no-check-certificate
    wget -nc -P ./deploy/slim/prune https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/sen.pickle --no-check-certificate
    
    cd ./train_data/ && tar xf icdar2015_lite.tar && tar xf ic15_data.tar
    ln -s ./icdar2015_lite ./icdar2015
    cd ../
    cd ./inference && tar xf rec_inference.tar && cd ../
    if [ ${model_name} == "en_server_pgnetA" ]; then
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/total_text_lite.tar --no-check-certificate
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_server_pgnetA.tar && cd ../
        cd ./train_data && tar xf total_text_lite.tar && ln -s total_text_lite total_text && cd ../
    fi
    if [ ${model_name} == "det_r50_vd_sast_icdar15_v2.0" ] || [ ${model_name} == "det_r50_vd_sast_totaltext_v2.0" ]; then
        wget -nc -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/total_text_lite.tar --no-check-certificate
        cd ./train_data && tar xf total_text_lite.tar && ln -s total_text_lite total_text  && cd ../
    fi
    if [ ${model_name} == "det_mv3_db_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_mv3_db_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "det_r50_db_v2.0" ]; then
        wget -nc -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_db_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_ppocr_mobile_v2.0_rec_FPGM" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_ppocr_mobile_v2.0_rec_train.tar && cd ../
    fi

elif [ ${MODE} = "whole_train_whole_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams --no-check-certificate
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    cd ./train_data/ && tar xf icdar2015.tar && tar xf ic15_data.tar && cd ../
    if [ ${model_name} == "ch_PPOCRv2_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
    if [ ${model_name} == "en_server_pgnetA" ]; then
        wget -nc -P ./train_data/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/total_text_lite.tar  --no-check-certificate
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_server_pgnetA.tar && cd ../
        cd ./train_data && tar xf total_text.tar && ln -s total_text_lite total_text  && cd ../
    fi
    if [ ${model_name} == "det_r50_vd_sast_totaltext_v2.0" ]; then
        wget -nc -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/total_text_lite.tar  --no-check-certificate
        cd ./train_data && tar xf total_text.tar && ln -s total_text_lite total_text  && cd ../
    fi
elif [ ${MODE} = "lite_train_whole_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams --no-check-certificate
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_infer.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    cd ./train_data/ && tar xf icdar2015_infer.tar && tar xf ic15_data.tar
    ln -s ./icdar2015_infer ./icdar2015
    cd ../
    if [ ${model_name} == "ch_PPOCRv2_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
elif [ ${MODE} = "whole_infer" ];then
    wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar --no-check-certificate
    wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar --no-check-certificate
    cd ./inference && tar xf rec_inference.tar  && tar xf ch_det_data_50.tar && cd ../
    if [ ${model_name} = "ch_ppocr_mobile_v2.0_det" ]; then
        eval_model_name="ch_ppocr_mobile_v2.0_det_train"
        rm -rf ./train_data/icdar2015
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2.0_det_PACT" ]; then
        eval_model_name="ch_ppocr_mobile_v2.0_det_prune_infer"
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_server_v2.0_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_train.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ch_ppocr_mobile_v2.0" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ch_ppocr_server_v2.0" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2.0_rec_PACT" ]; then
        eval_model_name="ch_ppocr_mobile_v2.0_rec_slim_infer"
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2.0_rec_FPGM" ]; then
        eval_model_name="ch_PP-OCRv2_rec_infer"
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && cd ../
    fi 
    if [[ ${model_name} =~ "ch_PPOCRv2_det" ]]; then
        eval_model_name="ch_PP-OCRv2_det_infer"
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [[ ${model_name} =~ "PPOCRv2_ocr_rec" ]]; then
        eval_model_name="ch_PP-OCRv2_rec_infer"
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_PP-OCRv2_rec_slim_quant_infer.tar && cd ../
    fi   
    if [ ${model_name} == "en_server_pgnetA" ]; then
        wget -nc -P ./inference/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar  --no-check-certificate
        cd ./inference && tar xf en_server_pgnetA.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [ ${model_name} == "det_r50_vd_sast_icdar15_v2.0" ]; then
        wget -nc -P  ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_sast_icdar15_v2.0_train.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [ ${model_name} == "rec_mv3_none_none_ctc_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mv3_none_none_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r34_vd_none_none_ctc_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r34_vd_none_none_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_mv3_none_bilstm_ctc_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mv3_none_bilstm_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r34_vd_none_bilstm_ctc_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r34_vd_none_bilstm_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_mv3_tps_bilstm_ctc_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mv3_tps_bilstm_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r34_vd_tps_bilstm_ctc_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_ppocr_server_v2.0_rec" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar --no-check-certificate
        cd ./inference/ && tar xf ch_ppocr_server_v2.0_rec_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_ppocr_mobile_v2.0_rec" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar --no-check-certificate
        cd ./inference/ && tar xf ch_ppocr_mobile_v2.0_rec_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_mtb_nrtr" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mtb_nrtr_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_mv3_tps_bilstm_att_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mv3_tps_bilstm_att_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r34_vd_tps_bilstm_att_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r34_vd_tps_bilstm_att_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r31_sar" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_sar_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r31_sar_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r50_fpn_vd_none_srn" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r50_vd_srn_train.tar && cd ../
    fi
    
    if [ ${model_name} == "det_r50_vd_sast_totaltext_v2.0" ]; then
        wget -nc -P  ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_sast_totaltext_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "det_mv3_db_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_mv3_db_v2.0_train.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [ ${model_name} == "det_r50_db_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_db_v2.0_train.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [ ${model_name} == "det_mv3_pse_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_mv3_pse_v2.0_train.tar & cd ../
    fi
    if [ ${model_name} == "det_r50_vd_pse_v2.0" ]; then
        wget -nc -P ./inference/  https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_pse_v2.0_train.tar & cd ../
    fi
    if [ ${model_name} == "det_mv3_east_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_mv3_east_v2.0_train.tar & cd ../
    fi
    if [ ${model_name} == "det_r50_vd_east_v2.0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_east_v2.0_train.tar & cd ../
    fi
fi

if [ ${MODE} = "klquant_whole_infer" ]; then
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar --no-check-certificate
    cd ./train_data/ && tar xf icdar2015_lite.tar && rm -rf ./icdar2015 && ln -s ./icdar2015_lite ./icdar2015 && cd ../
    if [ ${model_name} = "ch_ppocr_mobile_v2.0_det_KL" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && cd ../ 
    fi
    if [ ${model_name} = "PPOCRv2_ocr_rec_kl" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar  --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data.tar && cd ../
        cd ./inference && tar xf rec_inference.tar && tar xf ch_PP-OCRv2_rec_infer.tar && cd ../
    fi
    if [ ${model_name} = "PPOCRv2_ocr_det_kl" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    fi 
    if [ ${model_name} = "ch_ppocr_mobile_v2.0_rec_KL" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data.tar && cd ../
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf rec_inference.tar &&  cd ../ 
    fi 
fi

if [ ${MODE} = "cpp_infer" ];then
    if [ ${model_name} = "ocr_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2.0_rec" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf rec_inference.tar && cd ../
    elif  [ ${model_name} = "ocr_system" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    fi 
fi

if [ ${MODE} = "serving_infer" ];then
    # prepare serving env
    python_name_list=$(func_parser_value "${lines[2]}")
    IFS='|'
    array=(${python_name_list})
    python_name=${array[0]}
    wget -nc https://paddle-serving.bj.bcebos.com/chain/paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    ${python_name} -m pip install install paddle_serving_server_gpu-0.0.0.post101-py3-none-any.whl
    ${python_name} -m pip install paddle_serving_client==0.6.1
    ${python_name} -m pip install paddle-serving-app==0.6.3
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar
    cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_det_infer.tar && cd ../
fi

if [ ${MODE} = "paddle2onnx_infer" ];then
    # prepare serving env
    python_name=$(func_parser_value "${lines[2]}")
    ${python_name} -m pip install install paddle2onnx
    ${python_name} -m pip install onnxruntime==1.4.0
    # wget model
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar
    # wget data
    wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar
    wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar
    cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && tar xf rec_inference.tar && cd ../
    
fi
