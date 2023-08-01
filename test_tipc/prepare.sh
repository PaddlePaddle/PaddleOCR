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
    python_name_list=$(func_parser_value "${lines[2]}")
    array=(${python_name_list}) 
    python_name=python
    ${python_name} -m pip install -r requirements.txt
    if [[ ${model_name} =~ "ch_ppocr_mobile_v2_0_det" || ${model_name} =~ "det_mv3_db_v2_0" ]];then
        wget -nc -P  ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams  --no-check-certificate
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
        if [[ ${model_name} =~ "ch_ppocr_mobile_v2_0_det" ]];then
            # expand gt.txt 2 times
            cd ./train_data/icdar2015/text_localization
            for i in `seq 2`;do cp train_icdar2015_label.txt dup$i.txt;done
            cat dup* > train_icdar2015_label.txt && rm -rf dup*
            cd ../../../
        fi
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv4_mobile_det" ]];then
        wget -nc -P  ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/PPLCNetV3_x0_75_ocr_det.pdparams  --no-check-certificate
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv4_server_det" ]];then
        wget -nc -P  ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/PPHGNet_small_ocr_det.pdparams  --no-check-certificate
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv4_mobile_rec" ]];then
        rm -rf ./train_data/ic15_data
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/ic15_data_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data_benckmark.tar
        ln -s ./ic15_data_benckmark ./ic15_data
        cd ic15_data
        mv rec_gt_train4w.txt rec_gt_train.txt
        cd ../
        cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv4_server_rec" ]];then
        rm -rf ./train_data/ic15_data
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/ic15_data_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data_benckmark.tar
        ln -s ./ic15_data_benckmark ./ic15_data
        cd ic15_data
        mv rec_gt_train4w.txt rec_gt_train.txt
        cd ../
        cd ../
    fi
    if [[ ${model_name} =~ "ch_ppocr_server_v2_0_det" || ${model_name} =~ "ch_PP-OCRv3_det" ]];then
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv2_det" ]];then
        wget  -nc -P  ./pretrain_models/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_ppocr_server_v2.0_det_train.tar  && cd ../
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
    fi
    if [[ ${model_name} =~ "det_r50_vd_east_v2_0" ]]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf det_r50_vd_east_v2.0_train.tar && cd ../
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
    fi
    if [[ ${model_name} =~ "det_r50_db_v2_0" || ${model_name} =~ "det_r50_vd_pse_v2_0" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/ResNet50_vd_ssld_pretrained.pdparams --no-check-certificate
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
    fi
    if [[ ${model_name} =~ "det_r18_db_v2_0" ]];then
        wget -nc -P ./pretrain_models/  https://paddleocr.bj.bcebos.com/pretrained/ResNet18_vd_pretrained.pdparams  --no-check-certificate
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
    fi
    if [[ ${model_name} =~ "ch_ppocr_mobile_v2_0_rec" || ${model_name} =~ "ch_ppocr_server_v2_0_rec" || ${model_name} =~ "ch_PP-OCRv2_rec" || ${model_name} =~ "rec_mv3_none_bilstm_ctc_v2_0" || ${model_name} =~ "ch_PP-OCRv3_rec" ]];then
        rm -rf ./train_data/ic15_data
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/ic15_data_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data_benckmark.tar
        ln -s ./ic15_data_benckmark ./ic15_data
        cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv2_rec" || ${model_name} =~ "ch_PP-OCRv3_rec" || ${model_name} =~ "ch_PP-OCRv4_mobile_rec" || ${model_name} =~ "ch_PP-OCRv4_server_rec" ]];then
        rm -rf ./train_data/ic15_data
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/ic15_data_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data_benckmark.tar
        ln -s ./ic15_data_benckmark ./ic15_data
        cd ic15_data
        mv rec_gt_train4w.txt rec_gt_train.txt
        cd ../
        cd ../
    fi
    if [[ ${model_name} == "en_table_structure" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_ppocr_mobile_v2.0_table_structure_train.tar  && cd ../
        rm -rf ./train_data/pubtabnet
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/pubtabnet_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf pubtabnet_benckmark.tar
        ln -s ./pubtabnet_benckmark ./pubtabnet
        cd ../
    fi
    if [[ ${model_name} == "slanet" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_ppstructure_mobile_v2.0_SLANet_train.tar  && cd ../
        rm -rf ./train_data/pubtabnet
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/pubtabnet_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf pubtabnet_benckmark.tar
        ln -s ./pubtabnet_benckmark ./pubtabnet
        cd ../
    fi
    if [[ ${model_name} == "det_r50_dcn_fce_ctw_v2_0" ]]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf det_r50_dcn_fce_ctw_v2.0_train.tar && cd ../
        rm -rf ./train_data/icdar2015
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/icdar2015_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf icdar2015_benckmark.tar
        ln -s ./icdar2015_benckmark ./icdar2015
        cd ../
    fi
    if [ ${model_name} == "layoutxlm_ser" ] || [ ${model_name} == "vi_layoutxlm_ser" ]; then
        ${python_name} -m pip install -r ppstructure/kie/requirements.txt
        ${python_name} -m pip install opencv-python -U
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar --no-check-certificate
        cd ./train_data/ && tar xf XFUND.tar
        # expand gt.txt 10 times
        cd XFUND/zh_train
        for i in `seq 10`;do cp train.json dup$i.txt;done
        cat dup* > train.json && rm -rf dup*
        cd ../../
        
        cd ../
    fi
    if [ ${model_name} == "table_master" ];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf table_structure_tablemaster_train.tar  && cd ../
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/StructureLabel_val_500.tar --no-check-certificate
        cd ./train_data/ && tar xf StructureLabel_val_500.tar
        cd ../
    fi
    if [ ${model_name} == "rec_svtrnet" ]; then
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/ic15_data_benckmark.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data_benckmark.tar
        ln -s ./ic15_data_benckmark ./ic15_data
        cd ic15_data
        mv rec_gt_train4w.txt rec_gt_train.txt

        for i in `seq 10`;do cp rec_gt_train.txt dup$i.txt;done
        cat dup* > rec_gt_train.txt && rm -rf dup*

        cd ../
        cd ../
    fi
fi

if [ ${MODE} = "lite_train_lite_infer" ];then
    python_name_list=$(func_parser_value "${lines[2]}")
    array=(${python_name_list}) 
    python_name=${array[0]}
    ${python_name} -m pip install -r requirements.txt
    ${python_name} -m pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ${python_name} -m pip install paddleslim
    # pretrain lite train data
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams  --no-check-certificate
    wget -nc -P ./pretrain_models/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar  --no-check-certificate
    cd ./pretrain_models/
    tar xf det_mv3_db_v2.0_train.tar
    cd ../
    if [[ ${model_name} =~ "ch_PP-OCRv2_det" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv3_det" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv3_det_distill_train.tar && cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv4_mobile_det" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/PPLCNetV3_x0_75_ocr_det.pdparams --no-check-certificate
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv4_server_det" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/PPHGNet_small_ocr_det.pdparams --no-check-certificate
    fi
    if [ ${model_name} == "en_table_structure" ] || [ ${model_name} == "en_table_structure_PACT" ];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_ppocr_mobile_v2.0_table_structure_train.tar  && cd ../
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar --no-check-certificate
        cd ./inference/ && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar && cd ../
    fi
    if [[ ${model_name} =~ "slanet" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_ppstructure_mobile_v2.0_SLANet_train.tar  && cd ../
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar --no-check-certificate
        cd ./inference/ && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar && cd ../
    fi
    if [[ ${model_name} =~ "det_r50_db_plusplus" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams --no-check-certificate
    fi
    if [ ${model_name} == "table_master" ];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf table_structure_tablemaster_train.tar  && cd ../
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/StructureLabel_val_500.tar --no-check-certificate
        cd ./train_data/ && tar xf StructureLabel_val_500.tar && cd ../
    fi
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    rm -rf ./train_data/pubtabnet
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/pubtabnet.tar --no-check-certificate
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar --no-check-certificate
    wget -nc -P ./deploy/slim/prune https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/sen.pickle --no-check-certificate
    
    cd ./train_data/ && tar xf icdar2015_lite.tar && tar xf ic15_data.tar && tar xf pubtabnet.tar
    ln -s ./icdar2015_lite ./icdar2015
    wget -nc -P ./ic15_data/ https://paddleocr.bj.bcebos.com/dataset/rec_gt_train_lite.txt --no-check-certificate
    wget -nc -P ./ic15_data/ https://paddleocr.bj.bcebos.com/dataset/rec_gt_test_lite.txt --no-check-certificate
    mv ic15_data/rec_gt_train_lite.txt ic15_data/rec_gt_train.txt
    mv ic15_data/rec_gt_test_lite.txt ic15_data/rec_gt_test.txt
    cd ../
    cd ./inference && tar xf rec_inference.tar && cd ../
    if [ ${model_name} == "ch_PP-OCRv2_det" ] || [ ${model_name} == "ch_PP-OCRv2_det_PACT" ]; then
        wget  -nc -P  ./pretrain_models/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_ppocr_server_v2.0_det_train.tar  && cd ../
    fi
    if [ ${model_name} == "ch_PP-OCRv2_rec" ] || [ ${model_name} == "ch_PP-OCRv2_rec_PACT" ]; then
        wget  -nc -P  ./pretrain_models/  https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_rec_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_PP-OCRv3_rec" ] || [ ${model_name} == "ch_PP-OCRv3_rec_PACT" ]; then
        wget  -nc -P  ./pretrain_models/  https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv3_rec_train.tar && cd ../
    fi
    if [ ${model_name} == "det_r18_db_v2_0" ]; then
        wget -nc -P ./pretrain_models/  https://paddleocr.bj.bcebos.com/pretrained/ResNet18_vd_pretrained.pdparams  --no-check-certificate
    fi
    if [ ${model_name} == "en_server_pgnetA" ]; then
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/total_text_lite.tar --no-check-certificate
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_server_pgnetA.tar && cd ../
        cd ./train_data && tar xf total_text_lite.tar && ln -s total_text_lite total_text && cd ../
    fi
    if [ ${model_name} == "det_r50_vd_sast_icdar15_v2_0" ] || [ ${model_name} == "det_r50_vd_sast_totaltext_v2_0" ]; then
        wget -nc -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams --no-check-certificate
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/total_text_lite.tar --no-check-certificate
        cd ./train_data && tar xf total_text_lite.tar && ln -s total_text_lite total_text  && cd ../
        cd ./pretrain_models && tar xf det_r50_vd_sast_icdar15_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "det_mv3_db_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_mv3_db_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "det_r50_db_v2_0" ]; then
        wget -nc -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_db_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_ppocr_mobile_v2_0_rec_FPGM" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_ppocr_mobile_v2.0_rec_train.tar && cd ../
        ${python_name} -m pip install paddleslim
    fi
    if [ ${model_name} == "ch_ppocr_mobile_v2_0_det_FPGM" ]; then
        ${python_name} -m pip install paddleslim
    fi
    if [ ${model_name} == "det_r50_vd_pse_v2_0" ]; then
        wget -nc -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams --no-check-certificate
    fi
    if [ ${model_name} == "det_mv3_east_v2_0" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf det_mv3_east_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "det_r50_vd_east_v2_0" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf det_r50_vd_east_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "det_r50_dcn_fce_ctw_v2_0" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf det_r50_dcn_fce_ctw_v2.0_train.tar & cd ../
    fi
    if [ ${model_name} == "rec_r32_gaspin_bilstm_att" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/rec_r32_gaspin_bilstm_att_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf rec_r32_gaspin_bilstm_att_train.tar && cd ../
    fi
    if [[ ${model_name} =~ "layoutxlm_ser" ]]; then
        ${python_name} -m pip install -r ppstructure/kie/requirements.txt
        ${python_name} -m pip install opencv-python -U
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar --no-check-certificate
        cd ./train_data/ && tar xf XFUND.tar
        cd ../

        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ser_LayoutXLM_xfun_zh.tar  && cd ../
    fi
    if [[ ${model_name} =~ "vi_layoutxlm_ser" ]]; then
        ${python_name} -m pip install -r ppstructure/kie/requirements.txt
        ${python_name} -m pip install opencv-python -U
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar --no-check-certificate
        cd ./train_data/ && tar xf XFUND.tar
        cd ../
        if [ ${model_name} == "vi_layoutxlm_ser_PACT" ]; then
            wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_pretrained.tar --no-check-certificate
            cd ./pretrain_models/ && tar xf ser_vi_layoutxlm_xfund_pretrained.tar  && cd ../
        fi
    fi
    if [ ${model_name} == "det_r18_ct" ]; then
        wget -nc -P ./pretrain_models/  https://paddleocr.bj.bcebos.com/pretrained/ResNet18_vd_pretrained.pdparams  --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/ct_tipc/total_text_lite2.tar --no-check-certificate
        cd ./train_data && tar xf total_text_lite2.tar && ln -s total_text_lite2 total_text && cd ../
    fi
    if [ ${model_name} == "sr_telescope" ]; then
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/TextZoom.tar --no-check-certificate
        cd ./train_data/ && tar xf TextZoom.tar && cd ../
    fi
    if [ ${model_name} == "rec_d28_can" ]; then
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/CROHME_lite.tar --no-check-certificate
        cd ./train_data/ && tar xf CROHME_lite.tar && cd ../
    fi

elif [ ${MODE} = "whole_train_whole_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams --no-check-certificate
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    rm -rf ./train_data/pubtabnet
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/pubtabnet.tar --no-check-certificate
    cd ./train_data/ && tar xf icdar2015.tar && tar xf ic15_data.tar && tar xf pubtabnet.tar 
    wget -nc -P ./ic15_data/ https://paddleocr.bj.bcebos.com/dataset/rec_gt_train_lite.txt --no-check-certificate
    wget -nc -P ./ic15_data/ https://paddleocr.bj.bcebos.com/dataset/rec_gt_test_lite.txt --no-check-certificate
    cd ../
    if [ ${model_name} == "ch_PP-OCRv2_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_PP-OCRv3_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv3_det_distill_train.tar && cd ../
    fi
    if [ ${model_name} == "en_server_pgnetA" ]; then
        wget -nc -P ./train_data/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/total_text_lite.tar  --no-check-certificate
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_server_pgnetA.tar && cd ../
        cd ./train_data && tar xf total_text.tar && ln -s total_text_lite total_text  && cd ../
    fi
    if [ ${model_name} == "det_r50_vd_sast_totaltext_v2_0" ]; then
        wget -nc -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/total_text_lite.tar  --no-check-certificate
        cd ./train_data && tar xf total_text.tar && ln -s total_text_lite total_text  && cd ../
    fi
    if [[ ${model_name} =~ "en_table_structure" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_ppocr_mobile_v2.0_table_structure_train.tar  && cd ../
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar --no-check-certificate
        cd ./inference/ && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar && cd ../
    fi
elif [ ${MODE} = "lite_train_whole_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams --no-check-certificate
    rm -rf ./train_data/icdar2015
    rm -rf ./train_data/ic15_data
    rm -rf ./train_data/pubtabnet
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_infer.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/pubtabnet.tar --no-check-certificate
    cd ./train_data/ && tar xf icdar2015_infer.tar && tar xf ic15_data.tar && tar xf pubtabnet.tar
    ln -s ./icdar2015_infer ./icdar2015
    wget -nc -P ./ic15_data/ https://paddleocr.bj.bcebos.com/dataset/rec_gt_train_lite.txt --no-check-certificate
    wget -nc -P ./ic15_data/ https://paddleocr.bj.bcebos.com/dataset/rec_gt_test_lite.txt --no-check-certificate
    cd ../
    if [ ${model_name} == "ch_PP-OCRv2_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_PP-OCRv3_det" ]; then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf ch_PP-OCRv3_det_distill_train.tar && cd ../
    fi
    if [[ ${model_name} =~ "en_table_structure" ]];then
        wget -nc -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar --no-check-certificate
        cd ./pretrain_models/ && tar xf en_ppocr_mobile_v2.0_table_structure_train.tar  && cd ../
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar --no-check-certificate
        cd ./inference/ && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar && cd ../
    fi
elif [ ${MODE} = "whole_infer" ];then
    python_name_list=$(func_parser_value "${lines[2]}")
    array=(${python_name_list}) 
    python_name=${array[0]}
    ${python_name} -m pip install paddleslim
    ${python_name} -m pip install -r requirements.txt
    wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar --no-check-certificate
    wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar --no-check-certificate
    cd ./inference && tar xf rec_inference.tar  && tar xf ch_det_data_50.tar && cd ../
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar --no-check-certificate
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/pubtabnet.tar --no-check-certificate
    cd ./train_data/ && tar xf XFUND.tar && tar xf pubtabnet.tar && cd ../
    head -n 2 train_data/XFUND/zh_val/val.json > train_data/XFUND/zh_val/val_lite.json
    mv train_data/XFUND/zh_val/val_lite.json train_data/XFUND/zh_val/val.json
    if [ ${model_name} = "ch_ppocr_mobile_v2_0_det" ]; then
        eval_model_name="ch_ppocr_mobile_v2.0_det_train"
        rm -rf ./train_data/icdar2015
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2_0_det_PACT" ]; then
        eval_model_name="ch_ppocr_mobile_v2.0_det_prune_infer"
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_server_v2_0_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_train.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ch_ppocr_mobile_v2_0" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ch_ppocr_server_v2_0" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2_0_rec_PACT" ]; then
        eval_model_name="ch_ppocr_mobile_v2.0_rec_slim_infer"
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2_0_rec_FPGM" ]; then
        eval_model_name="ch_PP-OCRv2_rec_infer"
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && cd ../
    fi 
    if [[ ${model_name} =~ "ch_PP-OCRv2" ]]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && tar xf ch_PP-OCRv2_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv3" ]]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv2_det" ]]; then
        eval_model_name="ch_PP-OCRv2_det_infer"
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv3_det" ]]; then
        eval_model_name="ch_PP-OCRv3_det_infer"
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [[ ${model_name} =~ "ch_PP-OCRv2_rec" ]]; then
        eval_model_name="ch_PP-OCRv2_rec_infer"
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_PP-OCRv2_rec_slim_quant_infer.tar && cd ../
    fi   
    if [[ ${model_name} =~ "ch_PP-OCRv3_rec" ]]; then
        eval_model_name="ch_PP-OCRv3_rec_infer"
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar --no-check-certificate
        cd ./inference && tar xf ${eval_model_name}.tar && tar xf ch_PP-OCRv3_rec_slim_infer.tar && cd ../
    fi
    if [[ ${model_name} == "ch_PP-OCRv3_rec_PACT" ]]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_rec_slim_infer.tar && cd ../
    fi  
    if [ ${model_name} == "en_server_pgnetA" ]; then
        wget -nc -P ./inference/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar  --no-check-certificate
        cd ./inference && tar xf en_server_pgnetA.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [ ${model_name} == "det_r50_vd_sast_icdar15_v2_0" ]; then
        wget -nc -P  ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_sast_icdar15_v2.0_train.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [ ${model_name} == "rec_mv3_none_none_ctc_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mv3_none_none_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r34_vd_none_none_ctc_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r34_vd_none_none_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_mv3_none_bilstm_ctc_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mv3_none_bilstm_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r34_vd_none_bilstm_ctc_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r34_vd_none_bilstm_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_mv3_tps_bilstm_ctc_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mv3_tps_bilstm_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r34_vd_tps_bilstm_ctc_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_ppocr_server_v2_0_rec" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar --no-check-certificate
        cd ./inference/ && tar xf ch_ppocr_server_v2.0_rec_train.tar && cd ../
    fi
    if [ ${model_name} == "ch_ppocr_mobile_v2_0_rec" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar --no-check-certificate
        cd ./inference/ && tar xf ch_ppocr_mobile_v2.0_rec_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_mtb_nrtr" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mtb_nrtr_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_mv3_tps_bilstm_att_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf rec_mv3_tps_bilstm_att_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "rec_r34_vd_tps_bilstm_att_v2_0" ]; then
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
    
    if [ ${model_name} == "det_r50_vd_sast_totaltext_v2_0" ]; then
        wget -nc -P  ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_sast_totaltext_v2.0_train.tar && cd ../
    fi
    if [ ${model_name} == "det_mv3_db_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_mv3_db_v2.0_train.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [ ${model_name} == "det_r50_db_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_db_v2.0_train.tar && tar xf ch_det_data_50.tar && cd ../
    fi
    if [ ${model_name} == "det_mv3_pse_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_mv3_pse_v2.0_train.tar & cd ../
    fi
    if [ ${model_name} == "det_r50_vd_pse_v2_0" ]; then
        wget -nc -P ./inference/  https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar  --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_pse_v2.0_train.tar & cd ../
    fi
    if [ ${model_name} == "det_mv3_east_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_mv3_east_v2.0_train.tar & cd ../
    fi
    if [ ${model_name} == "det_r50_vd_east_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_r50_vd_east_v2.0_train.tar & cd ../
    fi
    if [ ${model_name} == "det_r50_dcn_fce_ctw_v2_0" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar --no-check-certificate
        cd ./inference/ && tar xf det_r50_dcn_fce_ctw_v2.0_train.tar & cd ../
    fi
    if [[ ${model_name} =~ "en_table_structure" ]];then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar --no-check-certificate

        cd ./inference/ && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar
        if [ ${model_name} == "en_table_structure" ]; then
            wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar --no-check-certificate
            tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar
        elif [ ${model_name} == "en_table_structure_PACT" ]; then
            wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_slim_infer.tar --no-check-certificate
            tar xf en_ppocr_mobile_v2.0_table_structure_slim_infer.tar
        fi
        cd ../
    fi
    if [[ ${model_name} =~ "slanet" ]];then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar --no-check-certificate
        cd ./inference/ && tar xf en_ppstructure_mobile_v2.0_SLANet_infer.tar && tar xf ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar && cd ../
    fi
    if [[ ${model_name} =~ "vi_layoutxlm_ser" ]]; then
        ${python_name} -m pip install -r ppstructure/kie/requirements.txt
        ${python_name} -m pip install opencv-python -U 
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_infer.tar --no-check-certificate
        cd ./inference/ && tar xf ser_vi_layoutxlm_xfund_infer.tar & cd ../
    fi
    if [[ ${model_name} =~ "layoutxlm_ser" ]]; then
        ${python_name} -m pip install -r ppstructure/kie/requirements.txt
        ${python_name} -m pip install opencv-python -U 
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh_infer.tar --no-check-certificate
        cd ./inference/ && tar xf ser_LayoutXLM_xfun_zh_infer.tar & cd ../
    fi
fi

if [[ ${model_name} =~ "KL" ]]; then
    wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar --no-check-certificate
    cd ./train_data/ && tar xf icdar2015_lite.tar && rm -rf ./icdar2015 && ln -s ./icdar2015_lite ./icdar2015 && cd ../
    if [ ${model_name} = "ch_ppocr_mobile_v2_0_det_KL" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && cd ../ 
    fi
    if [ ${model_name} = "ch_PP-OCRv2_rec_KL" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar  --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data.tar && cd ../
        cd ./inference && tar xf rec_inference.tar && tar xf ch_PP-OCRv2_rec_infer.tar && cd ../
    fi
    if [ ${model_name} = "ch_PP-OCRv3_rec_KL" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar  --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data.tar 
        wget -nc -P ./ic15_data/ https://paddleocr.bj.bcebos.com/dataset/rec_gt_train_lite.txt --no-check-certificate
        wget -nc -P ./ic15_data/ https://paddleocr.bj.bcebos.com/dataset/rec_gt_test_lite.txt --no-check-certificate
        cd ../
        cd ./inference && tar xf rec_inference.tar && tar xf ch_PP-OCRv3_rec_infer.tar && cd ../
    fi
    if [ ${model_name} = "ch_PP-OCRv2_det_KL" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    fi 
    if [ ${model_name} = "ch_PP-OCRv3_det_KL" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    fi 
    if [ ${model_name} = "ch_ppocr_mobile_v2_0_rec_KL" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ic15_data.tar --no-check-certificate
        cd ./train_data/ && tar xf ic15_data.tar && cd ../
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf rec_inference.tar &&  cd ../ 
    fi
    if [ ${model_name} = "en_table_structure_KL" ];then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar --no-check-certificate
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/pubtabnet.tar --no-check-certificate
        cd ./inference/ && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar && cd ../
        cd ./train_data/ && tar xf pubtabnet.tar && cd ../
    fi
    if [[ ${model_name} =~ "layoutxlm_ser_KL" ]]; then
        wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar --no-check-certificate
        cd ./train_data/ && tar xf XFUND.tar && cd ../
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh_infer.tar --no-check-certificate
        cd ./inference/ && tar xf ser_LayoutXLM_xfun_zh_infer.tar & cd ../
    fi
fi

if [ ${MODE} = "cpp_infer" ];then
    if [ ${model_name} = "ch_ppocr_mobile_v2_0_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2_0_det_KL" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_ppocr_mobile_v2.0_det_klquant_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_klquant_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2_0_det_PACT" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_ppocr_mobile_v2.0_det_pact_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_pact_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2_0_rec" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2_0_rec_KL" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_ppocr_mobile_v2.0_rec_klquant_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_rec_klquant_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_mobile_v2_0_rec_PACT" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_ppocr_mobile_v2.0_rec_pact_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_rec_pact_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_server_v2_0_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_ppocr_server_v2_0_rec" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv2_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv2_det_KL" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv2_det_klquant_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_klquant_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv2_det_PACT" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv2_det_pact_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_pact_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv2_rec" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_rec_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv2_rec_KL" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv2_rec_klquant_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_rec_klquant_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv2_rec_PACT" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv2_rec_pact_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_rec_pact_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv3_det" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv3_det_KL" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv3_det_klquant_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_klquant_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv3_det_PACT" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv3_det_pact_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_pact_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv3_rec" ]; then
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_rec_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv3_rec_KL" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv3_rec_klquant_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_rec_klquant_infer.tar && tar xf rec_inference.tar && cd ../
    elif [ ${model_name} = "ch_PP-OCRv3_rec_PACT" ]; then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv3_rec_pact_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_rec_pact_infer.tar && tar xf rec_inference.tar && cd ../
    elif  [ ${model_name} = "ch_ppocr_mobile_v2_0" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ch_ppocr_server_v2_0" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ch_PP-OCRv2" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && tar xf ch_PP-OCRv2_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif  [ ${model_name} = "ch_PP-OCRv3" ]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar  --no-check-certificate
        wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar  --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar && tar xf ch_det_data_50.tar && cd ../
    elif [[ ${model_name} =~ "en_table_structure" ]];then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar --no-check-certificate

        cd ./inference/ && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar
        if [ ${model_name} == "en_table_structure" ]; then
            wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar --no-check-certificate
            tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar
        elif [ ${model_name} == "en_table_structure_PACT" ]; then
            wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_slim_infer.tar --no-check-certificate
            tar xf en_ppocr_mobile_v2.0_table_structure_slim_infer.tar
        fi
        cd ../
    elif [[ ${model_name} =~ "slanet" ]];then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar --no-check-certificate
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar --no-check-certificate
        cd ./inference/ && tar xf ch_ppstructure_mobile_v2.0_SLANet_infer.tar && tar xf ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar && cd ../
    fi
fi

if [ ${MODE} = "serving_infer" ];then
    # prepare serving env
    python_name_list=$(func_parser_value "${lines[2]}")
    IFS='|'
    array=(${python_name_list})
    python_name=${array[0]}
    ${python_name} -m pip install paddle-serving-server-gpu
    ${python_name} -m pip install paddle_serving_client
    ${python_name} -m pip install paddle-serving-app
    ${python_name} -m pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    # wget model
    if [ ${model_name} == "ch_ppocr_mobile_v2_0_det_KL" ] || [ ${model_name} == "ch_ppocr_mobile_v2.0_rec_KL" ] ; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_ppocr_mobile_v2.0_det_klquant_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_ppocr_mobile_v2.0_rec_klquant_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_klquant_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_klquant_infer.tar && cd ../
    elif [ ${model_name} == "ch_PP-OCRv2_det_KL" ] || [ ${model_name} == "ch_PP-OCRv2_rec_KL" ] ; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv2_det_klquant_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv2_rec_klquant_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_klquant_infer.tar && tar xf ch_PP-OCRv2_rec_klquant_infer.tar && cd ../
    elif [ ${model_name} == "ch_PP-OCRv3_det_KL" ] || [ ${model_name} == "ch_PP-OCRv3_rec_KL" ] ; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv3_det_klquant_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv3_rec_klquant_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_klquant_infer.tar && tar xf ch_PP-OCRv3_rec_klquant_infer.tar && cd ../
    elif [ ${model_name} == "ch_ppocr_mobile_v2_0_det_PACT" ] || [ ${model_name} == "ch_ppocr_mobile_v2.0_rec_PACT" ] ; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_ppocr_mobile_v2.0_det_pact_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_ppocr_mobile_v2.0_rec_pact_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_pact_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_pact_infer.tar && cd ../
    elif [ ${model_name} == "ch_PP-OCRv2_det_PACT" ] || [ ${model_name} == "ch_PP-OCRv2_rec_PACT" ] ; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv2_det_pact_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv2_rec_pact_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_pact_infer.tar && tar xf ch_PP-OCRv2_rec_pact_infer.tar && cd ../
    elif [ ${model_name} == "ch_PP-OCRv3_det_PACT" ] || [ ${model_name} == "ch_PP-OCRv3_rec_PACT" ] ; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv3_det_pact_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/tipc_fake_model/ch_PP-OCRv3_rec_pact_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_pact_infer.tar && tar xf ch_PP-OCRv3_rec_pact_infer.tar && cd ../
    elif [[ ${model_name} =~ "ch_ppocr_mobile_v2_0" ]]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && cd ../
    elif [[ ${model_name} =~ "ch_ppocr_server_v2_0" ]]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && cd ../
    elif [[ ${model_name} =~ "ch_PP-OCRv2" ]]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && tar xf ch_PP-OCRv2_rec_infer.tar && cd ../
    elif [[ ${model_name} =~ "ch_PP-OCRv3" ]]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar && cd ../
    fi
    # wget data
    wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar  --no-check-certificate
    wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar  --no-check-certificate
    cd ./inference && tar xf ch_det_data_50.tar && tar xf rec_inference.tar && cd ../
fi

if [ ${MODE} = "paddle2onnx_infer" ];then
    # prepare serving env
    python_name=$(func_parser_value "${lines[2]}")
    ${python_name} -m pip install paddle2onnx onnxruntime onnx
    # wget model
    if [[ ${model_name} =~ "ch_ppocr_mobile_v2_0" ]]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && cd ../
    elif [[ ${model_name} =~ "ch_ppocr_server_v2_0" ]]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && cd ../
    elif [[ ${model_name} =~ "ch_PP-OCRv2" ]]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && tar xf ch_PP-OCRv2_rec_infer.tar && cd ../
    elif [[ ${model_name} =~ "ch_PP-OCRv3" ]]; then
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar --no-check-certificate
        wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar --no-check-certificate
        cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar && cd ../
    elif [[ ${model_name} =~ "slanet" ]];then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar --no-check-certificate
        cd ./inference/ && tar xf ch_ppstructure_mobile_v2.0_SLANet_infer.tar && cd ../
    elif [[ ${model_name} =~ "en_table_structure" ]];then
        wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar --no-check-certificate
        cd ./inference/ && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar && cd ../
    fi
    
    # wget data
    wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar
    wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar
    cd ./inference && tar xf ch_det_data_50.tar && tar xf rec_inference.tar && cd ../
    
fi
