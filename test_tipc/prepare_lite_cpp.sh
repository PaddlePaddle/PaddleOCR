#!/bin/bash
source ./test_tipc/common_func.sh
FILENAME=$1
dataline=$(cat ${FILENAME})
# parser params
IFS=$'\n'
lines=(${dataline})
IFS=$'\n'
paddlelite_library_source=$2

inference_cmd=$(func_parser_value "${lines[1]}")
DEVICE=$(func_parser_value "${lines[2]}")
det_lite_model_list=$(func_parser_value "${lines[3]}")
rec_lite_model_list=$(func_parser_value "${lines[4]}")
cls_lite_model_list=$(func_parser_value "${lines[5]}")

if [[ $inference_cmd =~ "det" ]]; then
    lite_model_list=${det_lite_model_list}
elif [[ $inference_cmd =~ "rec" ]]; then
    lite_model_list=(${rec_lite_model_list[*]} ${cls_lite_model_list[*]})
elif [[ $inference_cmd =~ "system" ]]; then
    lite_model_list=(${det_lite_model_list[*]} ${rec_lite_model_list[*]} ${cls_lite_model_list[*]})
else
    echo "inference_cmd is wrong, please check."
    exit 1
fi

if [ ${DEVICE} = "ARM_CPU" ]; then
    valid_targets="arm"
    paddlelite_library_url="https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10-rc/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.with_cv.tar.gz"
    end_index="66"
    compile_with_opencl="OFF"
elif [ ${DEVICE} = "ARM_GPU_OPENCL" ]; then
    valid_targets="opencl"
    paddlelite_library_url="https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10-rc/inference_lite_lib.armv8.clang.with_exception.with_extra.with_cv.opencl.tar.gz"
    end_index="71"
    compile_with_opencl="ON"
else
    echo "DEVICE only support ARM_CPU, ARM_GPU_OPENCL."
    exit 2    
fi

# prepare paddlelite model
pip install paddlelite==2.10-rc
current_dir=${PWD}
IFS="|"
model_path=./inference_models

for model in ${lite_model_list[*]}; do
    if [[ $model =~ "PP-OCRv2" ]]; then
        inference_model_url=https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/${model}.tar
    elif [[ $model =~ "v2.0" ]]; then
        inference_model_url=https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/${model}.tar
    elif [[ $model =~ "PP-OCRv3" ]]; then
        inference_model_url=https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/${model}.tar
    else 
        echo "Model is wrong, please check."
        exit 3
    fi
    inference_model=${inference_model_url##*/}
    wget -nc  -P ${model_path} ${inference_model_url}
    cd ${model_path} && tar -xf ${inference_model} && cd ../
    model_dir=${model_path}/${inference_model%.*}
    model_file=${model_dir}/inference.pdmodel
    param_file=${model_dir}/inference.pdiparams
    paddle_lite_opt --model_dir=${model_dir} --model_file=${model_file} --param_file=${param_file} --valid_targets=${valid_targets} --optimize_out=${model_dir}_opt
done

# prepare test data
data_url=https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015_lite.tar
data_file=${data_url##*/}
wget -nc  -P ./test_data ${data_url}
cd ./test_data && tar -xf ${data_file} && rm ${data_file} && cd ../

# prepare paddlelite predict library
if [[ ${paddlelite_library_source} = "download" ]]; then
    paddlelite_library_zipfile=$(echo $paddlelite_library_url | awk -F "/" '{print $NF}')
    paddlelite_library_file=${paddlelite_library_zipfile:0:${end_index}}
    wget ${paddlelite_library_url} && tar -xf ${paddlelite_library_zipfile}
    cd ${paddlelite_library_zipfile}
elif [[ ${paddlelite_library_source} = "compile" ]]; then
    git clone -b release/v2.10 https://github.com/PaddlePaddle/Paddle-Lite.git
    cd Paddle-Lite
    ./lite/tools/build_android.sh  --arch=armv8  --with_cv=ON --with_extra=ON --toolchain=clang --with_opencl=${compile_with_opencl}
    cd ../
    cp -r Paddle-Lite/build.lite.android.armv8.clang/inference_lite_lib.android.armv8/ .
    paddlelite_library_file=inference_lite_lib.android.armv8
else
    echo "paddlelite_library_source only support 'download' and 'compile'"
    exit 3
fi

# organize the required files  
mkdir -p  ${paddlelite_library_file}/demo/cxx/ocr/test_lite
cp -r ${model_path}/*_opt.nb test_data ${paddlelite_library_file}/demo/cxx/ocr/test_lite
cp ppocr/utils/ppocr_keys_v1.txt deploy/lite/config.txt ${paddlelite_library_file}/demo/cxx/ocr/test_lite
cp -r ./deploy/lite/* ${paddlelite_library_file}/demo/cxx/ocr/
cp ${paddlelite_library_file}/cxx/lib/libpaddle_light_api_shared.so ${paddlelite_library_file}/demo/cxx/ocr/test_lite
cp ${FILENAME} test_tipc/test_lite_arm_cpp.sh test_tipc/common_func.sh ${paddlelite_library_file}/demo/cxx/ocr/test_lite
cd ${paddlelite_library_file}/demo/cxx/ocr/
git clone https://github.com/cuicheng01/AutoLog.git

# compile and do some postprocess
make -j
sleep 1
make -j
cp ocr_db_crnn test_lite && cp test_lite/libpaddle_light_api_shared.so test_lite/libc++_shared.so
tar -cf test_lite.tar ./test_lite && cp test_lite.tar ${current_dir} && cd ${current_dir}
rm -rf ${paddlelite_library_file}* && rm -rf ${model_path}
