#!/bin/bash
source tests/common_func.sh

FILENAME=$1
dataline=$(awk 'NR==52, NR==66{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# parser cpp inference model 
use_opencv=$(func_parser_value "${lines[1]}")
cpp_infer_model_dir_list=$(func_parser_value "${lines[2]}")
cpp_infer_is_quant=$(func_parser_value "${lines[3]}")
# parser cpp inference 
inference_cmd=$(func_parser_value "${lines[4]}")
cpp_use_gpu_key=$(func_parser_key "${lines[5]}")
cpp_use_gpu_list=$(func_parser_value "${lines[5]}")
cpp_use_mkldnn_key=$(func_parser_key "${lines[6]}")
cpp_use_mkldnn_list=$(func_parser_value "${lines[6]}")
cpp_cpu_threads_key=$(func_parser_key "${lines[7]}")
cpp_cpu_threads_list=$(func_parser_value "${lines[7]}")
cpp_batch_size_key=$(func_parser_key "${lines[8]}")
cpp_batch_size_list=$(func_parser_value "${lines[8]}")
cpp_use_trt_key=$(func_parser_key "${lines[9]}")
cpp_use_trt_list=$(func_parser_value "${lines[9]}")
cpp_precision_key=$(func_parser_key "${lines[10]}")
cpp_precision_list=$(func_parser_value "${lines[10]}")
cpp_infer_model_key=$(func_parser_key "${lines[11]}")
cpp_image_dir_key=$(func_parser_key "${lines[12]}")
cpp_infer_img_dir=$(func_parser_value "${lines[12]}")
cpp_infer_key1=$(func_parser_key "${lines[13]}")
cpp_infer_value1=$(func_parser_value "${lines[13]}")
cpp_benchmark_key=$(func_parser_key "${lines[14]}")
cpp_benchmark_value=$(func_parser_value "${lines[14]}")
echo $use_opencv
echo $cpp_infer_img_dir

LOG_PATH="./tests/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_cpp.log"


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
                        set_infer_params1=$(func_set_params "${cpp_infer_key1}" "${cpp_infer_value1}")
                        command="${_script} ${cpp_use_gpu_key}=${use_gpu} ${cpp_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
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
                        set_infer_params1=$(func_set_params "${cpp_infer_key1}" "${cpp_infer_value1}")
                        command="${_script} ${cpp_use_gpu_key}=${use_gpu} ${set_tensorrt} ${set_precision} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
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


cd deploy/cpp_infer
if [ ${use_opencv} = "True" ]; then
    if [ -d "opencv-3.4.7/opencv3/" ] && [ $(md5sum opencv-3.4.7.tar.gz | awk -F ' ' '{print $1}') = "faa2b5950f8bee3f03118e600c74746a" ];then
        echo "################### build opencv skipped ###################"
    else
        echo "################### build opencv ###################"
        rm -rf opencv-3.4.7.tar.gz opencv-3.4.7/
        wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/opencv-3.4.7.tar.gz
        tar -xf opencv-3.4.7.tar.gz

        cd opencv-3.4.7/
        install_path=$(pwd)/opencv3

        rm -rf build
        mkdir build
        cd build

        cmake .. \
            -DCMAKE_INSTALL_PREFIX=${install_path} \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=OFF \
            -DWITH_IPP=OFF \
            -DBUILD_IPP_IW=OFF \
            -DWITH_LAPACK=OFF \
            -DWITH_EIGEN=OFF \
            -DCMAKE_INSTALL_LIBDIR=lib64 \
            -DWITH_ZLIB=ON \
            -DBUILD_ZLIB=ON \
            -DWITH_JPEG=ON \
            -DBUILD_JPEG=ON \
            -DWITH_PNG=ON \
            -DBUILD_PNG=ON \
            -DWITH_TIFF=ON \
            -DBUILD_TIFF=ON

        make -j
        make install
        cd ../
        echo "################### build opencv finished ###################"
    fi
fi


echo "################### build PaddleOCR demo ####################"
if [ ${use_opencv} = "True" ]; then
    OPENCV_DIR=$(pwd)/opencv-3.4.7/opencv3/
else
    OPENCV_DIR=''
fi
LIB_DIR=$(pwd)/Paddle/build/paddle_inference_install_dir/
CUDA_LIB_DIR=$(dirname `find /usr -name libcudart.so`)
CUDNN_LIB_DIR=$(dirname `find /usr -name libcudnn.so`)

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \

make -j
cd ../../../
echo "################### build PaddleOCR demo finished ###################"


# set cuda device
GPUID=$2
if [ ${#GPUID} -le 0 ];then
    env=" "
else
    env="export CUDA_VISIBLE_DEVICES=${GPUID}"
fi
set CUDA_VISIBLE_DEVICES
eval $env


echo "################### run test ###################"
export Count=0
IFS="|"
infer_quant_flag=(${cpp_infer_is_quant})
for infer_model in ${cpp_infer_model_dir_list[*]}; do
    #run inference
    is_quant=${infer_quant_flag[Count]}
    func_cpp_inference "${inference_cmd}" "${infer_model}" "${LOG_PATH}" "${cpp_infer_img_dir}" ${is_quant}
    Count=$(($Count + 1))
done
