ARM_ABI = arm8
export ARM_ABI

include ../Makefile.def

LITE_ROOT=../../../

THIRD_PARTY_DIR=${LITE_ROOT}/third_party

OPENCV_VERSION=opencv4.1.0

OPENCV_LIBS = ../../../third_party/${OPENCV_VERSION}/arm64-v8a/libs/libopencv_imgcodecs.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/libs/libopencv_imgproc.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/libs/libopencv_core.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/libtegra_hal.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/liblibjpeg-turbo.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/liblibwebp.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/liblibpng.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/liblibjasper.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/liblibtiff.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/libIlmImf.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/libtbb.a \
              ../../../third_party/${OPENCV_VERSION}/arm64-v8a/3rdparty/libs/libcpufeatures.a

OPENCV_INCLUDE = -I../../../third_party/${OPENCV_VERSION}/arm64-v8a/include

CXX_INCLUDES = $(INCLUDES) ${OPENCV_INCLUDE} -I$(LITE_ROOT)/cxx/include

CXX_LIBS = ${OPENCV_LIBS} -L$(LITE_ROOT)/cxx/lib/ -lpaddle_light_api_shared $(SYSTEM_LIBS)

###############################################################
# How to use one of static libaray:                           #
#  `libpaddle_api_full_bundled.a`                             #
#  `libpaddle_api_light_bundled.a`                            #
###############################################################
# Note: default use lite's shared library.                    #
###############################################################
# 1. Comment above line using `libpaddle_light_api_shared.so`
# 2. Undo comment below line using `libpaddle_api_light_bundled.a`

#CXX_LIBS = $(LITE_ROOT)/cxx/lib/libpaddle_api_light_bundled.a $(SYSTEM_LIBS)

ocr_db_crnn: fetch_opencv ocr_db_crnn.o
	$(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) ocr_db_crnn.o -o ocr_db_crnn  $(CXX_LIBS) $(LDFLAGS)

ocr_db_crnn.o: ocr_db_crnn.cc
	$(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o ocr_db_crnn.o -c ocr_db_crnn.cc

fetch_opencv:
	@ test -d ${THIRD_PARTY_DIR} ||  mkdir ${THIRD_PARTY_DIR}
	@ test -e ${THIRD_PARTY_DIR}/${OPENCV_VERSION}.tar.gz || \
      (echo "fetch opencv libs" && \
      wget -P ${THIRD_PARTY_DIR} https://paddle-inference-dist.bj.bcebos.com/${OPENCV_VERSION}.tar.gz)
	@ test -d ${THIRD_PARTY_DIR}/${OPENCV_VERSION} || \
      tar -zxvf ${THIRD_PARTY_DIR}/${OPENCV_VERSION}.tar.gz -C ${THIRD_PARTY_DIR}


.PHONY: clean
clean:
	rm -f ocr_db_crnn.o
	rm -f ocr_db_crnnn
