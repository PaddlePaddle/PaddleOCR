PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

# PP-OCR
add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc)
# 添加FastDeploy库依赖
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})

# Only Det
add_executable(infer_det ${PROJECT_SOURCE_DIR}/infer_det.cc)
# 添加FastDeploy库依赖
target_link_libraries(infer_det ${FASTDEPLOY_LIBS})

# Only Cls
add_executable(infer_cls ${PROJECT_SOURCE_DIR}/infer_cls.cc)
# 添加FastDeploy库依赖
target_link_libraries(infer_cls ${FASTDEPLOY_LIBS})

# Only Rec
add_executable(infer_rec ${PROJECT_SOURCE_DIR}/infer_rec.cc)
# 添加FastDeploy库依赖
target_link_libraries(infer_rec ${FASTDEPLOY_LIBS})
