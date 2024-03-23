PROJECT(infer_demo CSharp)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# Set the C# language version (defaults to 3.0 if not set).
set(CMAKE_CSharp_FLAGS "/langversion:10")
set(CMAKE_DOTNET_TARGET_FRAMEWORK "net6.0")
set(CMAKE_DOTNET_SDK "Microsoft.NET.Sdk")

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeployCSharp.cmake)


add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cs)

set_property(TARGET infer_demo PROPERTY VS_DOTNET_REFERENCES
    ${FASTDEPLOY_DOTNET_REFERENCES}
)

set_property(TARGET infer_demo
    PROPERTY VS_PACKAGE_REFERENCES  ${FASTDEPLOY_PACKAGE_REFERENCES})
