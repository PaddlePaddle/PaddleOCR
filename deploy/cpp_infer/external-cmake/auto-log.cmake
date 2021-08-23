find_package(Git REQUIRED)
message("${CMAKE_BUILD_TYPE}")

set(AUTOLOG_REPOSITORY     https://github.com/LDOUBLEV/AutoLog.git)
SET(AUTOLOG_INSTALL_DIR   ${CMAKE_CURRENT_BINARY_DIR}/install/Autolog)

ExternalProject_Add(
    extern_Autolog
    PREFIX autolog
    GIT_REPOSITORY ${AUTOLOG_REPOSITORY}
    GIT_TAG support_cpp_log
    DOWNLOAD_NO_EXTRACT True
    INSTALL_COMMAND cmake -E echo "Skipping install step."
)
