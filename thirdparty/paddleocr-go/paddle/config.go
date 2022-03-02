package paddle

// #include <stdbool.h>
// #include <stdlib.h>
// #include <paddle_c_api.h>
import "C"

import (
	"runtime"
	"unsafe"
)

type Precision C.Precision

const (
	Precision_FLOAT32 Precision = C.kFloat32
	Precision_INT8    Precision = C.kInt8
	Precision_HALF    Precision = C.kHalf
)

type AnalysisConfig struct {
	c *C.PD_AnalysisConfig
}

func NewAnalysisConfig() *AnalysisConfig {
	c_config := C.PD_NewAnalysisConfig()
	config := &AnalysisConfig{c: c_config}
	runtime.SetFinalizer(config, (*AnalysisConfig).finalize)
	return config
}

func (config *AnalysisConfig) finalize() {
	C.PD_DeleteAnalysisConfig(config.c)
}

func (config *AnalysisConfig) SetModel(model, params string) {
	c_model := C.CString(model)
	defer C.free(unsafe.Pointer(c_model))
	var c_params *C.char
	if params == "" {
		c_params = nil
	} else {
		c_params = C.CString(params)
		defer C.free(unsafe.Pointer(c_params))
	}

	C.PD_SetModel(config.c, c_model, c_params)
}

func (config *AnalysisConfig) ModelDir() string {
	return C.GoString(C.PD_ModelDir(config.c))
}

func (config *AnalysisConfig) ProgFile() string {
	return C.GoString(C.PD_ProgFile(config.c))
}

func (config *AnalysisConfig) ParamsFile() string {
	return C.GoString(C.PD_ParamsFile(config.c))
}

func (config *AnalysisConfig) EnableUseGpu(memory_pool_init_size_mb int, device_id int) {
	C.PD_EnableUseGpu(config.c, C.int(memory_pool_init_size_mb), C.int(device_id))
}

func (config *AnalysisConfig) DisableGpu() {
	C.PD_DisableGpu(config.c)
}

func (config *AnalysisConfig) UseGpu() bool {
	return ConvertCBooleanToGo(C.PD_UseGpu(config.c))
}

func (config *AnalysisConfig) GpuDeviceId() int {
	return int(C.PD_GpuDeviceId(config.c))
}

func (config *AnalysisConfig) MemoryPoolInitSizeMb() int {
	return int(C.PD_MemoryPoolInitSizeMb(config.c))
}

func (config *AnalysisConfig) EnableCudnn() {
	C.PD_EnableCUDNN(config.c)
}

func (config *AnalysisConfig) CudnnEnabled() bool {
	return ConvertCBooleanToGo(C.PD_CudnnEnabled(config.c))
}

func (config *AnalysisConfig) SwitchIrOptim(x bool) {
	C.PD_SwitchIrOptim(config.c, C.bool(x))
}

func (config *AnalysisConfig) IrOptim() bool {
	return ConvertCBooleanToGo(C.PD_IrOptim(config.c))
}

func (config *AnalysisConfig) SwitchUseFeedFetchOps(x bool) {
	C.PD_SwitchUseFeedFetchOps(config.c, C.bool(x))
}

func (config *AnalysisConfig) UseFeedFetchOpsEnabled() bool {
	return ConvertCBooleanToGo(C.PD_UseFeedFetchOpsEnabled(config.c))
}

func (config *AnalysisConfig) SwitchSpecifyInputNames(x bool) {
	C.PD_SwitchSpecifyInputNames(config.c, C.bool(x))
}

func (config *AnalysisConfig) SpecifyInputName() bool {
	return ConvertCBooleanToGo(C.PD_SpecifyInputName(config.c))
}

func (config *AnalysisConfig) EnableTensorRtEngine(workspace_size int, max_batch_size int, min_subgraph_size int, precision Precision, use_static bool, use_calib_mode bool) {
	C.PD_EnableTensorRtEngine(config.c, C.int(workspace_size), C.int(max_batch_size), C.int(min_subgraph_size), C.Precision(precision), C.bool(use_static), C.bool(use_calib_mode))
}

func (config *AnalysisConfig) TensorrtEngineEnabled() bool {
	return ConvertCBooleanToGo(C.PD_TensorrtEngineEnabled(config.c))
}

func (config *AnalysisConfig) SwitchIrDebug(x bool) {
	C.PD_SwitchIrDebug(config.c, C.bool(x))
}

func (config *AnalysisConfig) EnableMkldnn() {
	C.PD_EnableMKLDNN(config.c)
}

func (config *AnalysisConfig) SetCpuMathLibraryNumThreads(n int) {
	C.PD_SetCpuMathLibraryNumThreads(config.c, C.int(n))
}

func (config *AnalysisConfig) CpuMathLibraryNumThreads() int {
	return int(C.PD_CpuMathLibraryNumThreads(config.c))
}

func (config *AnalysisConfig) EnableMkldnnQuantizer() {
	C.PD_EnableMkldnnQuantizer(config.c)
}

func (config *AnalysisConfig) MkldnnQuantizerEnabled() bool {
	return ConvertCBooleanToGo(C.PD_MkldnnQuantizerEnabled(config.c))
}

// SetModelBuffer
// ModelFromMemory

func (config *AnalysisConfig) EnableMemoryOptim() {
	C.PD_EnableMemoryOptim(config.c)
}

func (config *AnalysisConfig) MemoryOptimEnabled() bool {
	return ConvertCBooleanToGo(C.PD_MemoryOptimEnabled(config.c))
}

func (config *AnalysisConfig) EnableProfile() {
	C.PD_EnableProfile(config.c)
}

func (config *AnalysisConfig) ProfileEnabled() bool {
	return ConvertCBooleanToGo(C.PD_ProfileEnabled(config.c))
}

func (config *AnalysisConfig) DisableGlogInfo() {
	C.PD_DisableGlogInfo(config.c)
}

func (config *AnalysisConfig) DeletePass(pass string) {
	c_pass := C.CString(pass)
	defer C.free(unsafe.Pointer(c_pass))
	C.PD_DeletePass(config.c, c_pass)
}

func (config *AnalysisConfig) SetInValid() {
	C.PD_SetInValid(config.c)
}

func (config *AnalysisConfig) IsValid() bool {
	return ConvertCBooleanToGo(C.PD_IsValid(config.c))
}
