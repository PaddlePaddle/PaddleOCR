package paddle

// #include <stdbool.h>
// #include "paddle_c_api.h"
import "C"

import (
	"reflect"
	"runtime"
	"unsafe"
)

type Predictor struct {
	c *C.PD_Predictor
}

func NewPredictor(config *AnalysisConfig) *Predictor {
	c_predictor := C.PD_NewPredictor((*config).c)
	predictor := &Predictor{c: c_predictor}
	runtime.SetFinalizer(predictor, (*Predictor).finalize)
	return predictor
}

func (predictor *Predictor) finalize() {
	C.PD_DeletePredictor(predictor.c)
}

func DeletePredictor(predictor *Predictor) {
	C.PD_DeletePredictor(predictor.c)
}

func (predictor *Predictor) GetInputNum() int {
	return int(C.PD_GetInputNum(predictor.c))
}

func (predictor *Predictor) GetOutputNum() int {
	return int(C.PD_GetOutputNum(predictor.c))
}

func (predictor *Predictor) GetInputName(n int) string {
	return C.GoString(C.PD_GetInputName(predictor.c, C.int(n)))
}

func (predictor *Predictor) GetOutputName(n int) string {
	return C.GoString(C.PD_GetOutputName(predictor.c, C.int(n)))
}

func (predictor *Predictor) GetInputTensors() [](*ZeroCopyTensor) {
	var result [](*ZeroCopyTensor)
	for i := 0; i < predictor.GetInputNum(); i++ {
		tensor := NewZeroCopyTensor()
		tensor.c.name = C.PD_GetInputName(predictor.c, C.int(i))
		result = append(result, tensor)
	}
	return result
}

func (predictor *Predictor) GetOutputTensors() [](*ZeroCopyTensor) {
	var result [](*ZeroCopyTensor)
	for i := 0; i < predictor.GetOutputNum(); i++ {
		tensor := NewZeroCopyTensor()
		tensor.c.name = C.PD_GetOutputName(predictor.c, C.int(i))
		result = append(result, tensor)
	}
	return result
}

func (predictor *Predictor) GetInputNames() []string {
	names := make([]string, predictor.GetInputNum())
	for i := 0; i < len(names); i++ {
		names[i] = predictor.GetInputName(i)
	}
	return names
}

func (predictor *Predictor) GetOutputNames() []string {
	names := make([]string, predictor.GetOutputNum())
	for i := 0; i < len(names); i++ {
		names[i] = predictor.GetOutputName(i)
	}
	return names
}

func (predictor *Predictor) SetZeroCopyInput(tensor *ZeroCopyTensor) {
	C.PD_SetZeroCopyInput(predictor.c, tensor.c)
}

func (predictor *Predictor) GetZeroCopyOutput(tensor *ZeroCopyTensor) {
	C.PD_GetZeroCopyOutput(predictor.c, tensor.c)
	tensor.name = C.GoString(tensor.c.name)
	var shape []int32
	shape_hdr := (*reflect.SliceHeader)(unsafe.Pointer(&shape))
	shape_hdr.Data = uintptr(unsafe.Pointer(tensor.c.shape.data))
	shape_hdr.Len = int(tensor.c.shape.length / C.sizeof_int)
	shape_hdr.Cap = int(tensor.c.shape.length / C.sizeof_int)
	tensor.Reshape(shape)
}

func (predictor *Predictor) ZeroCopyRun() {
	C.PD_ZeroCopyRun(predictor.c)
}
