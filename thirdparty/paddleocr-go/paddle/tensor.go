package paddle

// #include <stdbool.h>
// #include <stdlib.h>
// #include <string.h>
// #include <paddle_c_api.h>
import "C"

import (
	"reflect"
	"runtime"
	"unsafe"
)

type PaddleDType C.PD_DataType

const (
	FLOAT32  PaddleDType = C.PD_FLOAT32
	INT32    PaddleDType = C.PD_INT32
	INT64    PaddleDType = C.PD_INT64
	UINT8    PaddleDType = C.PD_UINT8
	UNKDTYPE PaddleDType = C.PD_UNKDTYPE
)

var types = []struct {
	gotype reflect.Type
	dtype  PaddleDType
}{
	{reflect.TypeOf(float32(0)), FLOAT32},
	{reflect.TypeOf(int32(0)), INT32},
	{reflect.TypeOf(int64(0)), INT64},
	{reflect.TypeOf(uint8(0)), UINT8},
}

func typeOfDataType(dtype PaddleDType) reflect.Type {
	var ret reflect.Type
	for _, t := range types {
		if t.dtype == dtype {
			ret = t.gotype
		}
	}
	return ret
}

func sizeofDataType(dtype PaddleDType) int32 {
	switch dtype {
	case UINT8:
		return int32(C.sizeof_uchar)
	case INT32:
		return int32(C.sizeof_int)
	case INT64:
		return int32(C.sizeof_longlong)
	case FLOAT32:
		return int32(C.sizeof_float)
	}
	return -1
}

func shapeAndTypeOf(val reflect.Value) (shape []int32, dt PaddleDType) {
	gotype := val.Type()
	for gotype.Kind() == reflect.Array || gotype.Kind() == reflect.Slice {
		shape = append(shape, int32(val.Len()))
		if val.Len() > 0 {
			val = val.Index(0)
		}
		gotype = gotype.Elem()
	}

	for _, t := range types {
		if gotype.Kind() == t.gotype.Kind() {
			return shape, PaddleDType(t.dtype)
		}
	}
	return shape, dt
}

type ZeroCopyTensor struct {
	c     *C.PD_ZeroCopyTensor
	name  string
	shape []int32
}

func NewZeroCopyTensor() *ZeroCopyTensor {
	c_tensor := C.PD_NewZeroCopyTensor()

	tensor := &ZeroCopyTensor{c: c_tensor}
	runtime.SetFinalizer(tensor, (*ZeroCopyTensor).finalize)
	return tensor
}

func (tensor *ZeroCopyTensor) finalize() {
	C.PD_DeleteZeroCopyTensor(tensor.c)
}

func (tensor *ZeroCopyTensor) Shape() []int32 {
	return tensor.shape
}

func (tensor *ZeroCopyTensor) Name() string {
	return C.GoString(tensor.c.name)
}

func (tensor *ZeroCopyTensor) Rename(name string) {
	tensor.name = name
	tensor.c.name = (*C.char)(unsafe.Pointer(tensor.c.name))
}

func (tensor *ZeroCopyTensor) Reshape(shape []int32) {
	tensor.shape = make([]int32, len(shape))
	copy(tensor.shape, shape)
	length := C.sizeof_int * C.size_t(len(shape))
	if tensor.c.shape.capacity < C.size_t(length) {
		if tensor.c.shape.capacity != C.size_t(0) {
			C.free(tensor.c.shape.data)
		}
		tensor.c.shape.data = C.malloc(length)
		tensor.c.shape.capacity = length
	}
	tensor.c.shape.length = length
	C.memcpy(tensor.c.shape.data, unsafe.Pointer(&shape[0]), length)
}

func (tensor *ZeroCopyTensor) DataType() PaddleDType {
	return PaddleDType(tensor.c.dtype)
}

func (tensor *ZeroCopyTensor) SetValue(value interface{}) {
	val := reflect.ValueOf(value)
	shape, dtype := shapeAndTypeOf(val)
	num := numel(shape)
	length := C.size_t(sizeofDataType(dtype) * num)
	if tensor.c.data.capacity < length {
		if tensor.c.data.capacity != C.size_t(0) {
			C.free(tensor.c.data.data)
		}
		tensor.c.data.data = C.malloc(length)
		tensor.c.data.capacity = length
	}
	tensor.c.data.length = length

	switch dtype {
	case PaddleDType(UINT8):
		data := val.Interface().([]uint8)
		C.memcpy(tensor.c.data.data, unsafe.Pointer(&data[0]), length)
	case PaddleDType(INT32):
		data := val.Interface().([]int32)
		C.memcpy(tensor.c.data.data, unsafe.Pointer(&data[0]), length)
	case PaddleDType(INT64):
		data := val.Interface().([]int64)
		C.memcpy(tensor.c.data.data, unsafe.Pointer(&data[0]), length)
	case PaddleDType(FLOAT32):
		data := val.Interface().([]float32)
		C.memcpy(tensor.c.data.data, unsafe.Pointer(&data[0]), length)
	}
	tensor.c.dtype = C.PD_DataType(dtype)
}

func (tensor *ZeroCopyTensor) tensorData() []byte {
	cbytes := tensor.c.data.data
	length := tensor.c.data.length
	var slice []byte
	if unsafe.Sizeof(unsafe.Pointer(nil)) == 8 {
		slice = (*[1<<50 - 1]byte)(unsafe.Pointer(cbytes))[:length:length]
	} else {
		slice = (*[1 << 30]byte)(unsafe.Pointer(cbytes))[:length:length]
	}
	return slice
}

func (tensor *ZeroCopyTensor) Value() interface{} {
	t := typeOfDataType(PaddleDType(tensor.c.dtype))
	data := tensor.tensorData()
	return decodeTensor(data, tensor.Shape(), t).Interface()
}

// It isn't safe to use reflect.SliceHeader as it uses a uintptr for Data and
// this is not inspected by the garbage collector
type sliceHeader struct {
	Data unsafe.Pointer
	Len  int
	Cap  int
}

func decodeTensor(raw []byte, shape []int32, t reflect.Type) reflect.Value {
	// Create a 1-dimensional slice of the base large enough for the data and
	// copy the data in.
	n := int(numel(shape))

	l := n * int(t.Size())
	typ := reflect.SliceOf(t)
	slice := reflect.MakeSlice(typ, n, n)
	baseBytes := *(*[]byte)(unsafe.Pointer(&sliceHeader{
		Data: unsafe.Pointer(slice.Pointer()),
		Len:  l,
		Cap:  l,
	}))
	copy(baseBytes, raw)

	if len(shape) == 0 {
		// for n
		return slice.Index(0)
	}
	if len(shape) == 1 {
		// for {}
		return slice
	}
	// for {{} {}} {{} {}} {{} {}}
	if n == 0 {
		n = int(numel(shape[:len(shape)-1]))
	}
	for i := len(shape) - 2; i >= 0; i-- {
		underlyingSize := typ.Elem().Size()
		typ = reflect.SliceOf(typ)
		subsliceLen := int(shape[i+1])
		if subsliceLen != 0 {
			n = n / subsliceLen
		}
		data := unsafe.Pointer(slice.Pointer())
		nextSlice := reflect.MakeSlice(typ, n, n)

		for j := 0; j < n; j++ {
			// This is equivalent to nSlice[j] = slice[j*subsliceLen: (j+1)*subsliceLen]
			setSliceInSlice(nextSlice, j, sliceHeader{
				Data: unsafe.Pointer(uintptr(data) + (uintptr(j*subsliceLen) * underlyingSize)),
				Len:  subsliceLen,
				Cap:  subsliceLen,
			})
		}

		slice = nextSlice
	}
	return slice
}

// setSliceInSlice sets slice[index] = content.
func setSliceInSlice(slice reflect.Value, index int, content sliceHeader) {
	const sliceSize = unsafe.Sizeof(sliceHeader{})
	// We must cast slice.Pointer to uninptr & back again to avoid GC issues.
	// See https://github.com/google/go-cmp/issues/167#issuecomment-546093202
	*(*sliceHeader)(unsafe.Pointer(uintptr(unsafe.Pointer(slice.Pointer())) + (uintptr(index) * sliceSize))) = content
}

func (tensor *ZeroCopyTensor) Lod() []uint {
	var val []uint
	valHdr := (*reflect.SliceHeader)(unsafe.Pointer(&val))
	valHdr.Data = uintptr(unsafe.Pointer(tensor.c.lod.data))
	valHdr.Len = int(tensor.c.lod.length / C.sizeof_size_t)
	valHdr.Cap = int(tensor.c.lod.length / C.sizeof_size_t)
	return val
}
