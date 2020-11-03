package paddle

// #include <stdbool.h>
// #include <stdlib.h>
// #include <string.h>
// #include <paddle_c_api.h>
import "C"

import (
	"bytes"
	"encoding/binary"
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

func TypeOfShape(dtype PaddleDType, shape []int32) reflect.Type {
	var ret reflect.Type
	for _, t := range types {
		if dtype == PaddleDType(t.dtype) {
			ret = t.gotype
			break
		}
	}

	if ret == nil {
		panic(bug("Data %v type is not support", dtype))
	}

	for range shape {
		ret = reflect.SliceOf(ret)
	}
	return ret
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
	//tensor.c.name = C.CString(tensor.name)
	//defer C.free(unsafe.Pointer(tensor.c.name))
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
	shape, dtype := ShapeAndTypeOf(val)
	num := numel(shape)
	length := C.size_t(SizeofDataType(dtype) * num)
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

func TypeOf(dtype PaddleDType, shape []int32) reflect.Type {
	var ret reflect.Type
	for _, t := range types {
		if t.dtype == dtype {
			ret = t.gotype
			break
		}
	}

	for range shape {
		ret = reflect.SliceOf(ret)
	}
	return ret
}

func (tensor *ZeroCopyTensor) Value() interface{} {
	t := TypeOf(PaddleDType(tensor.c.dtype), tensor.shape)
	value := reflect.New(t)
	c_bytes := tensor.c.data.data
	length := tensor.c.data.length
	var slice []byte
	if unsafe.Sizeof(unsafe.Pointer(nil)) == 8 {
		slice = (*[1<<50 - 1]byte)(unsafe.Pointer(c_bytes))[:length:length]
	} else {
		slice = (*[1 << 30]byte)(unsafe.Pointer(c_bytes))[:length:length]
	}
	r := bytes.NewReader(slice)
	DecodeTensor(r, tensor.Shape(), t, value)
	return reflect.Indirect(value).Interface()
}

func (tensor *ZeroCopyTensor) Lod() []uint {
	var val []uint
	valHdr := (*reflect.SliceHeader)(unsafe.Pointer(&val))
	valHdr.Data = uintptr(unsafe.Pointer(tensor.c.lod.data))
	valHdr.Len = int(tensor.c.lod.length / C.sizeof_size_t)
	valHdr.Cap = int(tensor.c.lod.length / C.sizeof_size_t)
	return val
}

func Endian() binary.ByteOrder {
	buf := [2]byte{}
	*(*uint16)(unsafe.Pointer(&buf[0])) = uint16(0xABCD)

	var endian binary.ByteOrder

	switch buf {
	case [2]byte{0xCD, 0xAB}:
		endian = binary.LittleEndian
	case [2]byte{0xAB, 0xCD}:
		endian = binary.BigEndian
	default:
		panic("Could not determine native endianness.")
	}
	return endian
}

func DecodeTensor(r *bytes.Reader, shape []int32, t reflect.Type, ptr reflect.Value) {
	switch t.Kind() {
	case reflect.Uint8, reflect.Int32, reflect.Int64, reflect.Float32:
		binary.Read(r, Endian(), ptr.Interface())
	case reflect.Slice:
		value := reflect.Indirect(ptr)
		value.Set(reflect.MakeSlice(t, int(shape[0]), int(shape[0])))
		if len(shape) == 1 && value.Len() > 0 {
			switch value.Index(0).Kind() {
			case reflect.Uint8, reflect.Int32, reflect.Int64, reflect.Float32:
				binary.Read(r, Endian(), value.Interface())
				return
			}
		}

		for i := 0; i < value.Len(); i++ {
			DecodeTensor(r, shape[1:], t.Elem(), value.Index(i).Addr())
		}
	}
}

func SizeofDataType(dtype PaddleDType) int32 {
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

func ShapeAndTypeOf(val reflect.Value) (shape []int32, dt PaddleDType) {
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
