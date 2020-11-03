package paddle

// #cgo CFLAGS: -I../paddle_c/paddle/include
// #cgo LDFLAGS: -L${SRCDIR}/../paddle_c/paddle/lib -Wl,-rpath=\$ORIGIN/paddle_c/paddle/lib -lpaddle_fluid_c
// #include <stdbool.h>
// #include "paddle_c_api.h"
import "C"
import "fmt"

func ConvertCBooleanToGo(b C.bool) bool {
	var c_false C.bool
	if b != c_false {
		return true
	}
	return false
}

func numel(shape []int32) int32 {
	n := int32(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

func bug(format string, args ...interface{}) error {
	return fmt.Errorf("Bug %v", fmt.Sprintf(format, args...))
}
