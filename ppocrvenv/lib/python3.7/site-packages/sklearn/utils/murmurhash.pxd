"""Export fast murmurhash C/C++ routines + cython wrappers"""

cimport numpy as np

# The C API is disabled for now, since it requires -I flags to get
# compilation to work even when these functions are not used.
#cdef extern from "MurmurHash3.h":
#    void MurmurHash3_x86_32(void* key, int len, unsigned int seed,
#                            void* out)
#
#    void MurmurHash3_x86_128(void* key, int len, unsigned int seed,
#                             void* out)
#
#    void MurmurHash3_x64_128(void* key, int len, unsigned int seed,
#                             void* out)


cpdef np.uint32_t murmurhash3_int_u32(int key, unsigned int seed)
cpdef np.int32_t murmurhash3_int_s32(int key, unsigned int seed)
cpdef np.uint32_t murmurhash3_bytes_u32(bytes key, unsigned int seed)
cpdef np.int32_t murmurhash3_bytes_s32(bytes key, unsigned int seed)
