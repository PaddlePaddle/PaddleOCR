
namespace fastertransformer {

const unsigned int WARP_REDUCE_MASK = 0xffffffff;
const float CUDA_FLOAT_INF_NEG = -100000000.f;
const unsigned int WARP_SIZE = 32;

template <typename T>
__forceinline__ __device__ T warpReduceMax(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE));
  return val;
}


/* Calculate the maximum of all elements in a block */
template <typename T>
__forceinline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane]
                                                 : CUDA_FLOAT_INF_NEG;
  val = warpReduceMax<T>(val);
  return val;
}

/* Calculate the rough topk-th value in a block, rough but safe */
template <typename T, int K>
__forceinline__ __device__ T blockRoughTopK(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  val = warpReduceMax(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  // we do not care about result of threadIdx.x bigger than (blockDim.x >> 5)
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : CUDA_FLOAT_INF_NEG;

  // K should be 2, 4, 6, 8, 16 or 32
  for (int mask = 16; mask >= K; mask >>= 1)
    val = max(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, 32));
  for (int mask = (K >> 1); mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, 32));

  return val;
}
}