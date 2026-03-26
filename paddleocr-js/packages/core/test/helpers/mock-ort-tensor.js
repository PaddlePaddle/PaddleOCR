/** Minimal ORT `Tensor` mock; optional `tensorCalls` records each `{ type, dims, size }`. */
export function createMockOrtTensorClass(tensorCalls = null) {
  return class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
      tensorCalls?.push({ type, dims, size: data.length });
    }
  };
}
