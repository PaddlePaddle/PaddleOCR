#include "predictor_input.h"

namespace ppredictor {

void PredictorInput::set_dims(std::vector<int64_t> dims) {
  // yolov3
  if (_net_flag == 101 && _index == 1) {
    _tensor->Resize({1, 2});
    _tensor->mutable_data<int>()[0] = (int)dims.at(2);
    _tensor->mutable_data<int>()[1] = (int)dims.at(3);
  } else {
    _tensor->Resize(dims);
  }
  _is_dims_set = true;
}

float *PredictorInput::get_mutable_float_data() {
  if (!_is_dims_set) {
    LOGE("PredictorInput::set_dims is not called");
  }
  return _tensor->mutable_data<float>();
}

void PredictorInput::set_data(const float *input_data, int input_float_len) {
  float *input_raw_data = get_mutable_float_data();
  memcpy(input_raw_data, input_data, input_float_len * sizeof(float));
}
} // namespace ppredictor
