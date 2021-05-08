#pragma once

#include "common.h"
#include <paddle_api.h>
#include <vector>

namespace ppredictor {
class PredictorOutput {
public:
  PredictorOutput() {}
  PredictorOutput(std::unique_ptr<const paddle::lite_api::Tensor> &&tensor,
                  int index, int net_flag)
      : _tensor(std::move(tensor)), _index(index), _net_flag(net_flag) {}

  const float *get_float_data() const;
  const int *get_int_data() const;
  int64_t get_size() const;
  const std::vector<std::vector<uint64_t>> get_lod() const;
  const std::vector<int64_t> get_shape() const;

  std::vector<float> data;    // return float, or use data_int
  std::vector<int> data_int;  // several layers return int ，or use data
  std::vector<int64_t> shape; // PaddleLite output shape
  std::vector<std::vector<uint64_t>> lod; // PaddleLite output lod

private:
  std::unique_ptr<const paddle::lite_api::Tensor> _tensor;
  int _index;
  int _net_flag;
};
}
