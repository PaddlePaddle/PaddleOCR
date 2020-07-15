#pragma once

#include <paddle_api.h>
#include <vector>
#include "common.h"

namespace ppredictor {
class PredictorOutput {
public:
    PredictorOutput(){

    }
    PredictorOutput(std::unique_ptr<const paddle::lite_api::Tensor> &&tensor, int index, int net_flag) :
        _tensor(std::move(tensor)), _index(index), _net_flag(net_flag) {

    }

    const float* get_float_data() const;
    const int* get_int_data() const;
    int64_t get_size() const;
    const std::vector<std::vector<uint64_t>> get_lod() const;
    const  std::vector<int64_t> get_shape() const;

    std::vector<float> data; // 通常是float返回，与下面的data_int二选一
    std::vector<int> data_int; // 少数层是int返回，与 data二选一
    std::vector<int64_t> shape; // PaddleLite输出层的shape
    std::vector<std::vector<uint64_t>> lod; // PaddleLite输出层的lod

private:
    std::unique_ptr<const paddle::lite_api::Tensor> _tensor;
    int _index;
    int _net_flag;
};
}

