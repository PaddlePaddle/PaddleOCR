#pragma once

#include <paddle_api.h>
#include <vector>
#include "common.h"

namespace ppredictor {
class PredictorInput {
public:
    PredictorInput(std::unique_ptr<paddle::lite_api::Tensor> &&tensor, int index, int net_flag) :
        _tensor(std::move(tensor)), _index(index),_net_flag(net_flag) {

    }


    void set_dims(std::vector<int64_t> dims);

    float *get_mutable_float_data();

    void set_data(const float *input_data, int input_float_len);

private:
    std::unique_ptr<paddle::lite_api::Tensor> _tensor;
    bool _is_dims_set = false;
    int _index;
    int _net_flag;
};
}
