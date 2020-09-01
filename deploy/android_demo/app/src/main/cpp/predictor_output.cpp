#include "predictor_output.h"
namespace ppredictor {
const float* PredictorOutput::get_float_data() const{
    return _tensor->data<float>();
}

const int* PredictorOutput::get_int_data() const{
    return _tensor->data<int>();
}

const std::vector<std::vector<uint64_t>> PredictorOutput::get_lod() const{
    return _tensor->lod();
}

int64_t PredictorOutput::get_size() const{
    if (_net_flag == NET_OCR) {
        return _tensor->shape().at(2) *  _tensor->shape().at(3);
    } else {
        return  product(_tensor->shape());
    }
}

const std::vector<int64_t> PredictorOutput::get_shape()  const{
    return _tensor->shape();

}
}