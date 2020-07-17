#include "ppredictor.h"
#include "common.h"

namespace ppredictor {
PPredictor::PPredictor(int thread_num, int net_flag, paddle::lite_api::PowerMode mode) :
    _thread_num(thread_num), _net_flag(net_flag), _mode(mode) {
}

int PPredictor::init_nb(const std::string &model_content) {
    paddle::lite_api::MobileConfig config;
    config.set_model_from_buffer(model_content);
    return _init(config);
}

int PPredictor::init_from_file(const std::string &model_content){
    paddle::lite_api::MobileConfig config;
    config.set_model_from_file(model_content);
    return _init(config);
}

template<typename ConfigT>
int PPredictor::_init(ConfigT &config) {
    config.set_threads(_thread_num);
    config.set_power_mode(_mode);
    _predictor = paddle::lite_api::CreatePaddlePredictor(config);
    LOGI("paddle instance created");
    return RETURN_OK;
}

PredictorInput PPredictor::get_input(int index) {
    PredictorInput input{_predictor->GetInput(index), index, _net_flag};
    _is_input_get = true;
    return input;
}

std::vector<PredictorInput> PPredictor::get_inputs(int num) {
    std::vector<PredictorInput> results;
    for (int i = 0; i < num; i++) {
        results.emplace_back(get_input(i));
    }
    return results;
}

PredictorInput PPredictor::get_first_input() {
    return get_input(0);
}

std::vector<PredictorOutput> PPredictor::infer() {
    LOGI("infer Run start %d", _net_flag);
    std::vector<PredictorOutput> results;
    if (!_is_input_get) {
        return results;
    }
    _predictor->Run();
    LOGI("infer Run end");

    for (int i = 0; i < _predictor->GetOutputNames().size(); i++) {
        std::unique_ptr<const paddle::lite_api::Tensor> output_tensor = _predictor->GetOutput(i);
        LOGI("output tensor[%d] size %ld", i, product(output_tensor->shape()));
        PredictorOutput result{std::move(output_tensor), i, _net_flag};
        results.emplace_back(std::move(result));
    }
    return results;
}

NET_TYPE PPredictor::get_net_flag() const {
    return (NET_TYPE) _net_flag;
}

}