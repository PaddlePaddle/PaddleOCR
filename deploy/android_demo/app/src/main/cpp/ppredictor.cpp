#include "ppredictor.h"
#include "common.h"

namespace ppredictor {
PPredictor::PPredictor(int use_opencl, int thread_num, int net_flag,
                       paddle::lite_api::PowerMode mode)
    : _use_opencl(use_opencl), _thread_num(thread_num), _net_flag(net_flag), _mode(mode) {}

int PPredictor::init_nb(const std::string &model_content) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_buffer(model_content);
  return _init(config);
}

int PPredictor::init_from_file(const std::string &model_content) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(model_content);
  return _init(config);
}

template <typename ConfigT> int PPredictor::_init(ConfigT &config) {
  bool is_opencl_backend_valid = paddle::lite_api::IsOpenCLBackendValid(/*check_fp16_valid = false*/);
  if (is_opencl_backend_valid) {
    if (_use_opencl != 0) {
      // Make sure you have write permission of the binary path.
      // We strongly recommend each model has a unique binary name.
      const std::string bin_path = "/data/local/tmp/";
      const std::string bin_name = "lite_opencl_kernel.bin";
      config.set_opencl_binary_path_name(bin_path, bin_name);

      // opencl tune option
      // CL_TUNE_NONE: 0
      // CL_TUNE_RAPID: 1
      // CL_TUNE_NORMAL: 2
      // CL_TUNE_EXHAUSTIVE: 3
      const std::string tuned_path = "/data/local/tmp/";
      const std::string tuned_name = "lite_opencl_tuned.bin";
      config.set_opencl_tune(paddle::lite_api::CL_TUNE_NORMAL, tuned_path, tuned_name);

      // opencl precision option
      // CL_PRECISION_AUTO: 0, first fp16 if valid, default
      // CL_PRECISION_FP32: 1, force fp32
      // CL_PRECISION_FP16: 2, force fp16
      config.set_opencl_precision(paddle::lite_api::CL_PRECISION_FP32);
      LOGI("device: running on gpu.");
    }
  } else {
    LOGI("device: running on cpu.");
    // you can give backup cpu nb model instead
    // config.set_model_from_file(cpu_nb_model_dir);
  }
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

PredictorInput PPredictor::get_first_input() { return get_input(0); }

std::vector<PredictorOutput> PPredictor::infer() {
  LOGI("infer Run start %d", _net_flag);
  std::vector<PredictorOutput> results;
  if (!_is_input_get) {
    return results;
  }
  _predictor->Run();
  LOGI("infer Run end");

  for (int i = 0; i < _predictor->GetOutputNames().size(); i++) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        _predictor->GetOutput(i);
    LOGI("output tensor[%d] size %ld", i, product(output_tensor->shape()));
    PredictorOutput result{std::move(output_tensor), i, _net_flag};
    results.emplace_back(std::move(result));
  }
  return results;
}

NET_TYPE PPredictor::get_net_flag() const { return (NET_TYPE)_net_flag; }
}