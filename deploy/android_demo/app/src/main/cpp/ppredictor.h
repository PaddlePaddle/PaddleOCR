#pragma once

#include "paddle_api.h"
#include "predictor_input.h"
#include "predictor_output.h"

namespace ppredictor {

/**
 * PaddleLite Preditor Common Interface
 */
class PPredictor_Interface {
public:
  virtual ~PPredictor_Interface() {}

  virtual NET_TYPE get_net_flag() const = 0;
};

/**
 * Common Predictor
 */
class PPredictor : public PPredictor_Interface {
public:
  PPredictor(
      int thread_num, int net_flag = 0,
      paddle::lite_api::PowerMode mode = paddle::lite_api::LITE_POWER_HIGH);

  virtual ~PPredictor() {}

  /**
   * init paddlitelite opt model，nb format ，or use ini_paddle
   * @param model_content
   * @return 0
   */
  virtual int init_nb(const std::string &model_content);

  virtual int init_from_file(const std::string &model_content);

  std::vector<PredictorOutput> infer();

  std::shared_ptr<paddle::lite_api::PaddlePredictor> get_predictor() {
    return _predictor;
  }

  virtual std::vector<PredictorInput> get_inputs(int num);

  virtual PredictorInput get_input(int index);

  virtual PredictorInput get_first_input();

  virtual NET_TYPE get_net_flag() const;

protected:
  template <typename ConfigT> int _init(ConfigT &config);

private:
  int _thread_num;
  paddle::lite_api::PowerMode _mode;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> _predictor;
  bool _is_input_get = false;
  int _net_flag;
};
}
