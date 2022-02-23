/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>

#include "helper.h"

#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

using namespace paddle_infer;

DEFINE_int32(batch_size, 1, "Batch size to do inference. ");
DEFINE_int32(gpu_id, 0, "The gpu id to do inference. ");
DEFINE_string(model_dir,
              "./infer_model/",
              "The directory to the inference model. ");
DEFINE_string(vocab_file,
              "./vocab_all.bpe.33708",
              "The path to the vocabulary file. ");
DEFINE_string(data_file,
              "./newstest2014.tok.bpe.33708.en",
              "The path to the input data file. ");

std::string model_dir = "";
std::string vocab_file = "";
std::string data_file = "";

const int EOS_IDX = 1;
const int PAD_IDX = 0;
const int MAX_LENGTH = 256;
const int N_BEST = 1;

int batch_size = 1;
int gpu_id = 0;

namespace paddle {
namespace inference {

struct DataInput {
  std::vector<int64_t> src_data;
};

struct DataResult {
  std::string result_q;
};

bool get_result_tensor(const std::unique_ptr<paddle_infer::Tensor>& seq_ids,
                       std::vector<DataResult>& dataresultvec,
                       std::unordered_map<int, std::string>& num2word_dict) {
  std::vector<int> output_shape = seq_ids->shape();
  int batch_size = output_shape[1];
  int beam_num = output_shape[2];
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  std::vector<int> seq_ids_out;
  seq_ids_out.resize(out_num);
  seq_ids->CopyToCpu(seq_ids_out.data());

  dataresultvec.resize(batch_size * N_BEST);
  auto max_output_length = output_shape[0];

  for (int bsz = 0; bsz < output_shape[1]; ++bsz) {
    for (int k = 0; k < N_BEST; ++k) {
      dataresultvec[bsz * N_BEST + k].result_q = "";
      for (int len = 0; len < max_output_length; ++len) {
        if (seq_ids_out[len * batch_size * beam_num + bsz * beam_num + k] ==
            EOS_IDX)
          break;
        dataresultvec[bsz * N_BEST + k].result_q =
            dataresultvec[bsz * N_BEST + k].result_q +
            num2word_dict[seq_ids_out[len * batch_size * beam_num +
                                      bsz * beam_num + k]] +
            " ";
      }
    }
  }
  return true;
}

class DataReader {
public:
  explicit DataReader(const std::string& path)
      : file(new std::ifstream(path)) {}

  bool NextBatch(std::shared_ptr<paddle_infer::Predictor>& predictor,
                 const int& batch_size,
                 std::vector<std::string>& source_query_vec) {
    std::string line;
    std::vector<std::string> word_data;
    std::vector<DataInput> data_input_vec;
    int max_len = 0;
    for (int i = 0; i < batch_size; i++) {
      if (!std::getline(*file, line)) {
        break;
      }
      DataInput data_input;
      split(line, ' ', &word_data);
      std::string query_str = "";
      for (int j = 0; j < word_data.size(); ++j) {
        if (j >= MAX_LENGTH) {
          break;
        }
        query_str += word_data[j];
        if (word2num_dict.find(word_data[j]) == word2num_dict.end()) {
          data_input.src_data.push_back(word2num_dict["<unk>"]);
        } else {
          data_input.src_data.push_back(word2num_dict[word_data[j]]);
        }
      }
      source_query_vec.push_back(query_str);
      data_input.src_data.push_back(EOS_IDX);
      max_len = std::max(max_len, static_cast<int>(data_input.src_data.size()));
      max_len = std::min(max_len, MAX_LENGTH);
      data_input_vec.push_back(data_input);
    }
    if (data_input_vec.empty()) {
      return false;
    }
    return TensorMoreBatch(
        predictor, data_input_vec, max_len, data_input_vec.size());
  }

  bool GetWordDict() {
    std::ifstream fin(vocab_file);
    std::string line;
    int k = 0;
    while (std::getline(fin, line)) {
      word2num_dict[line] = k;
      num2word_dict[k] = line;
      k += 1;
    }

    fin.close();

    return true;
  }

  std::unordered_map<std::string, int> word2num_dict;
  std::unordered_map<int, std::string> num2word_dict;
  std::unique_ptr<std::ifstream> file;

private:
  bool TensorMoreBatch(std::shared_ptr<paddle_infer::Predictor>& predictor,
                       std::vector<DataInput>& data_input_vec,
                       int max_len,
                       int batch_size) {
    auto src_word_t = predictor->GetInputHandle("src_word");
    std::vector<int64_t> src_word_vec;
    src_word_vec.resize(max_len * batch_size);
    for (int i = 0; i < batch_size; ++i) {
      for (int k = 0; k < max_len; ++k) {
        if (k < data_input_vec[i].src_data.size()) {
          src_word_vec[i * max_len + k] = data_input_vec[i].src_data[k];
        } else {
          src_word_vec[i * max_len + k] = PAD_IDX;
        }
      }
    }
    src_word_t->Reshape({batch_size, max_len});
    src_word_t->CopyFromCpu(src_word_vec.data());

    // NOTE: If the saved model supports force decoding, a nullptr must be
    // given to trg_word to ensure predictor work properly when not
    // using force decoding.
    /*
     * auto trg_word_t = predictor->GetInputHandle("trg_word");
     * trg_word_t->Reshape({0, 0});
     * trg_word_t->CopyFromCpu((int*)nullptr);
     */

    return true;
  }
};


template <typename... Args>
void SummaryConfig(const paddle_infer::Config& config,
                   double infer_time,
                   int num_batches,
                   int num_samples) {
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "batch_size: " << batch_size;
  LOG(INFO) << "num_of_samples: " << num_samples;
  LOG(INFO) << "----------------------- Conf info -----------------------";
  LOG(INFO) << "runtime_device: " << (config.use_gpu() ? "gpu" : "cpu");
  LOG(INFO) << "ir_optim: " << (config.ir_optim() ? "true" : "false");
  LOG(INFO) << "----------------------- Perf info -----------------------";
  LOG(INFO) << "average_latency(ms): " << infer_time / num_samples << ", "
            << "QPS: " << num_samples / (infer_time / 1000.0);
}


void Main(int batch_size, int gpu_id) {
  Config config;
  config.SetModel(model_dir + "/transformer.pdmodel",
                  model_dir + "/transformer.pdiparams");

  config.EnableUseGpu(100, gpu_id);

  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  // When using fp16, fc_elementwise_layernorm_fuse_pass causes a little
  // different translation results with original dygraph prediction, maybe you
  // can turn off the IR optimization for same results as following:
  // config.SwitchIrOptim(false);
  auto predictor = CreatePredictor(config);
  DataReader reader(data_file);
  reader.GetWordDict();

  double whole_time = 0;
  Timer timer;
  int num_batches = 0;
  int num_samples = 0;
  std::vector<std::string> source_query_vec;
  std::ofstream out("predict.txt");

  while (reader.NextBatch(predictor, batch_size, source_query_vec)) {
    timer.tic();
    predictor->Run();
    std::vector<DataResult> dataresultvec;
    auto output_names = predictor->GetOutputNames();
    get_result_tensor(predictor->GetOutputHandle(output_names[0]),
                      dataresultvec,
                      reader.num2word_dict);

    whole_time += timer.toc();
    num_batches++;

    if (out.is_open()) {
      for (int i = 0; i < dataresultvec.size(); ++i) {
        out << dataresultvec[i].result_q << "\n";
      }
    }
    num_samples += dataresultvec.size();

    source_query_vec.clear();
  }
  SummaryConfig(config, whole_time, num_batches, num_samples);
  out.close();
}
}  // namespace inference
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  batch_size = FLAGS_batch_size;
  gpu_id = FLAGS_gpu_id;

  model_dir = FLAGS_model_dir;
  vocab_file = FLAGS_vocab_file;
  data_file = FLAGS_data_file;

  paddle::inference::Main(batch_size, gpu_id);

  return 0;
}
