#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <codecvt>
#include <cstring>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef GPT_ON_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

#include "helper.h"

#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

DEFINE_int32(batch_size, 1, "Batch size to do inference. ");
DEFINE_int32(gpu_id, 0, "The gpu id to do inference. ");
DEFINE_string(model_dir,
              "./infer_model/",
              "The directory to the inference model. ");
DEFINE_string(vocab_file,
              "./infer_model/vocab.txt",
              "The path to the vocabulary file. ");
DEFINE_string(start_token, "<|endoftext|>", "The start token of GPT.");
DEFINE_string(end_token, "<|endoftext|>", "The end token of GPT.");

using namespace paddle_infer;

std::string model_dir = "";
std::string vocab_file = "";

const int BOS_IDX = 50256;
const int EOS_IDX = 50256;
const int PAD_IDX = 50256;
const int MAX_LENGTH = 256;
std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;

int batch_size = 1;
int gpu_id = 0;

namespace paddle {
namespace inference {

struct DataInput {
  std::vector<int64_t> src_data;
};

struct DataResult {
  std::wstring result_q;
};

bool get_result_tensor(const std::unique_ptr<paddle_infer::Tensor>& seq_ids,
                       std::vector<DataResult>& dataresultvec,
                       std::unordered_map<int, std::u32string>& num2word_dict,
                       std::unordered_map<char32_t, int>& byte_decoder) {
  // NOTE: Add SentencePiece to do some postprocess on cpm model.
  // sentencepiece::SentencePieceProcessor processor;
  // max_length * batch_size
  std::vector<int> output_shape = seq_ids->shape();
  int batch_size = output_shape[1];
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  std::vector<int> seq_ids_out;
  seq_ids_out.resize(out_num);
  seq_ids->CopyToCpu(seq_ids_out.data());

  dataresultvec.resize(batch_size);
  auto max_output_length = output_shape[0];

  for (int bsz = 0; bsz < batch_size; ++bsz) {
    std::u32string tmp_result_q = U"";
    for (int len = 1; len < max_output_length; ++len) {
      tmp_result_q =
          tmp_result_q + num2word_dict[seq_ids_out[len * batch_size + bsz]];
    }

    for (int i = 0; i < tmp_result_q.length(); ++i) {
      char32_t tmp = tmp_result_q[i];
      if (byte_decoder.find(tmp) != byte_decoder.end()) {
        dataresultvec[bsz].result_q = dataresultvec[bsz].result_q +
                                      static_cast<wchar_t>(byte_decoder[tmp]);
      } else {
        std::cout << "Should not reach here. " << std::endl;
        exit(-1);
      }
    }
  }
  return true;
}

std::unordered_map<char32_t, int> convert_unicode() {
  char32_t c0 = U'!';
  char32_t c1 = U'~';
  char32_t c2 = U'¡';
  char32_t c3 = U'¬';
  char32_t c4 = U'®';
  char32_t c5 = U'ÿ';

  int a0 = c0;
  int a1 = c1;
  int a2 = c2;
  int a3 = c3;
  int a4 = c4;
  int a5 = c5;

  std::unordered_map<char32_t, int> ret;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    char32_t key;
    if (b < a0 || (b > a1 && b < a2) || (b < a3 && b > a4) || b > a5) {
      key = static_cast<char32_t>(256 + n);
      ret.insert(std::pair<char32_t, int>(key, b));
      n++;
    } else {
      key = static_cast<char32_t>(b);
      ret.insert(std::pair<char32_t, int>(key, b));
    }
  }

  return ret;
}

class DataReader {
public:
  DataReader() {}

  bool NextBatch(std::shared_ptr<paddle_infer::Predictor>& predictor,
                 const int& batch_size,
                 const std::u32string& start_token,
                 const std::u32string& end_token,
                 const int& num_batches,
                 std::vector<std::u32string>& source_query_vec) {
    if (current_batches++ >= num_batches) {
      return false;
    }

    for (int i = 0; i < batch_size; ++i) {
      source_query_vec.push_back(start_token);
    }

    std::u32string line;
    std::vector<std::u32string> word_data;
    std::vector<DataInput> data_input_vec;
    int max_len = 0;
    for (int i = 0; i < batch_size; i++) {
      DataInput data_input;
      data_input.src_data.push_back(word2num_dict[start_token]);
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
      std::u32string tmp = convert.from_bytes(line);
      word2num_dict[tmp] = k;
      num2word_dict[k] = tmp;
      k += 1;
    }

    fin.close();

    return true;
  }

  int GetCurrentBatch() { return current_batches; }

  std::unordered_map<std::u32string, int> word2num_dict;
  std::unordered_map<int, std::u32string> num2word_dict;
  std::unique_ptr<std::ifstream> file;

private:
  bool TensorMoreBatch(std::shared_ptr<paddle_infer::Predictor>& predictor,
                       std::vector<DataInput>& data_input_vec,
                       int max_len,
                       int batch_size) {
    auto ids_name = predictor->GetInputNames();
    auto ids_t = predictor->GetInputHandle(ids_name[0]);
    std::vector<int> ids_vec;
    ids_vec.resize(max_len * batch_size);
    for (int i = 0; i < batch_size; ++i) {
      for (int k = 0; k < max_len; ++k) {
        if (k < data_input_vec[i].src_data.size()) {
          ids_vec[i * max_len + k] = data_input_vec[i].src_data[k];
        } else {
          ids_vec[i * max_len + k] = PAD_IDX;
        }
      }
    }
    ids_t->Reshape({batch_size, max_len});
    ids_t->CopyFromCpu(ids_vec.data());

    return true;
  }

  int current_batches = 0;
};


template <typename... Args>
void SummaryConfig(const paddle_infer::Config& config,
                   double infer_time,
                   int num_batches,
                   int num_samples) {
  LOG(INFO) << "----------------------- Perf info -----------------------";
  LOG(INFO) << "batch_size: " << batch_size;
  LOG(INFO) << "average_latency(ms): " << infer_time / num_samples << ", "
            << "QPS: " << num_samples / (infer_time / 1000.0);
}


void Main(const int& batch_size,
          const int& gpu_id,
          const std::u32string& start_token,
          const std::u32string& end_token) {
  Config config;
  config.SetModel(model_dir + "/gpt.pdmodel", model_dir + "/gpt.pdiparams");

  config.EnableUseGpu(100, gpu_id);

  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  auto predictor = CreatePredictor(config);
  DataReader reader;
  reader.GetWordDict();

  double whole_time = 0;
  Timer timer;
  int num_batches = 100;
  int warmup = 50;
  std::vector<std::u32string> source_query_vec;
  auto byte_decoder = convert_unicode();

  while (reader.NextBatch(predictor,
                          batch_size,
                          start_token,
                          end_token,
                          num_batches,
                          source_query_vec)) {
    int crt_batch = reader.GetCurrentBatch();
    if (crt_batch >= warmup) {
      timer.tic();
    }
    predictor->Run();

    if (crt_batch >= warmup) {
      whole_time += timer.toc();
    }

    std::vector<DataResult> dataresultvec;
    auto output_names = predictor->GetOutputNames();
    get_result_tensor(predictor->GetOutputHandle(output_names[0]),
                      dataresultvec,
                      reader.num2word_dict,
                      byte_decoder);

    for (int i = 0; i < batch_size; ++i) {
      std::wcout << dataresultvec[i].result_q;
      std::cout << std::endl;
    }
    source_query_vec.clear();
  }
  std::cout << std::endl;
  SummaryConfig(config,
                whole_time,
                num_batches - warmup,
                (num_batches - warmup) * batch_size);
}
}  // namespace inference
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  batch_size = FLAGS_batch_size;
  gpu_id = FLAGS_gpu_id;

  model_dir = FLAGS_model_dir;
  vocab_file = FLAGS_vocab_file;

  paddle::inference::Main(batch_size,
                          gpu_id,
                          convert.from_bytes(FLAGS_start_token),
                          convert.from_bytes(FLAGS_end_token));

  return 0;
}
