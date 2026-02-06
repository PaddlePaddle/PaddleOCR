// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "doc_preprocessor.h"

#include "src/utils/args.h"
#include "src/utils/yaml_config.h"

#define COPY_PARAMS(field) to.field = from.field;

DocPreprocessor::DocPreprocessor(const DocPreprocessorParams &params)
    : params_(params) {
  auto status = CheckParams();
  if (!status.ok()) {
    INFOE("Init DocPreprocessor fail : %s", status.ToString().c_str());
    exit(-1);
  }
  CreatePipeline();
};
std::vector<std::unique_ptr<BaseCVResult>>
DocPreprocessor::Predict(const std::vector<std::string> &input) {
  return pipeline_infer_->Predict(input);
}
void DocPreprocessor::CreatePipeline() {
  pipeline_infer_ = std::unique_ptr<BasePipeline>(
      new DocPreprocessorPipeline(ToDocPreprocessorPipelineParams(params_)));
}

absl::Status DocPreprocessor::CheckParams() {
  if (!params_.doc_orientation_classify_model_dir.has_value() &&
      !(params_.use_doc_orientation_classify.has_value() &&
        !params_.use_doc_orientation_classify.value())) {
    return absl::NotFoundError("Require doc orientation classify model dir.");
  }
  if (!params_.doc_unwarping_model_dir.has_value() &&
      !(params_.use_doc_unwarping.has_value() &&
        !params_.use_doc_unwarping.value())) {
    return absl::NotFoundError("Require doc unwarping model dir.");
  }
  return absl::OkStatus();
}

DocPreprocessorPipelineParams DocPreprocessor::ToDocPreprocessorPipelineParams(
    const DocPreprocessorParams &from) {
  DocPreprocessorPipelineParams to;
  COPY_PARAMS(doc_orientation_classify_model_name)
  COPY_PARAMS(doc_orientation_classify_model_dir)
  COPY_PARAMS(doc_unwarping_model_name)
  COPY_PARAMS(doc_unwarping_model_dir)
  COPY_PARAMS(use_doc_orientation_classify)
  COPY_PARAMS(use_doc_unwarping)
  COPY_PARAMS(device)
  COPY_PARAMS(enable_mkldnn)
  COPY_PARAMS(mkldnn_cache_capacity)
  COPY_PARAMS(precision)
  COPY_PARAMS(cpu_threads)
  COPY_PARAMS(thread_num)
  COPY_PARAMS(paddlex_config)
  return to;
}
