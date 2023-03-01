// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>

#include "fastdeploy_capi/vision.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void CpuInfer(const char *det_model_dir, const char *cls_model_dir,
              const char *rec_model_dir, const char *rec_label_file,
              const char *image_file) {
  char det_model_file[100];
  char det_params_file[100];

  char cls_model_file[100];
  char cls_params_file[100];

  char rec_model_file[100];
  char rec_params_file[100];

  int max_size = 99;
  snprintf(det_model_file, max_size, "%s%c%s", det_model_dir, sep,
           "inference.pdmodel");
  snprintf(det_params_file, max_size, "%s%c%s", det_model_dir, sep,
           "inference.pdiparams");

  snprintf(cls_model_file, max_size, "%s%c%s", cls_model_dir, sep,
           "inference.pdmodel");
  snprintf(cls_params_file, max_size, "%s%c%s", cls_model_dir, sep,
           "inference.pdiparams");

  snprintf(rec_model_file, max_size, "%s%c%s", rec_model_dir, sep,
           "inference.pdmodel");
  snprintf(rec_params_file, max_size, "%s%c%s", rec_model_dir, sep,
           "inference.pdiparams");

  FD_C_RuntimeOptionWrapper *det_option = FD_C_CreateRuntimeOptionWrapper();
  FD_C_RuntimeOptionWrapper *cls_option = FD_C_CreateRuntimeOptionWrapper();
  FD_C_RuntimeOptionWrapper *rec_option = FD_C_CreateRuntimeOptionWrapper();
  FD_C_RuntimeOptionWrapperUseCpu(det_option);
  FD_C_RuntimeOptionWrapperUseCpu(cls_option);
  FD_C_RuntimeOptionWrapperUseCpu(rec_option);

  FD_C_DBDetectorWrapper *det_model = FD_C_CreateDBDetectorWrapper(
      det_model_file, det_params_file, det_option, FD_C_ModelFormat_PADDLE);
  FD_C_ClassifierWrapper *cls_model = FD_C_CreateClassifierWrapper(
      cls_model_file, cls_params_file, cls_option, FD_C_ModelFormat_PADDLE);
  FD_C_RecognizerWrapper *rec_model = FD_C_CreateRecognizerWrapper(
      rec_model_file, rec_params_file, rec_label_file, rec_option,
      FD_C_ModelFormat_PADDLE);

  FD_C_PPOCRv3Wrapper *ppocr_v3 =
      FD_C_CreatePPOCRv3Wrapper(det_model, cls_model, rec_model);
  if (!FD_C_PPOCRv3WrapperInitialized(ppocr_v3)) {
    printf("Failed to initialize.\n");
    FD_C_DestroyRuntimeOptionWrapper(det_option);
    FD_C_DestroyRuntimeOptionWrapper(cls_option);
    FD_C_DestroyRuntimeOptionWrapper(rec_option);
    FD_C_DestroyClassifierWrapper(cls_model);
    FD_C_DestroyDBDetectorWrapper(det_model);
    FD_C_DestroyRecognizerWrapper(rec_model);
    FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
    return;
  }

  FD_C_Mat im = FD_C_Imread(image_file);

  FD_C_OCRResult *result = (FD_C_OCRResult *)malloc(sizeof(FD_C_OCRResult));

  if (!FD_C_PPOCRv3WrapperPredict(ppocr_v3, im, result)) {
    printf("Failed to predict.\n");
    FD_C_DestroyRuntimeOptionWrapper(det_option);
    FD_C_DestroyRuntimeOptionWrapper(cls_option);
    FD_C_DestroyRuntimeOptionWrapper(rec_option);
    FD_C_DestroyClassifierWrapper(cls_model);
    FD_C_DestroyDBDetectorWrapper(det_model);
    FD_C_DestroyRecognizerWrapper(rec_model);
    FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
    FD_C_DestroyMat(im);
    free(result);
    return;
  }

  // print res
  char res[2000];
  FD_C_OCRResultStr(result, res);
  printf("%s", res);
  FD_C_Mat vis_im = FD_C_VisOcr(im, result);
  FD_C_Imwrite("vis_result.jpg", vis_im);
  printf("Visualized result saved in ./vis_result.jpg\n");

  FD_C_DestroyRuntimeOptionWrapper(det_option);
  FD_C_DestroyRuntimeOptionWrapper(cls_option);
  FD_C_DestroyRuntimeOptionWrapper(rec_option);
  FD_C_DestroyClassifierWrapper(cls_model);
  FD_C_DestroyDBDetectorWrapper(det_model);
  FD_C_DestroyRecognizerWrapper(rec_model);
  FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
  FD_C_DestroyOCRResult(result);
  FD_C_DestroyMat(im);
  FD_C_DestroyMat(vis_im);
}

void GpuInfer(const char *det_model_dir, const char *cls_model_dir,
              const char *rec_model_dir, const char *rec_label_file,
              const char *image_file) {
  char det_model_file[100];
  char det_params_file[100];

  char cls_model_file[100];
  char cls_params_file[100];

  char rec_model_file[100];
  char rec_params_file[100];

  int max_size = 99;
  snprintf(det_model_file, max_size, "%s%c%s", det_model_dir, sep,
           "inference.pdmodel");
  snprintf(det_params_file, max_size, "%s%c%s", det_model_dir, sep,
           "inference.pdiparams");

  snprintf(cls_model_file, max_size, "%s%c%s", cls_model_dir, sep,
           "inference.pdmodel");
  snprintf(cls_params_file, max_size, "%s%c%s", cls_model_dir, sep,
           "inference.pdiparams");

  snprintf(rec_model_file, max_size, "%s%c%s", rec_model_dir, sep,
           "inference.pdmodel");
  snprintf(rec_params_file, max_size, "%s%c%s", rec_model_dir, sep,
           "inference.pdiparams");

  FD_C_RuntimeOptionWrapper *det_option = FD_C_CreateRuntimeOptionWrapper();
  FD_C_RuntimeOptionWrapper *cls_option = FD_C_CreateRuntimeOptionWrapper();
  FD_C_RuntimeOptionWrapper *rec_option = FD_C_CreateRuntimeOptionWrapper();
  FD_C_RuntimeOptionWrapperUseGpu(det_option, 0);
  FD_C_RuntimeOptionWrapperUseGpu(cls_option, 0);
  FD_C_RuntimeOptionWrapperUseGpu(rec_option, 0);

  FD_C_DBDetectorWrapper *det_model = FD_C_CreateDBDetectorWrapper(
      det_model_file, det_params_file, det_option, FD_C_ModelFormat_PADDLE);
  FD_C_ClassifierWrapper *cls_model = FD_C_CreateClassifierWrapper(
      cls_model_file, cls_params_file, cls_option, FD_C_ModelFormat_PADDLE);
  FD_C_RecognizerWrapper *rec_model = FD_C_CreateRecognizerWrapper(
      rec_model_file, rec_params_file, rec_label_file, rec_option,
      FD_C_ModelFormat_PADDLE);

  FD_C_PPOCRv3Wrapper *ppocr_v3 =
      FD_C_CreatePPOCRv3Wrapper(det_model, cls_model, rec_model);
  if (!FD_C_PPOCRv3WrapperInitialized(ppocr_v3)) {
    printf("Failed to initialize.\n");
    FD_C_DestroyRuntimeOptionWrapper(det_option);
    FD_C_DestroyRuntimeOptionWrapper(cls_option);
    FD_C_DestroyRuntimeOptionWrapper(rec_option);
    FD_C_DestroyClassifierWrapper(cls_model);
    FD_C_DestroyDBDetectorWrapper(det_model);
    FD_C_DestroyRecognizerWrapper(rec_model);
    FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
    return;
  }

  FD_C_Mat im = FD_C_Imread(image_file);

  FD_C_OCRResult *result = (FD_C_OCRResult *)malloc(sizeof(FD_C_OCRResult));

  if (!FD_C_PPOCRv3WrapperPredict(ppocr_v3, im, result)) {
    printf("Failed to predict.\n");
    FD_C_DestroyRuntimeOptionWrapper(det_option);
    FD_C_DestroyRuntimeOptionWrapper(cls_option);
    FD_C_DestroyRuntimeOptionWrapper(rec_option);
    FD_C_DestroyClassifierWrapper(cls_model);
    FD_C_DestroyDBDetectorWrapper(det_model);
    FD_C_DestroyRecognizerWrapper(rec_model);
    FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
    FD_C_DestroyMat(im);
    free(result);
    return;
  }

  // print res
  char res[2000];
  FD_C_OCRResultStr(result, res);
  printf("%s", res);
  FD_C_Mat vis_im = FD_C_VisOcr(im, result);
  FD_C_Imwrite("vis_result.jpg", vis_im);
  printf("Visualized result saved in ./vis_result.jpg\n");

  FD_C_DestroyRuntimeOptionWrapper(det_option);
  FD_C_DestroyRuntimeOptionWrapper(cls_option);
  FD_C_DestroyRuntimeOptionWrapper(rec_option);
  FD_C_DestroyClassifierWrapper(cls_model);
  FD_C_DestroyDBDetectorWrapper(det_model);
  FD_C_DestroyRecognizerWrapper(rec_model);
  FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
  FD_C_DestroyOCRResult(result);
  FD_C_DestroyMat(im);
  FD_C_DestroyMat(vis_im);
}
int main(int argc, char *argv[]) {
  if (argc < 7) {
    printf("Usage: infer_demo path/to/det_model path/to/cls_model "
           "path/to/rec_model path/to/rec_label_file path/to/image "
           "run_option, "
           "e.g ./infer_demo ./ch_PP-OCRv3_det_infer "
           "./ch_ppocr_mobile_v3.0_cls_infer ./ch_PP-OCRv3_rec_infer "
           "./ppocr_keys_v1.txt ./12.jpg 0\n");
    printf(
        "The data type of run_option is int, 0: run with cpu; 1: run with gpu"
        "\n");
    return -1;
  }

  if (atoi(argv[6]) == 0) {
    CpuInfer(argv[1], argv[2], argv[3], argv[4], argv[5]);
  } else if (atoi(argv[6]) == 1) {
    GpuInfer(argv[1], argv[2], argv[3], argv[4], argv[5]);
  }
  return 0;
}
