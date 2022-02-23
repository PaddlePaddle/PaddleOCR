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


#include "fastertransformer/utils/common.h"


/**
 * The CublasHandle class defines the `GetInstance` method that serves as an
 * alternative to constructor and lets clients access the same instance of this
 * class over and over.
 */
class CublasHandle {
  /**
   * The CublasHandle's constructor should always be private to prevent direct
   * construction calls with the `new` operator.
   */
private:
  CublasHandle() {
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);
  }

public:
  /**
   * CublasHandle should not be cloneable.
   */
  CublasHandle(CublasHandle& other) = delete;

  /**
   * CublasHandle should not be assignable.
   */
  void operator=(const CublasHandle&) = delete;

  /**
   * This is the static method that controls the access to the singleton
   * instance. On the first run, it creates a singleton object and places it
   * into the static field. On subsequent runs, it returns the client existing
   * object stored in the static field.
   */
  static CublasHandle* GetInstance();

  cublasHandle_t cublas_handle_;
  cublasLtHandle_t cublaslt_handle_;

  ~CublasHandle();
};
