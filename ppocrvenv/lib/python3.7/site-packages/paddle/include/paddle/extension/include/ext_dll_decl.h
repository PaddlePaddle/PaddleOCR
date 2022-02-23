// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#if defined(_WIN32)
#ifndef PD_DLL_DECL
#ifdef PADDLE_DLL_EXPORT
#define PD_DLL_DECL __declspec(dllexport)
#else
#define PD_DLL_DECL __declspec(dllimport)
#endif  // PADDLE_DLL_EXPORT
#endif  // PD_DLL_DECL
#else
#define PD_DLL_DECL
#endif  // _WIN32
