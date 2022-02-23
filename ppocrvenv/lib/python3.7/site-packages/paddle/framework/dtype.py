# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..fluid.core import VarDesc

dtype = VarDesc.VarType
dtype.__qualname__ = "dtype"
dtype.__module__ = "paddle"

uint8 = VarDesc.VarType.UINT8
int8 = VarDesc.VarType.INT8
int16 = VarDesc.VarType.INT16
int32 = VarDesc.VarType.INT32
int64 = VarDesc.VarType.INT64

float32 = VarDesc.VarType.FP32
float64 = VarDesc.VarType.FP64
float16 = VarDesc.VarType.FP16
bfloat16 = VarDesc.VarType.BF16

complex64 = VarDesc.VarType.COMPLEX64
complex128 = VarDesc.VarType.COMPLEX128

bool = VarDesc.VarType.BOOL

__all__ = []
