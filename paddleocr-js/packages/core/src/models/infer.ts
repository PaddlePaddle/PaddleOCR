/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { InferenceSession, Tensor } from "onnxruntime-web";

export async function runInference(
  session: InferenceSession,
  inputTensor: Tensor
): Promise<Tensor> {
  const inputName = session.inputNames[0];
  const outputMap = await session.run({ [inputName]: inputTensor });
  return outputMap[session.outputNames[0]];
}
