/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export {
  DEFAULT_DET_MODEL_PARSE_FALLBACKS,
  DEFAULT_DET_MODEL_CONFIG,
  createDetModel,
  createDetModelSession,
  parseDetModelConfigText
} from "./det";
export {
  DEFAULT_REC_MODEL_PARSE_FALLBACKS,
  DEFAULT_REC_RUNTIME_LIMITS,
  DEFAULT_REC_MODEL_CONFIG,
  createRecModel,
  createRecModelSession,
  parseRecModelConfigText
} from "./rec";
export type { RecRuntimeOverrides } from "./rec";
