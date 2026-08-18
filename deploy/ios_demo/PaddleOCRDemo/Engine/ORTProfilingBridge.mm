// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#import "ORTProfilingBridge.h"

#import "ort_session_internal.h"

#import "cxx_api.h"
#import "error_utils.h"

#import <objc/runtime.h>

#include <optional>

@implementation ORTProfilingBridge

+ (BOOL)enableProfilingOnSessionOptions:(ORTSessionOptions *)options
                             pathPrefix:(NSString *)pathPrefix
                                  error:(NSError **)error {
  if (options == nil || pathPrefix.length == 0) {
    ORTSaveCodeAndDescriptionToError(ORT_INVALID_ARGUMENT, @"ORTProfilingBridge: options and non-empty pathPrefix required", error);
    return NO;
  }
  try {
    [options CXXAPIOrtSessionOptions].EnableProfiling(pathPrefix.fileSystemRepresentation);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error);
}

+ (NSString *)endProfilingOnSession:(ORTSession *)session error:(NSError **)error {
  if (session == nil) {
    ORTSaveCodeAndDescriptionToError(ORT_INVALID_ARGUMENT, @"ORTProfilingBridge: session is nil", error);
    return nil;
  }
  try {
    Ivar ivar = class_getInstanceVariable([ORTSession class], "_session");
    if (ivar == nullptr) {
      ORTSaveCodeAndDescriptionToError(ORT_FAIL,
                                     @"ORTProfilingBridge: ORTSession has no _session ivar — onnxruntime-objc mismatch?",
                                     error);
      return nil;
    }
    uintptr_t base = reinterpret_cast<uintptr_t>(session);
    auto *opt = reinterpret_cast<std::optional<Ort::Session> *>(base + ivar_getOffset(ivar));
    if (opt == nullptr || !opt->has_value()) {
      ORTSaveCodeAndDescriptionToError(ORT_FAIL, @"ORTProfilingBridge: ORTSession is not ready", error);
      return nil;
    }
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr path = opt->value().EndProfilingAllocated(allocator);
    const char *cstr = path.get();
    if (cstr == nullptr) {
      return nil;
    }
    return [NSString stringWithUTF8String:cstr];
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error);
}

@end
