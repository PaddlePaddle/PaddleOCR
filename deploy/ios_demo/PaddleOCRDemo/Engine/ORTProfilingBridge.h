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

#import <Foundation/Foundation.h>

@class ORTSession;
@class ORTSessionOptions;

NS_ASSUME_NONNULL_BEGIN

/// Bridges ONNX Runtime C++ profiling APIs that are not exposed on
/// `onnxruntime-objc`'s public surface.
///
/// - Warning: `endProfilingOnSession` reads the private `_session` ivar on
/// `ORTSession` (layout depends on the
///   installed `onnxruntime-objc` version). Re-verify after pod upgrades.
@interface ORTProfilingBridge : NSObject

+ (BOOL)enableProfilingOnSessionOptions:(ORTSessionOptions *)options
                             pathPrefix:(NSString *)pathPrefix
                                  error:(NSError *_Nullable __autoreleasing *)
                                            error
    NS_SWIFT_NAME(enableProfiling(sessionOptions:pathPrefix:));

/// Ends profiling and returns the path to the written JSON (see ORT
/// `Session::EndProfilingAllocated`).
+ (nullable NSString *)
    endProfilingOnSession:(ORTSession *)session
                    error:(NSError *_Nullable __autoreleasing *)error
    NS_SWIFT_NAME(endProfiling(session:));

@end

NS_ASSUME_NONNULL_END
