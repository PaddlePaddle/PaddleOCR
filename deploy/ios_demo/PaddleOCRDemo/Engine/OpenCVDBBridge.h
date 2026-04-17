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

NS_ASSUME_NONNULL_BEGIN

@interface PDBMiniBoxResult : NSObject
@property(nonatomic) NSArray<NSValue *> *corners;
@property(nonatomic) float minSide;
@end

@interface PDBOpenCVDBBridge : NSObject

+ (NSArray<NSArray<NSValue *> *> *)findContoursWithBinaryMask:(NSData *)data
                                                        width:(NSInteger)width
                                                       height:(NSInteger)height
                                                maxCandidates:
                                                    (NSInteger)maxCandidates
    NS_SWIFT_NAME(findContours(binaryMask:width:height:maxCandidates:));

+ (nullable PDBMiniBoxResult *)miniBoxFromPoints:(NSArray<NSValue *> *)points
    NS_SWIFT_NAME(miniBox(fromPoints:));

+ (float)contourAreaFromPoints:(NSArray<NSValue *> *)points
    NS_SWIFT_NAME(contourArea(fromPoints:));

+ (float)arcLengthFromPoints:(NSArray<NSValue *> *)points
                      closed:(BOOL)closed
    NS_SWIFT_NAME(arcLength(fromPoints:closed:));

@end

NS_ASSUME_NONNULL_END
