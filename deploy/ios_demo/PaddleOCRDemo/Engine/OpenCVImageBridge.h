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

@interface PDBPerspectiveCropOutput : NSObject
@property(nonatomic) NSData *rgbData;
@property(nonatomic) NSInteger width;
@property(nonatomic) NSInteger height;
@end

@interface PDBOpenCVImageBridge : NSObject

/// `cv::resize(..., INTER_LINEAR)` on 8-bit **3-channel** row-major data
/// (`CV_8UC3`; stores BGR channel order for OpenCV).
+ (NSData *)resizeRGBU8:(NSData *)data
               srcWidth:(NSInteger)srcWidth
              srcHeight:(NSInteger)srcHeight
               dstWidth:(NSInteger)dstWidth
              dstHeight:(NSInteger)dstHeight NS_SWIFT_NAME(resizeRGBU8(_:srcWidth:srcHeight:dstWidth:dstHeight:));

/// Pad RGB image to `padWidth` x `padHeight` with zeros on bottom/right.
+ (NSData *)padRGBU8:(NSData *)data
               width:(NSInteger)width
              height:(NSInteger)height
            padWidth:(NSInteger)padWidth
           padHeight:(NSInteger)padHeight
    NS_SWIFT_NAME(padRGBU8(_:width:height:padWidth:padHeight:));

/// Minimum-area rectangle, corner ordering, perspective warp (`INTER_CUBIC`,
/// `BORDER_REPLICATE`), then 90° CCW rotation when height/width ≥ 1.5. Input
/// buffer is row-major **BGR**.
+ (nullable PDBPerspectiveCropOutput *)
    quadTextLineCropBGR:(NSData *)src
               srcWidth:(NSInteger)srcWidth
              srcHeight:(NSInteger)srcHeight
                   quad:(NSArray<NSValue *> *)quad
    NS_SWIFT_NAME(quadTextLineCropBGR(_:srcWidth:srcHeight:quad:));

/// Mean of the float probability map inside the quadrilateral: `fillPoly` mask
/// then `cv::mean`.
+ (float)meanPredInQuad:(NSData *)pred
                  width:(NSInteger)width
                 height:(NSInteger)height
                    box:(NSArray<NSArray<NSNumber *> *> *)box
    NS_SWIFT_NAME(meanPredInQuad(_:width:height:box:));

@end

NS_ASSUME_NONNULL_END
