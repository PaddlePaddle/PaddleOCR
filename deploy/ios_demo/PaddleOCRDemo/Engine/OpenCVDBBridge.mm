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

#import "OpenCVDBBridge.h"
// Include imgproc before UIKit: Objective-C's `NO` macro breaks OpenCV enums named `NO` in opencv.hpp.
#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>
#import <UIKit/UIKit.h>
#include <algorithm>

@implementation PDBMiniBoxResult
@end

@implementation PDBOpenCVDBBridge

+ (NSArray<NSArray<NSValue *> *> *)findContoursWithBinaryMask:(NSData *)data
                                                         width:(NSInteger)width
                                                        height:(NSInteger)height
                                                 maxCandidates:(NSInteger)maxCandidates {
    if (width <= 0 || height <= 0 || data.length < (NSUInteger)(width * height)) {
        return @[];
    }

    const uint8_t *src = static_cast<const uint8_t *>(data.bytes);
    cv::Mat mat((int)height, (int)width, CV_8UC1);
    for (int y = 0; y < (int)height; y++) {
        uint8_t *row = mat.ptr<uint8_t>(y);
        for (int x = 0; x < (int)width; x++) {
            row[x] = src[y * (int)width + x] ? 255 : 0;
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const NSInteger n = std::min((NSInteger)contours.size(), maxCandidates);
    NSMutableArray *out = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
    for (NSInteger i = 0; i < n; i++) {
        const auto &c = contours[(size_t)i];
        NSMutableArray *pts = [NSMutableArray arrayWithCapacity:c.size()];
        for (const auto &pt : c) {
            [pts addObject:[NSValue valueWithCGPoint:CGPointMake(pt.x, pt.y)]];
        }
        [out addObject:pts];
    }
    return out;
}

+ (nullable PDBMiniBoxResult *)miniBoxFromPoints:(NSArray<NSValue *> *)points {
    if (points.count < 2) {
        return nil;
    }

    std::vector<cv::Point2f> pts;
    pts.reserve(points.count);
    for (NSValue *v in points) {
        CGPoint p = v.CGPointValue;
        pts.emplace_back((float)p.x, (float)p.y);
    }

    cv::RotatedRect rr = cv::minAreaRect(pts);
    float w = rr.size.width;
    float h = rr.size.height;
    if (w <= 0 && h <= 0) {
        return nil;
    }

    cv::Point2f cvPts[4];
    rr.points(cvPts);

    // Stable sort by x only (min-area box corner order).
    int ord[4] = {0, 1, 2, 3};
    std::stable_sort(ord, ord + 4, [&](int a, int b) { return cvPts[a].x < cvPts[b].x; });
    cv::Point2f sortedPts[4];
    for (int i = 0; i < 4; i++) {
        sortedPts[i] = cvPts[ord[i]];
    }

    NSInteger index1, index2, index3, index4;
    cv::Point2f p0 = sortedPts[0];
    cv::Point2f p1 = sortedPts[1];
    if (p1.y > p0.y) {
        index1 = 0;
        index4 = 1;
    } else {
        index1 = 1;
        index4 = 0;
    }
    cv::Point2f p2 = sortedPts[2];
    cv::Point2f p3 = sortedPts[3];
    if (p3.y > p2.y) {
        index2 = 2;
        index3 = 3;
    } else {
        index2 = 3;
        index3 = 2;
    }

    PDBMiniBoxResult *res = [[PDBMiniBoxResult alloc] init];
    res.corners = @[
        [NSValue valueWithCGPoint:CGPointMake(sortedPts[index1].x, sortedPts[index1].y)],
        [NSValue valueWithCGPoint:CGPointMake(sortedPts[index2].x, sortedPts[index2].y)],
        [NSValue valueWithCGPoint:CGPointMake(sortedPts[index3].x, sortedPts[index3].y)],
        [NSValue valueWithCGPoint:CGPointMake(sortedPts[index4].x, sortedPts[index4].y)],
    ];
    res.minSide = std::min(w, h);
    return res;
}

+ (float)contourAreaFromPoints:(NSArray<NSValue *> *)points {
    if (points.count < 2) {
        return 0;
    }
    std::vector<cv::Point2f> pts;
    pts.reserve(points.count);
    for (NSValue *v in points) {
        CGPoint p = v.CGPointValue;
        pts.emplace_back((float)p.x, (float)p.y);
    }
    return cv::contourArea(pts);
}

+ (float)arcLengthFromPoints:(NSArray<NSValue *> *)points closed:(BOOL)closed {
    if (points.count < 2) {
        return 0;
    }
    std::vector<cv::Point2f> pts;
    pts.reserve(points.count);
    for (NSValue *v in points) {
        CGPoint p = v.CGPointValue;
        pts.emplace_back((float)p.x, (float)p.y);
    }
    return (float)cv::arcLength(pts, closed);
}

@end
