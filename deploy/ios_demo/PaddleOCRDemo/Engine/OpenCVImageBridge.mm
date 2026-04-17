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

#import "OpenCVImageBridge.h"
#import <algorithm>
#import <cfloat>
#import <cmath>
#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>
#import <UIKit/UIKit.h>

@implementation PDBPerspectiveCropOutput
@end

@implementation PDBOpenCVImageBridge

+ (NSData *)resizeRGBU8:(NSData *)data
               srcWidth:(NSInteger)srcWidth
              srcHeight:(NSInteger)srcHeight
               dstWidth:(NSInteger)dstWidth
              dstHeight:(NSInteger)dstHeight {
    if (srcWidth <= 0 || srcHeight <= 0 || dstWidth <= 0 || dstHeight <= 0) {
        return [NSData data];
    }
    if (data.length < (NSUInteger)(srcWidth * srcHeight * 3)) {
        return [NSData data];
    }
    cv::Mat src((int)srcHeight, (int)srcWidth, CV_8UC3, (void *)data.bytes);
    cv::Mat dst;
    cv::resize(src, dst, cv::Size((int)dstWidth, (int)dstHeight), 0, 0, cv::INTER_LINEAR);
    return [NSData dataWithBytes:dst.data length:(NSUInteger)(dst.rows * dst.cols * 3)];
}

+ (NSData *)padRGBU8:(NSData *)data
               width:(NSInteger)width
              height:(NSInteger)height
            padWidth:(NSInteger)padWidth
           padHeight:(NSInteger)padHeight {
    if (width <= 0 || height <= 0 || padWidth < width || padHeight < height) {
        return data ?: [NSData data];
    }
    if (data.length < (NSUInteger)(width * height * 3)) {
        return [NSData data];
    }
    cv::Mat src((int)height, (int)width, CV_8UC3, (void *)data.bytes);
    int bot = (int)(padHeight - height);
    int right = (int)(padWidth - width);
    cv::Mat dst;
    cv::copyMakeBorder(src, dst, 0, bot, 0, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return [NSData dataWithBytes:dst.data length:(NSUInteger)(dst.rows * dst.cols * 3)];
}

+ (nullable PDBPerspectiveCropOutput *)quadTextLineCropBGR:(NSData *)src
                                                 srcWidth:(NSInteger)srcWidth
                                                srcHeight:(NSInteger)srcHeight
                                                     quad:(NSArray<NSValue *> *)quad {
    if (srcWidth <= 0 || srcHeight <= 0) {
        return nil;
    }
    if (quad.count != 4 || src.length < (NSUInteger)(srcWidth * srcHeight * 3)) {
        return nil;
    }

    cv::Mat im((int)srcHeight, (int)srcWidth, CV_8UC3, (void *)src.bytes);

    std::vector<cv::Point2f> raw(4);
    for (int i = 0; i < 4; i++) {
        CGPoint p = [quad[i] CGPointValue];
        raw[i] = cv::Point2f((float)(int32_t)p.x, (float)(int32_t)p.y);
    }

    cv::RotatedRect rr = cv::minAreaRect(raw);
    cv::Point2f boxPts[4];
    rr.points(boxPts);
    std::vector<cv::Point2f> sorted(boxPts, boxPts + 4);
    std::sort(sorted.begin(), sorted.end(),
              [](const cv::Point2f &a, const cv::Point2f &b) { return a.x < b.x; });

    int index_a, index_b, index_c, index_d;
    if (sorted[1].y > sorted[0].y) {
        index_a = 0;
        index_d = 1;
    } else {
        index_a = 1;
        index_d = 0;
    }
    if (sorted[3].y > sorted[2].y) {
        index_b = 2;
        index_c = 3;
    } else {
        index_b = 3;
        index_c = 2;
    }

    cv::Point2f srcPts[4] = {sorted[index_a], sorted[index_b], sorted[index_c], sorted[index_d]};

    double w1 = cv::norm(srcPts[0] - srcPts[1]);
    double w2 = cv::norm(srcPts[2] - srcPts[3]);
    double h1 = cv::norm(srcPts[0] - srcPts[3]);
    double h2 = cv::norm(srcPts[1] - srcPts[2]);

    int img_crop_width = (int)std::max(w1, w2);
    int img_crop_height = (int)std::max(h1, h2);
    if (img_crop_width <= 0 || img_crop_height <= 0) {
        return nil;
    }

    cv::Point2f dstPts[4] = {
        cv::Point2f(0, 0),
        cv::Point2f((float)img_crop_width, 0),
        cv::Point2f((float)img_crop_width, (float)img_crop_height),
        cv::Point2f(0, (float)img_crop_height),
    };

    cv::Mat M = cv::getPerspectiveTransform(srcPts, dstPts);
    cv::Mat warped;
    cv::warpPerspective(im, warped, M, cv::Size(img_crop_width, img_crop_height), cv::INTER_CUBIC,
                          cv::BORDER_REPLICATE);

    cv::Mat outMat;
    double ratio = (double)warped.rows / (double)warped.cols;
    if (ratio >= 1.5) {
        cv::rotate(warped, outMat, cv::ROTATE_90_COUNTERCLOCKWISE);
    } else {
        outMat = warped;
    }

    PDBPerspectiveCropOutput *result = [[PDBPerspectiveCropOutput alloc] init];
    result.rgbData = [NSData dataWithBytes:outMat.data length:(NSUInteger)(outMat.rows * outMat.cols * 3)];
    result.width = outMat.cols;
    result.height = outMat.rows;
    return result;
}

+ (float)meanPredInQuad:(NSData *)pred
                  width:(NSInteger)width
                 height:(NSInteger)height
                    box:(NSArray<NSArray<NSNumber *> *> *)box {
    if (width <= 0 || height <= 0 || box.count != 4) {
        return 0;
    }
    size_t need = (size_t)(width * height) * sizeof(float);
    if (pred.length < need) {
        return 0;
    }

    float xminF = FLT_MAX, xmaxF = -FLT_MAX, yminF = FLT_MAX, ymaxF = -FLT_MAX;
    for (NSArray<NSNumber *> *pt in box) {
        if (pt.count < 2) {
            return 0;
        }
        float x = [pt[0] floatValue];
        float y = [pt[1] floatValue];
        xminF = std::min(xminF, x);
        xmaxF = std::max(xmaxF, x);
        yminF = std::min(yminF, y);
        ymaxF = std::max(ymaxF, y);
    }

    int w = (int)width;
    int h = (int)height;
    // Clamp ROI to bitmap bounds before masking and mean (avoids out-of-range reads).
    int xmin = std::max(0, std::min((int)std::floor(xminF), w - 1));
    int xmax = std::max(0, std::min((int)std::ceil(xmaxF), w - 1));
    int ymin = std::max(0, std::min((int)std::floor(yminF), h - 1));
    int ymax = std::max(0, std::min((int)std::ceil(ymaxF), h - 1));
    if (xmax < xmin || ymax < ymin) {
        return 0;
    }

    const float *p = (const float *)pred.bytes;
    cv::Mat bitmap(h, w, CV_32FC1, (void *)const_cast<float *>(p));
    cv::Rect roi(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
    cv::Mat crop = bitmap(roi);

    cv::Mat mask = cv::Mat::zeros(crop.rows, crop.cols, CV_8UC1);
    std::vector<cv::Point> pts;
    pts.reserve(4);
    for (NSArray<NSNumber *> *pt in box) {
        float fx = [pt[0] floatValue] - static_cast<float>(xmin);
        float fy = [pt[1] floatValue] - static_cast<float>(ymin);
        int px = static_cast<int>(fx);
        int py = static_cast<int>(fy);
        pts.push_back(cv::Point(px, py));
    }
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(pts);
    cv::fillPoly(mask, contours, cv::Scalar(1));

    cv::Scalar m = cv::mean(crop, mask);
    return (float)m[0];
}

@end
