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

#import "PDBPolygonOffsetBridge.h"
#import <UIKit/UIKit.h>
#include <cmath>
#include "clipper.hpp"
@implementation PDBPolygonOffsetBridge
+ (NSArray<NSArray<NSValue *> *> *)inflateClosedPolygonWith:(NSArray<NSValue *> *)points
                                                   distance:(double)distance {
    if (points.count < 2) {
        return @[];
    }
    using namespace ClipperLib;
    // Default miter limit and arc tolerance for ClipperOffset (round join).
    ClipperOffset co(2.0, 0.25);
    Path path;
    path.reserve(points.count);
    for (NSValue *v in points) {
        CGPoint p = v.CGPointValue;
        cInt ix = static_cast<cInt>(std::lround((double)p.x));
        cInt iy = static_cast<cInt>(std::lround((double)p.y));
        path.push_back(IntPoint(ix, iy));
    }
    co.AddPath(path, jtRound, etClosedPolygon);
    Paths solution;
    co.Execute(solution, distance);
    if (solution.empty()) {
        return @[];
    }
    const Path &out = solution[0];
    NSMutableArray *ring = [NSMutableArray arrayWithCapacity:out.size()];
    for (const IntPoint &pt : out) {
        [ring addObject:[NSValue valueWithCGPoint:CGPointMake((CGFloat)pt.X, (CGFloat)pt.Y)]];
    }
    return @[ ring ];
}
@end
