//
// Created by chenxiaoyu on 2018/5/5.
// Copyright (c) 2018 baidu. All rights reserved.
//

#include "BoxLayer.h"
#import "Helpers.h"

@implementation BoxLayer {

}

#define MAIN_COLOR UIColorFromRGB(0x3B85F5)
- (void)renderOcrPolygon:(OcrData *)d withHeight:(CGFloat)originHeight withWidth:(CGFloat)originWidth withLabel:(bool)withLabel {

    if ([d.polygonPoints count] != 4) {
        NSLog(@"poloygonPoints size is not 4");
        return;
    }

    CGPoint startPoint = [d.polygonPoints[0] CGPointValue];
    NSString *text = d.label;

    CGFloat x = startPoint.x * originWidth;
    CGFloat y = startPoint.y * originHeight;
    CGFloat width = originWidth - x;
    CGFloat height = originHeight - y;


    UIFont *font = [UIFont systemFontOfSize:16];
    NSDictionary *attrs = @{
//            NSStrokeColorAttributeName: [UIColor blackColor],
            NSForegroundColorAttributeName: [UIColor whiteColor],
//            NSStrokeWidthAttributeName : @((float) -6.0),
            NSFontAttributeName: font
    };


    if (withLabel) {
        NSAttributedString *displayStr = [[NSAttributedString alloc] initWithString:text attributes:attrs];
        CATextLayer *textLayer = [[CATextLayer alloc] init];
        textLayer.wrapped = YES;
        textLayer.string = displayStr;
        textLayer.frame = CGRectMake(x + 2, y + 2, width, height);
        textLayer.contentsScale = [[UIScreen mainScreen] scale];

        // 加阴影显得有点乱
//    textLayer.shadowColor = [MAIN_COLOR CGColor];
//    textLayer.shadowOffset = CGSizeMake(2.0, 2.0);
//    textLayer.shadowOpacity = 0.8;
//    textLayer.shadowRadius = 0.0;

        [self addSublayer:textLayer];
    }


    UIBezierPath *path = [UIBezierPath new];


    [path moveToPoint:CGPointMake(startPoint.x * originWidth, startPoint.y * originHeight)];
    for (NSValue *val in d.polygonPoints) {
        CGPoint p = [val CGPointValue];
        [path addLineToPoint:CGPointMake(p.x * originWidth, p.y * originHeight)];
    }
    [path closePath];

    self.path = path.CGPath;
    self.strokeColor = MAIN_COLOR.CGColor;
    self.lineWidth = 2.0;
    self.fillColor = [MAIN_COLOR colorWithAlphaComponent:0.2].CGColor;
    self.lineJoin = kCALineJoinBevel;

}

- (void)renderSingleBox:(OcrData *)data withHeight:(CGFloat)originHeight withWidth:(CGFloat)originWidth {
    [self renderOcrPolygon:data withHeight:originHeight withWidth:originWidth withLabel:YES];
}


@end
