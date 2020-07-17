//
// Created by chenxiaoyu on 2018/5/5.
// Copyright (c) 2018 baidu. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "OcrData.h"


@interface BoxLayer : CAShapeLayer

/**
 * 绘制OCR的结果
 */
-(void) renderOcrPolygon: (OcrData *)data withHeight:(CGFloat)originHeight withWidth:(CGFloat)originWidth withLabel:(bool) withLabel;



@end
