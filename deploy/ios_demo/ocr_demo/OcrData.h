//
// Created by Lv,Xiangxiang on 2020/7/11.
// Copyright (c) 2020 Li,Xiaoyang(SYS). All rights reserved.
//

#import <Foundation/Foundation.h>


@interface OcrData : NSObject
@property(nonatomic, copy) NSString *label;
@property(nonatomic) int category;
@property(nonatomic) float accuracy;
@property(nonatomic) NSArray *polygonPoints;

@end