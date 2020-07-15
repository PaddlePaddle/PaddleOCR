//
//  Helpers.h
//  EasyDLDemo
//
//  Created by chenxiaoyu on 2018/5/14.
//  Copyright © 2018年 baidu. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIImage.h>

#define UIColorFromRGB(rgbValue) \
[UIColor colorWithRed:((float)((rgbValue & 0xFF0000) >> 16))/255.0 \
green:((float)((rgbValue & 0x00FF00) >>  8))/255.0 \
blue:((float)((rgbValue & 0x0000FF) >>  0))/255.0 \
alpha:1.0]

#define SCREEN_HEIGHT  [UIScreen mainScreen].bounds.size.height
#define SCREEN_WIDTH  [UIScreen mainScreen].bounds.size.width

#define HIGHLIGHT_COLOR UIColorFromRGB(0xF5A623)

//#define BTN_HIGHTLIGH_TEXT_COLOR UIColorFromRGB(0xF5A623)


@interface Helpers : NSObject {


}

@end
