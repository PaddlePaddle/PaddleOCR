from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.

ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory
#img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
img_path = './img/0048.jpg'
result = ocr.ocr(img_path, cls=True)

# print("---------result------------")
# print(type(result))
# print(result)
# print("---------result------------")

for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
print("---------------------------")
print(boxes)
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
#im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result_0048.jpg')


# save to file
righttop_location = [topright[2] for topright in boxes]
# right to left order [right][top]
righttop_order = [sorted(righttop_location,reverse=True)]

topdown_location = [topdown[1] for topdown in righttop_order]
# top to down order [top]
topdown_order = [sorted(topdown_location,reverse=True)]

box_order = []
box_x = []
box_y = []
box_ylist = []
print("-----------righttop_order----------------")
print(righttop_order)
# print("------------ori_box---------------")
# print(box_x)
# print(righttop_order) 
# print("-----------topdown_order----------------")
# print(topdown_order)

    
# a = [[[783.0, 894.0], [750.0, 899.0], [646.0, 894.0],[783.0, 892.0], [749.0, 894.0],  [608.0, 376.0], [575.0, 321.0], [535.0, 241.0]]]
# print("-----------a----------------")
# print(sorted(a,reverse=True))

# for onerighttop in righttop_order[0]:
#     if not(onerighttop[0] in box_order):
#         box_x.append(onerighttop[0])
#         box_y.append(onerighttop[1])
#     else:
#         box_ylist = []
#         for onexidx in box_x:
#             if onexidx == onerighttop[0]:
#                 box_ylist.append(box_y[onexidx])
#         box_y.remove(onerighttop[1])
#         box_y.append(box_ylist)

# print("------------box_x---------------")
# print(len(box_x))
# print(box_x)
# print("------------box_y---------------")
# print(len(box_y))
# print(box_y)
print("================righttop_order===================")
print(righttop_order)

# 搭配 with 寫入檔案
with open("output.txt", "w") as f:
  # get each box righttop(x, y)
  for righttop in righttop_order[0]:
     print("================righttop===================")
     print(righttop)

    # boxes = [line[0] for line in result]
    # print("---------------------------")
    # print(boxes)
    # txts = [line[1][0] for line in result]
     for line in result:
        print("----------line-----------------")
        print(line)
        for line[0][2] in line[0]:
            print("----------line[0]-----------------")
            print(line[0][2])
        # print(line[0][2][1][0])
        # if line[0][2][2] == righttop:
        #     print(line[1][0])
           
        # print("================topright[2]===================")
        #    print("================type(line[1][0]===================")
        #    print(line[1][0])
            # f.write(line[1][0])
            # break