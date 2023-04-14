import paddle
from PIL import Image, ImageOps
import numpy as np
import json
import os
import cv2


def load_coco_text_anno(json_file):
    def XYWH2XYXY(box):
        x, y, w, h = box
        return np.array([x, y, x + w, y + h])

    if not os.path.exists(json_file):
        raise ValueError(f"{json_file} does not exists!")

    _mode = os.path.basename(os.path.abspath(
        json_file))[:-5]  # 'test' or 'train'
    root_path = os.path.join(
        os.path.dirname(os.path.abspath(json_file)), f"{_mode}_images")

    data = json.load(open(json_file, 'r'))
    data_gt = {}
    for dic in data['images']:
        data_gt[dic['id']] = {}
        data_gt[dic['id']]['im_file'] = os.path.join(root_path,
                                                     dic['file_name'])
        data_gt[dic['id']]['id'] = dic['id']
        data_gt[dic['id']]['height'] = dic['height']
        data_gt[dic['id']]['width'] = dic['width']
        data_gt[dic['id']]['gt_bbox'] = []
        data_gt[dic['id']]['rec'] = []
        data_gt[dic['id']]['bezier_pts'] = []
        data_gt[dic['id']]['category_id'] = []
        data_gt[dic['id']]['iscrowd'] = []

    for label in data['annotations']:
        data_gt[label['image_id']]['gt_bbox'].append(XYWH2XYXY(label['bbox']))
        data_gt[label['image_id']]['rec'].append(label['rec'])
        data_gt[label['image_id']]['bezier_pts'].append(
            np.array(label['bezier_pts']))
        data_gt[label['image_id']]['iscrowd'].append(label['iscrowd'])
        data_gt[label['image_id']]['category_id'].append(label['category_id'])

    for k in data_gt.keys():
        data_gt[k]['gt_bbox'] = np.array(data_gt[k]['gt_bbox']).astype(
            np.float32)
        data_gt[k]['bezier_pts'] = np.array(data_gt[k]['bezier_pts']).astype(
            np.float32)

    return data_gt


def test_paddle_berizer(scale=1):
    test_anno = load_coco_text_anno("/paddle/Datasets/totaltext/test.json")
    k = list(test_anno.keys())[2]
    data = test_anno[k]

    output_size = (48, 320)

    bezier_pts = data['bezier_pts']
    bbox = data['gt_bbox']
    img_file = data['im_file']
    print("img_file, ", img_file)
    img = cv2.imread(img_file)

    for i in range(bezier_pts.shape[0]):
        bez = np.array(bezier_pts[0]).reshape([-1, 2]).astype(np.int32)
        cv2.polylines(img, [bez], True, color=(0, 255, 0), thickness=2)

    cv2.imwrite("res_arr_hwc_src.png", img)

    bezier_layer = paddle.nn.BezierAlign(output_size, 1, 1)
    beziers_pd = np.array(bezier_pts).astype(np.float32)
    print("berzier_pd", beziers_pd)
    beziers_pd = paddle.to_tensor(beziers_pd, stop_gradient=False)
    bezier_nums = paddle.to_tensor(
        np.array([bezier_pts.shape[0]]), stop_gradient=False).astype(np.int32)

    img_ins = np.expand_dims(np.array(img), axis=0)
    print(img_ins.shape)
    try:
        im_arrs_pd = img_ins.transpose(0, 3, 1, 2).astype("float32")
    except:
        im_arrs_pd = np.transpose(img_ins, [0, 2, 2, 1]).astype(np.float32)

    ins = paddle.to_tensor(im_arrs_pd, stop_gradient=False)
    res = bezier_layer(ins, beziers_pd, bezier_nums)
    res_arr = res.numpy()
    print(res_arr.shape)
    res_arr_hwc = np.transpose(res_arr, [0, 2, 3, 1])[0, :, :, :]
    cv2.imwrite("res_arr_hwc.png", res_arr_hwc.astype(np.uint8))


def test_paddle_torch_bezier(scale=1):
    image_size = (2560, 2560)  #(2560, 2560)  # H x W
    output_size = (32, 100)

    input_size = (image_size[0] // scale, image_size[1] // scale)

    beziers = [[]]
    im_arrs = []
    down_scales = []

    imgfile = '/paddle/abcnet/abcnet_pd/bezier/1019.jpg'
    im = Image.open(imgfile)
    # im.show()
    # pad
    w, h = im.size
    down_scale = get_size(image_size, w, h)
    down_scales.append(down_scale)
    if down_scale > 1:
        im = im.resize((int(w / down_scale), int(h / down_scale)),
                       Image.ANTIALIAS)
        w, h = im.size
    padding = (0, 0, image_size[1] - w, image_size[0] - h)
    im = ImageOps.expand(im, padding)
    im = im.resize((input_size[1], input_size[0]), Image.ANTIALIAS)
    im_arrs.append(np.array(im))

    cps = [
        152.0, 209.0, 134.1, 34.18, 365.69, 66.2, 377.0, 206.0, 345.0, 214.0,
        334.31, 109.71, 190.03, 80.12, 203.0, 214.0
    ]  # 1019

    cps = np.array(cps)[[1, 0, 3, 2, 5, 4, 7, 6, 15, 14, 13, 12, 11, 10, 9, 8]]
    beziers[0].append(cps)

    # paddle predict 
    import paddle
    #paddle.device.set_device("cpu")
    bezier_layer = paddle.nn.BezierAlign(output_size, 1 / scale, 1)
    beziers_pd = [np.array(cps).astype(np.float32)]
    beziers_pd = paddle.to_tensor(beziers_pd, stop_gradient=False)
    print("paddle beziers: ", beziers_pd, beziers_pd.shape)
    bezier_nums = paddle.to_tensor(
        np.array([1]), stop_gradient=False).astype(np.int32)

    im_arrs_pd = im_arrs.transpose(0, 3, 1, 2).astype("float32")
    ins = paddle.to_tensor(im_arrs_pd, stop_gradient=False)

    res = bezier_layer(ins, beziers_pd, bezier_nums)
    loss = paddle.mean(res)
    loss.backward()

    print(res.shape)
    print("results: ", res)
    print(ins.gradient())

    np.testing.assert_allclose(
        th_x.detach().cpu().numpy(), res.numpy(), rtol=1e-7, atol=1e-7)


test_paddle_berizer()
