import os
import numpy as np
import cv2
import time


def visualize_e2e_result(im_fn, poly_list, seq_strs, src_im):
    """
    """
    result_path = './out'
    im_basename = os.path.basename(im_fn)
    im_prefix = im_basename[:im_basename.rfind('.')]
    vis_det_img = src_im.copy()
    valid_set = 'partvgg'
    gt_dir = "/Users/hongyongjie/Downloads/part_vgg_synth/train"
    text_path = os.path.join(gt_dir, im_prefix + '.txt')
    fid = open(text_path, 'r')
    lines = [line.strip() for line in fid.readlines()]
    for line in lines:
        if valid_set == 'partvgg':
            tokens = line.strip().split('\t')[0].split(',')
            # tokens = line.strip().split(',')
            coords = tokens[:]
            coords = list(map(float, coords))
            gt_poly = np.array(coords).reshape(1, 4, 2)
        elif valid_set == 'totaltext':
            tokens = line.strip().split('\t')[0].split(',')
            coords = tokens[:]
            coords_len = len(coords) / 2
            coords = list(map(float, coords))
            gt_poly = np.array(coords).reshape(1, coords_len, 2)
        cv2.polylines(
            vis_det_img,
            np.array(gt_poly).astype(np.int32),
            isClosed=True,
            color=(255, 0, 0),
            thickness=2)

    for detected_poly, recognized_str in zip(poly_list, seq_strs):
        cv2.polylines(
            vis_det_img,
            np.array(detected_poly[np.newaxis, ...]).astype(np.int32),
            isClosed=True,
            color=(0, 0, 255),
            thickness=2)
        cv2.putText(
            vis_det_img,
            recognized_str,
            org=(int(detected_poly[0, 0]), int(detected_poly[0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    cv2.imwrite("{}/{}_detection.jpg".format(result_path, im_prefix),
                vis_det_img)


def visualization_output(src_image,
                         f_tcl,
                         f_chars,
                         output_dir,
                         image_prefix=None):
    """
    """
    # restore BGR image, CHW -> HWC
    im_mean = [0.485, 0.456, 0.406]
    im_std = [0.229, 0.224, 0.225]

    im_mean = np.array(im_mean).reshape((3, 1, 1))
    im_std = np.array(im_std).reshape((3, 1, 1))
    src_image *= im_std
    src_image += im_mean
    src_image = src_image.transpose([1, 2, 0])
    src_image = src_image[:, :, ::-1] * 255  # BGR -> RGB
    H, W, _ = src_image.shape

    file_prefix = image_prefix if image_prefix is not None else str(
        int(time.time() * 1000))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # visualization f_tcl
    tcl_file_name = os.path.join(output_dir, file_prefix + '_0_tcl.jpg')
    vis_tcl_img = src_image.copy()
    f_tcl_resized = cv2.resize(f_tcl, dsize=(W, H))
    vis_tcl_img[:, :, 1] = f_tcl_resized * 255
    cv2.imwrite(tcl_file_name, vis_tcl_img)

    # visualization char maps
    vis_char_img = src_image.copy()
    # CHW -> HWC
    char_file_name = os.path.join(output_dir, file_prefix + '_1_chars.jpg')
    f_chars = np.argmax(f_chars, axis=2)[:, :, np.newaxis].astype('float32')
    f_chars[f_chars < 95] = 1.0
    f_chars[f_chars == 95] = 0.0
    f_chars_resized = cv2.resize(f_chars, dsize=(W, H))
    vis_char_img[:, :, 1] = f_chars_resized * 255
    cv2.imwrite(char_file_name, vis_char_img)


def visualize_point_result(im_fn, point_list, point_pair_list, src_im, gt_dir,
                           result_path):
    """
    """
    im_basename = os.path.basename(im_fn)
    im_prefix = im_basename[:im_basename.rfind('.')]
    vis_det_img = src_im.copy()

    # draw gt bbox on the image.
    text_path = os.path.join(gt_dir, im_prefix + '.txt')
    fid = open(text_path, 'r')
    lines = [line.strip() for line in fid.readlines()]
    for line in lines:
        tokens = line.strip().split('\t')
        coords = tokens[0].split(',')
        coords_len = len(coords)
        coords = list(map(float, coords))
        gt_poly = np.array(coords).reshape(1, coords_len / 2, 2)
        cv2.polylines(
            vis_det_img,
            np.array(gt_poly).astype(np.int32),
            isClosed=True,
            color=(255, 255, 255),
            thickness=1)

    for point, point_pair in zip(point_list, point_pair_list):
        cv2.line(
            vis_det_img,
            tuple(point_pair[0]),
            tuple(point_pair[1]), (0, 255, 255),
            thickness=1)
        cv2.circle(vis_det_img, tuple(point), 2, (0, 0, 255))
        cv2.circle(vis_det_img, tuple(point_pair[0]), 2, (255, 0, 0))
        cv2.circle(vis_det_img, tuple(point_pair[1]), 2, (0, 255, 0))

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    cv2.imwrite("{}/{}_border_points.jpg".format(result_path, im_prefix),
                vis_det_img)


def resize_image(im, max_side_len=512):
    """
    resize image to a size multiple of max_stride which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # Fix the longer side
    if resize_h > resize_w:
        ratio = float(max_side_len) / resize_h
    else:
        ratio = float(max_side_len) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def resize_image_min(im, max_side_len=512):
    """
    """
    print('--> Using resize_image_min')
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # Fix the longer side
    if resize_h < resize_w:
        ratio = float(max_side_len) / resize_h
    else:
        ratio = float(max_side_len) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)


def resize_image_for_totaltext(im, max_side_len=512):
    """
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h
    ratio = 1.25
    if h * ratio > max_side_len:
        ratio = float(max_side_len) / resize_h
        # Fix the longer side
        # if resize_h > resize_w:
        #    ratio = float(max_side_len) / resize_h
        # else:
        #    ratio = float(max_side_len) / resize_w
    ###
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)


def point_pair2poly(point_pair_list):
    """
    Transfer vertical point_pairs into poly point in clockwise.
    """
    pair_length_list = []
    for point_pair in point_pair_list:
        pair_length = np.linalg.norm(point_pair[0] - point_pair[1])
        pair_length_list.append(pair_length)
    pair_length_list = np.array(pair_length_list)
    pair_info = (pair_length_list.max(), pair_length_list.min(),
                 pair_length_list.mean())

    # constract poly
    point_num = len(point_pair_list) * 2
    point_list = [0] * point_num
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2), pair_info


def shrink_quad_along_width(quad, begin_width_ratio=0., end_width_ratio=1.):
    """
    Generate shrink_quad_along_width.
    """
    ratio_pair = np.array(
        [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
    p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
    p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
    return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])


def expand_poly_along_width(poly, shrink_ratio_of_width=0.3):
    """
    expand poly along width.
    """
    point_num = poly.shape[0]
    left_quad = np.array(
        [poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
    left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                 (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    right_quad = np.array(
        [
            poly[point_num // 2 - 2], poly[point_num // 2 - 1],
            poly[point_num // 2], poly[point_num // 2 + 1]
        ],
        dtype=np.float32)
    right_ratio = 1.0 + \
                  shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                  (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    poly[0] = left_quad_expand[0]
    poly[-1] = left_quad_expand[-1]
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2] = right_quad_expand[2]
    return poly


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x**2, axis=axis))
    return np.sqrt(np.sum(x**2))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))


def generate_direction_info(image_fn,
                            H,
                            W,
                            ratio_h,
                            ratio_w,
                            max_length=640,
                            out_scale=4,
                            gt_dir=None):
    """
    """
    im_basename = os.path.basename(image_fn)
    im_prefix = im_basename[:im_basename.rfind('.')]
    instance_direction_map = np.zeros(shape=[H // out_scale, W // out_scale, 3])

    if gt_dir is None:
        gt_dir = '/home/vis/huangzuming/data/SYNTH_DATA/part_vgg_synth_icdar/processed/val/poly'

    # get gt label map
    text_path = os.path.join(gt_dir, im_prefix + '.txt')
    fid = open(text_path, 'r')
    lines = [line.strip() for line in fid.readlines()]
    for label_idx, line in enumerate(lines, start=1):
        coords, txt = line.strip().split('\t')
        if txt == '###':
            continue
        tokens = coords.strip().split(',')
        coords = list(map(float, tokens))
        poly = np.array(coords).reshape(4, 2) * np.array(
            [ratio_w, ratio_h]).reshape(1, 2) / out_scale
        mid_idx = poly.shape[0] // 2
        direct_vector = (
            (poly[mid_idx] + poly[mid_idx - 1]) - (poly[0] + poly[-1])) / 2.0

        direct_vector /= len(txt)
        # l2_distance = norm2(direct_vector)
        # avg_char_distance = l2_distance / len(txt)
        avg_char_distance = 1.0

        direct_label = (direct_vector[0], direct_vector[1], avg_char_distance)
        cv2.fillPoly(instance_direction_map,
                     poly.round().astype(np.int32)[np.newaxis, :, :],
                     direct_label)
    instance_direction_map = instance_direction_map.transpose([2, 0, 1])
    return instance_direction_map[:2, ...]
