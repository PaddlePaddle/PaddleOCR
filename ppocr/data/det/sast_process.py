#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import math
import cv2
import numpy as np
import json


class SASTProcessTrain(object):
    """
    SAST process function for training
    """
    def __init__(self, params):
        self.img_set_dir = params['img_set_dir']
        self.min_crop_side_ratio = params['min_crop_side_ratio']
        self.min_crop_size = params['min_crop_size']
        image_shape = params['image_shape']
        self.input_size = image_shape[1]
        self.min_text_size = params['min_text_size']
        self.max_text_size = params['max_text_size']

    def convert_label_infor(self, label_infor):
        label_infor = label_infor.decode()
        label_infor = label_infor.encode('utf-8').decode('utf-8-sig')
        substr = label_infor.strip("\n").split("\t")
        img_path = self.img_set_dir + substr[0]
        label = json.loads(substr[1])
        nBox = len(label)
        wordBBs, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            wordBB = label[bno]['points']
            txt = label[bno]['transcription']
            wordBBs.append(wordBB)
            txts.append(txt)
            if txt == '###':
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        wordBBs = np.array(wordBBs, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)
        return img_path, wordBBs, txt_tags, txts

    def quad_area(self, poly):
        """
        compute area of a polygon
        :param poly:
        :return:
        """
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        return np.sum(edge) / 2.

    def gen_quad_from_poly(self, poly):
        """
        Generate min area quad from poly.
        """
        point_num = poly.shape[0]
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        if True:
            rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
            center_point = rect[0]
            box = np.array(cv2.boxPoints(rect))

            first_point_idx = 0
            min_dist = 1e4
            for i in range(4):
                dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                    np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                    np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                    np.linalg.norm(box[(i + 3) % 4] - poly[-1])
                if dist < min_dist:
                    min_dist = dist
                    first_point_idx = i
            for i in range(4):
                min_area_quad[i] = box[(first_point_idx + i) % 4]

        return min_area_quad

    def check_and_validate_polys(self, polys, tags, xxx_todo_changeme):
        """
        check so that the text poly is in the same direction,
        and also filter some invalid polygons
        :param polys:
        :param tags:
        :return:
        """
        (h, w) = xxx_todo_changeme
        if polys.shape[0] == 0:
            return polys, np.array([]), np.array([])
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

        validated_polys = []
        validated_tags = []
        hv_tags = []
        for poly, tag in zip(polys, tags):
            quad = self.gen_quad_from_poly(poly)
            p_area = self.quad_area(quad)
            if abs(p_area) < 1:
                print('invalid poly')
                continue
            if p_area > 0:
                if tag == False:
                    print('poly in wrong direction')
                    tag = True # reversed cases should be ignore
                poly = poly[(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1), :]
                quad = quad[(0, 3, 2, 1), :]

            len_w = np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])
            len_h = np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])
            hv_tag = 1
        
            if len_w * 2.0 <  len_h:
                hv_tag = 0

            validated_polys.append(poly)
            validated_tags.append(tag)
            hv_tags.append(hv_tag)
        return np.array(validated_polys), np.array(validated_tags), np.array(hv_tags)

    def crop_area(self, im, polys, tags, hv_tags, txts, crop_background=False, max_tries=25):
        """
        make random crop from the input image
        :param im:
        :param polys:
        :param tags:
        :param crop_background:
        :param max_tries: 50 -> 25
        :return:
        """
        h, w, _ = im.shape
        pad_h = h // 10
        pad_w = w // 10
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w: maxx + pad_w] = 1
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h: maxy + pad_h] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            return im, polys, tags, hv_tags, txts
        for i in range(max_tries):
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            # if xmax - xmin < ARGS.min_crop_side_ratio * w or \
            #   ymax - ymin < ARGS.min_crop_side_ratio * h:
            if xmax - xmin < self.min_crop_size or \
            ymax - ymin < self.min_crop_size:
                # area too small
                continue
            if polys.shape[0] != 0:
                poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                    & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
                selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polys = []
            if len(selected_polys) == 0:
                # no text in this area
                if crop_background:
                    txts_tmp = []
                    for selected_poly in selected_polys:
                        txts_tmp.append(txts[selected_poly])
                    txts = txts_tmp 
                    return im[ymin : ymax + 1, xmin : xmax + 1, :], \
                        polys[selected_polys], tags[selected_polys], hv_tags[selected_polys], txts
                else:
                    continue
            im = im[ymin: ymax + 1, xmin: xmax + 1, :]
            polys = polys[selected_polys]
            tags = tags[selected_polys]
            hv_tags = hv_tags[selected_polys]
            txts_tmp = []
            for selected_poly in selected_polys:
                txts_tmp.append(txts[selected_poly])
            txts = txts_tmp 
            polys[:, :, 0] -= xmin
            polys[:, :, 1] -= ymin
            return im, polys, tags, hv_tags, txts

        return im, polys, tags, hv_tags, txts

    def generate_direction_map(self, poly_quads, direction_map):
        """
        """
        width_list = []
        height_list = []
        for quad in poly_quads:
            quad_w = (np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[2] - quad[3])) / 2.0
            quad_h = (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[2] - quad[1])) / 2.0
            width_list.append(quad_w)
            height_list.append(quad_h)
        norm_width = max(sum(width_list) / (len(width_list) +  1e-6), 1.0)
        average_height = max(sum(height_list) / (len(height_list) + 1e-6), 1.0)

        for quad in poly_quads:
            direct_vector_full = ((quad[1] + quad[2]) - (quad[0] + quad[3])) / 2.0
            direct_vector = direct_vector_full / (np.linalg.norm(direct_vector_full) + 1e-6) * norm_width
            direction_label = tuple(map(float, [direct_vector[0], direct_vector[1], 1.0 / (average_height + 1e-6)]))
            cv2.fillPoly(direction_map, quad.round().astype(np.int32)[np.newaxis, :, :], direction_label)
        return direction_map

    def calculate_average_height(self, poly_quads):
        """
        """
        height_list = []
        for quad in poly_quads:
            quad_h = (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[2] - quad[1])) / 2.0
            height_list.append(quad_h)
        average_height = max(sum(height_list) / len(height_list), 1.0)
        return average_height

    def generate_tcl_label(self, hw, polys, tags, ds_ratio,
                            tcl_ratio=0.3, shrink_ratio_of_width=0.15):
        """
        Generate polygon.
        """
        h, w = hw
        h, w = int(h * ds_ratio), int(w * ds_ratio)
        polys = polys * ds_ratio

        score_map = np.zeros((h, w,), dtype=np.float32)
        tbo_map = np.zeros((h, w, 5), dtype=np.float32)
        training_mask = np.ones((h, w,), dtype=np.float32)
        direction_map = np.ones((h, w, 3)) * np.array([0, 0, 1]).reshape([1, 1, 3]).astype(np.float32)

        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0] 
            tag = poly_tag[1]

            # generate min_area_quad
            min_area_quad, center_point = self.gen_min_area_quad_from_poly(poly)
            min_area_quad_h = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[3]) +
                                    np.linalg.norm(min_area_quad[1] - min_area_quad[2]))
            min_area_quad_w = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[1]) +
                                    np.linalg.norm(min_area_quad[2] - min_area_quad[3]))

            if min(min_area_quad_h, min_area_quad_w) < self.min_text_size * ds_ratio \
                or min(min_area_quad_h, min_area_quad_w) > self.max_text_size * ds_ratio:
                continue

            if tag:
                # continue
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0.15)
            else:
                tcl_poly = self.poly2tcl(poly, tcl_ratio)
                tcl_quads = self.poly2quads(tcl_poly)
                poly_quads = self.poly2quads(poly)
                # stcl map
                stcl_quads, quad_index = self.shrink_poly_along_width(tcl_quads, shrink_ratio_of_width=shrink_ratio_of_width,
                                                                expand_height_ratio=1.0 / tcl_ratio)
                # generate tcl map
                cv2.fillPoly(score_map, np.round(stcl_quads).astype(np.int32), 1.0)

                # generate tbo map
                for idx, quad in enumerate(stcl_quads):
                    quad_mask = np.zeros((h, w), dtype=np.float32)
                    quad_mask = cv2.fillPoly(quad_mask, np.round(quad[np.newaxis, :, :]).astype(np.int32), 1.0)
                    tbo_map = self.gen_quad_tbo(poly_quads[quad_index[idx]], quad_mask, tbo_map)
        return score_map, tbo_map, training_mask

    def generate_tvo_and_tco(self, hw, polys, tags, tcl_ratio=0.3, ds_ratio=0.25):
        """
        Generate tcl map, tvo map and tbo map.
        """
        h, w = hw
        h, w = int(h * ds_ratio), int(w * ds_ratio)
        polys = polys * ds_ratio
        poly_mask = np.zeros((h, w), dtype=np.float32)

        tvo_map = np.ones((9, h, w), dtype=np.float32)
        tvo_map[0:-1:2] = np.tile(np.arange(0, w), (h, 1))
        tvo_map[1:-1:2] = np.tile(np.arange(0, w), (h, 1)).T
        poly_tv_xy_map = np.zeros((8, h, w), dtype=np.float32)

        # tco map
        tco_map = np.ones((3, h, w), dtype=np.float32)
        tco_map[0] = np.tile(np.arange(0, w), (h, 1))
        tco_map[1] = np.tile(np.arange(0, w), (h, 1)).T
        poly_tc_xy_map = np.zeros((2, h, w), dtype=np.float32)

        poly_short_edge_map = np.ones((h, w), dtype=np.float32)

        for poly, poly_tag in zip(polys, tags):

            if poly_tag == True:
                continue

            # adjust point order for vertical poly
            poly = self.adjust_point(poly)

            # generate min_area_quad
            min_area_quad, center_point = self.gen_min_area_quad_from_poly(poly)
            min_area_quad_h = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[3]) +
                                    np.linalg.norm(min_area_quad[1] - min_area_quad[2]))
            min_area_quad_w = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[1]) +
                                    np.linalg.norm(min_area_quad[2] - min_area_quad[3]))

            # generate tcl map and text, 128 * 128
            tcl_poly = self.poly2tcl(poly, tcl_ratio)

            # generate poly_tv_xy_map
            for idx in range(4):
                cv2.fillPoly(poly_tv_xy_map[2 * idx],
                            np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32),
                            float(min(max(min_area_quad[idx, 0], 0), w)))
                cv2.fillPoly(poly_tv_xy_map[2 * idx + 1],
                            np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32),
                            float(min(max(min_area_quad[idx, 1], 0), h)))

            # generate poly_tc_xy_map
            for idx in range(2):
                cv2.fillPoly(poly_tc_xy_map[idx],
                            np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32), float(center_point[idx]))

            # generate poly_short_edge_map
            cv2.fillPoly(poly_short_edge_map,
                        np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32),
                        float(max(min(min_area_quad_h, min_area_quad_w), 1.0)))

            # generate poly_mask and training_mask
            cv2.fillPoly(poly_mask, np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32), 1)

        tvo_map *= poly_mask
        tvo_map[:8] -= poly_tv_xy_map
        tvo_map[-1] /= poly_short_edge_map
        tvo_map = tvo_map.transpose((1, 2, 0))

        tco_map *= poly_mask
        tco_map[:2] -= poly_tc_xy_map
        tco_map[-1] /= poly_short_edge_map
        tco_map = tco_map.transpose((1, 2, 0))

        return tvo_map, tco_map

    def adjust_point(self, poly):
        """
        adjust point order.
        """
        point_num = poly.shape[0]
        if point_num == 4:
            len_1 = np.linalg.norm(poly[0] - poly[1])
            len_2 = np.linalg.norm(poly[1] - poly[2])
            len_3 = np.linalg.norm(poly[2] - poly[3])
            len_4 = np.linalg.norm(poly[3] - poly[0])

            if (len_1 + len_3) * 1.5 < (len_2 + len_4):
                poly = poly[[1, 2, 3, 0], :]

        elif point_num > 4:
            vector_1 = poly[0] - poly[1]
            vector_2 = poly[1] - poly[2]
            cos_theta = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2) + 1e-6)
            theta = np.arccos(np.round(cos_theta, decimals=4))

            if abs(theta) > (70 / 180 * math.pi):
                index = list(range(1, point_num)) + [0]
                poly = poly[np.array(index), :]
        return poly

    def gen_min_area_quad_from_poly(self, poly):
        """
        Generate min area quad from poly.
        """
        point_num = poly.shape[0]
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        if point_num == 4:
            min_area_quad = poly
            center_point = np.sum(poly, axis=0) / 4
        else:
            rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
            center_point = rect[0]
            box = np.array(cv2.boxPoints(rect))

            first_point_idx = 0
            min_dist = 1e4
            for i in range(4):
                dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                    np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                    np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                    np.linalg.norm(box[(i + 3) % 4] - poly[-1])
                if dist < min_dist:
                    min_dist = dist
                    first_point_idx = i

            for i in range(4):
                min_area_quad[i] = box[(first_point_idx + i) % 4]

        return min_area_quad, center_point

    def shrink_quad_along_width(self, quad, begin_width_ratio=0., end_width_ratio=1.):
        """
        Generate shrink_quad_along_width.
        """
        ratio_pair = np.array([[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
        p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
        p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
        return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])

    def shrink_poly_along_width(self, quads, shrink_ratio_of_width, expand_height_ratio=1.0):
        """
        shrink poly with given length.
        """
        upper_edge_list = []

        def get_cut_info(edge_len_list, cut_len):
            for idx, edge_len in enumerate(edge_len_list):
                cut_len -= edge_len
                if cut_len <= 0.000001:
                    ratio = (cut_len + edge_len_list[idx]) / edge_len_list[idx]
                    return idx, ratio

        for quad in quads:
            upper_edge_len = np.linalg.norm(quad[0] - quad[1])
            upper_edge_list.append(upper_edge_len)

        # length of left edge and right edge.
        left_length = np.linalg.norm(quads[0][0] - quads[0][3]) * expand_height_ratio
        right_length = np.linalg.norm(quads[-1][1] - quads[-1][2]) * expand_height_ratio

        shrink_length = min(left_length, right_length, sum(upper_edge_list)) * shrink_ratio_of_width
        # shrinking length
        upper_len_left = shrink_length
        upper_len_right = sum(upper_edge_list) - shrink_length

        left_idx, left_ratio = get_cut_info(upper_edge_list, upper_len_left)
        left_quad = self.shrink_quad_along_width(quads[left_idx], begin_width_ratio=left_ratio, end_width_ratio=1)
        right_idx, right_ratio = get_cut_info(upper_edge_list, upper_len_right)
        right_quad = self.shrink_quad_along_width(quads[right_idx], begin_width_ratio=0, end_width_ratio=right_ratio)
        
        out_quad_list = []
        if left_idx == right_idx:
            out_quad_list.append([left_quad[0], right_quad[1], right_quad[2], left_quad[3]])
        else:
            out_quad_list.append(left_quad)
            for idx in range(left_idx + 1, right_idx):
                out_quad_list.append(quads[idx])
            out_quad_list.append(right_quad)

        return np.array(out_quad_list), list(range(left_idx, right_idx + 1))

    def vector_angle(self, A, B):
        """
        Calculate the angle between vector AB and x-axis positive direction.
        """
        AB = np.array([B[1] - A[1], B[0] - A[0]])
        return np.arctan2(*AB)

    def theta_line_cross_point(self, theta, point):
        """
        Calculate the line through given point and angle in ax + by + c =0 form.
        """
        x, y = point
        cos = np.cos(theta)
        sin = np.sin(theta)
        return [sin, -cos, cos * y - sin * x]

    def line_cross_two_point(self, A, B):
        """
        Calculate the line through given point A and B in ax + by + c =0 form.
        """
        angle = self.vector_angle(A, B)
        return self.theta_line_cross_point(angle, A)

    def average_angle(self, poly):
        """
        Calculate the average angle between left and right edge in given poly.
        """
        p0, p1, p2, p3 = poly
        angle30 = self.vector_angle(p3, p0)
        angle21 = self.vector_angle(p2, p1)
        return (angle30 + angle21) / 2

    def line_cross_point(self, line1, line2):
        """
        line1 and line2 in  0=ax+by+c form, compute the cross point of line1 and line2
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        d = a1 * b2 - a2 * b1

        if d == 0:
            #print("line1", line1)
            #print("line2", line2)
            print('Cross point does not exist')
            return np.array([0, 0], dtype=np.float32)
        else:
            x = (b1 * c2 - b2 * c1) / d
            y = (a2 * c1 - a1 * c2) / d

        return np.array([x, y], dtype=np.float32)

    def quad2tcl(self, poly, ratio):
        """
        Generate center line by poly clock-wise point. (4, 2)
        """
        ratio_pair = np.array([[0.5 - ratio / 2], [0.5 + ratio / 2]], dtype=np.float32)
        p0_3 = poly[0] + (poly[3] - poly[0]) * ratio_pair
        p1_2 = poly[1] + (poly[2] - poly[1]) * ratio_pair
        return np.array([p0_3[0], p1_2[0], p1_2[1], p0_3[1]])

    def poly2tcl(self, poly, ratio):
        """
        Generate center line by poly clock-wise point.
        """
        ratio_pair = np.array([[0.5 - ratio / 2], [0.5 + ratio / 2]], dtype=np.float32)
        tcl_poly = np.zeros_like(poly)
        point_num = poly.shape[0]

        for idx in range(point_num // 2):
            point_pair = poly[idx] + (poly[point_num - 1 - idx] - poly[idx]) * ratio_pair
            tcl_poly[idx] = point_pair[0]
            tcl_poly[point_num - 1 - idx] = point_pair[1]
        return tcl_poly

    def gen_quad_tbo(self, quad, tcl_mask, tbo_map):
        """
        Generate tbo_map for give quad.
        """
        # upper and lower line function: ax + by + c = 0;
        up_line = self.line_cross_two_point(quad[0], quad[1])
        lower_line = self.line_cross_two_point(quad[3], quad[2])

        quad_h = 0.5 * (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2]))
        quad_w = 0.5 * (np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[2] - quad[3]))

        # average angle of left and right line.
        angle = self.average_angle(quad)

        xy_in_poly = np.argwhere(tcl_mask == 1)
        for y, x in xy_in_poly:
            point = (x, y)
            line = self.theta_line_cross_point(angle, point)
            cross_point_upper = self.line_cross_point(up_line, line)
            cross_point_lower = self.line_cross_point(lower_line, line)
            ##FIX, offset reverse
            upper_offset_x, upper_offset_y = cross_point_upper - point
            lower_offset_x, lower_offset_y = cross_point_lower - point
            tbo_map[y, x, 0] = upper_offset_y
            tbo_map[y, x, 1] = upper_offset_x
            tbo_map[y, x, 2] = lower_offset_y
            tbo_map[y, x, 3] = lower_offset_x
            tbo_map[y, x, 4] = 1.0 / max(min(quad_h, quad_w), 1.0) * 2
        return tbo_map

    def poly2quads(self, poly):
        """
        Split poly into quads.
        """
        quad_list = []
        point_num = poly.shape[0]

        # point pair
        point_pair_list = []
        for idx in range(point_num // 2):
            point_pair = [poly[idx], poly[point_num - 1 - idx]]
            point_pair_list.append(point_pair)

        quad_num = point_num // 2 - 1
        for idx in range(quad_num):
            # reshape and adjust to clock-wise
            quad_list.append((np.array(point_pair_list)[[idx, idx + 1]]).reshape(4, 2)[[0, 2, 3, 1]])

        return np.array(quad_list)

    def extract_polys(self, poly_txt_path):
        """
        Read text_polys, txt_tags, txts from give txt file.
        """
        text_polys, txt_tags, txts = [], [], []

        with open(poly_txt_path, 'rb') as f:
            for line in f.readlines():
                poly_str, txt = line.strip().split('\t')
                poly = map(float, poly_str.split(','))
                text_polys.append(np.array(poly, dtype=np.float32).reshape(-1, 2))
                txts.append(txt)
                if txt == '###':
                    txt_tags.append(True)
                else:
                    txt_tags.append(False)

        return np.array(map(np.array, text_polys)), \
            np.array(txt_tags, dtype=np.bool), txts

    def __call__(self, label_infor):
        infor = self.convert_label_infor(label_infor)
        im_path, text_polys, text_tags, text_strs = infor
        im = cv2.imread(im_path)
        if im is None:
            return None
        if text_polys.shape[0] == 0:
            return None

        h, w, _ = im.shape
        text_polys, text_tags, hv_tags = self.check_and_validate_polys(text_polys, text_tags, (h, w))

        if text_polys.shape[0] == 0:
            return None

        #set aspect ratio and keep area fix
        asp_scales = np.arange(1.0, 1.55, 0.1)
        asp_scale = np.random.choice(asp_scales)

        if np.random.rand() < 0.5:
            asp_scale = 1.0 / asp_scale
        asp_scale = math.sqrt(asp_scale)
        
        asp_wx = asp_scale
        asp_hy = 1.0 / asp_scale
        im = cv2.resize(im, dsize=None, fx=asp_wx, fy=asp_hy)
        text_polys[:, :, 0] *= asp_wx
        text_polys[:, :, 1] *= asp_hy

        h, w, _ = im.shape
        if max(h, w) > 2048:
            rd_scale = 2048.0 / max(h, w)
            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale
        h, w, _ = im.shape
        if min(h, w) < 16:
            return None

        #no background
        im, text_polys, text_tags, hv_tags, text_strs = self.crop_area(im, \
            text_polys, text_tags, hv_tags, text_strs, crop_background=False)
        if text_polys.shape[0] == 0:
            return None
        #continue for all ignore case
        if np.sum((text_tags * 1.0)) >= text_tags.size:
            return None
        new_h, new_w, _ = im.shape
        if (new_h is None) or (new_w is None):
            return None
        #resize image
        std_ratio = float(self.input_size) / max(new_w, new_h)
        rand_scales = np.array([0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0, 1.0, 1.0, 1.0])
        rz_scale = std_ratio * np.random.choice(rand_scales)
        im = cv2.resize(im, dsize=None, fx=rz_scale, fy=rz_scale)
        text_polys[:, :, 0] *= rz_scale
        text_polys[:, :, 1] *= rz_scale
        
        #add gaussian blur
        if np.random.rand() < 0.1 * 0.5:
            ks = np.random.permutation(5)[0] + 1
            ks = int(ks/2)*2 + 1
            im =  cv2.GaussianBlur(im, ksize=(ks, ks), sigmaX=0, sigmaY=0)
        #add brighter
        if np.random.rand() < 0.1 * 0.5:
            im = im * (1.0 + np.random.rand() * 0.5)
            im = np.clip(im, 0.0, 255.0)
        #add darker
        if np.random.rand() < 0.1 * 0.5:
            im = im * (1.0 - np.random.rand() * 0.5)
            im = np.clip(im, 0.0, 255.0)
        
        # Padding the im to [input_size, input_size]
        new_h, new_w, _ = im.shape
        if min(new_w, new_h) < self.input_size * 0.5:
            return None

        im_padded = np.ones((self.input_size, self.input_size, 3), dtype=np.float32)
        im_padded[:, :, 2] = 0.485 * 255
        im_padded[:, :, 1] = 0.456 * 255
        im_padded[:, :, 0] = 0.406 * 255

        # Random the start position
        del_h = self.input_size - new_h
        del_w = self.input_size - new_w
        sh, sw = 0, 0
        if del_h > 1:
            sh = int(np.random.rand() * del_h)
        if del_w > 1:
            sw = int(np.random.rand() * del_w)

        # Padding
        im_padded[sh: sh + new_h, sw: sw + new_w, :] = im.copy()
        text_polys[:, :, 0] += sw
        text_polys[:, :, 1] += sh

        score_map, border_map, training_mask = self.generate_tcl_label((self.input_size, self.input_size), 
                            text_polys, text_tags, 0.25)
        
        # SAST head
        tvo_map, tco_map = self.generate_tvo_and_tco((self.input_size, self.input_size), text_polys, text_tags,  tcl_ratio=0.3, ds_ratio=0.25)
        # print("test--------tvo_map shape:", tvo_map.shape)

        im_padded[:, :, 2] -= 0.485 * 255
        im_padded[:, :, 1] -= 0.456 * 255
        im_padded[:, :, 0] -= 0.406 * 255
        im_padded[:, :, 2] /= (255.0 * 0.229) 
        im_padded[:, :, 1] /= (255.0 * 0.224) 
        im_padded[:, :, 0] /= (255.0 * 0.225) 
        im_padded = im_padded.transpose((2, 0, 1))

        return im_padded[::-1, :, :], score_map[np.newaxis, :, :], border_map.transpose((2, 0, 1)), training_mask[np.newaxis, :, :], tvo_map.transpose((2, 0, 1)), tco_map.transpose((2, 0, 1))

    
class SASTProcessTest(object):
    """
    SAST process function for test
    """
    def __init__(self, params):
        super(SASTProcessTest, self).__init__()
        if 'max_side_len' in params:
            self.max_side_len = params['max_side_len']
        else:
            self.max_side_len = 2400

    def resize_image(self, im):
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
            ratio = float(self.max_side_len) / resize_h
        else:
            ratio = float(self.max_side_len) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)

    def __call__(self, im):
        src_h, src_w, _ = im.shape
        im, (ratio_h, ratio_w) = self.resize_image(im)
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im[:, :, ::-1].astype(np.float32)
        im = im / 255
        im -= img_mean
        im /= img_std
        im = im.transpose((2, 0, 1))
        im = im[np.newaxis, :]
        return [im, (ratio_h, ratio_w, src_h, src_w)]
