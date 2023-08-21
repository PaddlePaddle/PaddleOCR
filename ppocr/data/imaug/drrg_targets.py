# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/textdet_targets/drrg_targets.py
"""

import cv2
import numpy as np
from ppocr.utils.utility import check_install
from numpy.linalg import norm


class DRRGTargets(object):
    def __init__(self,
                 orientation_thr=2.0,
                 resample_step=8.0,
                 num_min_comps=9,
                 num_max_comps=600,
                 min_width=8.0,
                 max_width=24.0,
                 center_region_shrink_ratio=0.3,
                 comp_shrink_ratio=1.0,
                 comp_w_h_ratio=0.3,
                 text_comp_nms_thr=0.25,
                 min_rand_half_height=8.0,
                 max_rand_half_height=24.0,
                 jitter_level=0.2,
                 **kwargs):

        super().__init__()
        self.orientation_thr = orientation_thr
        self.resample_step = resample_step
        self.num_max_comps = num_max_comps
        self.num_min_comps = num_min_comps
        self.min_width = min_width
        self.max_width = max_width
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.comp_shrink_ratio = comp_shrink_ratio
        self.comp_w_h_ratio = comp_w_h_ratio
        self.text_comp_nms_thr = text_comp_nms_thr
        self.min_rand_half_height = min_rand_half_height
        self.max_rand_half_height = max_rand_half_height
        self.jitter_level = jitter_level
        self.eps = 1e-8

    def vector_angle(self, vec1, vec2):
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps)
        return np.arccos(
            np.clip(
                np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def vector_slope(self, vec):
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + self.eps))

    def vector_sin(self, vec):
        assert len(vec) == 2
        return vec[1] / (norm(vec) + self.eps)

    def vector_cos(self, vec):
        assert len(vec) == 2
        return vec[0] / (norm(vec) + self.eps)

    def find_head_tail(self, points, orientation_thr):

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            pad_points = np.vstack([points, points[0]])
            edge_vec = pad_points[1:] - pad_points[:-1]

            theta_sum = []
            adjacent_vec_theta = []
            for i, edge_vec1 in enumerate(edge_vec):
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(adjacent_edge_vec[0],
                                                        adjacent_edge_vec[1])
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(
                    pad_points[1:] - poly_center, axis=-1),
                norm(
                    pad_points[:-1] - poly_center, axis=-1))
            dist_score = edge_dist / (np.max(edge_dist) + self.eps)
            position_score = np.zeros(len(edge_vec))
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            score += 0.1 * position_score
            pad_score = np.concatenate([score, score])
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[(i + 2):(i + len(
                    score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(score_matrix.argmax(),
                                                          score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                    points[3] - points[2]) < self.vector_slope(points[
                        2] - points[1]) + self.vector_slope(points[0] - points[
                            3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] - points[
                vertical_edge_inds[0][1]]) + norm(points[vertical_edge_inds[1][
                    0]] - points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(points[horizontal_edge_inds[0][
                0]] - points[horizontal_edge_inds[0][1]]) + norm(points[
                    horizontal_edge_inds[1][0]] - points[horizontal_edge_inds[1]
                                                         [1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def reorder_poly_edge(self, points):

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        head_inds, tail_inds = self.find_head_tail(points, self.orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        return head_edge, tail_edge, top_sideline, bot_sideline

    def cal_curve_length(self, line):

        assert line.ndim == 2
        assert len(line) >= 2

        edges_length = np.sqrt((line[1:, 0] - line[:-1, 0])**2 + (line[
            1:, 1] - line[:-1, 1])**2)
        total_length = np.sum(edges_length)
        return edges_length, total_length

    def resample_line(self, line, n):

        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 2

        edges_length, total_length = self.cal_curve_length(line)
        t_org = np.insert(np.cumsum(edges_length), 0, 0)
        unit_t = total_length / (n - 1)
        t_equidistant = np.arange(1, n - 1, dtype=np.float32) * unit_t
        edge_ind = 0
        points = [line[0]]
        for t in t_equidistant:
            while edge_ind < len(edges_length) - 1 and t > t_org[edge_ind + 1]:
                edge_ind += 1
            t_l, t_r = t_org[edge_ind], t_org[edge_ind + 1]
            weight = np.array(
                [t_r - t, t - t_l], dtype=np.float32) / (t_r - t_l + self.eps)
            p_coords = np.dot(weight, line[[edge_ind, edge_ind + 1]])
            points.append(p_coords)
        points.append(line[-1])
        resampled_line = np.vstack(points)

        return resampled_line

    def resample_sidelines(self, sideline1, sideline2, resample_step):

        assert sideline1.ndim == sideline2.ndim == 2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        assert isinstance(resample_step, float)

        _, length1 = self.cal_curve_length(sideline1)
        _, length2 = self.cal_curve_length(sideline2)

        avg_length = (length1 + length2) / 2
        resample_point_num = max(int(float(avg_length) / resample_step) + 1, 3)

        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2

    def dist_point2line(self, point, line):

        assert isinstance(line, tuple)
        point1, point2 = line
        d = abs(np.cross(point2 - point1, point - point1)) / (
            norm(point2 - point1) + 1e-8)
        return d

    def draw_center_region_maps(self, top_line, bot_line, center_line,
                                center_region_mask, top_height_map,
                                bot_height_map, sin_map, cos_map,
                                region_shrink_ratio):

        assert top_line.shape == bot_line.shape == center_line.shape
        assert (center_region_mask.shape == top_height_map.shape ==
                bot_height_map.shape == sin_map.shape == cos_map.shape)
        assert isinstance(region_shrink_ratio, float)

        h, w = center_region_mask.shape
        for i in range(0, len(center_line) - 1):

            top_mid_point = (top_line[i] + top_line[i + 1]) / 2
            bot_mid_point = (bot_line[i] + bot_line[i + 1]) / 2

            sin_theta = self.vector_sin(top_mid_point - bot_mid_point)
            cos_theta = self.vector_cos(top_mid_point - bot_mid_point)

            tl = center_line[i] + (top_line[i] - center_line[i]
                                   ) * region_shrink_ratio
            tr = center_line[i + 1] + (top_line[i + 1] - center_line[i + 1]
                                       ) * region_shrink_ratio
            br = center_line[i + 1] + (bot_line[i + 1] - center_line[i + 1]
                                       ) * region_shrink_ratio
            bl = center_line[i] + (bot_line[i] - center_line[i]
                                   ) * region_shrink_ratio
            current_center_box = np.vstack([tl, tr, br, bl]).astype(np.int32)

            cv2.fillPoly(center_region_mask, [current_center_box], color=1)
            cv2.fillPoly(sin_map, [current_center_box], color=sin_theta)
            cv2.fillPoly(cos_map, [current_center_box], color=cos_theta)

            current_center_box[:, 0] = np.clip(current_center_box[:, 0], 0,
                                               w - 1)
            current_center_box[:, 1] = np.clip(current_center_box[:, 1], 0,
                                               h - 1)
            min_coord = np.min(current_center_box, axis=0).astype(np.int32)
            max_coord = np.max(current_center_box, axis=0).astype(np.int32)
            current_center_box = current_center_box - min_coord
            box_sz = (max_coord - min_coord + 1)

            center_box_mask = np.zeros((box_sz[1], box_sz[0]), dtype=np.uint8)
            cv2.fillPoly(center_box_mask, [current_center_box], color=1)

            inds = np.argwhere(center_box_mask > 0)
            inds = inds + (min_coord[1], min_coord[0])
            inds_xy = np.fliplr(inds)
            top_height_map[(inds[:, 0], inds[:, 1])] = self.dist_point2line(
                inds_xy, (top_line[i], top_line[i + 1]))
            bot_height_map[(inds[:, 0], inds[:, 1])] = self.dist_point2line(
                inds_xy, (bot_line[i], bot_line[i + 1]))

    def generate_center_mask_attrib_maps(self, img_size, text_polys):

        assert isinstance(img_size, tuple)

        h, w = img_size

        center_lines = []
        center_region_mask = np.zeros((h, w), np.uint8)
        top_height_map = np.zeros((h, w), dtype=np.float32)
        bot_height_map = np.zeros((h, w), dtype=np.float32)
        sin_map = np.zeros((h, w), dtype=np.float32)
        cos_map = np.zeros((h, w), dtype=np.float32)

        for poly in text_polys:
            polygon_points = poly
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            if self.vector_slope(center_line[-1] - center_line[0]) > 2:
                if (center_line[-1] - center_line[0])[1] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]
            else:
                if (center_line[-1] - center_line[0])[0] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]

            line_head_shrink_len = np.clip(
                (norm(top_line[0] - bot_line[0]) * self.comp_w_h_ratio),
                self.min_width, self.max_width) / 2
            line_tail_shrink_len = np.clip(
                (norm(top_line[-1] - bot_line[-1]) * self.comp_w_h_ratio),
                self.min_width, self.max_width) / 2
            num_head_shrink = int(line_head_shrink_len // self.resample_step)
            num_tail_shrink = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > num_head_shrink + num_tail_shrink + 2:
                center_line = center_line[num_head_shrink:len(center_line) -
                                          num_tail_shrink]
                resampled_top_line = resampled_top_line[num_head_shrink:len(
                    resampled_top_line) - num_tail_shrink]
                resampled_bot_line = resampled_bot_line[num_head_shrink:len(
                    resampled_bot_line) - num_tail_shrink]
            center_lines.append(center_line.astype(np.int32))

            self.draw_center_region_maps(
                resampled_top_line, resampled_bot_line, center_line,
                center_region_mask, top_height_map, bot_height_map, sin_map,
                cos_map, self.center_region_shrink_ratio)

        return (center_lines, center_region_mask, top_height_map,
                bot_height_map, sin_map, cos_map)

    def generate_rand_comp_attribs(self, num_rand_comps, center_sample_mask):

        assert isinstance(num_rand_comps, int)
        assert num_rand_comps > 0
        assert center_sample_mask.ndim == 2

        h, w = center_sample_mask.shape

        max_rand_half_height = self.max_rand_half_height
        min_rand_half_height = self.min_rand_half_height
        max_rand_height = max_rand_half_height * 2
        max_rand_width = np.clip(max_rand_height * self.comp_w_h_ratio,
                                 self.min_width, self.max_width)
        margin = int(
            np.sqrt((max_rand_height / 2)**2 + (max_rand_width / 2)**2)) + 1

        if 2 * margin + 1 > min(h, w):

            assert min(h, w) > (np.sqrt(2) * (self.min_width + 1))
            max_rand_half_height = max(min(h, w) / 4, self.min_width / 2 + 1)
            min_rand_half_height = max(max_rand_half_height / 4,
                                       self.min_width / 2)

            max_rand_height = max_rand_half_height * 2
            max_rand_width = np.clip(max_rand_height * self.comp_w_h_ratio,
                                     self.min_width, self.max_width)
            margin = int(
                np.sqrt((max_rand_height / 2)**2 + (max_rand_width / 2)**2)) + 1

        inner_center_sample_mask = np.zeros_like(center_sample_mask)
        inner_center_sample_mask[margin:h - margin, margin:w - margin] = \
            center_sample_mask[margin:h - margin, margin:w - margin]
        kernel_size = int(np.clip(max_rand_half_height, 7, 21))
        inner_center_sample_mask = cv2.erode(
            inner_center_sample_mask,
            np.ones((kernel_size, kernel_size), np.uint8))

        center_candidates = np.argwhere(inner_center_sample_mask > 0)
        num_center_candidates = len(center_candidates)
        sample_inds = np.random.choice(num_center_candidates, num_rand_comps)
        rand_centers = center_candidates[sample_inds]

        rand_top_height = np.random.randint(
            min_rand_half_height,
            max_rand_half_height,
            size=(len(rand_centers), 1))
        rand_bot_height = np.random.randint(
            min_rand_half_height,
            max_rand_half_height,
            size=(len(rand_centers), 1))

        rand_cos = 2 * np.random.random(size=(len(rand_centers), 1)) - 1
        rand_sin = 2 * np.random.random(size=(len(rand_centers), 1)) - 1
        scale = np.sqrt(1.0 / (rand_cos**2 + rand_sin**2 + 1e-8))
        rand_cos = rand_cos * scale
        rand_sin = rand_sin * scale

        height = (rand_top_height + rand_bot_height)
        width = np.clip(height * self.comp_w_h_ratio, self.min_width,
                        self.max_width)

        rand_comp_attribs = np.hstack([
            rand_centers[:, ::-1], height, width, rand_cos, rand_sin,
            np.zeros_like(rand_sin)
        ]).astype(np.float32)

        return rand_comp_attribs

    def jitter_comp_attribs(self, comp_attribs, jitter_level):
        """Jitter text components attributes.

        Args:
            comp_attribs (ndarray): The text component attributes.
            jitter_level (float): The jitter level of text components
                attributes.

        Returns:
            jittered_comp_attribs (ndarray): The jittered text component
                attributes (x, y, h, w, cos, sin, comp_label).
        """

        assert comp_attribs.shape[1] == 7
        assert comp_attribs.shape[0] > 0
        assert isinstance(jitter_level, float)

        x = comp_attribs[:, 0].reshape((-1, 1))
        y = comp_attribs[:, 1].reshape((-1, 1))
        h = comp_attribs[:, 2].reshape((-1, 1))
        w = comp_attribs[:, 3].reshape((-1, 1))
        cos = comp_attribs[:, 4].reshape((-1, 1))
        sin = comp_attribs[:, 5].reshape((-1, 1))
        comp_labels = comp_attribs[:, 6].reshape((-1, 1))

        x += (np.random.random(size=(len(comp_attribs), 1)) - 0.5) * (
            h * np.abs(cos) + w * np.abs(sin)) * jitter_level
        y += (np.random.random(size=(len(comp_attribs), 1)) - 0.5) * (
            h * np.abs(sin) + w * np.abs(cos)) * jitter_level

        h += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
              ) * h * jitter_level
        w += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
              ) * w * jitter_level

        cos += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
                ) * 2 * jitter_level
        sin += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
                ) * 2 * jitter_level

        scale = np.sqrt(1.0 / (cos**2 + sin**2 + 1e-8))
        cos = cos * scale
        sin = sin * scale

        jittered_comp_attribs = np.hstack([x, y, h, w, cos, sin, comp_labels])

        return jittered_comp_attribs

    def generate_comp_attribs(self, center_lines, text_mask, center_region_mask,
                              top_height_map, bot_height_map, sin_map, cos_map):
        """Generate text component attributes.

        Args:
            center_lines (list[ndarray]): The list of text center lines .
            text_mask (ndarray): The text region mask.
            center_region_mask (ndarray): The text center region mask.
            top_height_map (ndarray): The map on which the distance from points
                to top side lines will be drawn for each pixel in text center
                regions.
            bot_height_map (ndarray): The map on which the distance from points
                to bottom side lines will be drawn for each pixel in text
                center regions.
            sin_map (ndarray): The sin(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
            cos_map (ndarray): The cos(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).

        Returns:
            pad_comp_attribs (ndarray): The padded text component attributes
                of a fixed size.
        """

        assert isinstance(center_lines, list)
        assert (
            text_mask.shape == center_region_mask.shape == top_height_map.shape
            == bot_height_map.shape == sin_map.shape == cos_map.shape)

        center_lines_mask = np.zeros_like(center_region_mask)
        cv2.polylines(center_lines_mask, center_lines, 0, 1, 1)
        center_lines_mask = center_lines_mask * center_region_mask
        comp_centers = np.argwhere(center_lines_mask > 0)

        y = comp_centers[:, 0]
        x = comp_centers[:, 1]

        top_height = top_height_map[y, x].reshape(
            (-1, 1)) * self.comp_shrink_ratio
        bot_height = bot_height_map[y, x].reshape(
            (-1, 1)) * self.comp_shrink_ratio
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))

        top_mid_points = comp_centers + np.hstack(
            [top_height * sin, top_height * cos])
        bot_mid_points = comp_centers - np.hstack(
            [bot_height * sin, bot_height * cos])

        width = (top_height + bot_height) * self.comp_w_h_ratio
        width = np.clip(width, self.min_width, self.max_width)
        r = width / 2

        tl = top_mid_points[:, ::-1] - np.hstack([-r * sin, r * cos])
        tr = top_mid_points[:, ::-1] + np.hstack([-r * sin, r * cos])
        br = bot_mid_points[:, ::-1] + np.hstack([-r * sin, r * cos])
        bl = bot_mid_points[:, ::-1] - np.hstack([-r * sin, r * cos])
        text_comps = np.hstack([tl, tr, br, bl]).astype(np.float32)

        score = np.ones((text_comps.shape[0], 1), dtype=np.float32)
        text_comps = np.hstack([text_comps, score])
        check_install('lanms', 'lanms-neo')
        from lanms import merge_quadrangle_n9 as la_nms
        text_comps = la_nms(text_comps, self.text_comp_nms_thr)

        if text_comps.shape[0] >= 1:
            img_h, img_w = center_region_mask.shape
            text_comps[:, 0:8:2] = np.clip(text_comps[:, 0:8:2], 0, img_w - 1)
            text_comps[:, 1:8:2] = np.clip(text_comps[:, 1:8:2], 0, img_h - 1)

            comp_centers = np.mean(
                text_comps[:, 0:8].reshape((-1, 4, 2)), axis=1).astype(np.int32)
            x = comp_centers[:, 0]
            y = comp_centers[:, 1]

            height = (top_height_map[y, x] + bot_height_map[y, x]).reshape(
                (-1, 1))
            width = np.clip(height * self.comp_w_h_ratio, self.min_width,
                            self.max_width)

            cos = cos_map[y, x].reshape((-1, 1))
            sin = sin_map[y, x].reshape((-1, 1))

            _, comp_label_mask = cv2.connectedComponents(
                center_region_mask, connectivity=8)
            comp_labels = comp_label_mask[y, x].reshape(
                (-1, 1)).astype(np.float32)

            x = x.reshape((-1, 1)).astype(np.float32)
            y = y.reshape((-1, 1)).astype(np.float32)
            comp_attribs = np.hstack(
                [x, y, height, width, cos, sin, comp_labels])
            comp_attribs = self.jitter_comp_attribs(comp_attribs,
                                                    self.jitter_level)

            if comp_attribs.shape[0] < self.num_min_comps:
                num_rand_comps = self.num_min_comps - comp_attribs.shape[0]
                rand_comp_attribs = self.generate_rand_comp_attribs(
                    num_rand_comps, 1 - text_mask)
                comp_attribs = np.vstack([comp_attribs, rand_comp_attribs])
        else:
            comp_attribs = self.generate_rand_comp_attribs(self.num_min_comps,
                                                           1 - text_mask)

        num_comps = (np.ones(
            (comp_attribs.shape[0], 1),
            dtype=np.float32) * comp_attribs.shape[0])
        comp_attribs = np.hstack([num_comps, comp_attribs])

        if comp_attribs.shape[0] > self.num_max_comps:
            comp_attribs = comp_attribs[:self.num_max_comps, :]
            comp_attribs[:, 0] = self.num_max_comps

        pad_comp_attribs = np.zeros(
            (self.num_max_comps, comp_attribs.shape[1]), dtype=np.float32)
        pad_comp_attribs[:comp_attribs.shape[0], :] = comp_attribs

        return pad_comp_attribs

    def generate_text_region_mask(self, img_size, text_polys):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            polygon = np.array(poly, dtype=np.int32).reshape((1, -1, 2))
            cv2.fillPoly(text_region_mask, polygon, 1)

        return text_region_mask

    def generate_effective_mask(self, mask_size: tuple, polygons_ignore):
        """Generate effective mask by setting the ineffective regions to 0 and
        effective regions to 1.

        Args:
            mask_size (tuple): The mask size.
            polygons_ignore (list[[ndarray]]: The list of ignored text
                polygons.

        Returns:
            mask (ndarray): The effective mask of (height, width).
        """
        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in polygons_ignore:
            instance = poly.astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)

        return mask

    def generate_targets(self, data):
        """Generate the gt targets for DRRG.

        Args:
            data (dict): The input result dictionary.

        Returns:
            data (dict): The output result dictionary.
        """

        assert isinstance(data, dict)

        image = data['image']
        polygons = data['polys']
        ignore_tags = data['ignore_tags']
        h, w, _ = image.shape

        polygon_masks = []
        polygon_masks_ignore = []
        for tag, polygon in zip(ignore_tags, polygons):
            if tag is True:
                polygon_masks_ignore.append(polygon)
            else:
                polygon_masks.append(polygon)

        gt_text_mask = self.generate_text_region_mask((h, w), polygon_masks)
        gt_mask = self.generate_effective_mask((h, w), polygon_masks_ignore)
        (center_lines, gt_center_region_mask, gt_top_height_map,
         gt_bot_height_map, gt_sin_map,
         gt_cos_map) = self.generate_center_mask_attrib_maps((h, w),
                                                             polygon_masks)

        gt_comp_attribs = self.generate_comp_attribs(
            center_lines, gt_text_mask, gt_center_region_mask,
            gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map)

        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_top_height_map': gt_top_height_map,
            'gt_bot_height_map': gt_bot_height_map,
            'gt_sin_map': gt_sin_map,
            'gt_cos_map': gt_cos_map
        }

        data.update(mapping)
        data['gt_comp_attribs'] = gt_comp_attribs
        return data

    def __call__(self, data):
        data = self.generate_targets(data)
        return data
