# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import cv2
from collections import defaultdict


class MergeBoxes:
    def __init__(self, image_shape):
        self.image_shape = image_shape  # (height, width)
        self.distance_threshold = [20, 25]  # (vertical, horizontal) Maximum gap
        self.height_overlap_threshold = 0.6  # Minimum height overlap rate
        pass

    def merge(self, boxes):
        # Prepare
        data = np.array(boxes)
        data = sorted(data, key=lambda item: (item[0][1], item[0][0]))
        removed = defaultdict(lambda: False)

        # Check if is horizontal
        for index in range(len(data)):
            box = data[index]
            if not self.is_horizontal(box):
                data[index] = np.array([box[1], box[2], box[3], box[0]])  # Rotate 90°
                pass

        # Loop
        current = 0
        while current < len(data):
            if removed[current]:
                current += 1
                continue
            current_box = data[current]

            # Look forward
            forward = current + 1
            while forward < len(data):
                if removed[forward] or current == forward:
                    forward += 1
                    continue
                forward_box = data[forward]
                if self.should_merge(current_box, forward_box):
                    data[current] = self.merge_two_boxes(current_box, forward_box)
                    removed[forward] = True
                    # Recheck
                    current_box = data[current]
                    forward = 0
                    continue
                forward += 1

            # Next
            current += 1
            pass

        # Filter
        results = []
        for index in range(len(data)):
            if not removed[index]:
                results.append(data[index])
        results = sorted(results, key=lambda item: (item[0][1], item[0][0]))
        return np.array(results)

    def should_merge(self, first_box, second_box):
        # Check distance
        distance = self.calculate_distance(first_box, second_box)
        if self.distance_threshold[0] < distance[0] or self.distance_threshold[1] < distance[1]:
            return False
        # Check height overlap rate
        if self.calculate_height_overlap_rate(first_box, second_box) < self.height_overlap_threshold \
                and self.calculate_height_overlap_rate(second_box, first_box) < self.height_overlap_threshold:
            return False
        return True

    def merge_two_boxes(self, first_box, second_box):
        # Merge
        points = np.array([first_box, second_box]).reshape((-1, 1, 2)).astype(np.float32)
        box, _ = self.contour_to_box(points)

        # Clip bo
        box = np.array(box)
        box[:, 0] = np.clip(np.round(box[:, 0]), 0, self.image_shape[1])
        box[:, 1] = np.clip(np.round(box[:, 1]), 0, self.image_shape[0])
        return box

    @staticmethod
    def calculate_angle(box):
        begin, end = box[0], box[1]
        vector = end - begin
        if vector[0] == 0:
            return 90
        angle = math.degrees(np.arctan(vector[1] / vector[0]))
        return angle

    @staticmethod
    def calculate_ratio(box):
        width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
        return max(width, height) / min(width, height)
        pass

    @staticmethod
    def calculate_distance(first_box, second_box):
        # List centres
        positions = [
            [(first_box[0] + first_box[3]) / 2, True],
            [(first_box[1] + first_box[2]) / 2, True],
            [(second_box[0] + second_box[3]) / 2, False],
            [(second_box[1] + second_box[2]) / 2, False],
        ]
        positions = sorted(positions, key=lambda item: (item[0][0], item[0][1]))

        # Calculate
        segment = (positions[0][0], positions[3][0])
        vertical_distance = np.linalg.norm(MergeBoxes.project_position(positions[1][0], segment) - positions[1][0])
        vertical_distance += np.linalg.norm(MergeBoxes.project_position(positions[2][0], segment) - positions[2][0])
        if positions[0][1] == positions[3][1]:  # Containing
            horizontal_distance = 0
        elif positions[0][1] != positions[1][1]:  # Intersecting
            horizontal_distance = 0
        else:  # Separating
            horizontal_distance = np.linalg.norm(positions[1][0] - positions[2][0])
        return vertical_distance, horizontal_distance

    @staticmethod
    def calculate_height_overlap_rate(first_box, second_box):
        first_segment = (first_box[1], first_box[2])
        second_segment = (second_box[0], second_box[3])
        first_height = np.linalg.norm(first_segment[0] - first_segment[1])
        second_height = np.linalg.norm(second_segment[0] - second_segment[1])
        overlap = MergeBoxes.project_overlap(first_segment, second_segment) + MergeBoxes.project_overlap(second_segment, first_segment)
        return overlap / (first_height + second_height)

    @staticmethod
    def project_overlap(from_segment, to_segment):
        # Project
        positions = [
            [MergeBoxes.project_position(from_segment[0], to_segment), False],
            [MergeBoxes.project_position(from_segment[1], to_segment), False],
            [to_segment[0], True],
            [to_segment[1], True],
        ]

        # Calculate overlap
        positions = sorted(positions, key=lambda item: (item[0][0], item[0][1]))
        if positions[0][1] != positions[1][1] and positions[2][1] != positions[3][1]:
            begin = np.array(positions[1][0])
            end = np.array(positions[2][0])
            return np.linalg.norm(begin - end)
        return 0

    @staticmethod
    def project_position(point, segment):
        # y = k * x + b
        vector = segment[1] - segment[0]
        if vector[0] == 0:  # or 1e-6
            # k does not exist
            return np.array([segment[0][0], point[1]])
        else:
            # k = (y2 - y1) / (x2 - x1)
            # b = y1 - k * x1
            k = vector[1] / vector[0]
            b = segment[0][1] - k * segment[0][0]

            # Project
            x = (k * (point[1] - b) + point[0]) / (k * k + 1)
            y = k * x + b
            return np.array([x, y])


    @staticmethod
    def contour_to_box(contour, fix_direction=False):
        bounding_box = cv2.minAreaRect(contour)  # [center, [side], angle]

        # left top, 
        points = list(cv2.boxPoints(bounding_box))
        points = sorted(points, key=lambda x: x[0])
        if points[0][1] < points[1][1]:
            index_0 = 0
            index_3 = 1
        else:
            index_0 = 1
            index_3 = 0
        if points[2][1] < points[3][1]:
            index_1 = 2
            index_2 = 3
        else:
            index_1 = 3
            index_2 = 2
        box = [points[index_0], points[index_1], points[index_2], points[index_3]]

        # Rotate
        if fix_direction:
            if MergeBoxes.is_horizontal(box):
                box = [box[1], box[2], box[3], box[0]]

        return box, (min(bounding_box[1]), max(bounding_box[1]))

    @staticmethod
    def is_horizontal(box):
        width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
        if height * 1.0 / width >= 1.5:
            return False
        return True

