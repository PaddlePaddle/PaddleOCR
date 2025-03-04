# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import numpy as np
import time
import math
import cv2
import fastdeploy as fd

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


def get_rotate_crop_image(img, box):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    points = []
    for i in range(4):
        points.append([box[2 * i], box[2 * i + 1]])
    points = np.array(points, dtype=np.float32)
    img = img.astype(np.float32)
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])
        print("model_config:", self.model_config)

        self.input_names = []
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("postprocess input names:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("postprocess output names:", self.output_names)
        self.postprocessor = fd.vision.ocr.DBDetectorPostprocessor()
        self.cls_preprocessor = fd.vision.ocr.ClassifierPreprocessor()
        self.rec_preprocessor = fd.vision.ocr.RecognizerPreprocessor()
        self.cls_threshold = 0.9

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            infer_outputs = pb_utils.get_input_tensor_by_name(
                request, self.input_names[0]
            )
            im_infos = pb_utils.get_input_tensor_by_name(request, self.input_names[1])
            ori_imgs = pb_utils.get_input_tensor_by_name(request, self.input_names[2])

            infer_outputs = infer_outputs.as_numpy()
            im_infos = im_infos.as_numpy()
            ori_imgs = ori_imgs.as_numpy()

            results = self.postprocessor.run([infer_outputs], im_infos)
            batch_rec_texts = []
            batch_rec_scores = []
            batch_box_list = []
            for i_batch in range(len(results)):
                cls_labels = []
                cls_scores = []
                rec_texts = []
                rec_scores = []

                box_list = fd.vision.ocr.sort_boxes(results[i_batch])
                image_list = []
                if len(box_list) == 0:
                    image_list.append(ori_imgs[i_batch])
                else:
                    for box in box_list:
                        crop_img = get_rotate_crop_image(ori_imgs[i_batch], box)
                        image_list.append(crop_img)

                batch_box_list.append(box_list)

                cls_pre_tensors = self.cls_preprocessor.run(image_list)
                cls_dlpack_tensor = cls_pre_tensors[0].to_dlpack()
                cls_input_tensor = pb_utils.Tensor.from_dlpack("x", cls_dlpack_tensor)

                inference_request = pb_utils.InferenceRequest(
                    model_name="cls_pp",
                    requested_output_names=["cls_labels", "cls_scores"],
                    inputs=[cls_input_tensor],
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    # Extract the output tensors from the inference response.
                    cls_labels = pb_utils.get_output_tensor_by_name(
                        inference_response, "cls_labels"
                    )
                    cls_labels = cls_labels.as_numpy()

                    cls_scores = pb_utils.get_output_tensor_by_name(
                        inference_response, "cls_scores"
                    )
                    cls_scores = cls_scores.as_numpy()

                for index in range(len(image_list)):
                    if (
                        cls_labels[index] == 1
                        and cls_scores[index] > self.cls_threshold
                    ):
                        image_list[index] = cv2.rotate(
                            image_list[index].astype(np.float32), 1
                        )
                        image_list[index] = image_list[index].astype(np.uint8)

                rec_pre_tensors = self.rec_preprocessor.run(image_list)
                rec_dlpack_tensor = rec_pre_tensors[0].to_dlpack()
                rec_input_tensor = pb_utils.Tensor.from_dlpack("x", rec_dlpack_tensor)

                inference_request = pb_utils.InferenceRequest(
                    model_name="rec_pp",
                    requested_output_names=["rec_texts", "rec_scores"],
                    inputs=[rec_input_tensor],
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    # Extract the output tensors from the inference response.
                    rec_texts = pb_utils.get_output_tensor_by_name(
                        inference_response, "rec_texts"
                    )
                    rec_texts = rec_texts.as_numpy()

                    rec_scores = pb_utils.get_output_tensor_by_name(
                        inference_response, "rec_scores"
                    )
                    rec_scores = rec_scores.as_numpy()

                    batch_rec_texts.append(rec_texts)
                    batch_rec_scores.append(rec_scores)

            out_tensor_0 = pb_utils.Tensor(
                self.output_names[0], np.array(batch_rec_texts, dtype=np.object_)
            )
            out_tensor_1 = pb_utils.Tensor(
                self.output_names[1], np.array(batch_rec_scores)
            )
            out_tensor_2 = pb_utils.Tensor(
                self.output_names[2], np.array(batch_box_list)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
