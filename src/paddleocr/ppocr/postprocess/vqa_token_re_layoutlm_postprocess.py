# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle


class VQAReTokenLayoutLMPostProcess(object):
    """Convert between text-label and text-index"""

    def __init__(self, **kwargs):
        super(VQAReTokenLayoutLMPostProcess, self).__init__()

    def __call__(self, preds, label=None, *args, **kwargs):
        pred_relations = preds["pred_relations"]
        if isinstance(preds["pred_relations"], paddle.Tensor):
            pred_relations = pred_relations.numpy()
        pred_relations = self.decode_pred(pred_relations)

        if label is not None:
            return self._metric(pred_relations, label)
        else:
            return self._infer(pred_relations, *args, **kwargs)

    def _metric(self, pred_relations, label):
        return pred_relations, label[-1], label[-2]

    def _infer(self, pred_relations, *args, **kwargs):
        ser_results = kwargs["ser_results"]
        entity_idx_dict_batch = kwargs["entity_idx_dict_batch"]

        # merge relations and ocr info
        results = []
        for pred_relation, ser_result, entity_idx_dict in zip(
            pred_relations, ser_results, entity_idx_dict_batch
        ):
            result = []
            used_tail_id = []
            for relation in pred_relation:
                if relation["tail_id"] in used_tail_id:
                    continue
                used_tail_id.append(relation["tail_id"])
                ocr_info_head = ser_result[entity_idx_dict[relation["head_id"]]]
                ocr_info_tail = ser_result[entity_idx_dict[relation["tail_id"]]]
                result.append((ocr_info_head, ocr_info_tail))
            results.append(result)
        return results

    def decode_pred(self, pred_relations):
        pred_relations_new = []
        for pred_relation in pred_relations:
            pred_relation_new = []
            pred_relation = pred_relation[1 : pred_relation[0, 0, 0] + 1]
            for relation in pred_relation:
                relation_new = dict()
                relation_new["head_id"] = relation[0, 0]
                relation_new["head"] = tuple(relation[1])
                relation_new["head_type"] = relation[2, 0]
                relation_new["tail_id"] = relation[3, 0]
                relation_new["tail"] = tuple(relation[4])
                relation_new["tail_type"] = relation[5, 0]
                relation_new["type"] = relation[6, 0]
                pred_relation_new.append(relation_new)
            pred_relations_new.append(pred_relation_new)
        return pred_relations_new


class DistillationRePostProcess(VQAReTokenLayoutLMPostProcess):
    """
    DistillationRePostProcess
    """

    def __init__(self, model_name=["Student"], key=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name
        self.key = key

    def __call__(self, preds, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            output[name] = super().__call__(pred, *args, **kwargs)
        return output
