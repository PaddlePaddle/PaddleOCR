# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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


class VQAReTokenRelation(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        """
        build relations
        """
        entities = data["entities"]
        relations = data["relations"]
        id2label = data.pop("id2label")
        empty_entity = data.pop("empty_entity")
        entity_id_to_index_map = data.pop("entity_id_to_index_map")

        relations = list(set(relations))
        relations = [
            rel
            for rel in relations
            if rel[0] not in empty_entity and rel[1] not in empty_entity
        ]
        kv_relations = []
        for rel in relations:
            pair = [id2label[rel[0]], id2label[rel[1]]]
            if pair == ["question", "answer"]:
                kv_relations.append(
                    {
                        "head": entity_id_to_index_map[rel[0]],
                        "tail": entity_id_to_index_map[rel[1]],
                    }
                )
            elif pair == ["answer", "question"]:
                kv_relations.append(
                    {
                        "head": entity_id_to_index_map[rel[1]],
                        "tail": entity_id_to_index_map[rel[0]],
                    }
                )
            else:
                continue
        relations = sorted(
            [
                {
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "start_index": self.get_relation_span(rel, entities)[0],
                    "end_index": self.get_relation_span(rel, entities)[1],
                }
                for rel in kv_relations
            ],
            key=lambda x: x["head"],
        )

        data["relations"] = relations
        return data

    def get_relation_span(self, rel, entities):
        bound = []
        for entity_index in [rel["head"], rel["tail"]]:
            bound.append(entities[entity_index]["start"])
            bound.append(entities[entity_index]["end"])
        return min(bound), max(bound)
