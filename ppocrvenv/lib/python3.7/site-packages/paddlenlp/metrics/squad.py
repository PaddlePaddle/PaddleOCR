# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

import collections
import re
import string
import json
import numpy as np


def compute_prediction(examples,
                       features,
                       predictions,
                       version_2_with_negative=False,
                       n_best_size=20,
                       max_answer_length=30,
                       null_score_diff_threshold=0.0):
    """
    Post-processes the predictions of a question-answering model to convert 
    them to answers that are substrings of the original contexts. This is 
    the base postprocessing functions for models that only return start and 
    end logits.

    Args:
        examples (list): List of raw squad-style data (see `run_squad.py 
            <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/
            machine_reading_comprehension/SQuAD/run_squad.py>`__ for more 
            information).
        features (list): List of processed squad-style features (see 
            `run_squad.py <https://github.com/PaddlePaddle/PaddleNLP/blob/
            develop/examples/machine_reading_comprehension/SQuAD/run_squad.py>`__
            for more information).
        predictions (tuple): The predictions of the model. Should be a tuple
            of two list containing the start logits and the end logits.
        version_2_with_negative (bool, optional): Whether the dataset contains
            examples with no answers. Defaults to False.
        n_best_size (int, optional): The total number of candidate predictions
            to generate. Defaults to 20.
        max_answer_length (int, optional): The maximum length of predicted answer.
            Defaults to 20.
        null_score_diff_threshold (float, optional): The threshold used to select
            the null answer. Only useful when `version_2_with_negative` is True.
            Defaults to 0.0.
    
    Returns:
        A tuple of three dictionaries containing final selected answer, all n_best 
        answers along with their probability and scores, and the score_diff of each 
        example.
    """
    assert len(
        predictions
    ) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features), "Number of predictions should be equal to number of features."

    # Build a map example to its corresponding features.
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    scores_diff_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example['id']]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction[
                    "score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:
                                                     -1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1].tolist(
            )
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (start_index >= len(offset_mapping) or
                            end_index >= len(offset_mapping) or
                            offset_mapping[start_index] is None or
                            offset_mapping[end_index] is None or
                            offset_mapping[start_index] == (0, 0) or
                            offset_mapping[end_index] == (0, 0)):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(
                            str(start_index), False):
                        continue
                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0],
                                    offset_mapping[end_index][1]),
                        "score":
                        start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    })
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"],
            reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0)
                                               for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]:offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and
                                     predictions[0]["text"] == ""):
            predictions.insert(0, {
                "text": "empty",
                "start_logit": 0.0,
                "end_logit": 0.0,
                "score": 0.0
            })

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred[
                "start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(
                score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [{
            k: (float(v)
                if isinstance(v, (np.float16, np.float32, np.float64)) else v)
            for k, v in pred.items()
        } for pred in predictions]

    return all_predictions, all_nbest_json, scores_diff_json


def make_qid_to_has_ans(examples):
    qid_to_has_ans = {}
    for example in examples:
        qid_to_has_ans[example['id']] = not example.get('is_impossible', False)
    return qid_to_has_ans


def normalize_answer(s):
    #Lower text and remove punctuation, articles and extra whitespace.
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if not s:
        return ''
    else:
        return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred, is_whitespace_splited=True):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()

    if not is_whitespace_splited:
        gold_toks = gold_toks[0] if gold_toks else ""
        pred_toks = pred_toks[0] if pred_toks else ""

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples, preds, is_whitespace_splited=True):
    exact_scores = {}
    f1_scores = {}
    for example in examples:
        qid = example['id']
        gold_answers = [
            text for text in example['answers'] if normalize_answer(text)
        ]
        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = ['']
        if qid not in preds:
            print('Missing prediction for %s' % qid)
            continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(
            compute_f1(a, a_pred, is_whitespace_splited) for a in gold_answers)

    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores: continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs,
                         qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs,
                                                qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs,
                                          qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def squad_evaluate(examples,
                   preds,
                   na_probs=None,
                   na_prob_thresh=1.0,
                   is_whitespace_splited=True):
    '''
    Computes and prints the f1 score and em score of input prediction.

    Args:
        examples (list): List of raw squad-style data (see `run_squad.py
            <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/
            machine_reading_comprehension/SQuAD/run_squad.py>`__ for more
            information).
        preds (dict): Dictionary of final predictions. Usually generated by
            `compute_prediction`.
        na_probs (dict, optional): Dictionary of score_diffs of each example.
            Used to decide if answer exits and compute best score_diff
            threshold of null. Defaults to None.
        na_prob_thresh (float, optional): The threshold used to select the
            null answer. Defaults to 1.0.
        is_whitespace_splited (bool, optional): Whether the predictions and references
            can be tokenized by whitespace. Usually set True for English and
            False for Chinese. Defaults to True.
    '''

    if not na_probs:
        na_probs = {k: 0.0 for k in preds}

    qid_to_has_ans = make_qid_to_has_ans(examples)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(examples, preds, is_whitespace_splited)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                          na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                       na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(
            exact_thresh, f1_thresh, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(
            exact_thresh, f1_thresh, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
        find_all_best_thresh(out_eval, preds, exact_raw, f1_raw, na_probs,
                             qid_to_has_ans)

    print(json.dumps(out_eval, indent=2))
