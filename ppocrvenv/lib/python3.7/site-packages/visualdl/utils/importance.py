# Copyright (c) 2020 VisualDL Authors. All Rights Reserve.
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
# =======================================================================
from functools import reduce

import numpy as np
import pandas as pd

from visualdl.server.log import logger


def calc_hyper_param_importance(df, hyper_param, target):
    new_df = df[[hyper_param, target]]
    no_missing_value_df = new_df.dropna()

    # Can not calc pearson correlation coefficient when number of samples is less or equal than 2
    if len(no_missing_value_df) <= 2:
        logger.error("Number of samples is less or equal than 2.")
        return 0

    correlation = no_missing_value_df[target].corr(no_missing_value_df[hyper_param])
    if np.isnan(correlation):
        logger.warning("Correlation is nan!")
        return 0

    return abs(correlation)


def calc_all_hyper_param_importance(hparams, metrics):
    results = {}
    for metric in metrics:
        for hparam in hparams:
            flattened_lineage = {hparam['name']: hparam['values'], metric['name']: metric['values']}
            result = calc_hyper_param_importance(pd.DataFrame(flattened_lineage), hparam['name'], metric['name'])
            # print('%s - %s : result=' % (hparam, metric), result)
            if hparam['name'] not in results.keys():
                results[hparam['name']] = result
            else:
                results[hparam['name']] += result
    sum_score = reduce(lambda x, y: x+y, results.values())
    for key, value in results.items():
        results[key] = value/sum_score
    result = [{'name': key, 'value': value} for key, value in results.items()]
    return result
