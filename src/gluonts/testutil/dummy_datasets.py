# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from random import randint
from typing import List, Tuple

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

# Mark:怎么造数据的脚本

def make_dummy_datasets_with_features(
    num_ts: int = 5,
    start: str = "2018-01-01",
    freq: str = "D",
    min_length: int = 5,
    max_length: int = 10,
    prediction_length: int = 3,
    cardinality: List[int] = [],
    num_feat_dynamic_real: int = 0,
    num_past_feat_dynamic_real: int = 0,
) -> Tuple[ListDataset, ListDataset]:

    data_iter_train = []
    data_iter_test = []

    # data_entry_train 是一个dict，字段名称用FieldName来标记
    # FieldName.FEAT_DYNAMIC_REAL 产生的数据形状和target长度相同
    for k in range(num_ts):
        ts_length = randint(min_length, max_length)  # 随机产生的一个长度
        data_entry_train = {
            FieldName.START: start,
            FieldName.TARGET: [0.0] * ts_length,
        }
        if len(cardinality) > 0:
            data_entry_train[FieldName.FEAT_STATIC_CAT] = [
                randint(0, c) for c in cardinality
            ]
        if num_past_feat_dynamic_real > 0:
            data_entry_train[FieldName.PAST_FEAT_DYNAMIC_REAL] = [
                [float(1 + k)] * ts_length
                for k in range(num_past_feat_dynamic_real)
            ]
        # Since used directly in predict and not in make_evaluate_predictions,
        # where the test target would be chopped（切片）, test and train target have the same lengths
        data_entry_test = data_entry_train.copy()
        if num_feat_dynamic_real > 0:
            data_entry_train[FieldName.FEAT_DYNAMIC_REAL] = [
                [float(1 + k)] * ts_length
                for k in range(num_feat_dynamic_real)
            ]   #
            data_entry_test[FieldName.FEAT_DYNAMIC_REAL] = [
                [float(1 + k)] * (ts_length + prediction_length)
                for k in range(num_feat_dynamic_real)
            ]
        data_iter_train.append(data_entry_train)
        data_iter_test.append(data_entry_test)

    return (
        ListDataset(data_iter=data_iter_train, freq=freq),
        ListDataset(data_iter=data_iter_test, freq=freq),
    )
