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

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.model.trivial.mean import MovingAveragePredictor

# Third-party imports
import numpy as np
import pytest


# 注意，MovingAveragePredictor是支持freq='10S'的

def get_predictions(
    target, prediction_length=1, context_length=1, freq="D", start="2020"
):
    mp = MovingAveragePredictor(
        prediction_length=prediction_length,
        context_length=context_length,
        freq=freq,
    )

    ds = ListDataset([{"target": target, "start": start}], freq=freq)
    item = next(iter(ds))
    predictions = mp.predict_item(item).mean

    return predictions

# 1.利用这个可以把数据传到testing函数中去
# 2.这儿不用写遍历，直接针对各种情况来做判断了
@pytest.mark.parametrize(
    "data, expected_output, prediction_length, context_length",
    [
        ([1, 1, 1], [1], 1, 1),
        ([1, 1, 1], [1, 1], 2, 1),
        ([1, 1, 1], [1, 1, 1], 3, 1),
        ([1, 1, 1], [1], 1, 2),
        ([1, 1, 1], [1, 1], 2, 2),
        ([1, 1, 1], [1, 1, 1], 3, 2),
        ([1, 1, 1], [1], 1, 3),
        ([1, 1, 1], [1, 1], 2, 3),
        ([1, 1, 1], [1, 1, 1], 3, 3),
        ([], [np.nan] * 1, 1, 1),
        ([], [np.nan] * 2, 2, 1),
        ([], [np.nan] * 3, 3, 1),
        ([np.nan], [np.nan] * 1, 1, 1),
        ([1, 3, np.nan], [2], 1, 3),
        ([1, 3, np.nan], [2, 2.5], 2, 3),
        ([1, 3, np.nan], [2, 2.5, 2.25], 3, 3),
        ([1, 2, 3], [3], 1, 1),
        ([1, 2, 3], [3, 3], 2, 1),
        ([1, 2, 3], [3, 3, 3], 3, 1),
        ([1, 2, 3], [2.5], 1, 2),
        ([1, 2, 3], [2.5, 2.75], 2, 2),
        ([1, 2, 3], [2.5, 2.75, 2.625], 3, 2),
        ([1, 2, 3], [2], 1, 3),
        ([1, 2, 3], [2, 7 / 3], 2, 3),
        ([1, 2, 3], [2, 7 / 3, 22 / 9], 3, 3), # 2 = (1+2+3) /3 , 7 / 3 = (2+3 + 2) / 3 ,22 /9 = (3 + 2 + 7/3 ) / 3 移动平均，用到了预测部分的值
        ([1, 1, 1], [1], 1, None), # 不指定context_length时，也就是context_length=None的时候，就是全部来求平均,所以，预测的时候，预测值都是相同的。
        ([1, 1, 1], [1, 1], 2, None),
        ([1, 1, 1], [1, 1, 1], 3, None),
        ([1, 3, np.nan], [2], 1, None),
        ([1, 3, np.nan], [2, 2], 2, None),
        ([1, 3, np.nan], [2, 2, 2], 3, None),
    ],
)
def testing(data, expected_output, prediction_length, context_length):

    predictions = get_predictions(
        target=data,
        prediction_length=prediction_length,
        context_length=context_length,
    )

    np.testing.assert_equal(predictions, expected_output)
