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
from typing import List, Optional

# Third-party imports
import numpy as np
from pandas.tseries.frequencies import to_offset


def _make_lags(middle: int, delta: int) -> np.ndarray:
    """
    Create a set of lags around a middle point including +/- delta
    比如，_make_lags(10,0.5) = [9.5,10.5],10刚好在中间
    """
    return np.arange(middle - delta, middle + delta + 1).tolist()


def get_lags_for_frequency(
    freq_str: str, lag_ub: int = 1200, num_lags: Optional[int] = None
) -> List[int]:
    """
    Generates a list of lags that that are appropriate for the given frequency string.

    By default all frequencies have the following lags: [1, 2, 3, 4, 5, 6, 7].
    Remaining lags correspond to the same `season` (+/- `delta`) in previous `k` cycles.
    Here `delta` and `k` are chosen according to the existing code.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    lag_ub  up-bound
        The maximum value for a lag.

    num_lags
        Maximum number of lags; by default all generated lags are returned
    """

    # Lags are target values at the same `season` (+/- delta) but in the previous cycle.
    def _make_lags_for_minute(multiple, num_cycles=3):
        # We use previous ``num_cycles`` hours to generate lags
        return [
            _make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_hour(multiple, num_cycles=7):
        # We use previous ``num_cycles`` days to generate lags
        return [
            _make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_day(multiple, num_cycles=4):
        # We use previous ``num_cycles`` weeks to generate lags
        # We use the last month (in addition to 4 weeks) to generate lag.
        return [
            _make_lags(k * 7 // multiple, 1) for k in range(1, num_cycles + 1)
        ] + [_make_lags(30 // multiple, 1)]

    def _make_lags_for_week(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        # Additionally, we use previous 4, 8, 12 weeks
        return [
            _make_lags(k * 52 // multiple, 1) for k in range(1, num_cycles + 1)
        ] + [[4 // multiple, 8 // multiple, 12 // multiple]]

    def _make_lags_for_month(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        # 1m - [[11, 12, 13], [23, 24, 25], [35, 36, 37]]，以 multiple=1为例子，考虑了12个月、24个月、36个月的滞后项数据，并且左右摆动1个单位
        # 2m - 半个月 - [[5, 6, 7], [11, 12, 13], [17, 18, 19]]
        return [
            _make_lags(k * 12 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    # multiple, granularity = get_granularity(freq_str)
    offset = to_offset(freq_str)

    if offset.name == "M":
        lags = _make_lags_for_month(offset.n)
    elif offset.name == "W-SUN":
        lags = _make_lags_for_week(offset.n)
    elif offset.name == "D":
        lags = _make_lags_for_day(offset.n) + _make_lags_for_week(
            offset.n / 7.0
        )
    elif offset.name == "B":
        # todo find good lags for business day
        lags = []
    elif offset.name == "H":
        lags = (
            _make_lags_for_hour(offset.n)
            + _make_lags_for_day(offset.n / 24.0)
            + _make_lags_for_week(offset.n / (24.0 * 7))
        )
    # minutes
    elif offset.name == "T":
        lags = (
            _make_lags_for_minute(offset.n)   # offset.n 越大，比如1min,是以[60,120,180]为中心，2min是以[30,60,120]为中心，数值越大，lag项越小
            + _make_lags_for_hour(offset.n / 60.0)
            + _make_lags_for_day(offset.n / (60.0 * 24))
            + _make_lags_for_week(offset.n / (60.0 * 24 * 7))
        )
    else:
        raise Exception("invalid frequency")

    # flatten lags list and filter
    lags = [
        int(lag) for sub_list in lags for lag in sub_list if 7 < lag <= lag_ub
    ]
    lags = [1, 2, 3, 4, 5, 6, 7] + sorted(list(set(lags)))  # 默认是考虑1-7个滞后项的,其实滞后项的选择，带有人为的性质,需要跟进具体的业务问题，考虑用多久的滞后期，自带的周期性起始并不好

    return lags[:num_lags]
