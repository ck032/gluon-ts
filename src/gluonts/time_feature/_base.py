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

from typing import List

# Third-party imports
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# First-party imports
from gluonts.core.component import validated


class TimeFeature:
    """
    Base class for features that only depend on time.

    时间方面的特征
    """

    @validated()
    def __init__(self, normalized: bool = True):
        # 正常情况下，需要标准化
        self.normalized = normalized

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MinuteOfHour(TimeFeature):
    """
    Minute of hour encoded as value between [-0.5, 0.5]

    因为分钟是0-59，所以除以59，这样0分的标准化的值是-0.5,59分钟的标准话的值是0.5
    同理，
    MinuteOfHour - 分钟
    HourOfDay - 小时
    DayOfWeek - 周几
    DayOfMonth - 几号
    DayOfYear - 1年内的第几天
    MonthOfYear - 月份
    WeekOfYear - 周数
    都被标准化进入到模型中
    如果不标准化的话，直接输出的是float类型的数值
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.minute / 59.0 - 0.5
        else:
            return index.minute.map(float)


class HourOfDay(TimeFeature):
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.hour / 23.0 - 0.5
        else:
            return index.hour.map(float)


class DayOfWeek(TimeFeature):
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.dayofweek / 6.0 - 0.5
        else:
            return index.dayofweek.map(float)


class DayOfMonth(TimeFeature):
    """
    Day of month encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.day / 30.0 - 0.5
        else:
            return index.day.map(float)


class DayOfYear(TimeFeature):
    """
    Day of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.dayofyear / 364.0 - 0.5
        else:
            return index.dayofyear.map(float)


class MonthOfYear(TimeFeature):
    """
    Month of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.month / 11.0 - 0.5
        else:
            return index.month.map(float)


class WeekOfYear(TimeFeature):
    """
    Week of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.weekofyear / 51.0 - 0.5
        else:
            return index.weekofyear.map(float)


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    """

    features_by_offsets = {
        offsets.YearOffset: [],
        offsets.MonthOffset: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:

        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
    """
    raise RuntimeError(supported_freq_msg)
