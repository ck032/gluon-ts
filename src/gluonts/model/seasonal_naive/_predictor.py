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
from typing import Optional

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.evaluation import get_seasonality
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.support.pandas import forecast_start


class SeasonalNaivePredictor(RepresentablePredictor):
    """
    Seasonal naïve forecaster.

    For each time series :math:`y`, this predictor produces a forecast
    :math:`\\tilde{y}(T+k) = y(T+k-h)`, where :math:`T` is the forecast time,
    :math:`k = 0, ...,` `prediction_length - 1`, and :math:`h =`
    `season_length`.

     ref = data["target"][
              -SEASON_LENGTH: -SEASON_LENGTH + PREDICTION_LENGTH
              ]

     SeasonalNaivePredictor：在原始target数据中倒着从-SEASON_LENGTH开始取，取PREDICTION_LENGTH个值就是预测值了
     假设，SEASON_LENGTH=30,预测长度PREDICTION_LENGTH=10，就从一个月之前取10条记录就可以了
     这个算法好简单

    If `prediction_length > season_length`, then the season is repeated
    multiple times. If a time series is shorter than season_length, then the
    mean observed value is used as prediction.

    Parameters
    ----------
    freq
        Frequency of the input data
    prediction_length
        Number of time points to predict
    season_length
        Length of the seasonality pattern of the input data
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        season_length: Optional[int] = None,
    ) -> None:
        super().__init__(freq=freq, prediction_length=prediction_length)

        assert (
            season_length is None or season_length > 0
        ), "The value of `season_length` should be > 0"

        self.freq = freq
        self.prediction_length = prediction_length
        self.season_length = (
            season_length
            if season_length is not None
            else get_seasonality(freq) # 这样利用小括号的写法值得学习，另外，可以通过分析freq，得到season_length
        ) # 这半个小括号都是单独成行

    def predict_item(self, item: DataEntry) -> Forecast:
        target = np.asarray(item["target"], np.float32)
        len_ts = len(target)
        forecast_start_time = forecast_start(item) # 预测的起始时间

        assert (
            len_ts >= 1
        ), "all time series should have at least one data point"

        if len_ts >= self.season_length: # 这个是常态
            indices = [
                len_ts - self.season_length + k % self.season_length
                for k in range(self.prediction_length)
            ]
            samples = target[indices].reshape((1, self.prediction_length))
        else:
            samples = np.full(
                shape=(1, self.prediction_length), fill_value=target.mean()
            ) # np.null，所有的元素都用target的均值填充

        return SampleForecast(samples, forecast_start_time, self.freq)
