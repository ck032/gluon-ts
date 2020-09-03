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

# Third-party imports
import statsmodels.api as sm

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.support.pandas import forecast_start
from gluonts.time_feature import get_seasonality


def seasonality_test(past_ts_data: np.array, season_length: int) -> bool:
    """
    Test the time-series for seasonal patterns by performing a 90% auto-correlation test:

    # TODO：这儿涉及到具体的算法，需要读算法文档才行
    测试时间序列是否有季节相关性

    As described here: https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    Code based on: https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
    """
    critical_z_score = 1.645  # corresponds to 90% confidence interval

    # 序列的长度小于3倍的season_length，直接没有相关性
    if len(past_ts_data) < 3 * season_length:
        return False
    else:
        # calculate auto-correlation for lags up to season_length
        # 怎么判断的？ 这儿用了statsmodels的包
        auto_correlations = sm.tsa.stattools.acf(
            past_ts_data, fft=False, nlags=season_length
        )
        auto_correlations[1:] = 2 * auto_correlations[1:] ** 2
        limit = (
            critical_z_score
            / np.sqrt(len(past_ts_data))
            * np.sqrt(np.cumsum(auto_correlations))
        )
        is_seasonal = (
            abs(auto_correlations[season_length]) > limit[season_length]
        )

    return is_seasonal


def naive_2(
    past_ts_data: np.ndarray,
    prediction_length: int,
    freq: Optional[str] = None,
    season_length: Optional[int] = None,
) -> np.ndarray:
    """
    Make seasonality adjusted time series prediction.
    时序的季节性调整

    If specified, `season_length` takes precedence.

    As described here: https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    Code based on: https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
    """
    assert freq is not None or season_length is not None, (
        "Either the frequency or season length of the time series "
        "has to be specified. "
    )
    season_length = (
        season_length if season_length is not None else get_seasonality(freq)
    )
    has_seasonality = False

    if season_length > 1:
        has_seasonality = seasonality_test(past_ts_data, season_length)

    # it has seasonality, then calculate the multiplicative seasonal component
    if has_seasonality:
        # TODO: think about maybe only using past_ts_data[- max(5*season_length, 2*prediction_length):] for speedup
        # 利用sm中的季节因素分解函数来做的
        # 1.获取季节调整指数
        # 2.利用季节调整指数调整原序列
        # 3.获取季节成分multiplicative_seasonal_component
        # 4.二者想乘得到计算结果
        seasonal_decomposition = sm.tsa.seasonal_decompose(
            x=past_ts_data, period=season_length, model="multiplicative"
        ).seasonal
        seasonality_normed_context = past_ts_data / seasonal_decomposition

        last_period = seasonal_decomposition[-season_length:]
        num_required_periods = (prediction_length // len(last_period)) + 1

        multiplicative_seasonal_component = np.tile(
            last_period, num_required_periods
        )[:prediction_length]  # np.tile是重复序列的作用
    else:
        seasonality_normed_context = past_ts_data
        multiplicative_seasonal_component = np.ones(
            prediction_length
        )  # i.e. no seasonality component

    # calculate naive forecast: (last value prediction_length times)
    # 没有季节相关性是，直接以原始数据的最后的一个值，做预测
    # 都是取seasonality_normed_context[-1],最后一个值
    naive_forecast = (
        np.ones(prediction_length) * seasonality_normed_context[-1]
    )

    forecast = np.mean(naive_forecast) * multiplicative_seasonal_component

    return forecast


class Naive2Predictor(RepresentablePredictor):
    """
    Naïve 2 forecaster as described in the M4 Competition Guide:
    https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf

    The python analogue implementation to:
    https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R#L118

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
        # super() __init__的方法继承了（且重写）父类，也就是RepresentablePredictor中的__init__中的两个参数(freq,prediction_length)
        # 并且子类可以重新有自己的参数，比如season_length

        assert (
            season_length is None or season_length > 0
        ), "The value of `season_length` should be > 0"

        self.freq = freq
        self.prediction_length = prediction_length
        self.season_length = (
            season_length
            if season_length is not None
            else get_seasonality(freq)
        )

    def predict_item(self, item: DataEntry) -> Forecast:
        past_ts_data = item["target"]
        forecast_start_time = forecast_start(item) # 获取到预测的起始时间

        assert (
            len(past_ts_data) >= 1
        ), "all time series should have at least one data point"

        prediction = naive_2(past_ts_data, self.prediction_length, self.freq)

        samples = np.array([prediction])  # sample是预测结果，是一个np.array

        return SampleForecast(samples, forecast_start_time, self.freq)
