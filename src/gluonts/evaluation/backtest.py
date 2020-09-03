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
####################################################
# 　评估脚本，最重要的是make_evaluation_predictions/evaluator
####################################################

# Standard library imports
import logging
import re
from typing import Dict, Iterator, NamedTuple, Optional, Tuple

# Third-party imports
import pandas as pd

# First-party imports
import gluonts  # noqa
from gluonts import transform
from gluonts.core.serde import load_code
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.stat import (
    DatasetStatistics,
    calculate_dataset_statistics,
)
from gluonts.evaluation import Evaluator
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from gluonts.support.util import maybe_len
from gluonts.transform import TransformedDataset


def make_evaluation_predictions(
        dataset: Dataset, predictor: Predictor, num_samples: int
) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
    """
    Return predictions on the last portion of predict_length time units of the
    target. Such portion is cut before making predictions, such a function can
    be used in evaluations where accuracy is evaluated on the last portion of*
    the target.

    Parameters
    ----------
    dataset
        Dataset where the evaluation will happen. Only the portion excluding
        the prediction_length portion is used when making prediction.

        针对的是test data，所以只有 prediction_length比例的部分被拿出来做预测

    predictor
        Model used to draw predictions.
    num_samples
        Number of samples to draw on the model when evaluating.

    Returns
    -------
    """

    # 注意，这三个字段都是predictor带出来的
    prediction_length = predictor.prediction_length  # 预测的长度
    freq = predictor.freq  # 预测的频率
    lead_time = predictor.lead_time  # 观察期

    # 注意，采用了闭包的写法，也就是函数嵌套函数。闭包是一个独立的运行环境。
    def add_ts_dataframe(
            data_iterator: Iterator[DataEntry],
    ) -> Iterator[DataEntry]:
        """把target序列转化为pd.DataFrame"""
        for data_entry in data_iterator:  # 一个data_entry可以看成是一个观测序列
            data = data_entry.copy()
            # 利用date_range函数，造出来index
            index = pd.date_range(
                start=data["start"],
                freq=freq,
                periods=data["target"].shape[-1],  # 取到长度，利用pd.date_range函数，构建时间的index 7
            )
            data["ts"] = pd.DataFrame(
                index=index, data=data["target"].transpose()  # ts表示ｔime_series的简写，注意这儿用了转置，说明原来的target行是观测，列是日期
            )
            yield data

    def ts_iter(dataset: Dataset) -> pd.DataFrame:
        # 针对dataset，yield每个target序列
        for data_entry in add_ts_dataframe(iter(dataset)):
            yield data_entry["ts"]

    def truncate_target(data):
        data = data.copy()
        target = data["target"]
        assert (
                target.shape[-1] >= prediction_length
        )  # handles multivariate case (target_dim, history_length)
        # data = np.arange(10) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # data[...,:-3] # array([0, 1, 2, 3, 4, 5, 6])
        data["target"] = target[..., : -prediction_length - lead_time] # predictor的lead_time是一个数值，应该是指模型的观察期，这样就把target给截断了
        return data

    # TODO filter out time series with target shorter than prediction length
    # TODO or fix the evaluator so it supports missing values instead (all
    # TODO the test set may be gone otherwise with such a filtering)

    # 这儿利用TransformedDataset的方法
    # 注意：truncate_target是一个普通的函数，利用AdhocTransform，把普通的函数注册为Transform
    dataset_trunc = TransformedDataset(
        dataset, transformations=[transform.AdhocTransform(truncate_target)]
    )
    # 返回的结果有两个部分：预测值(包含prediction_length＋lead_time），原始数据集
    return (
        predictor.predict(dataset_trunc, num_samples=num_samples),
        ts_iter(dataset),
    )


train_dataset_stats_key = "train_dataset_stats"
test_dataset_stats_key = "test_dataset_stats"
estimator_key = "estimator"
agg_metrics_key = "agg_metrics"

# logger信息格式化，message就是上面定义的几个
# 如果logger有指定的话，会自动序列化信息到logger中
def serialize_message(logger, message: str, variable):
    logger.info(f"gluonts[{message}]: {variable}")


# 这个函数，输出的是评估指标
# vs make_evaluation_predictions,输出的是预测值和真实值序列
# vs evaluator 多了序列化的功能
# vs Evaluator　类有更多的可自定义的内容，否则用 backtest_metrics、make_evaluation_predictions两个函数就好了
def backtest_metrics(
    test_dataset: Dataset,
    predictor: Predictor,
    evaluator=Evaluator(
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ),
    num_samples: int = 100,
    logging_file: Optional[str] = None,
):
    """
    Parameters
    ----------
    test_dataset
        Dataset to use for testing.
    predictor
        The predictor to test.
    evaluator
        Evaluator to use.
    num_samples
        Number of samples to use when generating sample-based forecasts.
        基于 sample-based 的预测，指定的sample数量。初步的理解是，针对多少个观测（也就是行）的数据来做验证。默认是１００个观测。
    logging_file
        If specified, information of the backtest is redirected to this file.

    Returns
    -------
    tuple
        A tuple of aggregate metrics and per-time-series metrics obtained by
        training `forecaster` on `train_dataset` and evaluating the resulting
        `evaluator` provided on the `test_dataset`.
    """

    # 这儿的logging写法是个范例
    if logging_file is not None:
        log_formatter = logging.Formatter(
            "[%(asctime)s %(levelname)s %(thread)d] %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(logging_file)
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)
    else:
        logger = logging.getLogger(__name__)

    test_statistics = calculate_dataset_statistics(test_dataset)
    serialize_message(logger, test_dataset_stats_key, test_statistics)

    # 由 predictor作用在test_data上，得到真实值和预测值序列
    forecast_it, ts_it = make_evaluation_predictions(
        test_dataset, predictor=predictor, num_samples=num_samples
    )

    # 计算评估指标
    agg_metrics, item_metrics = evaluator(
        ts_it, forecast_it, num_series=maybe_len(test_dataset)
    )

    # we only log aggregate metrics for now as item metrics may be very large
    # 记录汇总的评估指标
    for name, value in agg_metrics.items():
        serialize_message(logger, f"metric-{name}", value)

    if logging_file is not None:
        # Close the file handler to avoid letting the file open.
        # https://stackoverflow.com/questions/24816456/python-logging-wont-shutdown
        logger.removeHandler(handler)
        del logger, handler

    return agg_metrics, item_metrics


# TODO does it make sense to have this then?
class BacktestInformation(NamedTuple):
    train_dataset_stats: DatasetStatistics
    test_dataset_stats: DatasetStatistics
    estimator: Estimator
    agg_metrics: Dict[str, float]

    @staticmethod
    def make_from_log(log_file):
        with open(log_file, "r") as f:
            return BacktestInformation.make_from_log_contents(
                "\n".join(f.readlines())
            )

    @staticmethod
    def make_from_log_contents(log_contents):
        messages = dict(re.findall(r"gluonts\[(.*)\]: (.*)", log_contents))

        # avoid to fail if a key is missing for instance in the case a run did
        # not finish so that we can still get partial information
        try:
            return BacktestInformation(
                train_dataset_stats=eval(
                    messages[train_dataset_stats_key]
                ),  # TODO: use load
                test_dataset_stats=eval(
                    messages[test_dataset_stats_key]
                ),  # TODO: use load
                estimator=load_code(messages[estimator_key]),
                agg_metrics={
                    k: load_code(v)
                    for k, v in messages.items()
                    if k.startswith("metric-") and v != "nan"
                },
            )
        except Exception as error:
            logging.error(error)
            return None
