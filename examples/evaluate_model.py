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

"""
This example shows how to fit a model and evaluate its predictions.
"""

# 这样可以保证调试源码的时候，用的是git下载的源码，注意要到src目录
import sys
sys.path.insert(0,'/home/chenkai/Documents/git_projects/gluon-ts/src')


import pprint

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

if __name__ == "__main__":
    # 可用的数据集名称，recipes-食谱
    print(f"datasets available: {dataset_recipes.keys()}")

    # we pick m4_hourly as it only contains a few hundred time series
    # 这里采用m4_hourly数据集，因为数据在本地了，所以不需要重新下载
    # 这个数据集的train.target 有700个值,test = 700 + 48 = 748
    dataset = get_dataset("m4_hourly", regenerate=False)

    print('dataset:\n',dataset)
    print('dataset.metadata:\n',dataset.metadata)
    print(len(dataset.train), len(dataset.test))  # 414行

    print(next(iter(dataset.train))['target'].shape)
    print(next(iter(dataset.test))['target'].shape)

    # simple-MLP网络
    # 迭代了epochs=5次，num_batches_per_epoch=10,每迭代一次，需要size(dataset)/num_batches_per_epoch次iteration
    estimator = DeepAREstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        trainer=Trainer(epochs=5, num_batches_per_epoch=10),

    )

    predictor = estimator.train(dataset.train)

    print(predictor.lead_time)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_samples=100
    )

    # item_metrics可以理解为每个样本的评估
    # 比如，多次充电的数据，形成多个样本，每个样本用前700个数据点（温度值）来预测后48个数据点（温度值）
    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test) # 因为Evaluator定义了__call__，所以可以这么传参
    )

    pprint.pprint(agg_metrics)
    pprint.pprint(item_metrics)
