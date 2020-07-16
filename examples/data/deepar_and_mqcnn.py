# _*_ coding:utf-8 _*_

# @Author      : chenkai<chenkai15@geely.com>
# @Created Time: 2020/7/16 下午1:09
# @File        : deepar_and_mqcnn.py.py

docs = """
This is the sample code to check the environment for running GluonTS.

Before running sample.py, please check requirement.txt and install required packages.

Sample.py includes two deep learning models, DeepAR and MQCNN.

Papers about two models for time series forecasting:

DeepAR: https://arxiv.org/pdf/1704.04110.pdf

MQCNN: https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_conv/ (wavenet)"""


from functools import partial

import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator

epochs = 5
num_batches_per_epoch = 10
dataset_name = "m4_hourly"

dataset = get_dataset(dataset_name,regenerate=False)

# If you want to use GPU, please set ctx="gpu(0)"　/ ctx="cpu"
estimators = [
    partial(
        SimpleFeedForwardEstimator,
        trainer=Trainer(
            ctx="gpu",
            epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            learning_rate= 0.01
        )
    ),
    partial(
        MQCNNEstimator,
        trainer=Trainer(
            ctx="gpu",
            epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            learning_rate= 0.01
        )
    ),
]

results = []
for estimator in estimators:
    estimator = estimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq
    )
    predictor = estimator.train(dataset.train)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    eval_dict = agg_metrics
    eval_dict["dataset"] = dataset_name
    eval_dict["estimator"] = type(estimator).__name__
    results.append(eval_dict)

df = pd.DataFrame(results)
sub_df = df[
    [
        "dataset",
        "estimator",
        "mean_wQuantileLoss",
    ]
]
print(sub_df)