# _*_ coding:utf-8 _*_

# @Author      : chenkai<chenkai15@geely.com>
# @Created Time: 2020/7/27 上午10:10
# @File        : tune_params_with_fmin.py

import mxnet as mx
from mxnet import gluon

import matplotlib.pyplot as plt
import json
from hyperopt import hp

from gluonts.dataset.repository.datasets import get_dataset, default_dataset_path
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

from gluonts.model.deepar import DeepAREstimator
from gluonts.distribution.gaussian import GaussianOutput
from gluonts.distribution.student_t import StudentTOutput
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator


def get_data(name='m4_hourly', regenerate=False):
    dataset = get_dataset(dataset_name=name, regenerate=regenerate)
    return dataset


def yield_data_iter(dataset, min_seq_len=748):
    data_iter = []
    for entry in iter(dataset.test):
        if len(entry['target']) != min_seq_len:  # 原始数据集中的长度不同
            data_entry = {
                FieldName.START: entry['start'],
                FieldName.TARGET: entry['target'],
                FieldName.FEAT_STATIC_CAT: entry['feat_static_cat'],
                FieldName.ITEM_ID: entry['item_id'],
            }
            data_iter.append(data_entry)
            yield data_iter
        else:
            pass


def get_list_dataset(dataset):
    data_iter = yield_data_iter(dataset)

    list_dataset = ListDataset(data_iter=data_iter,
                               freq=dataset.metadata.freq,
                               one_dim_target=True)

    single_entry = next(iter(dataset.test))

    entry_info = {'len/item_ids': len(list_dataset.list_data),
                  'start': single_entry['start'],
                  'target_shape': single_entry['target'].shape,
                  'feat_static_cat': single_entry['feat_static_cat'],
                  'item_id': single_entry['item_id'],
                  'freq': single_entry['start'].freq}

    print(entry_info)

    return list_dataset


def hp_space():
    space_num_layers = (2, 3)  # 层数
    space_num_cells = (30, 40, 50)  # 每层的神经元个数
    space_embedding_dimension = (None, [10], [20])
    space_cell_type = ('lstm', 'gru')

    space = {
        'num_layers': hp.choice('nl', space_num_layers),
        'num_cells': hp.choice('nc', space_num_cells),
        'embedding_dimension': hp.choice('ed', space_embedding_dimension),
        'cell_type': hp.choice('ct', space_cell_type),
        'learning_rate': hp.uniform('lr', 0.0001, 0.001),
    }

    return space


def objective(args):
    """定义调优目标"""

    global dataset  # 在全局查找dataset

    num_layers = args['num_layers']
    num_cells = args['num_cells']
    embedding_dimension = args['embedding_dimension']
    cell_type = args['cell_type']
    learning_rate = args['learning_rate']

    kwargs = {
        'freq': dataset.metadata.freq,
        'prediction_length': dataset.metadata.prediction_length,
        'trainer': Trainer(ctx="gpu", learning_rate=learning_rate, epochs=1, hybridize=False),
        'num_layers': num_layers,
        'num_cells': num_cells,
        'use_feat_static_cat': True,
        'embedding_dimension': embedding_dimension,
        'cardinality': [int(dataset.metadata.feat_static_cat[0].cardinality)],
        'cell_type': cell_type,
    }
    Estimator = DeepAREstimator(**kwargs)
    predictor = Estimator.train(dataset.train)  # 在训练集上做训练

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,  # dataset.test,  # test dataset　# 测试集上做评估
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(dataset.test))

    return agg_metrics['RMSE']


def turn_model():
    from hyperopt import fmin, tpe

    space = hp_space()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=2, return_argmin=False)

    print(best)
    return best


def train(dataset, best):
    kwargs = {
        'freq': dataset.metadata.freq,
        'prediction_length': dataset.metadata.prediction_length,
        'trainer': Trainer(ctx="gpu", learning_rate=best['learning_rate'], epochs=100, hybridize=False),
        'num_layers': best['num_layers'],
        'num_cells': best['num_cells'],
        'use_feat_static_cat': True,
        'embedding_dimension': best['embedding_dimension'],
        'cardinality': [int(dataset.metadata.feat_static_cat[0].cardinality)],
        'cell_type': best['cell_type'],
    }
    Estimator = DeepAREstimator(**kwargs)
    predictor = Estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
        num_samples=100,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))
    print(json.dumps(agg_metrics, indent=4))
    return predictor, agg_metrics, item_metrics


def main(tune=False):
    dataset = get_data()
    list_dataset = get_list_dataset(dataset)

    if tune:
        best_params = turn_model()
    else:
        best_params = {'cell_type': 'gru', 'embedding_dimension': (20,), 'learning_rate': 0.0005708878601215623,
                       'num_cells': 40, 'num_layers': 2}

    predictor, agg_metrics, item_metrics = train(dataset, best_params)
    print(agg_metrics)
    return predictor


if __name__ == '__main__':
    predictor = main(tune=False)
