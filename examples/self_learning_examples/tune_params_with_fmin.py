# _*_ coding:utf-8 _*_

# @Author      : chenkai<chenkai15@geely.com>
# @Created Time: 2020/7/27 上午10:10
# @File        : tune_params_with_fmin.py

import mxnet as mx
from mxnet import gluon

import matplotlib.pyplot as plt
import json
import hyperopt

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


def yield_data_iter(dataset):
    data_iter = []
    for entry in iter(dataset.test):
        if len(entry['target']) == 748:  # 长度不同
            data_entry = {
                FieldName.START: entry['start'],
                FieldName.TARGET: entry['target'],
                FieldName.FEAT_STATIC_CAT: entry['feat_static_cat'],
                FieldName.ITEM_ID: entry['item_id'],
            }
        data_iter.append(data_entry)
        yield data_iter


def get_list_dataset(dataset):

    # data_iter = [{
    #     FieldName.START: entry['start'],
    #     FieldName.TARGET: entry['target'],
    #     FieldName.FEAT_STATIC_CAT: entry['feat_static_cat'],
    #     FieldName.ITEM_ID: entry['item_id'],
    # }
    #     for entry in iter(dataset.test) if len(entry['target']) == 748  # 长度不同
    # ]

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


if __name__ == '__main__':
    dataset = get_data()
    list_dataset = get_list_dataset(dataset)
