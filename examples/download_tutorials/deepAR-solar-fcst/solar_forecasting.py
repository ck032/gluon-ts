# _*_ coding:utf-8 _*_

# @Author      : chenkai<chenkai15@geely.com>
# @Created Time: 2020/7/16 下午3:17
# @File        : solar_forecasting.py.py
# 这个例子并不完整

# !/usr/bin/env python
# coding: utf-8

# In[1]:


import mxnet as mxt
import gluonts
import numpy as np
import pandas as pd
import os
import json

# 构建数据集
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset

from gluonts.distribution.distribution_output import DistributionOutput
from gluonts.distribution.gaussian import GaussianOutput
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer


def process_time(df, freq='1H'):
    # Convert timestamp into a pandas datatime object
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # df = df.set_index('Timestamp')

    # Extract units of time from the timestamp
    df['min'] = df.index.minute
    df['hour'] = df.index.hour
    df['wday'] = df.index.dayofweek
    df['mday'] = df.index.day - 1
    df['yday'] = df.index.dayofyear - 1
    df['month'] = df.index.month - 1
    df['year'] = df.index.year

    # Create a time of day to represent hours and minutes
    df['time'] = df['hour'] + (df['min'] / 60)
    df = df.drop(columns=['hour', 'min'])

    # Cyclical variable transformations

    # wday has period of 6
    df['wday_sin'] = np.sin(2 * np.pi * df['wday'] / 6)
    df['wday_cos'] = np.cos(2 * np.pi * df['wday'] / 6)

    # yday has period of 365
    df['yday_sin'] = np.sin(2 * np.pi * df['yday'] / 364)
    df['yday_cos'] = np.cos(2 * np.pi * df['yday'] / 364)

    # month has period of 12
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # time has period of 24
    df['time_sin'] = np.sin(2 * np.pi * df['time'] / 23)
    df['time_cos'] = np.cos(2 * np.pi * df['time'] / 23)

    df['past_hour_load_1'] = df['Generation'].shift(-1)
    df['past_hour_load_2'] = df['Generation'].shift(-2)
    df['past_hour_load_3'] = df['Generation'].shift(-3)
    df['past_hour_load_4'] = df['Generation'].shift(-4)
    df['past_hour_load_5'] = df['Generation'].shift(-5)
    df['past_hour_load_6'] = df['Generation'].shift(-6)
    df['past_hour_load_7'] = df['consumption_rate'].shift(-7)
    df['past_hour_load_8'] = df['consumption_rate'].shift(-8)
    df['past_hour_load_9'] = df['consumption_rate'].shift(-9)
    df['past_hour_load_10'] = df['consumption_rate'].shift(-10)
    df['past_hour_load_11'] = df['consumption_rate'].shift(-11)
    df['past_hour_load_12'] = df['consumption_rate'].shift(-12)

    # turn the index into a column
    # df = df.reset_index(level=0)

    return df


def is_df_missing_steps(df, freq='1H'):
    """检查是否有不一致的地方，通用函数"""
    index_steps = df.index
    start = df.index[0]
    end = df.index[-1]
    dt_series = pd.date_range(start=start, end=end, freq=freq)
    return not dt_series.equals(index_steps)


def get_missing_steps(df, freq='1H'):
    """检查是否有不一致的地方，通用函数"""
    index_steps = df.index
    start = df.index[0]
    end = df.index[-1]
    dt_series = pd.date_range(start=start, end=end, freq=freq)
    return dt_series.difference(index_steps)


# data_xlsx = pd.read_excel('EMHIRESPV_TSh_CF_Country_19862015.xlsx' ,nrows=1000)
# data_xlsx.to_excel("short_1000.xlsx",index=False)
data_xlsx = pd.read_excel('short_1000.xlsx')
data_xlsx.set_index('Date', inplace=True)
target_cols = list(set(data_xlsx.columns) - set(['Time_step', 'Year', 'Month', 'Day', 'Hour']))

print(data_xlsx[:-365][target_cols].to_numpy())
# 直接用dataframe的列形成数据集
# target_cols / 'SE'

# train_ds = ListDataset([{FieldName.TARGET: target,
#                      FieldName.START: start,
#                      FieldName.FEAT_DYNAMIC_REAL: fdr}
#                     for (target, start, fdr) in zip(
# target,
# custom_ds_metadata['start'],
# feat_dynamic_real)]


#
# train_ds = ListDataset([{
#     FieldName.TARGET: data_xlsx[:-365][target_cols].to_numpy(),  # 多个预测目标
#     FieldName.START: data_xlsx.index[0],
#     FieldName.FEAT_DYNAMIC_REAL: data_xlsx[:-365][['Year', 'Month', 'Day', 'Hour']].to_numpy()
#     # 额外的属性值(连续)－这儿的写法值得学习，但是把年月日纳入不太合适
#
# }],
#     freq='1H',
#     one_dim_target=False)  # 因为是多个预测目标，所以这儿的one_dim_target=False


estimator = DeepAREstimator(freq="1H",
                            prediction_length=24,
                            context_length=7 * 24,  # 4 * 7 * 24,
                            distr_output=GaussianOutput(),
                            trainer=Trainer(epochs=10, ctx='gpu'),
                            num_layers=1)

predictor = estimator.train(training_data=train_ds)
