#!/usr/bin/env python
# coding: utf-8

# # Extended Forecasting Tutorial 
# 
# In the extended forecasting tutorial we cover the following topics.
# 
# ---
# > ## Table of contents
# > 1. ### <span style="color:orange">Datasets</span>
#     1.1 Available datasets on GluonTS  
#     1.2 Create artificial datasets with GluonTS  
#     1.3 Use your time series and features
# > 2. ### <span style="color:orange">Transformation</span>
#     2.1 Define a transformation  
#     2.2 Transform a dataset 
# > 3. ### <span style="color:orange">Training an existing model</span>
#     3.1 Configuring an estimator  
#     3.2 Getting a predictor  
#     3.3 Saving/Loading an existing model  
# > 4. ### <span style="color:orange">Evaluation</span>
#     4.1 Getting the forecasts  
#     4.2 Compute metrics  
# > 5. ### <span style="color:orange">Create your own model</span>
#     5.1 Point forecasts with a simple feedforward network  
#     5.2 Probabilistic forecasting  
#     5.3 Add features and scaling   
#     5.4 From feedforward to RNN
#     
# ---

# In[52]:


# Third-party imports

import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from itertools import islice
from pathlib import Path


# In[53]:


mx.random.seed(0)
np.random.seed(0)


# # 1. Datasets
# 
# The first requirement to use GluonTS is to have an appropriate dataset. GluonTS offers three different options to practitioners that want to experiment with the various modules: 
# 
# - Use an available dataset provided by GluonTS
# - Create an artificial dataset using GluonTS
# - Convert your dataset to a GluonTS friendly format
# 
# In general, a dataset should satisfy some minimum format requirements to be compatible with GluonTS. In particular, it should be an iterable collection of data entries (time series), and each entry should have at least a `target` field, which contains the actual values of the time series, and a `start` field, which denotes the starting date of the time series. There are many more optional fields that we will go through in this tutorial.
# 
# The datasets provided by GluonTS come in the appropriate format and they can be used without any post processing. However, a custom dataset needs to be converted. Fortunately this is an easy task.
# 
# ## 1.1 Available datasets on GluonTS
# 
# 可以获取到的数据集
# 
# GluonTS comes with a number of available datasets.

# In[54]:


from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas


# In[55]:


print(f"Available datasets: {list(dataset_recipes.keys())}")


# To download one of the built-in datasets, simply call `get_dataset` with one of the above names. GluonTS can re-use the saved dataset so that it does not need to be downloaded again: simply set `regenerate=False`.
# 
# 可以指定下载数据集

# In[56]:


dataset = get_dataset("m4_hourly", regenerate=False)


# ### 1.1.1 What is in a dataset?
# 
# In general, the datasets provided by GluonTS are objects that consists of three main members:
# 
# - `dataset.train` is an iterable collection of data entries used for training. Each entry corresponds to one time series.
# - `dataset.test` is an iterable collection of data entries used for inference. The test dataset is an extended version of the train dataset that contains a window in the end of each time series that was not seen during training. This window has length equal to the recommended prediction length.
# - `dataset.metadata` contains metadata of the dataset such as the frequency of the time series, a recommended prediction horizon, associated features, etc.
# 
# First, let's see what the first entry of the train dataset contains. We should expect at least a `target` and a `start` field in each entry, and the target of the test entry to have an additional window equal to `prediction_length`.

# In[57]:


# get the first time series in the training set
# 利用next(iter()) 来获取数据的首行
train_entry = next(iter(dataset.train))
train_entry.keys()


# We observe that apart from the required fields there is one more `feat_static_cat` field (we can safely ignore the `source` field). This shows that the dataset has some features apart from the values of the time series. For now, we will ignore this field too. We will explain it in detail later with all the other optional fields.
# 
# We can similarly examine the first entry of the test dataset. We should expect exactly the same fields as in the train dataset.

# In[58]:


# get the first time series in the test set
# 测试集
test_entry = next(iter(dataset.test))
test_entry.keys()


# Moreover, we should expect that the target will have an additional window in the end with length equal to `prediction_length`. To better understand what this means we can visualize both the train and test time series.

# In[59]:


test_series = to_pandas(test_entry)
train_series = to_pandas(train_entry)

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

train_series.plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].legend(["train series"], loc="upper left")

test_series.plot(ax=ax[1])
ax[1].axvline(train_series.index[-1], color='r') # end of train dataset，训练数据集的结束位置
ax[1].grid(which="both")
ax[1].legend(["test series", "end of train series"], loc="upper left")

plt.show()


# * 预测长度=48，预测的频率是H

# In[60]:


print(f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}")
print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
print(f"Frequency of the time series: {dataset.metadata.freq}")


# ## 1.2 Create artificial datasets
# 
# * 利用`ComplexSeasonalTimeSeries`来构建数据集
# 
# We can easily create a complex artificial time series dataset using the `ComplexSeasonalTimeSeries` module.

# In[61]:


from gluonts.dataset.artificial import ComplexSeasonalTimeSeries
from gluonts.dataset.common import ListDataset


# In[62]:


artificial_dataset = ComplexSeasonalTimeSeries(
    num_series=10, # 数据集的item_size = 10
    prediction_length=21, # 预测的长度
    freq_str="H", # 预测的频率
    length_low=30,
    length_high=200,
    min_val=-10000,
    max_val=10000,
    is_integer=False,
    proportion_missing_values=0,
    is_noise=True,
    is_scale=True,
    percentage_unique_timestamps=1,
    is_out_of_bounds_date=True,
)


# We can access some important metadata of the artificial dataset as follows:

# In[63]:


print(f"prediction length: {artificial_dataset.metadata.prediction_length}")
print(f"frequency: {artificial_dataset.metadata.freq}")


# The artificial dataset that we created is a list of dictionaries. Each dictionary corresponds to a time series and it should contain the required fields.

# * keys 至少包括start,target,item_id

# In[64]:


print(f"type of train dataset: {type(artificial_dataset.train)}")
print(len(artificial_dataset.train))
print(f"train dataset fields: {artificial_dataset.train[0].keys()}")
print(len(artificial_dataset.test))
print(f"type of test dataset: {type(artificial_dataset.test)}")
print(f"test dataset fields: {artificial_dataset.test[0].keys()}")


# In[65]:


artificial_dataset.train[0]['start']


# In[66]:


print(artificial_dataset.test[0]['target'].shape ,artificial_dataset.train[0]['target'].shape ) # 刚好预测长度是21


# In[67]:


[artificial_dataset.train[i]['item_id']  for i in range(len(artificial_dataset.train))] # 和参数num_series相同


# In[68]:


artificial_dataset.train[0]['target'].min(),artificial_dataset.train[0]['target'].max()


# In order to use the artificially created datasets (list of dictionaries) we need to convert them to `ListDataset` objects.

# In[69]:


train_ds = ListDataset(artificial_dataset.train, 
                        freq=artificial_dataset.metadata.freq)


# In[70]:


test_ds = ListDataset(artificial_dataset.test, 
                       freq=artificial_dataset.metadata.freq)


# In[71]:


train_entry = next(iter(train_ds))
train_entry.keys() # 注意这儿多了一个'source'字段


# In[72]:


test_entry = next(iter(test_ds))
test_entry.keys()


# In[73]:


print(train_entry['source'],test_entry['source'])


# In[74]:


test_series = to_pandas(test_entry)
train_series = to_pandas(train_entry)

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

train_series.plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].legend(["train series"], loc="upper left")

test_series.plot(ax=ax[1])
ax[1].axvline(train_series.index[-1], color='r') # end of train dataset
ax[1].grid(which="both")
ax[1].legend(["test series", "end of train series"], loc="upper left")

plt.show()

# just fun
print(train_series.index[0],train_series.index[-1])
print(test_series.index[0],test_series.index[-1]) 


# ## 1.3 Use your time series and features
# 
# * 可以有更多的变量
# 
# Now, we will see how we can convert any custom dataset with any associated features to an appropriate format for GluonTS.
# 
# As already mentioned a dataset is required to have at least the `target` and the `start` fields. However, it may have more. Let's see what are all the available fields:

# In[75]:


from gluonts.dataset.field_names import FieldName


# In[76]:


[f"FieldName.{k} = '{v}'" for k, v in FieldName.__dict__.items() if not k.startswith('_')]


# * 三类fields
# 
# The fields are split into three categories: the required ones, the optional ones, and the ones that can be added by the `Transformation` (explained in a while).
# 
# * 必须包含（2个、开始时间、预测列）
# 
# Required:
# 
# - `start`: start date of the time series
# - `target`: values of the time series
# 
# 
# 
# 
# * 可选的（4个）
# 
# - `feat_static_cat` 静态分类，所属的类别
# - `feat_static_real` 静态real
# - `feat_dynamic_cat` 动态分类
# - `feat_dynamic_real`动态real,数值型时序特征(item_size,target.length)
# 
# 
# Optional:
# 
# - `feat_static_cat`: static (over time) categorical features, list with dimension equal to the number of features
# - `feat_static_real`: static (over time) real features, list with dimension equal to the number of features
# - `feat_dynamic_cat`: dynamic (over time) categorical features, array with shape equal to (number of features, target length)
# - `feat_dynamic_real`: dynamic (over time) real features, array with shape equal to (number of features, target length)
# 
# 
# * 添加的，转换而来的（6个）
# 
# 
# Added by `Transformation`:
# 
# - `time_feat`: time related features such as the month or the day 
# - `feat_dynamic_const`: expands a constant value feature along the time axis
# - `feat_dynamic_age`: age feature, i.e., a feature that its value is small for distant past timestamps and it monotonically increases the more we approach the current timestamp
# - `observed_values`: indicator for observed values, i.e., a feature that equals to 1 if the value is observed and 0 if the value is missing
# - `is_pad`: indicator for each time step that shows if it is padded (if the length is not enough) 
# - `forecast_start`: forecast start date
# 
# As a simple example, we can create a custom dataset to see how we can use some of these fields. The dataset consists of a target, a real dynamic feature (which in this example we set to be the target value one period earlier), and a static categorical feature that indicates the sinusoid type (different phase) that we used to create the target.

# In[77]:


def create_dataset(num_series, num_steps, period=24, mu=1, sigma=0.3):
    # create target: noise + pattern    
    # noise
    noise = np.random.normal(mu, sigma, size=(num_series, num_steps))
    
    # pattern - sinusoid with different phase
    sin_minumPi_Pi = np.sin(np.tile(np.linspace(-np.pi, np.pi, period), int(num_steps / period)))
    sin_Zero_2Pi = np.sin(np.tile(np.linspace(0, 2 * np.pi, 24), int(num_steps / period)))
    
    pattern = np.concatenate((np.tile(sin_minumPi_Pi.reshape(1, -1), 
                                      (int(np.ceil(num_series / 2)),1)), 
                              np.tile(sin_Zero_2Pi.reshape(1, -1), 
                                      (int(np.floor(num_series / 2)), 1))
                             ),
                             axis=0
                            )
    
    # target
    target = noise + pattern
    
    # create time features: use target one period earlier, append with zeros
    # feat_dynamic_real －　时序特征
    feat_dynamic_real = np.concatenate((np.zeros((num_series, period)), 
                                        target[:, :-period]
                                       ), 
                                       axis=1
                                      )
    
    # create categorical static feats: use the sinusoid type as a categorical feature
    # 0,1，刚好一半一半。feat_static_cat是针对item的，大小和item_size相同，也就是每个观测有一个所属的类别
    feat_static_cat = np.concatenate((np.zeros(int(np.ceil(num_series / 2))), 
                                      np.ones(int(np.floor(num_series / 2)))
                                     ),
                                     axis=0
                                    )
    
    return target, feat_dynamic_real, feat_static_cat
    


# In[78]:


# define the parameters of the dataset
custom_ds_metadata = {'num_series': 100, # 100个观察
                      'num_steps': 24 * 7,# 168个数值－target,这样形成的target.shape = (100,168)
                      'prediction_length': 24,#　预测的长度
                      'freq': '1H',
                      'start': [pd.Timestamp("01-01-2019", freq='1H') 
                                for _ in range(100)]
                     }


# In[79]:


data_out = create_dataset(custom_ds_metadata['num_series'], 
                          custom_ds_metadata['num_steps'],                                                      
                          custom_ds_metadata['prediction_length']
                         )

target, feat_dynamic_real, feat_static_cat = data_out


# * 以下部分是对上面代码的理解

# In[80]:


target.shape


# In[81]:


print(target[0].shape,target[0].min(),target[0].max())

# feature_static_cat相当于是每个观测，都有一个所属的类别
# feat_dynamic_real相当于和target的形状相同，相当于是额外的特征
# In[82]:


feat_dynamic_real.shape,feat_static_cat.shape 


# In[83]:


feat_static_cat


# In[84]:


feat_dynamic_real[1]


# We can easily create the train and test datasets by simply filling in the correct fields. Remember that for the train dataset we need to cut the last window.

# In[85]:


len(custom_ds_metadata['start']),custom_ds_metadata['start'][0] # 100个观察序列的起始值相同


# In[86]:


target[:, :-custom_ds_metadata['prediction_length']].shape # 把前１４４个拿出来作为训练集


# In[87]:


custom_ds_metadata['freq']


# In[88]:


# feat_dynamic_real的处理和target相同，但是feat_static_cat因为是和item的数量相同（１００），所以全部纳入进来
# 训练数据需要剔除掉预测的部分，也就是prediction_length

train_ds = ListDataset([{FieldName.TARGET: target, 
                         FieldName.START: start,
                         FieldName.FEAT_DYNAMIC_REAL: [fdr],
                         FieldName.FEAT_STATIC_CAT: [fsc]} 
                        for (target, start, fdr, fsc) in zip(target[:, :-custom_ds_metadata['prediction_length']], 
                                                             custom_ds_metadata['start'], 
                                                             feat_dynamic_real[:, :-custom_ds_metadata['prediction_length']], 
                                                             feat_static_cat)],
                      freq=custom_ds_metadata['freq'])


# In[89]:


# 注意，在设计中，test数据集包含全部的
# 注意，ＬistDataset中，是一个list，list中包含了dict
test_ds = ListDataset([{FieldName.TARGET: target, 
                        FieldName.START: start,
                        FieldName.FEAT_DYNAMIC_REAL: [fdr],
                        FieldName.FEAT_STATIC_CAT: [fsc]} 
                       for (target, start, fdr, fsc) in zip(target, 
                                                            custom_ds_metadata['start'], 
                                                            feat_dynamic_real, 
                                                            feat_static_cat)],
                     freq=custom_ds_metadata['freq'])


# Now, we can examine each entry of the train and test datasets. We should expect that they have the following fields: `target`, `start`, `feat_dynamic_real` and `feat_static_cat`.

# In[90]:


# 利用next(iter) 获取实际的数据
train_entry = next(iter(train_ds))
train_entry.keys()


# In[91]:


test_entry = next(iter(test_ds))
test_entry.keys()


# In[92]:


test_series = to_pandas(test_entry)
train_series = to_pandas(train_entry)

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

train_series.plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].legend(["train series"], loc="upper left")

test_series.plot(ax=ax[1])
ax[1].axvline(train_series.index[-1], color='r') # end of train dataset
ax[1].grid(which="both")
ax[1].legend(["test series", "end of train series"], loc="upper left")

plt.show()


# <span style="color:red">*For the rest of the tutorial we will use the custom dataset*</span>

# # 2. Transformation
# 
# ## 2.1 Define a transformation
# 
# The primary use case for a `Transformation` is for feature processing, e.g., adding a holiday feature and for defining the way the dataset will be split into appropriate windows during training and inference. 
# 
# 特征抽取，比如添加节假日的特征。拆分窗口。
# 
# In general, it gets an iterable collection of entries of a dataset and transform it to another iterable collection that can possibly contain more fields. The transformation is done by defining a set of "actions" to the raw dataset depending on what is useful to our model. This actions usually create some additional features or transform an existing feature. As an example, in the following we add the following transformations:
# 
# 输入的是dataset，输出的另外一个可迭代对象，包含了更多的特征。
# 利用一组“actions"来做特征抽取。
# 
# 
# - `AddObservedValuesIndicator`: Creates the `observed_values` field in the dataset, i.e., adds a feature that equals to 1 if the value is observed and 0 if the value is missing 
# 
# 特征值如果存在则为１，否则为０
# 
# 
# - `AddAgeFeature`: Creates the `feat_dynamic_age` field in the dataset, i.e., adds a feature that its value is small for distant past timestamps and it monotonically increases the more we approach the current timestamp  
# 
# 很早之前的值比较小，但是最近的时间戳的值会越来越大
# 
# The `Transformation` may not define an additional field in the dataset. However, it **always** needs to define how the datasets are going to be split in example windows during training and testing. This is done with the `InstanceSplitter` that is configured as follows (skipping the obvious fields):
# 
# `Transformation` 并没有增加dataset中的额外字段。但是它总是定义了如何拆分训练和测试的时间窗口。
# 这是利用`InstanceSplitter`来做的。
# 
# - `is_pad_field`: indicator if the time series is padded (if the length is not enough)
# - `train_sampler`: defines how the training windows are cut/sampled
# - `time_series_fields`: contains the time dependent features that need to be split in the same manner as the target
# 

# In[93]:


from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetFieldIfNotPresent,
)


# In[106]:


def create_transformation(freq, context_length, prediction_length):
    return Chain(
        [
            # 是否出现
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            
            # 时间上的加权，越远的值越小
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            
            # 拆分
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                past_length=context_length,
                future_length=prediction_length,
                time_series_fields=[
                    FieldName.FEAT_AGE,
                    FieldName.FEAT_DYNAMIC_REAL,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
        ]
    )


# ## 2.2 Transform a dataset
# 
# Now, we can create a transformation object by applying the above transformation to the custom dataset we have created.

# In[107]:


transformation = create_transformation(custom_ds_metadata['freq'], 
                                       2 * custom_ds_metadata['prediction_length'], # 用过去多长时间来预测未来　can be any appropriate value
                                       custom_ds_metadata['prediction_length'])


# In[108]:


train_tf = transformation(iter(train_ds), is_train=True)


# In[98]:


type(train_tf)


# As expected, the output is another iterable object. We can easily examine what is contained in an entry of the transformed dataset. When `is_train=True` in the transformation, the `InstanceSplitter` iterates over the transformed dataset and cuts windows by selecting randomly a time series and a starting point on that time series (this "randomness" is defined by the `train_sampler`).

# In[109]:


train_tf_entry = next(iter(train_tf))
[k for k in train_tf_entry.keys()]


# The transformer has done what we asked. In particular it has added:
# 
# - a field for observed values (`observed_values`)  
# - a field for the age feature (`feat_dynamic_age`)
# - some extra useful fields (`past_is_pad`, `forecast_start`)
# 
# It has done one more important thing: it has split the window into past and future and has added the corresponding prefixes to all time dependent fields. This way we can easily use e.g., the `past_target` field as input and the `future_target` field to calculate the error of our predictions. Of course, the length of the past is equal to the `context_length` and of the future equal to the `prediction_length`.
# 
# 也就是所谓的context_length，上下文长度

# In[110]:


print(f"past target shape: {train_tf_entry['past_target'].shape}") # 过去的，target = 48 = 2 * custom_ds_metadata['prediction_length']
print(f"future target shape: {train_tf_entry['future_target'].shape}") # 未来的，target = 24
print(f"past observed values shape: {train_tf_entry['past_observed_values'].shape}") # 48
print(f"future observed values shape: {train_tf_entry['future_observed_values'].shape}") #24
print(f"past age feature shape: {train_tf_entry['past_feat_dynamic_age'].shape}") #过去的，48，这个是一个实数，递增的
print(f"future age feature shape: {train_tf_entry['future_feat_dynamic_age'].shape}")#未来的,24，值越来越大
print(train_tf_entry['feat_static_cat'])


# Just for comparison, let's see again what were the fields in the original dataset before the transformation:

# In[122]:


[k for k in next(iter(train_ds)).keys()]


# In[124]:


[k for k in train_tf_entry.keys() if k  in next(iter(train_ds)).keys()]


# In[125]:


[k for k in train_tf_entry.keys() if k  not in next(iter(train_ds)).keys()]


# Now, we can move on and see how the test dataset is split. As we saw, the transformation splits the windows into past and future. However, during inference (`is_train=False` in the transformation), the splitter always cuts the last window (of length `context_length`) of the dataset so it can be used to predict the subsequent unknown values of length `prediction_length`. 
# 
# So, how is the test dataset split in past and future since we do not know the future target? And what about the time dependent features?

# In[126]:


test_tf = transformation(iter(test_ds), is_train=False)


# In[127]:


test_tf_entry = next(iter(test_tf))
[k for k in test_tf_entry.keys()]


# In[128]:


print(f"past target shape: {test_tf_entry['past_target'].shape}")
print(f"future target shape: {test_tf_entry['future_target'].shape}")
print(f"past observed values shape: {test_tf_entry['past_observed_values'].shape}")
print(f"future observed values shape: {test_tf_entry['future_observed_values'].shape}")
print(f"past age feature shape: {test_tf_entry['past_feat_dynamic_age'].shape}")
print(f"future age feature shape: {test_tf_entry['future_feat_dynamic_age'].shape}")
print(test_tf_entry['feat_static_cat'])


# The future target is empty but not the features - we always assume that we know the future features!
# 
# All the things we did manually here are done by an internal block called `DataLoader`. It gets as an input the raw dataset (in appropriate format) and the transformation object and it outputs the transformed iterable dataset batch by batch. The only thing that we need to worry about is setting the transformation fields correctly!
# 
# 
# `DataLoader`．只需要配置好transformation就好了。

# # 3. Training an existing model
# 
# GluonTS comes with a number of pre-built models. All the user needs to do is configure some hyperparameters. The existing models focus on (but are not limited to) probabilistic forecasting. Probabilistic forecasts are predictions in the form of a probability distribution, rather than simply a single point estimate. Having estimated the future distribution of each time step in the forecasting horizon, we can draw a sample from the distribution at each time step and thus create a "sample path" that can be seen as a possible realization of the future. In practice we draw multiple samples and create multiple sample paths which can be used for visualization, evaluation of the model, to derive statistics, etc.
# 
# 自带的pre-built模型，用户只要配置好超参数就可以了。
# 
# 
# ## 3.1 Configuring an estimator
# 
# We will begin with GulonTS's pre-built feedforward neural network estimator, a simple but powerful forecasting model. We will use this model to demonstrate the process of training a model, producing forecasts, and evaluating the results.
# 
# GluonTS's built-in feedforward neural network (`SimpleFeedForwardEstimator`) accepts an input window of length `context_length` and predicts the distribution of the values of the subsequent `prediction_length` values. In GluonTS parlance, the feedforward neural network model is an example of `Estimator`. In GluonTS, `Estimator` objects represent a forecasting model as well as details such as its coefficients, weights, etc.
# 
# In general, each estimator (pre-built or custom) is configured by a number of hyperparameters that can be either common (but not binding) among all estimators (e.g., the `prediction_length`) or specific for the particular estimator (e.g., number of layers for a neural network or the stride in a CNN).
# 
# Finally, each estimator is configured by a `Trainer`, which defines how the model will be trained i.e., the number of epochs, the learning rate, etc.

# In[129]:


from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer


# In[135]:


estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=custom_ds_metadata['prediction_length'],
    context_length=2*custom_ds_metadata['prediction_length'],
    freq=custom_ds_metadata['freq'],
    trainer=Trainer(ctx="cpu", 
                    epochs=5, 
                    learning_rate=1e-3, 
                    hybridize=False, 
                    num_batches_per_epoch=100
                   )
)


# ## 3.2 Getting a predictor
# 
# After specifying our estimator with all the necessary hyperparameters we can train it using our training dataset `dataset.train` by invoking the `train` method of the estimator. The training algorithm returns a fitted model (or a `Predictor` in GluonTS parlance) that can be used to construct forecasts.
# 
# We should emphasize here that a single model, as the one defined above, is trained over all the time series contained in the training dataset `train_ds`. This results in a **global** model, suitable for prediction for all the time series in `train_ds` and possibly for other unseen related time series.

# In[136]:


predictor = estimator.train(train_ds)


# ## 3.3 Saving/Loading an existing model
# 
# A fitted model, i.e., a `Predictor`, can be saved and loaded back easily:

# In[137]:


# save the trained model in tmp/
from pathlib import Path
predictor.serialize(Path("/tmp/"))


# In[138]:


# loads it back
from gluonts.model.predictor import Predictor
predictor_deserialized = Predictor.deserialize(Path("/tmp/"))


# # 4. Evaluation
# 
# ## 4.1 Getting the forecasts
# 
# With a predictor in hand, we can now predict the last window of the `dataset.test` and evaluate our model's performance.
# 
# GluonTS comes with the `make_evaluation_predictions` function that automates the process of prediction and model evaluation. Roughly, this function performs the following steps:
# 
# - Removes the final window of length `prediction_length` of the `dataset.test` that we want to predict
# - The estimator uses the remaining data to predict (in the form of sample paths) the "future" window that was just removed
# - The module outputs the forecast sample paths and the `dataset.test` (as python generator objects)

# In[139]:


from gluonts.evaluation.backtest import make_evaluation_predictions


# In[141]:


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


# First, we can convert these generators to lists to ease the subsequent computations.

# In[142]:


forecasts = list(forecast_it)
tss = list(ts_it)


# We can examine the first element of these lists (that corresponds to the first time series of the dataset). Let's start with the list containing the time series, i.e., `tss`. We expect the first entry of `tss` to contain the (target of the) first time series of `test_ds`.

# In[143]:


# first entry of the time series list
ts_entry = tss[0]


# In[144]:


# first 5 values of the time series (convert from pandas to numpy)
np.array(ts_entry[:5]).reshape(-1,)


# In[145]:


# first entry of test_ds
test_ds_entry = next(iter(test_ds))


# In[146]:


# first 5 values
test_ds_entry['target'][:5]


# The entries in the `forecast` list are a bit more complex. They are objects that contain all the sample paths in the form of `numpy.ndarray` with dimension `(num_samples, prediction_length)`, the start date of the forecast, the frequency of the time series, etc. We can access all these information by simply invoking the corresponding attribute of the forecast object.

# In[147]:


# first entry of the forecast list
forecast_entry = forecasts[0]


# In[148]:


print(f"Number of sample paths: {forecast_entry.num_samples}")
print(f"Dimension of samples: {forecast_entry.samples.shape}")
print(f"Start date of the forecast window: {forecast_entry.start_date}")
print(f"Frequency of the time series: {forecast_entry.freq}")


# We can also do calculations to summarize the sample paths, such as computing the mean or a quantile for each of the 24 time steps in the forecast window.

# In[149]:


print(f"Mean of the future window:\n {forecast_entry.mean}")
print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")


# `Forecast` objects have a `plot` method that can summarize the forecast paths as the mean, prediction intervals, etc. The prediction intervals are shaded in different colors as a "fan chart".

# In[150]:


def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150 
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


# In[56]:


plot_prob_forecasts(ts_entry, forecast_entry)


# ## 4.2 Compute metrics
# 
# We can also evaluate the quality of our forecasts numerically. In GluonTS, the `Evaluator` class can compute aggregate performance metrics, as well as metrics per time series (which can be useful for analyzing performance across heterogeneous time series).

# In[151]:


from gluonts.evaluation import Evaluator


# In[152]:


evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))


# Aggregate metrics aggregate both across time-steps and across time series.

# In[157]:


print(json.dumps(agg_metrics, indent=2))


# Individual metrics are aggregated only across time-steps.

# In[158]:


item_metrics.head()


# In[159]:


item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
plt.grid(which="both")
plt.show()


# # 5. Create your own model
# 
# 暂缓
# 
# For creating our own forecast model we need to:
# 
# - Define the training and prediction network
# - Define a new estimator that specifies any data processing and uses the networks
# 
# The training and prediction networks can be arbitrarily complex but they should follow some basic rules:
# 
# - Both should have a `hybrid_forward` method that defines what should happen when the network is called    
# - The training network's `hybrid_forward` should return a **loss** based on the prediction and the true values
# - The prediction network's `hybrid_forward` should return the predictions 
# 
# The estimator should also follow some rules:
# 
# - It should include a `create_transformation` method that defines all the possible feature transformations and how the data is split during training
# - It should include a `create_training_network` method that returns the training network configured with any necessary hyperparameters
# - It should include a `create_predictor` method that creates the prediction network, and returns a `Predictor` object 
# 
# A `Predictor` defines the `predictor.predict` method of a given predictor. This method takes the test dataset, it passes it through the prediction network to take the predictions, and yields the predictions. You can think of the `Predictor` object as a wrapper of the prediction network that defines its `predict` method. 
# 
# In this section, we will start simple by creating a feedforward network that is restricted to point forecasts. Then, we will add complexity to the network by expanding it to probabilistic forecasting, considering features and scaling of the time series, and in the end we will replace it with an RNN.
# 
# We need to emphasize that the way the following models are implemented and all the design choices that are made are neither binding nor necessarily optimal. Their sole purpose is to provide guidelines and hints on how to build a model. 
# 
# ## 5.1 Point forecasts with a simple feedforward network
# 
# We can create a simple training network that defines a neural network that takes as input a window of length `context_length` and predicts the subsequent window of dimension `prediction_length` (thus, the output dimension of the network is `prediction_length`). The `hybrid_forward` method of the training network returns the mean of the L1 loss. 
# 
# The prediction network is (and should be) identical to the training network (by inheriting the class) and its `hybrid_forward` method returns the predictions.

# In[62]:


class MyNetwork(gluon.HybridBlock):
    def __init__(self, prediction_length, num_cells, **kwargs):
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.num_cells = num_cells
    
        with self.name_scope():
            # Set up a 3 layer neural network that directly predicts the target values
            self.nn = mx.gluon.nn.HybridSequential()
            self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length, activation='softrelu'))


class MyTrainNetwork(MyNetwork):    
    def hybrid_forward(self, F, past_target, future_target):
        prediction = self.nn(past_target)
        # calculate L1 loss with the future_target to learn the median
        return (prediction - future_target).abs().mean(axis=-1)


class MyPredNetwork(MyTrainNetwork):
    # The prediction network only receives past_target and returns predictions
    def hybrid_forward(self, F, past_target):
        prediction = self.nn(past_target)
        return prediction.expand_dims(axis=1)


# The estimator class is configured by a few hyperparameters and implements the required methods.

# In[63]:


from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.core.component import validated
from gluonts.trainer import Trainer
from gluonts.support.util import copy_parameters
from gluonts.transform import ExpectedNumInstanceSampler, Transformation, InstanceSplitter
from mxnet.gluon import HybridBlock


# In[64]:


class MyEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        freq: str,
        num_cells: int,
        trainer: Trainer = Trainer()
    ) -> None:
        super().__init__(trainer=trainer)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.num_cells = num_cells
            
    def create_transformation(self):
        # Feature transformation that the model uses for input
        # Here we use a transformation that defines only how the train and test windows are cut
        return InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                )
    
    def create_training_network(self) -> MyTrainNetwork:
        return MyTrainNetwork(
            prediction_length=self.prediction_length,
            num_cells = self.num_cells
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = MyPredNetwork(
            prediction_length=self.prediction_length,
            num_cells=self.num_cells
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


# After defining the training and prediction network, as well as the estimator class, we can follow exactly the same steps as with the existing models, i.e., we can specify our estimator by passing all the required hyperparameters to the estimator class, train the estimator by invoking its `train` method to create a predictor, and finally use the `make_evaluation_predictions` function to generate our forecasts.

# In[65]:


estimator = MyEstimator(
    prediction_length=custom_ds_metadata['prediction_length'],
    context_length=2*custom_ds_metadata['prediction_length'],
    freq=custom_ds_metadata['freq'],
    num_cells=40,
    trainer=Trainer(ctx="cpu", 
                    epochs=5, 
                    learning_rate=1e-3, 
                    hybridize=False, 
                    num_batches_per_epoch=100
                   )
)


# The estimator can be trained using our training dataset `train_ds` just by invoking its `train` method. The training returns a predictor that can be used to predict.

# In[66]:


predictor = estimator.train(train_ds)


# In[67]:


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


# In[68]:


forecasts = list(forecast_it)
tss = list(ts_it)


# In[69]:


plot_prob_forecasts(tss[0], forecasts[0])


# Observe that we cannot actually see any prediction intervals in the predictions. This is expected since the model that we defined does not do probabilistic forecasting but it just gives point estimates. By requiring 100 sample paths (defined in `make_evaluation_predictions`) in such a network, we get 100 times the same output.
# 
# ## 5.2 Probabilistic forecasting
# 
# ### 5.2.1 How does a model learn a distribution?
# 
# Probabilistic forecasting requires that we learn the distribution of the future values of the time series and not the values themselves as in point forecasting. To achieve this, we need to specify the type of distribution that the future values follow. GluonTS comes with a number of different distributions that cover many use cases, such as Gaussian, Student-t and Uniform just to name a few.  
# 
# 
# In order to learn a distribution we need to learn its parameters. For example, in the simple case where we assume a Gaussian distribution, we need to learn the mean and the variance that fully specify the distribution.
# 
# Each distribution that is available in GluonTS is defined by the corresponding `Distribution` class (e.g., `Gaussian`). This class defines -among others- the parameters of the distribution, its (log-)likelihood and a sampling method (given the parameters). 
# 
# However, it is not straightforward how to connect a model with such a distribution and learn its parameters. For this, each distribution comes with a `DistributionOutput` class (e.g., `GaussianOutput`). The role of this class is to connect a model with a distribution. Its main usage is to take the output of the model and map it to the parameters of the distribution. You can think of it as an additional projection layer on top of the model. The parameters of this layer are optimized along with the rest of the network.
# 
# By including this projection layer, our model effectively learns the parameters of the (chosen) distribution of each time step. Such a model is usually optimized by choosing as a loss function the negative log-likelihood of the chosen distribution. After we optimize our model and learn the parameters we can sample or derive any other useful statistics from the learned distributions.
# 
# ### 5.2.2 Feedforward network for probabilistic forecasting
# 
# Let's see what changes we need to make to the previous model to make it probabilistic: 
# 
# - First, we need to change the output of the network. In the point forecast network the output was a vector of length `prediction_length` that gave directly the point estimates. Now, we need to output a set of features that the `DistributionOutput` will use to project to the distribution parameters. These features should be different for each time step at the prediction horizon. Therefore we need an overall output of `prediction_length * num_features` values.
# - The `DistributionOutput` takes as input a tensor and uses the last dimension as features to be projected to the distribution parameters. Here, we need a distribution object for each time step, i.e., `prediction_length` distribution objects. Given that the output of the network has `prediction_length * num_features` values, we can reshape it to `(prediction_length, num_features)` and get the required distributions, while the last axis of length `num_features` will be projected to the distribution parameters. 
# - We want the prediction network to output many sample paths for each time series. To achieve this we can repeat each time series as many times as the number of sample paths and do a standard forecast for each of them. 
# 
# Note that in all the tensors that we handle there is an initial dimension that refers to the batch, e.g., the output of the network has dimension `(batch_size, prediction_length * num_features)`.

# In[70]:


from gluonts.distribution.distribution_output import DistributionOutput
from gluonts.distribution.gaussian import GaussianOutput


# In[71]:


class MyProbNetwork(gluon.HybridBlock):
    def __init__(self, 
                 prediction_length, 
                 distr_output, 
                 num_cells, 
                 num_sample_paths=100, 
                 **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.proj_distr_args = distr_output.get_args_proj()

        with self.name_scope():
            # Set up a 2 layer neural network that its ouput will be projected to the distribution parameters
            self.nn = mx.gluon.nn.HybridSequential()
            self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length * self.num_cells, activation='relu'))


class MyProbTrainNetwork(MyProbNetwork):
    def hybrid_forward(self, F, past_target, future_target):
        # compute network output
        net_output = self.nn(past_target)

        # (batch, prediction_length * nn_features)  ->  (batch, prediction_length, nn_features)
        net_output = net_output.reshape(0, self.prediction_length, -1)

        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)

        # compute distribution
        distr = self.distr_output.distribution(distr_args)

        # negative log-likelihood
        loss = distr.loss(future_target)
        return loss


class MyProbPredNetwork(MyProbTrainNetwork):
    # The prediction network only receives past_target and returns predictions
    def hybrid_forward(self, F, past_target):
        # repeat past target: from (batch_size, past_target_length) to 
        # (batch_size * num_sample_paths, past_target_length)
        repeated_past_target = past_target.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        
        # compute network output
        net_output = self.nn(repeated_past_target)

        # (batch * num_sample_paths, prediction_length * nn_features)  ->  (batch * num_sample_paths, prediction_length, nn_features)
        net_output = net_output.reshape(0, self.prediction_length, -1)
       
        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)

        # compute distribution
        distr = self.distr_output.distribution(distr_args)

        # get (batch_size * num_sample_paths, prediction_length) samples
        samples = distr.sample()
        
        # reshape from (batch_size * num_sample_paths, prediction_length) to 
        # (batch_size, num_sample_paths, prediction_length)
        return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))


# The changes we need to do at the estimator are minor and they mainly reflect the additional `distr_output` parameter that our networks use.

# In[72]:


class MyProbEstimator(GluonEstimator):
    @validated()
    def __init__(
            self,
            prediction_length: int,
            context_length: int,
            freq: str,
            distr_output: DistributionOutput,
            num_cells: int,
            num_sample_paths: int = 100,
            trainer: Trainer = Trainer()
    ) -> None:
        super().__init__(trainer=trainer)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths

    def create_transformation(self):
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            train_sampler=ExpectedNumInstanceSampler(num_instances=1),
            past_length=self.context_length,
            future_length=self.prediction_length,
        )

    def create_training_network(self) -> MyProbTrainNetwork:
        return MyProbTrainNetwork(
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths
        )

    def create_predictor(
            self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = MyProbPredNetwork(
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


# In[73]:


estimator = MyProbEstimator(
    prediction_length=custom_ds_metadata['prediction_length'],
    context_length=2*custom_ds_metadata['prediction_length'],
    freq=custom_ds_metadata['freq'],
    distr_output=GaussianOutput(),
    num_cells=40,
    trainer=Trainer(ctx="cpu", 
                    epochs=5, 
                    learning_rate=1e-3, 
                    hybridize=False, 
                    num_batches_per_epoch=100
                   )
)


# In[74]:


predictor = estimator.train(train_ds)


# In[75]:


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


# In[76]:


forecasts = list(forecast_it)
tss = list(ts_it)


# In[77]:


plot_prob_forecasts(tss[0], forecasts[0])


# ## 5.3 Add features and scaling
# 
# In the previous networks we used only the target and did not leverage any of the features of the dataset. Here we expand the probabilistic network by including the `feat_dynamic_real` field of the dataset that could enhance the forecasting power of our model. We achieve this by concatenating the target and the features to an enhanced vector that forms the new network input. 
# 
# All the features that are available in a dataset can be potentially used as inputs to our model. However, for the purposes of this example we will restrict ourselves to using only one feature.
# 
# An important issue that a practitioner needs to deal with often is the different orders of magnitude in the values of the time series in a dataset. It is extremely helpful for a model to be trained and forecast values that lie roughly in the same value range. To address this issue, we add a `Scaler` to out model, that computes the scale of each time series. Then we can scale accordingly the values of the time series or any related features and use these as inputs to the network.

# In[78]:


from gluonts.block.scaler import MeanScaler, NOPScaler


# In[79]:


class MyProbNetwork(gluon.HybridBlock):
    def __init__(self, 
                 prediction_length, 
                 context_length, 
                 distr_output, 
                 num_cells, 
                 num_sample_paths=100, 
                 scaling=True, 
                 **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.proj_distr_args = distr_output.get_args_proj()
        self.scaling = scaling

        with self.name_scope():
            # Set up a 2 layer neural network that its ouput will be projected to the distribution parameters
            self.nn = mx.gluon.nn.HybridSequential()
            self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length * self.num_cells, activation='relu'))

            if scaling:
                self.scaler = MeanScaler(keepdims=True)
            else:
                self.scaler = NOPScaler(keepdims=True)

    def compute_scale(self, past_target, past_observed_values):
        # scale shape is (batch_size, 1)
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )

        return scale


class MyProbTrainNetwork(MyProbNetwork):
    def hybrid_forward(self, F, past_target, future_target, past_observed_values, past_feat_dynamic_real):
        # compute scale 
        scale = self.compute_scale(past_target, past_observed_values)

        # scale target and time features
        past_target_scale = F.broadcast_div(past_target, scale)
        past_feat_dynamic_real_scale = F.broadcast_div(past_feat_dynamic_real.squeeze(axis=-1), scale)

        # concatenate target and time features to use them as input to the network
        net_input = F.concat(past_target_scale, past_feat_dynamic_real_scale, dim=-1)

        # compute network output
        net_output = self.nn(net_input)

        # (batch, prediction_length * nn_features)  ->  (batch, prediction_length, nn_features)
        net_output = net_output.reshape(0, self.prediction_length, -1)

        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)

        # compute distribution
        distr = self.distr_output.distribution(distr_args, scale=scale)

        # negative log-likelihood
        loss = distr.loss(future_target)
        return loss


class MyProbPredNetwork(MyProbTrainNetwork):
    # The prediction network only receives past_target and returns predictions
    def hybrid_forward(self, F, past_target, past_observed_values, past_feat_dynamic_real):
        # repeat fields: from (batch_size, past_target_length) to
        # (batch_size * num_sample_paths, past_target_length)
        repeated_past_target = past_target.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        repeated_past_observed_values = past_observed_values.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        repeated_past_feat_dynamic_real = past_feat_dynamic_real.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        
        # compute scale
        scale = self.compute_scale(repeated_past_target, repeated_past_observed_values)

        # scale repeated target and time features
        repeated_past_target_scale = F.broadcast_div(repeated_past_target, scale)
        repeated_past_feat_dynamic_real_scale = F.broadcast_div(repeated_past_feat_dynamic_real.squeeze(axis=-1), scale)

        # concatenate target and time features to use them as input to the network
        net_input = F.concat(repeated_past_target_scale, repeated_past_feat_dynamic_real_scale, dim=-1)

        # compute network oputput
        net_output = self.nn(net_input)
        
        # (batch * num_sample_paths, prediction_length * nn_features)  ->  (batch * num_sample_paths, prediction_length, nn_features)
        net_output = net_output.reshape(0, self.prediction_length, -1)

        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)
        
        # compute distribution
        distr = self.distr_output.distribution(distr_args, scale=scale)

        # get (batch_size * num_sample_paths, prediction_length) samples
        samples = distr.sample()

        # reshape from (batch_size * num_sample_paths, prediction_length) to
        # (batch_size, num_sample_paths, prediction_length)
        return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))


# In[80]:


class MyProbEstimator(GluonEstimator):
    @validated()
    def __init__(
            self,
            prediction_length: int,
            context_length: int,
            freq: str,
            distr_output: DistributionOutput,
            num_cells: int,
            num_sample_paths: int = 100,
            scaling: bool = True,
            trainer: Trainer = Trainer()
    ) -> None:
        super().__init__(trainer=trainer)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.scaling = scaling

    def create_transformation(self):
        # Feature transformation that the model uses for input.
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_DYNAMIC_REAL,
                        FieldName.OBSERVED_VALUES,
                    ],
                ),

            ]
        )

    def create_training_network(self) -> MyProbTrainNetwork:
        return MyProbTrainNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling
        )

    def create_predictor(
            self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = MyProbPredNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


# In[81]:


estimator = MyProbEstimator(
    prediction_length=custom_ds_metadata['prediction_length'],
    context_length=2*custom_ds_metadata['prediction_length'],
    freq=custom_ds_metadata['freq'],
    distr_output=GaussianOutput(),
    num_cells=40,
    trainer=Trainer(ctx="cpu", 
                    epochs=5, 
                    learning_rate=1e-3, 
                    hybridize=False, 
                    num_batches_per_epoch=100
                   )
)


# In[82]:


predictor = estimator.train(train_ds)


# In[83]:


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


# In[84]:


forecasts = list(forecast_it)
tss = list(ts_it)


# In[85]:


plot_prob_forecasts(tss[0], forecasts[0])


# ## 5.4 From feedforward to RNN
# 
# In all the previous examples we have used a feedforward neural network as the base for our forecasting model. The main idea behind it was to use as an input to the network a window of the time series (of length `context_length`) and train the network to forecast the following window (of length `prediction_length`). 
# 
# In this section we will replace the feedforward network with a recurrent neural network (RNN). Due to the different nature of RNNs the structure of the networks will be a bit different. Let's see what are the major changes. 
# 
# ### 5.4.1 Training
# 
# The main idea behind RNN is the same as in the feedforward networks we already constructed: as we unrolll the RNN at each time step we use as an input past values of the time series and forecast the next value. We can enhance the input by using multiple past values (for example specific lags based on seasonality patterns) or available features. However, in this example we will keep things simple and just use the last value of the time series. The output of the network at each time step is the distribution of the value of the next time step, where the state of the RNN is used as the feature vector for the parameter projection of the distribution.
# 
# Due to the sequential nature of the RNN, the distinction between `past_` and `future_` in the cut window of the time series is not really necessary. Therefore, we can concatenate `past_target` and `future_target ` and treat it as a concrete `target` window that we wish to forecast. This means that the input to the RNN would be (sequentially) the window `target[-(context_length + prediction_length + 1):-1]` (one time step before the window we want to predict). As a consequence, we need to have `context_length + prediction_length + 1` available values at each window that we cut. We can define this in the `InstanceSplitter`. 
# 
# Overall, during training the steps are the following:
# 
# - We pass sequentially through the RNN the target values `target[-(context_length + prediction_length + 1):-1]` 
# - We use the state of the RNN at each time step as a feature vector and project it to the distribution parameter domain
# - The output at each time step is the distribution of the values of the next time step, which overall is the forecasted distribution for the window `target[-(context_length + prediction_length):]`
# 
# The above steps are implemented in the `unroll_encoder` method.
# 
# ### 5.4.2 Inference
# 
# During inference we know the values only of `past_target` therefore we cannot follow exactly the same steps as in training. However the main idea is very similar:
# 
# - We pass sequentially through the RNN the past target values `past_target[-(context_length + 1):]` that effectively updates the state of the RNN
# - In the last time step the output of the RNN is effectively the distribution of the next value of the time series (which we do not know). Therefore we sample (`num_sample_paths` times) from this distribution and use the samples as inputs to the RNN for the next time step
# - We repeat the previous step `prediction_length` times 
# 
# The first step is implemented in `unroll_encoder` and the last steps in the `sample_decoder` method.

# In[86]:


class MyProbRNN(gluon.HybridBlock):
    def __init__(self,
                 prediction_length,
                 context_length,
                 distr_output,
                 num_cells,
                 num_layers,
                 num_sample_paths=100,
                 scaling=True,
                 **kwargs
     ) -> None:
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.num_sample_paths = num_sample_paths
        self.proj_distr_args = distr_output.get_args_proj()
        self.scaling = scaling

        with self.name_scope():
            self.rnn = mx.gluon.rnn.HybridSequentialRNNCell()
            for k in range(self.num_layers):
                cell = mx.gluon.rnn.LSTMCell(hidden_size=self.num_cells)
                cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
                self.rnn.add(cell)

            if scaling:
                self.scaler = MeanScaler(keepdims=True)
            else:
                self.scaler = NOPScaler(keepdims=True)

    def compute_scale(self, past_target, past_observed_values):
        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )

        return scale

    def unroll_encoder(self, 
                       F, 
                       past_target, 
                       past_observed_values, 
                       future_target=None, 
                       future_observed_values=None):
        # overall target field
        # input target from -(context_length + prediction_length + 1) to -1
        if future_target is not None:  # during training
            target_in = F.concat(
                past_target, future_target, dim=-1
            ).slice_axis(
                axis=1, begin=-(self.context_length + self.prediction_length + 1), end=-1
            )

            # overall observed_values field
            # input observed_values corresponding to target_in
            observed_values_in = F.concat(
                past_observed_values, future_observed_values, dim=-1
            ).slice_axis(
                axis=1, begin=-(self.context_length + self.prediction_length + 1), end=-1
            )

            rnn_length = self.context_length + self.prediction_length
        else:  # during inference
            target_in = past_target.slice_axis(
                axis=1, begin=-(self.context_length + 1), end=-1
            )

            # overall observed_values field
            # input observed_values corresponding to target_in
            observed_values_in = past_observed_values.slice_axis(
                axis=1, begin=-(self.context_length + 1), end=-1
            )

            rnn_length = self.context_length

        # compute scale
        scale = self.compute_scale(target_in, observed_values_in)

        # scale target_in
        target_in_scale = F.broadcast_div(target_in, scale)

        # compute network output
        net_output, states = self.rnn.unroll(
            inputs=target_in_scale,
            length=rnn_length,
            layout="NTC",
            merge_outputs=True,
        )

        return net_output, states, scale


class MyProbTrainRNN(MyProbRNN):
    def hybrid_forward(self,
                       F,
                       past_target,
                       future_target,
                       past_observed_values,
                       future_observed_values):

        net_output, _, scale = self.unroll_encoder(F,
                                                   past_target,
                                                   past_observed_values,
                                                   future_target,
                                                   future_observed_values)

        # output target from -(context_length + prediction_length) to end
        target_out = F.concat(
            past_target, future_target, dim=-1
        ).slice_axis(
            axis=1, begin=-(self.context_length + self.prediction_length), end=None
        )

        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)

        # compute distribution
        distr = self.distr_output.distribution(distr_args, scale=scale)

        # negative log-likelihood
        loss = distr.loss(target_out)
        return loss


class MyProbPredRNN(MyProbTrainRNN):
    def sample_decoder(self, F, past_target, states, scale):
        # repeat fields: from (batch_size, past_target_length) to
        # (batch_size * num_sample_paths, past_target_length)
        repeated_states = [
            s.repeat(repeats=self.num_sample_paths, axis=0)
            for s in states
        ]
        repeated_scale = scale.repeat(repeats=self.num_sample_paths, axis=0)

        # first decoder input is the last value of the past_target, i.e.,
        # the previous value of the first time step we want to forecast
        decoder_input = past_target.slice_axis(
            axis=1, begin=-1, end=None
        ).repeat(
            repeats=self.num_sample_paths, axis=0
        )

        # list with samples at each time step
        future_samples = []

        # for each future time step we draw new samples for this time step and update the state
        # the drawn samples are the inputs to the rnn at the next time step
        for k in range(self.prediction_length):
            rnn_outputs, repeated_states = self.rnn.unroll(
                inputs=decoder_input,
                length=1,
                begin_state=repeated_states,
                layout="NTC",
                merge_outputs=True,
            )

            # project network output to distribution parameters domain
            distr_args = self.proj_distr_args(rnn_outputs)

            # compute distribution
            distr = self.distr_output.distribution(distr_args, scale=repeated_scale)

            # draw samples (batch_size * num_samples, 1)
            new_samples = distr.sample()

            # append the samples of the current time step
            future_samples.append(new_samples)

            # update decoder input for the next time step
            decoder_input = new_samples

        samples = F.concat(*future_samples, dim=1)

        # (batch_size, num_samples, prediction_length)
        return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))

    def hybrid_forward(self, F, past_target, past_observed_values):
        # unroll encoder over context_length
        net_output, states, scale = self.unroll_encoder(F,
                                                        past_target,
                                                        past_observed_values)

        samples = self.sample_decoder(F, past_target, states, scale)

        return samples


# In[87]:


class MyProbRNNEstimator(GluonEstimator):
    @validated()
    def __init__(
            self,
            prediction_length: int,
            context_length: int,
            freq: str,
            distr_output: DistributionOutput,
            num_cells: int,
            num_layers: int,
            num_sample_paths: int = 100,
            scaling: bool = True,
            trainer: Trainer = Trainer()
    ) -> None:
        super().__init__(trainer=trainer)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.num_sample_paths = num_sample_paths
        self.scaling = scaling

    def create_transformation(self):
        # Feature transformation that the model uses for input.
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length + 1,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_DYNAMIC_REAL,
                        FieldName.OBSERVED_VALUES,
                    ],
                ),

            ]
        )

    def create_training_network(self) -> MyProbTrainRNN:
        return MyProbTrainRNN(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_layers=self.num_layers,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling
        )

    def create_predictor(
            self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = MyProbPredRNN(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_layers=self.num_layers,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


# In[88]:


estimator = MyProbRNNEstimator(
        prediction_length=24,
        context_length=48,
        freq="1H",
        num_cells=40,
        num_layers=2,
        distr_output=GaussianOutput(),
        trainer=Trainer(ctx="cpu",
                        epochs=5,
                        learning_rate=1e-3,
                        hybridize=False,
                        num_batches_per_epoch=100
                       )
    )


# In[89]:


predictor = estimator.train(train_ds)


# In[90]:


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


# In[91]:


forecasts = list(forecast_it)
tss = list(ts_it)


# In[92]:


plot_prob_forecasts(tss[0], forecasts[0])

