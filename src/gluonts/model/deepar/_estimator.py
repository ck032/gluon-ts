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

from typing import List, Optional

# Standard library imports
import numpy as np

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import DType, validated
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.trainer import Trainer
from gluonts.support.util import copy_parameters
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SetField,
    Transformation,
    VstackFeatures,
)
from gluonts.transform.feature import (
    DummyValueImputation,
    MissingValueImputation,
)

# Relative imports
from ._network import DeepARPredictionNetwork, DeepARTrainingNetwork


class DeepAREstimator(GluonEstimator):
    """
    Construct a DeepAR estimator.

    This implements an RNN-based model, close to the one described in
    [SFG17]_.

    RNN 模型

    *Note:* the code of this model is unrelated to the implementation behind
    `SageMaker's DeepAR Forecasting Algorithm
    <https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html>`_.

    和 `SageMaker's DeepAR Forecasting Algorithm 无关,bala bala~~~~~~~~~

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    trainer
        Trainer object to be used (default: Trainer())
    context_length
        Number of steps to unroll（展开） the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
        默认值是  context_length = prediction_length
    num_layers
        Number of RNN layers (default: 2)
        默认是2层
    num_cells
        Number of RNN cells for each layer (default: 40)
        每层的cell个数，默认是40个
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm')
        cell类型的选择：`lstm` 或者 `gru`
    dropoutcell_type
        Type of dropout cells to use 
        (available: 'ZoneoutCell', 'RNNZoneoutCell', 'VariationalDropoutCell' or 'VariationalZoneoutCell';
        default: 'ZoneoutCell')
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    use_feat_dynamic_real
        Whether to use the ``feat_dynamic_real`` field from the data
        (default: False)
    use_feat_static_cat
        Whether to use the ``feat_static_cat`` field from the data
        (default: False)
    use_feat_static_real
        Whether to use the ``feat_static_real`` field from the data
        (default: False)
    cardinality
        Number of values of each categorical feature.
        This must be set if ``use_feat_static_cat == True`` (default: None)
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: [min(50, (cat+1)//2) for cat in cardinality])
        MARK:cardinality 、 embedding_dimension 、 use_feat_static_cat 和 离散值有关系
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput())
        输出的分布
    scaling
        Whether to automatically scale the target values (default: true)
        是否标准化目标变量，默认为True
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq)
        目标变量的滞后项，Optional[List[int]]，入参可以在这儿配置，如果是None,那么会根据freq来确定
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
        时间方面的特征，默认的根据freq来确定
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 100)
        推断时，加速用的
    imputation_method
        One of the methods from ImputationStrategy
<<<<<<< HEAD
        缺失项的填充方式
=======
    alpha
        The scaling coefficient of the activation regularization
    beta
        The scaling coefficient of the temporal activation regularization
>>>>>>> master
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        dropoutcell_type: str = "ZoneoutCell",
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        use_feat_static_real: bool = False,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),  # 默认的是t分布
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        imputation_method: Optional[MissingValueImputation] = None,
        dtype: DType = np.float32,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__(trainer=trainer, dtype=dtype)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert num_layers > 0, "The value of `num_layers` should be > 0"
        assert num_cells > 0, "The value of `num_cells` should be > 0"
        supported_dropoutcell_types = [
            "ZoneoutCell",
            "RNNZoneoutCell",
            "VariationalDropoutCell",
            "VariationalZoneoutCell",
        ]
        assert (
            dropoutcell_type in supported_dropoutcell_types
        ), f"`dropoutcell_type` should be one of {supported_dropoutcell_types}"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"
        assert (cardinality and use_feat_static_cat) or (
            not (cardinality or use_feat_static_cat)
        ), "You should set `cardinality` if and only if `use_feat_static_cat=True`"
        assert cardinality is None or all(
            [c > 0 for c in cardinality]
        ), "Elements of `cardinality` should be > 0"
        assert embedding_dimension is None or all(
            [e > 0 for e in embedding_dimension]
        ), "Elements of `embedding_dimension` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"
        assert alpha >= 0, "The value of `alpha` should be >= 0"
        assert beta >= 0, "The value of `beta` should be >= 0"

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )  # 如果不指定的话，默认用prediction_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.distr_output.dtype = dtype
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.dropoutcell_type = dropoutcell_type
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real
        self.cardinality = (
            cardinality if cardinality and use_feat_static_cat else [1]
        )  # Optional[List[int]]
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )  # 注意理解下这里，cardinality - 基数的意思，每个离散的特征，针对每个离散特征，会得到一个embedding_dimension的list[int]
        self.scaling = scaling
        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_lags_for_frequency(freq_str=freq)
        )  # 这儿可以指定采用的滞后项
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        # MARK:实际上只考虑了部分序列的数据，作为past_data,并不是全部的data，也不是想象中的并行切数据
        self.history_length = self.context_length + max(self.lags_seq)  # 上下文长度 +　最大的滞后项长度

        self.num_parallel_samples = num_parallel_samples

        self.imputation_method = (
            imputation_method
            if imputation_method is not None
            else DummyValueImputation(self.distr_output.value_in_support)
        )  # 缺失项的填充值

        self.alpha = alpha
        self.beta = beta

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "use_feat_dynamic_real": stats.num_feat_dynamic_real > 0,
            "use_feat_static_cat": bool(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    # 下面的代码很简单，就是数据做转化，然后把参数传入，做训练，和预测
    def create_transformation(self) -> Transformation:
        # 不考虑的变量名称
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            # 1-Fields方面的transform
            [RemoveFields(field_names=remove_field_names)]  # 删除特征的操作
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat  # 默认是False,所以如果用的话，那么会有SetField的操作,默认是置为[0]
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.use_feat_static_real
                else []
            )
            + [
                # 2-转化-convert操作：转化为np.array
                # 注意：FEAT_STATIC_CAT　、FEAT_STATIC_CAT　默认都是[0.0]
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                    dtype=self.dtype,
                ),

                # 3-feature转化
                # 目标变量进行缺失值填充，并且添加指示变量 "observed_values"：nan值为0
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                    imputation_method=self.imputation_method,
                ),
                # 由List[TimeFeature]，对dataentry做处理，形成的时间频率方面的特征
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,  # self.time_features是一些时间方面的特征，List[TimeFeature]
                    pred_length=self.prediction_length,
                ),
                # 目标列，添加一个自增长序列
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                    dtype=self.dtype,
                ),
                # 纵向合并特征
                # input_fields中包括3个-（FEAT_TIME、FEAT_AGE、FEAT_DYNAMIC_REAL）,这些都是和时间有关系的特征
                # output_fields就只剩下一个FEAT_TIME,采用的是np.vstack，相当于纵向合并特征，形成一个np.array
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                                 + (
                                     [FieldName.FEAT_DYNAMIC_REAL]
                                     if self.use_feat_dynamic_real
                                     else []
                                 ),
                ),
                # 4-数据集拆分转化
                # 把时序问题转变为结构化的有监督学习问题
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    # num_instances和训练过程中的采样数有关，很奇怪为啥这儿是1，而不是更大
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    # past_length： self.history_length = self.context_length + max(self.lags_seq)
                    # 上下文长度 + 最大滞后项的长度
                    past_length=self.history_length,
                    # future_length 就是 prediction_length
                    future_length=self.prediction_length,
                    # 产生past_,future_的特征字段
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                    dummy_value=self.distr_output.value_in_support,
                ),
            ]
        )

    def create_training_network(self) -> DeepARTrainingNetwork:
        return DeepARTrainingNetwork(
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropoutcell_type=self.dropoutcell_type,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            dtype=self.dtype,
            alpha=self.alpha,
            beta=self.beta,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = DeepARPredictionNetwork(
            num_parallel_samples=self.num_parallel_samples,  # 除了这个入参以外，其他的和DeepARTrainingNetwork没有区别
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropoutcell_type=self.dropoutcell_type,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            dtype=self.dtype,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            dtype=self.dtype,
        )
