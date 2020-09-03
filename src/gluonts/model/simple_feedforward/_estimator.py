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
from typing import List, Optional

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import RepresentableBlockPredictor
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.trainer import Trainer
from gluonts.transform import (
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    Transformation,
    AddObservedValuesIndicator,
)
from gluonts.model.forecast_generator import DistributionForecastGenerator  # todo:理解forecast_generator
from gluonts.transform.feature import (
    DummyValueImputation,
    MissingValueImputation,
)


# Relative imports
from ._network import (
    SimpleFeedForwardSamplingNetwork,
    SimpleFeedForwardDistributionNetwork,
    SimpleFeedForwardTrainingNetwork,
)


class SimpleFeedForwardEstimator(GluonEstimator):
    """
    SimpleFeedForwardEstimator shows how to build a simple MLP model predicting
    the next target time-steps given the previous ones.

    简单的MLP模型

    Given that we want to define a gluon model trainable by SGD, we inherit the
    parent class `GluonEstimator` that handles most of the logic for fitting a
    neural-network.

    GluonEstimator 处理了大部分拟合NN的工作，所以只需要做好以下的工作就可以了：

    1.数据转换 - create_transformation -> Transformation
    2.训练网络 - create_training_network -> HybridBlock
    3.预测 - create_predictor -> Predictor

    We thus only have to define:

    1. How the data is transformed before being fed to our model::

        def create_transformation(self) -> Transformation

    2. How the training happens::

        def create_training_network(self) -> HybridBlock

    3. how the predictions can be made for a batch given a trained network::

        def create_predictor(
             self,
             transformation: Transformation,
             trained_net: HybridBlock,
        ) -> Predictor


    Parameters
    ----------
    freq
        Time time granularity of the data
    prediction_length
        Length of the prediction horizon
    trainer
        Trainer object to be used (default: Trainer())
    num_hidden_dimensions
        Number of hidden nodes in each layer (default: [40, 40])
    context_length
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    distr_output
        Distribution to fit (default: StudentTOutput())
    batch_normalization
        Whether to use batch normalization (default: False)
    mean_scaling
        Scale the network input by the data mean and the network output by
        its inverse (default: True)
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 100)
    """

    # @validated() 用来保证参数可以通过Pydantic验证，以及允许序列化/打印模型
    # 都是针对__init__来用的
    # Pydantic 是一个使用Python类型提示来进行数据验证和设置管理的库
    # PEP 484 从Python3.5开始引入了类型提示的功能
    # PEP 526 使用Python3.6中的变量注释语法对其进行了拓展
    # 所有的参数，除了freq,prediction_length，都有默认值，以方便模型之间进行比较

    # The validated() decorator makes sure that parameters are checked by
    # Pydantic and allows to serialize/print models. Note that all parameters
    # have defaults except for `freq` and `prediction_length`. which is
    # recommended in GluonTS to allow to compare models easily.
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        sampling: bool = True,
        trainer: Trainer = Trainer(),
        num_hidden_dimensions: Optional[List[int]] = None,
        context_length: Optional[int] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        imputation_method: Optional[MissingValueImputation] = None,
        batch_normalization: bool = False,
        mean_scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        """
        Defines an estimator. All parameters should be serializable.
        """
        super().__init__(trainer=trainer)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert num_hidden_dimensions is None or (
            [d > 0 for d in num_hidden_dimensions]
        ), "Elements of `num_hidden_dimensions` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"

        self.num_hidden_dimensions = (
            num_hidden_dimensions
            if num_hidden_dimensions is not None
            else list([40, 40])
        )
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.freq = freq
        self.distr_output = distr_output
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        self.num_parallel_samples = num_parallel_samples
        self.sampling = sampling
        self.imputation_method = (
            imputation_method
            if imputation_method is not None
            else DummyValueImputation(self.distr_output.value_in_support)
        )

    # here we do only a simple operation to convert the input data to a form
    # that can be digested by our model by only splitting the target in two, a
    # conditioning part and a to-predict part, for each training example.
    # for a more complex transformation example, see the `gluonts.model.deepar`
    # transformation that includes time features, age feature, observed values
    # indicator, ...

    # AddObservedValuesIndicator - 缺失值填充，和加入标识变量
    # InstanceSplitter - 把 target查分为2个部分，条件部分，和预测部分

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                    imputation_method=self.imputation_method,
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=[FieldName.OBSERVED_VALUES],
                ),
            ]
        )

    # defines the network, we get to see one batch to initialize it.
    # the network should return at least one tensor that is used as a loss to minimize in the training loop.
    # several tensors can be returned for instance for analysis, see DeepARTrainingNetwork for an example.

    # 定义网络（network)
    # 网络返回至少一个tensor，在训练过程中来减小loss
    # 还返回很多其他的tensors，用来做分析用
    def create_training_network(self) -> HybridBlock:
        return SimpleFeedForwardTrainingNetwork(
            num_hidden_dimensions=self.num_hidden_dimensions,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            batch_normalization=self.batch_normalization,
            mean_scaling=self.mean_scaling,
        )

    # we now define how the prediction happens given that we are provided a
    # training network.
    def create_predictor(self, transformation, trained_network):
        if self.sampling is True:
            # 与下面相比，少了一个forecast_generator。也就是默认是SampleForecastGenerator()
            prediction_network = SimpleFeedForwardSamplingNetwork(
                num_hidden_dimensions=self.num_hidden_dimensions,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                distr_output=self.distr_output,
                batch_normalization=self.batch_normalization,
                mean_scaling=self.mean_scaling,
                params=trained_network.collect_params(),
                num_parallel_samples=self.num_parallel_samples,
            )

            return RepresentableBlockPredictor(
                input_transform=transformation,
                prediction_net=prediction_network,
                batch_size=self.trainer.batch_size,
                freq=self.freq,
                prediction_length=self.prediction_length,
                ctx=self.trainer.ctx,
            )

        else:
            prediction_network = SimpleFeedForwardDistributionNetwork(
                num_hidden_dimensions=self.num_hidden_dimensions,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                distr_output=self.distr_output,
                batch_normalization=self.batch_normalization,
                mean_scaling=self.mean_scaling,
                params=trained_network.collect_params(),
                num_parallel_samples=self.num_parallel_samples,
            )
            return RepresentableBlockPredictor(
                input_transform=transformation,
                prediction_net=prediction_network,
                batch_size=self.trainer.batch_size,
                forecast_generator=DistributionForecastGenerator(
                    self.distr_output
                ),  # 与上面相比，多了一个forecast_generator
                freq=self.freq,
                prediction_length=self.prediction_length,
                ctx=self.trainer.ctx,
            )
