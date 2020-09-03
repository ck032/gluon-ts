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

from typing import List, Optional, Tuple, Union

# Third-party imports
import mxnet as mx
from mxnet.gluon.rnn import ZoneoutCell
from mxnet.gluon.contrib.rnn import VariationalDropoutCell

# Standard library imports
import numpy as np

from gluonts.core.component import DType, validated
from gluonts.model.common import Tensor

# First-party imports
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import Distribution, DistributionOutput
from gluonts.mx.distribution.distribution import getF
from gluonts.support.util import weighted_average
from gluonts.mx.block.dropout import VariationalZoneoutCell, RNNZoneoutCell
from gluonts.mx.block.regularization import (
    ActivationRegularizationLoss,
    TemporalActivationRegularizationLoss,
)


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class DeepARNetwork(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: List[int],
        lags_seq: List[int],
        dropoutcell_type: str = "ZoneoutCell",
        scaling: bool = True,
        dtype: DType = np.float32,  # RNN精度
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropoutcell_type = dropoutcell_type
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)  # 总计有几个离散特征
        self.scaling = scaling
        self.dtype = dtype

        assert len(cardinality) == len(
            embedding_dimension
        ), "embedding_dimension should be a list with the same size as cardinality"

        assert len(set(lags_seq)) == len(
            lags_seq
        ), "no duplicated lags allowed!"
        lags_seq.sort()  # 滞后项排序，且不允许重复

        self.lags_seq = lags_seq

        self.distr_output = distr_output
        RnnCell = {"lstm": mx.gluon.rnn.LSTMCell, "gru": mx.gluon.rnn.GRUCell}[
            self.cell_type
        ]  # 根据名称获取到实例化对象（字典形式）

        self.target_shape = distr_output.event_shape

        # TODO: is the following restriction needed?
        assert (
            len(self.target_shape) <= 1
        ), "Argument `target_shape` should be a tuple with 1 element at most"

        Dropout = {
            "ZoneoutCell": ZoneoutCell,
            "RNNZoneoutCell": RNNZoneoutCell,
            "VariationalDropoutCell": VariationalDropoutCell,
            "VariationalZoneoutCell": VariationalZoneoutCell,
        }[self.dropoutcell_type]


        with self.name_scope():
            self.proj_distr_args = distr_output.get_args_proj()
            # 实例化rnn，根据传入的cell_type,num_cells,dropout_rate,对每一层的cell进行迭代
            self.rnn = mx.gluon.rnn.HybridSequentialRNNCell()
            for k in range(num_layers):
                cell = RnnCell(hidden_size=num_cells)
                # 迭代：除了第一层，其他层都是取mx.gluon.rnn.ResidualCell(cell)
                cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
                # we found that adding dropout to outputs doesn't improve the performance, so we only drop states
                if "Zoneout" in self.dropoutcell_type:
                    cell = (
                        Dropout(cell, zoneout_states=dropout_rate)
                        if dropout_rate > 0.0
                        else cell
                    )
                elif "Dropout" in self.dropoutcell_type:
                    cell = (
                        Dropout(cell, drop_states=dropout_rate)
                        if dropout_rate > 0.0
                        else cell
                    )
                # 获取到完整的self.rnn
                self.rnn.add(cell)
            self.rnn.cast(dtype=dtype)

            # 对离散特征做embedding
            # 对每个离散特征，设定好embedding的大小（int）类型
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=embedding_dimension,
                dtype=self.dtype,
            )

            # 标准化
            if scaling:
                self.scaler = MeanScaler(keepdims=True)  # 可以简单理解为：绝对值然后取平均值，进行标准化
            else:
                self.scaler = NOPScaler(keepdims=True)  # 不需要标准化

    @staticmethod
    def get_lagged_subsequences(
        F,
        sequence: Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> Tensor:
        """
        Returns lagged subsequences of a given sequence.

        给定序列的滞后项

        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: sequence_length - lag_index - subsequences_length >= 0
        # for all lag_index, hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, "
            f"found lag {max(indices)} while history length is only "
            f"{sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:  # indices是 List[int]
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(
                F.slice_axis(
                    sequence, axis=1, begin=begin_index, end=end_index
                )
            )

        return F.stack(*lagged_values, axis=-1)

    def unroll_encoder(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        feat_static_real: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Optional[
            Tensor
        ],  # (batch_size, prediction_length, num_features)
        future_target: Optional[
            Tensor
        ],  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[Tensor, List, Tensor, Tensor]:
        """
        Unrolls the LSTM encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of past_target
        and a vector of static features that was constructed and fed as input
        to the encoder.
        All tensor arguments should have NTC layout.
        """

        ###########################
        # 1.处理目标变量，获取滞后项
        ###########################
        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=None,
            )
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = F.concat(
                past_time_feat.slice_axis(
                    axis=1,
                    begin=self.history_length - self.context_length,
                    end=None,
                ),
                future_time_feat,
                dim=1,
            )
            sequence = F.concat(past_target, future_target, dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags = self.get_lagged_subsequences(
            F=F,
            sequence=sequence,  # 滞后项是对谁做的？要么是past_target,要么是past_target + future_target ，都是针对target的
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        ###########################
        # 2.标准化目标变量和特征
        ###########################
        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        # scale只应用在past_target、past_observed_values，不仅仅针对target，也针对past_observed_values
        # todo:past_observed_values在哪里出现的呢？
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )
        ###########################
        # 3.离散特征计算embedding
        ###########################
        # 对　feat_static_cat　应用　FeatureEmbedder
        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = F.concat(
            embedded_cat,
            feat_static_real,
            F.log(scale)
            if len(self.target_shape) == 0
            else F.log(scale.squeeze(axis=1)),
            dim=1,
        )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.expand_dims(axis=1).repeat(
            axis=1, repeats=subsequences_length
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = F.reshape(
            data=lags_scaled,
            shape=(
                -1,
                subsequences_length,
                len(self.lags_seq) * prod(self.target_shape),
            ),
        )

        ###########################
        # 4.处理好的特征做concat
        ###########################
        # (batch_size, sub_seq_len, input_dim)
        # (batch_size大小，序列长度，输入的维度）
        # 获取rnn的输入，包括3部分特征：
        # input_lags - 目标列的滞后项
        # time_feat - 时间方面的特征
        # repeated_static_feat - 离散embedding特征
        inputs = F.concat(input_lags, time_feat, repeated_static_feat, dim=-1)

        ###########################
        # 5.处理好的特征输入到rnn中
        ###########################
        # unroll encoder
        outputs, state = self.rnn.unroll(
            inputs=inputs,
            length=subsequences_length,
            layout="NTC",
            merge_outputs=True,
            begin_state=self.rnn.begin_state(
                func=F.zeros,
                dtype=self.dtype,
                batch_size=inputs.shape[0]
                if isinstance(inputs, mx.nd.NDArray)
                else 0,
            ),  # rnn的初始化状态
        )

        # outputs: (batch_size, seq_len, num_cells)
        # state: list of (batch_size, num_cells) tensors
        # scale: (batch_size, 1, *target_shape)
        # static_feat: (batch_size, num_features + prod(target_shape))
        # 输出　self.rnn的outputs,状态state,标准化的scale，static_feat特征
        return outputs, state, scale, static_feat


class DeepARTrainingNetwork(DeepARNetwork):
    @validated()
    def __init__(self, alpha: float = 0, beta: float = 0, **kwargs) -> None:
        super().__init__(**kwargs)

        # regularization weights
        self.alpha = alpha
        self.beta = beta

        if alpha:
            self.ar_loss = ActivationRegularizationLoss(
                alpha, time_axis=1, batch_axis=0
            )
        if beta:
            self.tar_loss = TemporalActivationRegularizationLoss(
                beta, time_axis=1, batch_axis=0
            )

    def distribution(
        self,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
        return_rnn_outputs: bool = False,
    ) -> Union[Distribution, Tuple[Distribution, Tensor]]:
        """

        Returns the distribution predicted by the model on the range of
        past_target and future_target.

        The distribution is obtained by unrolling the network （循环网络的展开）with the true
        target, this is also the distribution that is being minimized during
        training. This can be used in anomaly detection, see for instance
        examples/anomaly_detection.py.

        Input arguments are the same as for the hybrid_forward method.

        Returns
        -------
        Distribution
            a distribution object whose mean has shape:
            (batch_size, context_length + prediction_length).
        Tensor
            (optional) when return_rnn_outputs=True, rnn_outputs will be returned
            so that it could be used for regularization
        """
        # unroll the decoder in "training mode"
        # i.e. by providing future data as well
        F = getF(feat_static_cat)

        rnn_outputs, _, scale, _ = self.unroll_encoder(
            F=F,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        distr_args = self.proj_distr_args(rnn_outputs)

        # return the output of rnn layers if return_rnn_outputs=True, so that it can be used for regularization later
        # assume no dropout for outputs, so can be directly used for activation regularization
        # MARK:根据传入的参数，计算目标变量的似然分布（likelihood of target)
        return (
            (
                self.distr_output.distribution(distr_args, scale=scale),
                rnn_outputs,
            )
            if return_rnn_outputs
            else self.distr_output.distribution(distr_args, scale=scale)
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Computes the loss for training DeepAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape, seq_len)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target : (batch_size, prediction_length, *target_shape)
        future_observed_values : (batch_size, prediction_length, *target_shape)

        Returns loss with shape (batch_size, context + prediction_length, 1)
        -------

        """

        outputs = self.distribution(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
            return_rnn_outputs=True,
        )
        # since return_rnn_outputs=True, assert:
        assert isinstance(outputs, tuple)
        distr, rnn_outputs = outputs

        # put together target sequence
        # (batch_size, seq_len, *target_shape)
        target = F.concat(
            past_target.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=None,
            ),
            future_target,
            dim=1,
        )

        # (batch_size, seq_len)
        loss = distr.loss(target)  # distr 是预测结果，和target相比较，得到损失

        # (batch_size, seq_len, *target_shape)
        observed_values = F.concat(
            past_observed_values.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=self.history_length,
            ),
            future_observed_values,
            dim=1,
        )

        # mask the loss at one time step iff one or more observations is missing in the target dimensions
        # (batch_size, seq_len)
        loss_weights = (
            observed_values
            if (len(self.target_shape) == 0)
            else observed_values.min(axis=-1, keepdims=False)
        )

        weighted_loss = weighted_average(
            F=F, x=loss, weights=loss_weights, axis=1
        )

        # need to mask possible nans and -inf
        loss = F.where(condition=loss_weights, x=loss, y=F.zeros_like(loss))

        # rnn_outputs is already merged into a single tensor
        assert not isinstance(rnn_outputs, list)
        # it seems that the trainer only uses the first return value for backward
        # so we only add regularization to weighted_loss
        if self.alpha:
            ar_loss = self.ar_loss(rnn_outputs)
            weighted_loss = weighted_loss + ar_loss
        if self.beta:
            tar_loss = self.tar_loss(rnn_outputs)
            weighted_loss = weighted_loss + tar_loss

        return weighted_loss, loss # hybrid_forward是用来计算损失的，包括weighted_loss、loss两个部分


class DeepARPredictionNetwork(DeepARNetwork):  # 这个是用来做预测用的（Decoder)
    @validated()
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one, at the first time-step
        # of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        F,
        static_feat: Tensor,
        past_target: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        begin_states: List,
    ) -> Tensor:
        """
        Computes sample paths by unrolling the LSTM starting with a initial
        input and state.

        解码的时候，给定初始输入和状态，然后按照分布进行采样

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch. Shape: (batch_size, 1, 1).
        begin_states : List
            list of initial states for the LSTM layers.
            the shape of each tensor of the list should be (batch_size, num_cells)
        Returns
        --------
        Tensor
            A tensor containing sampled paths.
            Shape: (batch_size, num_sample_paths, prediction_length).
        """

        ############################################################################
        # 采样
        # 假设 self.num_parallel_samples = 100,下面这些都是在做repeat，获取100次采样的结果
        # 每个结果不同
        # 包括 特征（past_target、time_feat、static_feat）、scale、RNN的begin_states
        ############################################################################
        # blows-up the dimension of each tensor to batch_size * self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_time_feat = time_feat.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_static_feat = static_feat.repeat(
            repeats=self.num_parallel_samples, axis=0
        ).expand_dims(axis=1)
        repeated_scale = scale.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_states = [
            s.repeat(repeats=self.num_parallel_samples, axis=0)
            for s in begin_states
        ]

        # 程序的目标就是获取到 future_samples
        future_samples = []

        # for each future time-units we draw new samples for this time-unit and update the state
        # Mark:输入不变，为啥输出会变呢？
        # 这儿就是解释！ 因为self.rnn的state会被更新掉
        # 在每个时间步进行采样
        for k in range(self.prediction_length):
            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = self.get_lagged_subsequences(
                F=F,
                sequence=repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = F.broadcast_div(
                lags, repeated_scale.expand_dims(axis=-1)
            )

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = F.reshape(
                data=lags_scaled,
                shape=(-1, 1, prod(self.target_shape) * len(self.lags_seq)),
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            decoder_input = F.concat(
                input_lags,
                repeated_time_feat.slice_axis(axis=1, begin=k, end=k + 1),
                repeated_static_feat,
                dim=-1,
            )

            # output shape: (batch_size * num_samples, 1, num_cells)
            # state shape: (batch_size * num_samples, num_cells)
            rnn_outputs, repeated_states = self.rnn.unroll(
                inputs=decoder_input,  # 解码的时候，输入是不变的
                length=1,
                begin_state=repeated_states,  # self.rnn的begin_state会倍更新掉，所以输出的结果不同，也就是输出的分布会不一样
                layout="NTC",
                merge_outputs=True,
            )

            distr_args = self.proj_distr_args(rnn_outputs)

            # compute likelihood of target given the predicted parameters
            # MARK:根据传入的参数，计算目标变量的似然分布（likelihood of target)
            distr = self.distr_output.distribution(
                distr_args, scale=repeated_scale
            )

            # (batch_size * num_samples, 1, *target_shape)
            # MARK:从分布中进行采样
            # TODO：采样也会导致输出变化？
            new_samples = distr.sample(dtype=self.dtype)  # 每个时间步的采样

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = F.concat(
                repeated_past_target, new_samples, dim=1
            )
            future_samples.append(new_samples)

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = F.concat(*future_samples, dim=1)  # 这是预测结果

        # (batch_size, num_samples, prediction_length, *target_shape)
        return samples.reshape(
            shape=(
                (-1, self.num_parallel_samples)
                + (self.prediction_length,)
                + self.target_shape
            )
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        feat_static_real: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Tensor,  # (batch_size, prediction_length, num_features)
    ) -> Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns
        -------
        Tensor
            Predicted samples
        """

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, state, scale, static_feat = self.unroll_encoder(
            F=F,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=None,
            future_target=None,
        )

        return self.sampling_decoder(
            F=F,
            past_target=past_target,
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            begin_states=state,
        )
