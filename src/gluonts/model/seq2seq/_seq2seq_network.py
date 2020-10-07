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

# Third-party imports
import mxnet as mx

from gluonts.core.component import validated
from gluonts.model.common import Tensor

# First-party imports
from gluonts.mx.block.decoder import Seq2SeqDecoder
from gluonts.mx.block.enc2dec import Seq2SeqEnc2Dec
from gluonts.mx.block.encoder import Seq2SeqEncoder
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.block.scaler import Scaler


class Seq2SeqNetworkBase(mx.gluon.HybridBlock):
    """
    Base network for the :class:`Seq2SeqEstimator`.

    Parameters
    ----------
    scaler : Scaler
        scale of the target time series, both as input or in the output
        distributions
    encoder : encoder
        see encoder.py for possible choices
    enc2dec : encoder to decoder
        see enc2dec.py for possible choices
    decoder : decoder
        see decoder.py for possible choices
    quantile_output : QuantileOutput
        quantile regression output
    kwargs : dict
        a dict of parameters to be passed to the parent initializer
    """

    @validated()
    def __init__(
        self,
        embedder: FeatureEmbedder,
        scaler: Scaler,
        encoder: Seq2SeqEncoder,
        enc2dec: Seq2SeqEnc2Dec,
        decoder: Seq2SeqDecoder,
        quantile_output: QuantileOutput,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.embedder = embedder
        self.scaler = scaler
        self.encoder = encoder
        self.enc2dec = enc2dec
        self.decoder = decoder
        self.quantile_output = quantile_output

        # 1.分位数的结果连接
        # 2.特别需要注意的是，这儿的损失是分位数损失
        with self.name_scope():
            self.quantile_proj = quantile_output.get_quantile_proj()
            self.loss = quantile_output.get_loss()

    def compute_decoder_outputs(
        self,
        F,
        past_target: Tensor,
        feat_static_cat: Tensor,
        past_feat_dynamic_real: Tensor,
        future_feat_dynamic_real: Tensor,
    ) -> Tensor:
        # 1.特征处理
        scaled_target, scale = self.scaler(
            past_target, F.ones_like(past_target)
        )

        embedded_cat = self.embedder(
            feat_static_cat
        )  # (batch_size, num_features * embedding_size)

        # 2.encoder，注意这里用的是 past_feat_dynamic_real
        encoder_output_static, encoder_output_dynamic = self.encoder(
            scaled_target, embedded_cat, past_feat_dynamic_real
        )

        # 3.encoder 2 decoder，注意这里用的是 future_feat_dynamic_real，future的特征是已知的
        # 这一部分可以理解为，是一个过渡，把encoder部分学习到的结果，和future部分的特征结合在一起，通过4.decoder部分把预测结果计算出来
        decoder_input_static, decoder_input_dynamic = self.enc2dec(
            encoder_output_static,
            encoder_output_dynamic,
            future_feat_dynamic_real,
        )

        # 4.decoder
        decoder_output = self.decoder(
            decoder_input_static, decoder_input_dynamic
        )

        # 5. scaled decoder output
        scaled_decoder_output = F.broadcast_mul(
            decoder_output, scale.expand_dims(-1).expand_dims(-1)
        )
        return scaled_decoder_output


class Seq2SeqTrainingNetwork(Seq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        feat_static_cat: Tensor,
        past_feat_dynamic_real: Tensor,
        future_feat_dynamic_real: Tensor,
    ) -> Tensor:
        """

        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: mx.nd.NDArray or mx.sym.Symbol
            past target
        future_target: mx.nd.NDArray or mx.sym.Symbol
            future target
        feat_static_cat: mx.nd.NDArray or mx.sym.Symbol
            static categorical features
        past_feat_dynamic_real: mx.nd.NDArray or mx.sym.Symbol
            past dynamic real-valued features
        future_feat_dynamic_real: mx.nd.NDArray or mx.sym.Symbol
            future dynamic real-valued features

        Returns
        -------
        mx.nd.NDArray or mx.sym.Symbol
           the computed loss
        """
        # 计算decoder的输出结果
        scaled_decoder_output = self.compute_decoder_outputs(
            F,
            past_target=past_target,
            feat_static_cat=feat_static_cat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            future_feat_dynamic_real=future_feat_dynamic_real,
        )
        projected = self.quantile_proj(scaled_decoder_output)
        # 计算损失，在future_target上计算损失
        loss = self.loss(future_target, projected)
        # TODO: there used to be "nansum" here, to be fully equivalent we
        # TODO: should have a "nanmean" here
        # TODO: shouldn't we sum and divide by the number of observed values
        # TODO: here?
        return loss


class Seq2SeqPredictionNetwork(Seq2SeqNetworkBase):
    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        feat_static_cat: Tensor,
        past_feat_dynamic_real: Tensor,
        future_feat_dynamic_real: Tensor,
    ) -> Tensor:
        """

        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: mx.nd.NDArray or mx.sym.Symbol
            past target
        feat_static_cat: mx.nd.NDArray or mx.sym.Symbol
            static categorical features
        past_feat_dynamic_real: mx.nd.NDArray or mx.sym.Symbol
            past dynamic real-valued features
        future_feat_dynamic_real: mx.nd.NDArray or mx.sym.Symbol
            future dynamic real-valued features

        Returns
        -------
        mx.nd.NDArray or mx.sym.Symbol
            the predicted sequence
        """
        # 和上面的相同，只不过这里直接获取预测结果
        scaled_decoder_output = self.compute_decoder_outputs(
            F,
            past_target=past_target,
            feat_static_cat=feat_static_cat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            future_feat_dynamic_real=future_feat_dynamic_real,
        )
        predictions = self.quantile_proj(scaled_decoder_output).swapaxes(2, 1)

        return predictions
