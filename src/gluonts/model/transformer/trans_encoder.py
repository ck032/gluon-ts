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
from typing import Dict

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.model.transformer.layers import (
    InputLayer,
    MultiHeadSelfAttention,
    TransformerFeedForward,
    TransformerProcessBlock,
)


class TransformerEncoder(HybridBlock):
    @validated()
    def __init__(self, encoder_length: int, config: Dict, **kwargs) -> None:

        super().__init__(**kwargs)

        # 编码器的长度，是指self.context_length
        self.encoder_length = encoder_length

        with self.name_scope():
            self.enc_input_layer = InputLayer(model_size=config["model_dim"])

            # 编码器－self-attention　注意力
            # 层的归一化操作
            self.enc_pre_self_att = TransformerProcessBlock(
                sequence=config["pre_seq"],      # pre_seq: str = "dn",包含d:drop out 、n: layer normalization  归一化
                dropout=config["dropout_rate"],  # dropout_rate: float = 0.1
                prefix="pretransformerprocessblock_",
            )
            self.enc_self_att = MultiHeadSelfAttention(
                att_dim_in=config["model_dim"],  # model_dim: int = 32
                heads=config["num_heads"],    # num_heads: int = 8
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="multiheadselfattention_",
            )
            self.enc_post_self_att = TransformerProcessBlock(
                sequence=config["post_seq"],  # post_seq: str = "drn", 顺序进行drop out 、残差链接、归一化
                dropout=config["dropout_rate"],
                prefix="postselfatttransformerprocessblock_",
            )
            # feed - forward
            # 编码器－feed-forward　前馈神经网络
            # 与解码器－decoder相同
            self.enc_ff = TransformerFeedForward(
                inner_dim=config["model_dim"] * config["inner_ff_dim_scale"], # inner_ff_dim_scale: int = 4 ,默认32 * 4
                out_dim=config["model_dim"], # 默认32
                act_type=config["act_type"],  # act_type: str = "softrelu"
                dropout=config["dropout_rate"],
                prefix="transformerfeedforward_",
            )
            self.enc_post_ff = TransformerProcessBlock(
                sequence=config["post_seq"],  # post_seq: str = "drn", 顺序进行drop out 、残差链接、归一化
                dropout=config["dropout_rate"],
                prefix="postfftransformerprocessblock_",
            )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, data: Tensor) -> Tensor:

        """
        A transformer encoder block consists of a self-attention and a feed-forward layer with pre/post process blocks
        in between.
        """

        #　一、input layer
        # 输入层
        inputs = self.enc_input_layer(data)

        # 二、self-attention
        # 注意力模块:存在路径依赖，无法并行
        # ３个步骤：
        # (1) self.enc_pre_self_att　，添加drop-out,norm操作
        # (2) self.enc_self_att　多头注意力　MultiHeadSelfAttention
        # (3) self.enc_post_self_att
        data_self_att, _ = self.enc_self_att(
            self.enc_pre_self_att(inputs, None)  #
        )
        data = self.enc_post_self_att(data_self_att, inputs)

        # 三、feed-forward:不存在路径依赖，可以并行
        # 2个步骤：
        # (1) self.enc_ff
        # (2) self.enc_post_ff
        data_ff = self.enc_ff(data)
        data = self.enc_post_ff(data_ff, data)

        return data
