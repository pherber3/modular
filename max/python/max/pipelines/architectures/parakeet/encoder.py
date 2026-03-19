# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Parakeet-CTC encoder and top-level model.

Contains the Conv2D subsampling, full encoder stack, and CTC head.
Conformer layer building blocks are in ``layers.py``; relative positional
encoding is in ``positional.py``.

Reference: HuggingFace Transformers ``modeling_parakeet.py``.
"""

from __future__ import annotations

import math

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.conv import Conv1D, Conv2d
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.sequential import Sequential

from .layers import ParakeetConformerLayer
from .model_config import ParakeetModelConfig
from .positional import compute_relative_position_encoding


class ParakeetSubsampling(Module):
    """Conv2D subsampling: 3 stages of stride-2 convolutions (8x downsample).

    Uses ``permute=False`` (NHWC input, RSCF weights) because grouped/depthwise
    Conv2d with ``permute=True`` has a compilation issue on CPU. The weight
    adapter pre-permutes subsampling weights from PyTorch FCRS to RSCF format.

    TODO: Revert to ``permute=True`` once grouped Conv2d compilation on CPU
    is fixed upstream (error: "All operation types must have the same shape").
    """

    def __init__(self, config: ParakeetModelConfig) -> None:
        super().__init__()
        channels = config.subsampling_conv_channels
        kernel_size = config.subsampling_conv_kernel_size
        stride = config.subsampling_conv_stride
        padding = (kernel_size - 1) // 2
        num_layers = int(math.log2(config.subsampling_factor))

        self.initial_conv = Conv2d(
            kernel_size=kernel_size,
            in_channels=1,
            out_channels=channels,
            dtype=config.dtype,
            stride=stride,
            padding=padding,
            device=config.device,
            has_bias=True,
            permute=False,
        )

        dw_pw_layers: list[Module] = []
        for _ in range(num_layers - 1):
            dw_pw_layers.append(
                Conv2d(
                    kernel_size=kernel_size,
                    in_channels=channels,
                    out_channels=channels,
                    dtype=config.dtype,
                    stride=stride,
                    padding=padding,
                    num_groups=channels,
                    device=config.device,
                    has_bias=True,
                    permute=False,
                )
            )
            dw_pw_layers.append(
                Conv2d(
                    kernel_size=1,
                    in_channels=channels,
                    out_channels=channels,
                    dtype=config.dtype,
                    stride=1,
                    padding=0,
                    device=config.device,
                    has_bias=True,
                    permute=False,
                )
            )

        self.dw_pw_stages = Sequential(dw_pw_layers)

        out_mel = config.num_mel_bins // (stride**num_layers)
        self.linear = Linear(
            channels * out_mel,
            config.hidden_size,
            config.dtype,
            config.device,
            has_bias=True,
        )

    def __call__(self, input_features: TensorValue) -> TensorValue:
        """
        Args:
            input_features: ``(batch, num_frames, num_mel_bins)``

        Returns:
            ``(batch, num_frames // subsampling_factor, hidden_size)``
        """
        batch = input_features.shape[0]

        # NHWC format: (batch, num_frames, mel_bins, 1)
        x = ops.unsqueeze(input_features, -1)

        x = ops.relu(self.initial_conv(x))

        layers = self.dw_pw_stages.layers
        for i in range(0, len(layers), 2):
            x = ops.relu(layers[i + 1](layers[i](x)))

        # x is NHWC: (batch, T', mel', channels)
        # HF reference flattens as (batch, T', channels, mel') from NCHW.
        # Swap mel' and channels to match the expected flatten order.
        x = x.transpose(2, 3)  # (batch, T', channels, mel')
        x = ops.reshape(
            x, [batch, x.shape[1], -1]
        )  # (batch, T', channels*mel')

        return self.linear(x)


class ParakeetEncoder(Module):
    """Parakeet encoder: Conv2D subsampling + N Conformer layers."""

    def __init__(self, config: ParakeetModelConfig) -> None:
        super().__init__()
        self.input_scale = (
            math.sqrt(config.hidden_size) if config.scale_input else 1.0
        )
        self.config = config

        self.subsampling = ParakeetSubsampling(config)
        self.layers = Sequential(
            [
                ParakeetConformerLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )

    def __call__(self, input_features: TensorValue) -> TensorValue:
        """
        Args:
            input_features: ``(batch, num_frames, num_mel_bins)``

        Returns:
            ``(batch, seq_len, hidden_size)`` where ``seq_len ~ num_frames // 8``.
        """
        hidden_states = self.subsampling(input_features)
        hidden_states = hidden_states * self.input_scale

        seq_len = hidden_states.shape[1]
        position_embeddings = compute_relative_position_encoding(
            seq_len,
            self.config.hidden_size,
            self.config.dtype,
            self.config.device,
        )

        # Iterate layers directly (not via Sequential.__call__) because
        # each conformer layer takes two arguments (hidden_states + position).
        for layer in self.layers.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        return hidden_states


class ParakeetForCTC(Module):
    """Parakeet encoder with CTC linear decoder head.

    The CTC head is a Conv1d(hidden_size, vocab_size, kernel_size=1),
    consistent with NVIDIA NeMo's decoding layer convention.
    """

    def __init__(self, config: ParakeetModelConfig) -> None:
        super().__init__()
        self.encoder = ParakeetEncoder(config)
        self.ctc_head = Conv1D(
            kernel_size=1,
            in_channels=config.hidden_size,
            out_channels=config.vocab_size,
            dtype=config.dtype,
            stride=1,
            padding=0,
            device=config.device,
            has_bias=True,
            permute=True,
        )

    def __call__(self, input_features: TensorValue) -> TensorValue:
        """
        Args:
            input_features: ``(batch, num_frames, num_mel_bins)``

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        hidden_states = self.encoder(input_features)

        logits = self.ctc_head(hidden_states.transpose(1, 2))
        logits = logits.transpose(1, 2)  # (batch, seq_len, vocab_size)

        return ops.cast(logits, DType.float32)
