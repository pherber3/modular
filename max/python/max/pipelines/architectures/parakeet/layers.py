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
"""Conformer layer building blocks for Parakeet.

Contains the reusable modules that make up each conformer layer:
BatchNorm1d, feed-forward, convolution module, relative attention,
and the full ConformerLayer that composes them.

Reference: HuggingFace Transformers ``modeling_parakeet.py``.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.conv import Conv1D
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import LayerNorm

from .model_config import ParakeetModelConfig
from .positional import rel_shift

# ---------------------------------------------------------------------------
# BatchNorm1d (inference-only)
# ---------------------------------------------------------------------------


class BatchNorm1d(Module):
    """Inference-only BatchNorm1d using pre-computed running statistics.

    Input shape: ``(batch, channels, length)`` (channel-first, matching Conv1D).
    """

    def __init__(
        self,
        num_features: int,
        dtype: DType,
        device: DeviceRef,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Weight(
            name="weight", dtype=dtype, shape=[num_features], device=device
        )
        self.bias = Weight(
            name="bias", dtype=dtype, shape=[num_features], device=device
        )
        self.running_mean = Weight(
            name="running_mean",
            dtype=dtype,
            shape=[num_features],
            device=device,
        )
        self.running_var = Weight(
            name="running_var",
            dtype=dtype,
            shape=[num_features],
            device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        # x: (batch, channels, length)
        # Broadcast running stats over batch and length dims.
        mean = ops.unsqueeze(ops.unsqueeze(self.running_mean, 0), -1)
        var = ops.unsqueeze(ops.unsqueeze(self.running_var, 0), -1)
        w = ops.unsqueeze(ops.unsqueeze(self.weight, 0), -1)
        b = ops.unsqueeze(ops.unsqueeze(self.bias, 0), -1)

        inv_std = ops.rsqrt(
            var + ops.constant(self.eps, x.dtype, device=x.device)
        )
        return (x - mean) * inv_std * w + b


# ---------------------------------------------------------------------------
# Feed-forward module
# ---------------------------------------------------------------------------


class ParakeetFeedForward(Module):
    """Two-layer FFN with SiLU activation."""

    def __init__(self, config: ParakeetModelConfig) -> None:
        super().__init__()
        self.linear1 = Linear(
            config.hidden_size,
            config.intermediate_size,
            config.dtype,
            config.device,
            has_bias=config.attention_bias,
        )
        self.linear2 = Linear(
            config.intermediate_size,
            config.hidden_size,
            config.dtype,
            config.device,
            has_bias=config.attention_bias,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.linear2(ops.silu(self.linear1(x)))


# ---------------------------------------------------------------------------
# Convolution module
# ---------------------------------------------------------------------------


class ParakeetConvModule(Module):
    """Conformer convolution module.

    Architecture: pointwise -> GLU -> depthwise -> BatchNorm -> SiLU -> pointwise.
    """

    def __init__(self, config: ParakeetModelConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size
        padding = (kernel_size - 1) // 2

        conv_bias = config.convolution_bias

        self.pointwise_conv1 = Conv1D(
            kernel_size=1,
            in_channels=channels,
            out_channels=2 * channels,
            dtype=config.dtype,
            stride=1,
            padding=0,
            device=config.device,
            has_bias=conv_bias,
            permute=True,
        )
        self.depthwise_conv = Conv1D(
            kernel_size=kernel_size,
            in_channels=channels,
            out_channels=channels,
            dtype=config.dtype,
            stride=1,
            padding=padding,
            num_groups=channels,
            device=config.device,
            has_bias=conv_bias,
            permute=True,
        )
        self.norm = BatchNorm1d(
            channels, dtype=config.dtype, device=config.device
        )
        self.pointwise_conv2 = Conv1D(
            kernel_size=1,
            in_channels=channels,
            out_channels=channels,
            dtype=config.dtype,
            stride=1,
            padding=0,
            device=config.device,
            has_bias=conv_bias,
            permute=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        # x: (batch, seq_len, channels) -> transpose for Conv1D
        x = x.transpose(1, 2)  # (batch, channels, seq_len)

        x = self.pointwise_conv1(x)  # (batch, 2*channels, seq_len)
        half = x.shape[1] // 2
        x_a, x_b = ops.split(x, [half, half], axis=1)
        x = x_a * ops.sigmoid(x_b)  # (batch, channels, seq_len)

        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = ops.silu(x)

        x = self.pointwise_conv2(x)  # (batch, channels, seq_len)
        return x.transpose(1, 2)  # (batch, seq_len, channels)


# ---------------------------------------------------------------------------
# Relative multi-head attention
# ---------------------------------------------------------------------------


class ParakeetAttention(Module):
    """Multi-head attention with relative positional encoding.

    Implements the four-term decomposition from Shaw et al. (2018):
    ``A = (Q + bias_u) @ K^T + rel_shift((Q + bias_v) @ R^T)``
    """

    def __init__(self, config: ParakeetModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(
            config.hidden_size,
            config.hidden_size,
            config.dtype,
            config.device,
            has_bias=config.attention_bias,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.hidden_size,
            config.dtype,
            config.device,
            has_bias=config.attention_bias,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.hidden_size,
            config.dtype,
            config.device,
            has_bias=config.attention_bias,
        )
        self.o_proj = Linear(
            config.hidden_size,
            config.hidden_size,
            config.dtype,
            config.device,
            has_bias=config.attention_bias,
        )
        self.relative_k_proj = Linear(
            config.hidden_size,
            config.hidden_size,
            config.dtype,
            config.device,
            has_bias=False,
        )
        self.bias_u = Weight(
            name="bias_u",
            dtype=config.dtype,
            shape=[self.num_heads, self.head_dim],
            device=config.device,
        )
        self.bias_v = Weight(
            name="bias_v",
            dtype=config.dtype,
            shape=[self.num_heads, self.head_dim],
            device=config.device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        position_embeddings: TensorValue,
    ) -> TensorValue:
        """
        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``
            position_embeddings: ``(1, 2*seq_len-1, hidden_size)``

        Returns:
            ``(batch, seq_len, hidden_size)``
        """
        batch = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Project Q, K, V and reshape to (batch, heads, seq_len, head_dim)
        q = ops.reshape(
            self.q_proj(hidden_states),
            [batch, seq_len, self.num_heads, self.head_dim],
        ).transpose(1, 2)
        k = ops.reshape(
            self.k_proj(hidden_states),
            [batch, seq_len, self.num_heads, self.head_dim],
        ).transpose(1, 2)
        v = ops.reshape(
            self.v_proj(hidden_states),
            [batch, seq_len, self.num_heads, self.head_dim],
        ).transpose(1, 2)

        # Bias shapes: (1, num_heads, 1, head_dim) for broadcasting
        bias_u = ops.reshape(self.bias_u, [1, self.num_heads, 1, self.head_dim])
        bias_v = ops.reshape(self.bias_v, [1, self.num_heads, 1, self.head_dim])

        # Content attention: (Q + bias_u) @ K^T * scale
        q_with_bias_u = q + bias_u
        matrix_ac = (q_with_bias_u @ k.transpose(-1, -2)) * self.scaling

        # Position attention: (Q + bias_v) @ R^T, then rel_shift
        # Position embeddings are batch-independent: (1, 2T-1, hidden_size)
        q_with_bias_v = q + bias_v
        rel_key = self.relative_k_proj(position_embeddings)
        pos_len = rel_key.shape[1]
        rel_key = ops.reshape(
            rel_key,
            [1, pos_len, self.num_heads, self.head_dim],
        )  # (1, 2T-1, heads, head_dim)
        # (batch, heads, seq_len, head_dim) @ (1, heads, head_dim, 2T-1)
        # Broadcasting handles the batch dimension.
        matrix_bd = q_with_bias_v @ rel_key.permute([0, 2, 3, 1])
        matrix_bd = rel_shift(matrix_bd)
        # Keep only first seq_len positions
        matrix_bd = ops.slice_tensor(
            matrix_bd,
            [slice(None), slice(None), slice(None), slice(0, seq_len)],
        )
        matrix_bd = matrix_bd * self.scaling

        attn_weights = matrix_ac + matrix_bd
        attn_weights = ops.softmax(attn_weights)

        attn_output = attn_weights @ v  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(
            1, 2
        )  # (batch, seq_len, heads, head_dim)
        attn_output = ops.reshape(attn_output, [batch, seq_len, -1])

        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Conformer layer
# ---------------------------------------------------------------------------


class ParakeetConformerLayer(Module):
    """Single Conformer layer with Macaron-style FF-Attn-Conv-FF structure."""

    def __init__(self, config: ParakeetModelConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        self.feed_forward1 = ParakeetFeedForward(config)
        self.self_attn = ParakeetAttention(config, layer_idx)
        self.conv = ParakeetConvModule(config)
        self.feed_forward2 = ParakeetFeedForward(config)

        self.norm_feed_forward1 = LayerNorm(
            hidden_size, devices=[config.device], dtype=config.dtype
        )
        self.norm_self_att = LayerNorm(
            hidden_size, devices=[config.device], dtype=config.dtype
        )
        self.norm_conv = LayerNorm(
            hidden_size, devices=[config.device], dtype=config.dtype
        )
        self.norm_feed_forward2 = LayerNorm(
            hidden_size, devices=[config.device], dtype=config.dtype
        )
        self.norm_out = LayerNorm(
            hidden_size, devices=[config.device], dtype=config.dtype
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        position_embeddings: TensorValue,
    ) -> TensorValue:
        residual = hidden_states
        hidden_states = self.feed_forward1(
            self.norm_feed_forward1(hidden_states)
        )
        hidden_states = residual + hidden_states * 0.5

        attn_output = self.self_attn(
            self.norm_self_att(hidden_states), position_embeddings
        )
        hidden_states = hidden_states + attn_output

        conv_output = self.conv(self.norm_conv(hidden_states))
        hidden_states = hidden_states + conv_output

        ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + ff2_output * 0.5

        return self.norm_out(hidden_states)
