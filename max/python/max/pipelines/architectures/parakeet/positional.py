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
"""Relative positional encoding for Parakeet conformer attention.

Implements sinusoidal relative position embeddings and the Shaw et al.
``rel_shift`` skewing trick used in Transformer-XL style attention.
"""

from __future__ import annotations

import math

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


def compute_relative_position_encoding(
    seq_length: TensorValue,
    hidden_size: int,
    dtype: DType,
    device: DeviceRef,
) -> TensorValue:
    """Compute sinusoidal relative position embeddings.

    Generates position IDs from ``seq_length - 1`` down to ``-seq_length + 1``
    (total ``2 * seq_length - 1`` positions), then computes interleaved sin/cos
    embeddings of dimension ``hidden_size``.

    Returns:
        Tensor of shape ``(1, 2 * seq_length - 1, hidden_size)``.
    """
    # Position IDs: [T-1, T-2, ..., 1, 0, -1, ..., -(T-1)]
    pos_length = seq_length * 2 - 1
    # Build positions as (T-1) - range(0, 2T-1) to keep everything on device.
    # ops.range produces a device tensor; subtracting it from another device
    # tensor avoids a CPU scalar ending up in the graph.
    indices = ops.cast(
        ops.range(0, pos_length, dtype=DType.int64, device=device),
        dtype,
    )
    # (seq_length - 1) as a device scalar via range(seq_length-1, seq_length)
    offset = ops.cast(
        ops.range(seq_length - 1, seq_length, dtype=DType.int64, device=device),
        dtype,
    )
    positions = offset - indices  # [T-1, T-2, ..., 0, -1, ..., -(T-1)]

    # Inverse frequencies: 1 / (10000^(2i/d))
    half_dim = hidden_size // 2
    freq_indices = ops.cast(
        ops.range(0, half_dim, dtype=DType.int64, device=device),
        dtype,
    )
    inv_freq = ops.exp(
        freq_indices
        * ops.constant(
            -math.log(10000.0) * 2.0 / hidden_size, dtype, device=device
        )
    )

    # Outer product: (2T-1,) x (d/2,) -> (2T-1, d/2)
    positions = ops.unsqueeze(positions, -1)  # (2T-1, 1)
    inv_freq = ops.unsqueeze(inv_freq, 0)  # (1, d/2)
    freqs = positions * inv_freq  # (2T-1, d/2)

    # Interleave sin and cos: stack then reshape
    sin_emb = ops.sin(freqs)  # (2T-1, d/2)
    cos_emb = ops.cos(freqs)  # (2T-1, d/2)
    pos_embed = ops.stack([sin_emb, cos_emb], axis=-1)  # (2T-1, d/2, 2)
    pos_embed = ops.reshape(pos_embed, [pos_length, hidden_size])  # (2T-1, d)

    return ops.unsqueeze(pos_embed, 0)  # (1, 2T-1, d)


def rel_shift(x: TensorValue) -> TensorValue:
    """Shaw et al. relative position shift (skewing trick).

    See appendix B of https://huggingface.co/papers/1901.02860.

    Input: ``(batch, heads, query_len, position_len)``.
    Output: same shape, with positions aligned for content-position attention.
    """
    batch = x.shape[0]
    heads = x.shape[1]
    query_len = x.shape[2]
    pos_len = x.shape[3]

    # Pad left by 1 on last dim: (B, H, Q, P+1)
    # Format: [before_d0, after_d0, before_d1, after_d1, ...]
    x = ops.pad(x, [0, 0, 0, 0, 0, 0, 1, 0])

    x = ops.reshape(x, [batch, heads, pos_len + 1, query_len])

    # Slice off first row and reshape back: (B, H, Q, P)
    x = ops.slice_tensor(
        x, [slice(None), slice(None), slice(1, None), slice(None)]
    )
    x = ops.reshape(x, [batch, heads, query_len, pos_len])

    return x
