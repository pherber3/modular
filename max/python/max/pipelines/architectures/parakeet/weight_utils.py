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
"""Shared weight transformation utilities for Parakeet ASR models.

Both CTC and TDT architectures use the same FastConformer encoder, so
depthwise and pointwise conv weight transforms are shared here.
"""

from __future__ import annotations

import numpy as np
from max.graph.weights import WeightData


def is_conformer_depthwise_weight(key: str) -> bool:
    """Check if a key is a conformer depthwise conv weight (needs FCS->SCF permute)."""
    return key.endswith(".weight") and ".conv.depthwise_conv." in key


def is_conformer_pointwise_weight(key: str) -> bool:
    """Check if a key is a conformer pointwise conv weight (needs squeeze).

    Pointwise convs are replaced with Linear layers. HF stores weights
    as (F, C, 1); Linear expects (F, C).
    """
    return key.endswith(".weight") and (
        ".conv.pointwise_conv1." in key or ".conv.pointwise_conv2." in key
    )


def apply_conformer_weight_transforms(
    key: str, weight_data: WeightData
) -> WeightData:
    """Apply depthwise permutation or pointwise squeeze if needed.

    Args:
        key: Weight name after any prefix stripping.
        weight_data: Raw weight data.

    Returns:
        Transformed weight data (or original if no transform needed).
    """
    if is_conformer_depthwise_weight(key):
        arr = np.from_dlpack(weight_data)
        return WeightData.from_numpy(
            np.ascontiguousarray(arr.transpose(2, 1, 0)),
            name=key,
        )

    if is_conformer_pointwise_weight(key):
        arr = np.from_dlpack(weight_data).copy()
        return WeightData.from_numpy(
            np.ascontiguousarray(arr.squeeze(-1)),
            name=key,
        )

    return weight_data
