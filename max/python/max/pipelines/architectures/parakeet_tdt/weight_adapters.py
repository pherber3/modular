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
"""Weight adapters for Parakeet-TDT models.

The conversion script (``scripts/convert_nemo.py``) has already remapped NeMo
weight names to MAX names and permuted Conv2d weights. This adapter only needs
to handle the safetensors → MAX WeightData conversion and skip any keys not
needed for the encoder graph (decoder/joint weights are loaded separately).

BatchNorm ``num_batches_tracked`` is already stripped by the conversion script.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from max.graph.weights import WeightData, Weights


def _is_subsampling_conv_weight(key: str) -> bool:
    """Check if a key is a subsampling Conv2d weight.

    These are already permuted to RSCF by the conversion script, but we
    check in case someone loads unconverted weights.
    """
    return key.endswith(".weight") and (
        "subsampling.initial_conv." in key or "subsampling.dw_pw_stages." in key
    )


def _is_conformer_depthwise_weight(key: str) -> bool:
    """Check if a key is a conformer depthwise conv weight (needs FCS->SCF permute)."""
    return key.endswith(".weight") and ".conv.depthwise_conv." in key


def _is_conformer_pointwise_weight(key: str) -> bool:
    """Check if a key is a conformer pointwise conv weight (needs squeeze).

    Pointwise convs are replaced with Linear layers. Weights stored as
    (F, C, 1) need to be squeezed to (F, C).
    """
    return key.endswith(".weight") and (
        ".conv.pointwise_conv1." in key or ".conv.pointwise_conv2." in key
    )


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    """Convert pre-converted safetensors state dict to MAX format.

    The heavy lifting (NeMo name remapping, Conv2d permutation) was done
    by ``scripts/convert_nemo.py``. This adapter passes weights through,
    skipping any unexpected keys. Depthwise Conv1D weights are permuted
    from PyTorch (F,C,K) to MAX (K,C,F) format.
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if "num_batches_tracked" in weight_name:
            continue

        # Strip "encoder." prefix — ParakeetEncoder is the root module
        max_name = weight_name.removeprefix("encoder.")
        weight_data = value.data()

        # Permute conformer depthwise conv weights: FCS -> SCF (K,C,F)
        if _is_conformer_depthwise_weight(max_name):
            arr = np.from_dlpack(weight_data)
            weight_data = WeightData.from_numpy(
                np.ascontiguousarray(arr.transpose(2, 1, 0)),
                name=max_name,
            )

        # Squeeze conformer pointwise conv weights: (F,C,1) -> (F,C)
        if _is_conformer_pointwise_weight(max_name):
            arr = np.from_dlpack(weight_data).copy()
            weight_data = WeightData.from_numpy(
                np.ascontiguousarray(arr.squeeze(-1)),
                name=max_name,
            )

        new_state_dict[max_name] = weight_data

    return new_state_dict
