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
"""Weight adapters for Parakeet-CTC models.

Converts HuggingFace safetensors weight names to MAX weight names.

Most weight names pass through unchanged since both HF and MAX use
``.weight`` and ``.bias`` for all layer types (including Conv1D/Conv2d).

Two transformations are needed for subsampling Conv2d layers:

1. **Index remapping**: HF stores conv layers interleaved with ReLU
   activations in a ``ModuleList`` (indices 0, 2, 3, 5, 6), while MAX
   uses ``initial_conv`` for index 0 and ``dw_pw_stages`` Sequential
   (indices 0-3) for the rest.

2. **Weight permutation**: Subsampling Conv2d uses ``permute=False``
   (to work around a grouped-conv compilation issue on CPU), so weights
   must be pre-permuted from PyTorch FCRS ``(F, C, R, S)`` to RSCF
   ``(R, S, C, F)`` format in the adapter.

BatchNorm ``num_batches_tracked`` parameters are skipped (not needed
at inference).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from max.graph.weights import WeightData, Weights

# HF subsampling ModuleList indices -> MAX remapped names.
_SUBSAMPLING_REMAP = {
    "0": "initial_conv",
    "2": "dw_pw_stages.0",
    "3": "dw_pw_stages.1",
    "5": "dw_pw_stages.2",
    "6": "dw_pw_stages.3",
}


def _remap_subsampling_index(key: str) -> str:
    """Remap HF subsampling layer indices to MAX structure."""
    prefix = "encoder.subsampling.layers."
    if not key.startswith(prefix):
        return key

    rest = key[len(prefix) :]
    dot_pos = rest.index(".")
    hf_idx = rest[:dot_pos]
    suffix = rest[dot_pos:]

    if hf_idx in _SUBSAMPLING_REMAP:
        return "encoder.subsampling." + _SUBSAMPLING_REMAP[hf_idx] + suffix
    return key


def _is_subsampling_conv_weight(key: str) -> bool:
    """Check if a key is a subsampling Conv2d weight (needs FCRS->RSCF permute)."""
    return key.endswith(".weight") and (
        "subsampling.initial_conv." in key or "subsampling.dw_pw_stages." in key
    )


def _is_conformer_depthwise_weight(key: str) -> bool:
    """Check if a key is a conformer depthwise conv weight (needs FCS->SCF permute).

    The depthwise conv is implemented manually (not via Conv1D) to avoid
    GPU compilation bugs. Weights must be pre-permuted from PyTorch
    (F, C/groups, K) to SCF (K, C/groups, F).
    """
    return key.endswith(".weight") and ".conv.depthwise_conv." in key


def _is_conformer_pointwise_weight(key: str) -> bool:
    """Check if a key is a conformer pointwise conv weight (needs squeeze).

    Pointwise convs are replaced with Linear layers. HF stores weights
    as (F, C, 1); Linear expects (F, C) — squeeze the kernel dim.
    """
    return key.endswith(".weight") and (
        ".conv.pointwise_conv1." in key or ".conv.pointwise_conv2." in key
    )


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    """Convert HuggingFace safetensors state dict to MAX format.

    Remaps subsampling layer indices, permutes subsampling Conv2d weights
    from FCRS to RSCF, and skips BatchNorm ``num_batches_tracked``.
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if "num_batches_tracked" in weight_name:
            continue

        max_name = _remap_subsampling_index(weight_name)
        weight_data = value.data()

        # Permute subsampling Conv2d weights: FCRS (F,C,R,S) -> RSCF (R,S,C,F)
        # Conv2d uses permute=False to avoid a grouped-conv compilation bug,
        # so weights must be pre-permuted to RSCF format here.
        if _is_subsampling_conv_weight(max_name):
            arr = np.from_dlpack(weight_data)
            weight_data = WeightData.from_numpy(
                np.ascontiguousarray(arr.transpose(2, 3, 1, 0)),
                name=max_name,
            )

        # Permute conformer depthwise conv weights: FCS -> SCF (K,C,F)
        if _is_conformer_depthwise_weight(max_name):
            arr = np.from_dlpack(weight_data)
            weight_data = WeightData.from_numpy(
                np.ascontiguousarray(arr.transpose(2, 1, 0)),
                name=max_name,
            )

        # Squeeze conformer pointwise conv weights: (F,C,1) -> (F,C)
        # Pointwise convs are now Linear layers.
        if _is_conformer_pointwise_weight(max_name):
            arr = np.from_dlpack(weight_data).copy()
            weight_data = WeightData.from_numpy(
                np.ascontiguousarray(arr.squeeze(-1)),
                name=max_name,
            )

        new_state_dict[max_name] = weight_data

    return new_state_dict
