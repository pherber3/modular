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

Two CTC-specific transformations:

1. **Index remapping**: HF stores conv layers interleaved with ReLU
   activations in a ``ModuleList`` (indices 0, 2, 3, 5, 6), while MAX
   uses ``initial_conv`` for index 0 and ``dw_pw_stages`` Sequential
   (indices 0-3) for the rest.

2. **Subsampling weight permutation**: Conv2d uses ``permute=False``
   (grouped-conv compilation workaround), so weights are pre-permuted
   from PyTorch FCRS to RSCF format.

Conformer depthwise/pointwise transforms are shared with TDT via
``weight_utils.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.config.config_enums import supported_encoding_dtype

from .weight_utils import apply_conformer_weight_transforms

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


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
    pipeline_config: PipelineConfig | None = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert HuggingFace safetensors state dict to MAX format."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if "num_batches_tracked" in weight_name:
            continue

        max_name = _remap_subsampling_index(weight_name)
        weight_data = value.data()

        # Permute subsampling Conv2d weights: FCRS -> RSCF
        if _is_subsampling_conv_weight(max_name):
            arr = np.from_dlpack(weight_data)
            weight_data = WeightData.from_numpy(
                np.ascontiguousarray(arr.transpose(2, 3, 1, 0)),
                name=max_name,
            )

        # Shared conformer transforms (depthwise permute, pointwise squeeze)
        weight_data = apply_conformer_weight_transforms(max_name, weight_data)

        new_state_dict[max_name] = weight_data

    # Handle dtype casting (e.g. float32 -> bfloat16)
    if pipeline_config is not None:
        model_config = pipeline_config.model
        if model_config._applied_dtype_cast_from:
            cast_from = model_config._applied_dtype_cast_from
            cast_to = model_config._applied_dtype_cast_to
            assert cast_to
            for key, weight_data in new_state_dict.items():
                if weight_data.dtype == supported_encoding_dtype(cast_from):
                    new_state_dict[key] = weight_data.astype(
                        supported_encoding_dtype(cast_to)
                    )

    return new_state_dict
