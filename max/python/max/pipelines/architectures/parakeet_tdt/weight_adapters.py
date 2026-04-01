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
weight names to MAX names and permuted Conv2d weights. This adapter strips
the ``encoder.`` prefix and applies shared conformer weight transforms.
"""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.config.config_enums import supported_encoding_dtype

from ..parakeet.weight_utils import apply_conformer_weight_transforms


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
    pipeline_config: PipelineConfig | None = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert pre-converted safetensors state dict to MAX format."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if "num_batches_tracked" in weight_name:
            continue

        max_name = weight_name.removeprefix("encoder.")
        weight_data = value.data()
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
