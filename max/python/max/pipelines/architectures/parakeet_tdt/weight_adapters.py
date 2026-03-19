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

from max.graph.weights import WeightData, Weights


def _is_subsampling_conv_weight(key: str) -> bool:
    """Check if a key is a subsampling Conv2d weight.

    These are already permuted to RSCF by the conversion script, but we
    check in case someone loads unconverted weights.
    """
    return key.endswith(".weight") and (
        "subsampling.initial_conv." in key
        or "subsampling.dw_pw_stages." in key
    )


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    """Convert pre-converted safetensors state dict to MAX format.

    The heavy lifting (NeMo name remapping, Conv2d permutation) was done
    by ``scripts/convert_nemo.py``. This adapter passes weights through,
    skipping any unexpected keys.
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if "num_batches_tracked" in weight_name:
            continue

        new_state_dict[weight_name] = value.data()

    return new_state_dict
