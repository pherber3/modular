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
"""Graph construction for Parakeet-CTC.

Builds the computation graph for the full Parakeet-CTC model:
mel spectrogram input -> encoder -> CTC logits output.
"""

from __future__ import annotations

from collections.abc import Mapping

from max.driver import DLPackArray
from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.weights import WeightData

from .encoder import ParakeetForCTC
from .model_config import ParakeetModelConfig


def build_graph(
    config: ParakeetModelConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
) -> Graph:
    """Build the computation graph for Parakeet-CTC.

    Args:
        config: Model configuration.
        state_dict: Weight name -> data mapping.

    Returns:
        Compiled graph accepting mel spectrogram input.
    """
    input_type = TensorType(
        DType.float32,
        shape=["batch_size", "num_frames", config.num_mel_bins],
        device=config.device,
    )

    with Graph("parakeet_ctc", input_types=[input_type]) as graph:
        model = ParakeetForCTC(config)
        model.load_state_dict(state_dict)
        input_features = graph.inputs[0].tensor
        logits = model(input_features)
        graph.output(logits)

    return graph
