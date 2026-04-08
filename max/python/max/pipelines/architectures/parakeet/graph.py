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
from max.graph import Graph, TensorType, ops
from max.graph.weights import WeightData

from .encoder import ParakeetForCTC
from .model_config import ParakeetModelConfig


def build_graph(
    config: ParakeetModelConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
    num_frames: int,
    pad_to_encoder_frames: int,
) -> Graph:
    """Build the computation graph for Parakeet-CTC.

    Args:
        config: Model configuration.
        state_dict: Weight name -> data mapping.
        num_frames: Fixed mel-frame count this graph compiles against.
            Each bucket builds its own graph at its own ``num_frames``.
        pad_to_encoder_frames: The global maximum encoder-frame count
            (across all buckets). The output logits are zero-padded along
            the time axis to this size so all bucket graphs return the
            same shape — downstream callers slice off the padded tail
            using the bucket's true ``encoder_frames``.

    Returns:
        Compiled graph accepting mel spectrogram input
        ``[1, num_frames, num_mel_bins]`` and producing logits
        ``[1, pad_to_encoder_frames, vocab_size]``.
    """
    input_type = TensorType(
        DType.float32,
        shape=[1, num_frames, config.num_mel_bins],
        device=config.device,
    )

    with Graph("parakeet_ctc", input_types=[input_type]) as graph:
        model = ParakeetForCTC(config)
        model.load_state_dict(state_dict)
        input_features = graph.inputs[0].tensor
        logits = model(input_features)
        # Zero-pad the time axis to the global max so every bucket's CTC
        # graph returns the same output shape. Uses ``ops.pad`` so the
        # compiler emits a single constant-pad kernel rather than
        # materializing a zero tensor + concat. The padded region holds
        # blank-token logits and is sliced off in numpy decode.
        bucket_encoder_frames = num_frames // 8
        pad_len = pad_to_encoder_frames - bucket_encoder_frames
        if pad_len > 0:
            # paddings: [before_batch, after_batch,
            #            before_time,  after_time,
            #            before_vocab, after_vocab]
            logits = ops.pad(logits, [0, 0, 0, pad_len, 0, 0])
        graph.output(logits)

    return graph
