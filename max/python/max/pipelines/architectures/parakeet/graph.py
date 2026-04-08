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
mel spectrogram input -> encoder -> on-device argmax -> int32 predicted
ids (zero-padded to the global max encoder-frame count across all
buckets).

The argmax over the vocab dimension happens inside the compiled graph,
mirroring what the TDT decoder step graph already does. This drops the
D2H transfer from ``(1, T, vocab_size)`` float32 (~1.6MB per sample) to
``(1, T)`` int32 (~1.6KB per sample). Host-side ``ctc_greedy_decode``
then only needs to dedup consecutive duplicates and strip blank tokens.
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
            (across all buckets). The output predicted_ids tensor is
            zero-padded along the time axis to this size so all bucket
            graphs return the same shape — downstream callers slice off
            the padded tail using the bucket's true ``encoder_frames``.

    Returns:
        Compiled graph accepting mel spectrogram input
        ``[1, num_frames, num_mel_bins]`` and producing
        ``[1, pad_to_encoder_frames]`` int32 predicted token ids
        (post on-device argmax over the vocab dimension).
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
        logits = model(input_features)  # (1, T, vocab_size), T = num_frames//8

        # On-device argmax — mirrors TDT pattern
        # (parakeet_tdt/decoder_graph.py:346-362). Drops D2H transfer
        # from ~1.6MB per sample (float32 logits) to ~1.6KB per sample
        # (int32 ids). Host-side ``ctc_greedy_decode`` then only handles
        # dedup + blank-strip + tokenizer decode.
        # ops.argmax is rank-preserving: (1, T, vocab) -> (1, T, 1).
        predicted_ids = ops.argmax(logits, axis=-1)  # (1, T, 1), int64
        predicted_ids = ops.squeeze(predicted_ids, -1)  # (1, T), int64
        predicted_ids = ops.cast(predicted_ids, DType.int32)  # (1, T), int32

        # Zero-pad the time axis to the global max so every bucket's CTC
        # graph returns the same output shape. Pad value is 0 by default;
        # since the host-side code slices to the bucket's true
        # ``encoder_frames`` before dedup, the padded tail is never read.
        bucket_encoder_frames = num_frames // 8
        pad_len = pad_to_encoder_frames - bucket_encoder_frames
        if pad_len > 0:
            # paddings: [before_batch, after_batch,
            #            before_time,  after_time]
            predicted_ids = ops.pad(predicted_ids, [0, 0, 0, pad_len])
        graph.output(predicted_ids)

    return graph
