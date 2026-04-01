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

Maps safetensors weight names to the names expected by:
  - The encoder graph (``ParakeetEncoder`` + ``enc_proj`` joint projection)
  - The decoder step graph (``DecoderStepGraph``)

Encoder weights have an ``encoder.`` prefix stripped. Decoder/joint weights
are remapped to match the graph module attribute names.
"""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData, Weights

# Decoder/joint weight name mappings: safetensors key → graph module key.
# The decoder step graph uses these attribute paths.
_DECODER_JOINT_MAPPINGS: dict[str, str] = {
    "decoder.prediction.embed.weight": "embed.weight",
    "decoder.prediction.dec_rnn.lstm.weight_ih_l0": "lstm_cell_0.weight_ih",
    "decoder.prediction.dec_rnn.lstm.weight_hh_l0": "lstm_cell_0.weight_hh",
    "decoder.prediction.dec_rnn.lstm.bias_ih_l0": "lstm_cell_0.bias_ih",
    "decoder.prediction.dec_rnn.lstm.bias_hh_l0": "lstm_cell_0.bias_hh",
    "decoder.prediction.dec_rnn.lstm.weight_ih_l1": "lstm_cell_1.weight_ih",
    "decoder.prediction.dec_rnn.lstm.weight_hh_l1": "lstm_cell_1.weight_hh",
    "decoder.prediction.dec_rnn.lstm.bias_ih_l1": "lstm_cell_1.bias_ih",
    "decoder.prediction.dec_rnn.lstm.bias_hh_l1": "lstm_cell_1.bias_hh",
    "joint.pred.weight": "pred_proj.weight",
    "joint.pred.bias": "pred_proj.bias",
    "joint.joint_net.2.weight": "joint_out.weight",
    "joint.joint_net.2.bias": "joint_out.bias",
}

# Encoder projection (joint.enc) goes into the encoder graph as enc_proj.
_ENC_PROJ_MAPPINGS: dict[str, str] = {
    "joint.enc.weight": "enc_proj.weight",
    "joint.enc.bias": "enc_proj.bias",
}


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensors state dict to MAX format.

    Handles three categories of weights:
      1. Encoder weights: strip ``encoder.`` prefix
      2. Encoder projection (``joint.enc``): map to ``enc_proj.*``
      3. Decoder/joint weights: map to decoder step graph attribute names
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if "num_batches_tracked" in weight_name:
            continue

        if weight_name in _ENC_PROJ_MAPPINGS:
            max_name = _ENC_PROJ_MAPPINGS[weight_name]
        elif weight_name in _DECODER_JOINT_MAPPINGS:
            max_name = _DECODER_JOINT_MAPPINGS[weight_name]
        elif weight_name.startswith("encoder."):
            max_name = weight_name.removeprefix("encoder.")
        else:
            continue

        new_state_dict[max_name] = value.data()

    return new_state_dict
