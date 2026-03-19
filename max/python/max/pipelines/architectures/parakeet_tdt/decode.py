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
"""TDT (Token-and-Duration Transducer) greedy decoding.

Implements the decode loop that iterates over encoder timesteps, running
the LSTM prediction network and joint network at each step to predict
both a token and a duration (number of encoder frames to skip).

Reference: NeMo ``GreedyTDTInfer._greedy_decode``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .decoder import JointNetwork, PredictionNetwork

NDFloat = npt.NDArray[np.floating]


def tdt_greedy_decode(
    encoder_output: NDFloat,
    prediction_net: PredictionNetwork,
    joint_net: JointNetwork,
    durations: list[int],
    vocab_size: int,
    blank_id: int,
    max_symbols_per_step: int = 10,
) -> list[list[int]]:
    """TDT greedy decode over a batch of encoder outputs.

    For each encoder timestep, the loop:
    1. Runs the prediction network on the previous token.
    2. Computes joint logits from encoder + predictor outputs.
    3. Splits logits into token logits and duration logits.
    4. Emits the token if non-blank, advances by the predicted duration.

    Args:
        encoder_output: Shape ``(batch, T, encoder_hidden)``.
        prediction_net: LSTM prediction network.
        joint_net: Joint network producing combined logits.
        durations: Duration values, e.g. ``[0, 1, 2, 3, 4]``.
        vocab_size: Number of vocabulary tokens (excluding blank).
        blank_id: Blank token ID (typically ``vocab_size``).
        max_symbols_per_step: Safety limit on tokens emitted per timestep.

    Returns:
        List of token ID sequences, one per batch element.
    """
    batch_size, T, _ = encoder_output.shape
    # Number of token classes = vocab + blank
    num_token_classes = vocab_size + 1
    results: list[list[int]] = []

    for b in range(batch_size):
        tokens: list[int] = []
        t = 0
        h_states, c_states = prediction_net.init_states()
        prev_token = blank_id

        while t < T:
            symbols_emitted = 0

            while symbols_emitted < max_symbols_per_step:
                pred_out, h_states, c_states = prediction_net(
                    prev_token, h_states, c_states
                )

                logits = joint_net(encoder_output[b, t], pred_out)

                token_logits = logits[:num_token_classes]
                dur_logits = logits[num_token_classes:]

                token = int(np.argmax(token_logits))
                dur_idx = int(np.argmax(dur_logits))
                duration = durations[dur_idx]

                if token != blank_id:
                    tokens.append(token)
                    prev_token = token
                    symbols_emitted += 1

                if token == blank_id or duration > 0:
                    t += max(duration, 1)
                    break
                # duration == 0 and non-blank: continue emitting at same t

            else:
                # Safety: hit max_symbols_per_step, force advance
                t += 1

        results.append(tokens)

    return results
