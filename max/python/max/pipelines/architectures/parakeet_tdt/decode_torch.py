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
"""TDT greedy decoding on GPU using PyTorch tensors.

Drop-in replacement for decode.py (numpy). All computation runs on GPU,
eliminating the GPU→CPU transfer previously required after the encoder.

Implements NeMo's torch_impl optimization pattern:
  - Pre-project encoder output once before the loop
  - Pre-project decoder output after each prediction step
  - Joint network only does add + ReLU + linear per step
  - All argmax/comparison ops run on GPU (no device sync)

For batch_size=1 (the common case in MAX), this uses a simple sequential
loop. The architecture supports future batched decoding using torch.where
masks (see NeMo's GreedyBatchedTDTLabelLoopingComputer).

Reference: NeMo ``tdt_label_looping.py::torch_impl()``
"""

from __future__ import annotations

import torch

from .decoder_torch import JointNetwork, PredictionNetwork


def tdt_greedy_decode(
    encoder_output: torch.Tensor,
    prediction_net: PredictionNetwork,
    joint_net: JointNetwork,
    durations: list[int],
    vocab_size: int,
    blank_id: int,
    max_symbols_per_step: int = 10,
) -> list[list[int]]:
    """TDT greedy decode over a batch of encoder outputs on GPU.

    Pre-projects encoder output once through the joint encoder projection,
    then runs the decode loop with pre-projected decoder outputs. This
    avoids redundant (1024→640) and (640→640) matmuls at every timestep.

    For each encoder timestep, the loop:
    1. Computes joint logits from pre-projected encoder + predictor outputs.
    2. Splits logits into token logits and duration logits.
    3. Emits the token if non-blank, updates prediction network.
    4. Advances time by the predicted duration.

    Args:
        encoder_output: Shape ``(batch, T, encoder_hidden)`` on GPU.
        prediction_net: PyTorch LSTM prediction network (on GPU).
        joint_net: PyTorch joint network (on GPU).
        durations: Duration values, e.g. ``[0, 1, 2, 3, 4]``.
        vocab_size: Number of vocabulary tokens (excluding blank).
        blank_id: Blank token ID (typically ``vocab_size``).
        max_symbols_per_step: Safety limit on tokens emitted per timestep.

    Returns:
        List of token ID sequences, one per batch element.
    """
    batch_size, T, _ = encoder_output.shape
    num_token_classes = vocab_size + 1
    durations_t = torch.tensor(durations, device=encoder_output.device)

    # Pre-project encoder output once: (batch, T, encoder_hidden) → (batch, T, joint_hidden)
    # This saves a (1024, 640) matmul at every decode step.
    encoder_projected = joint_net.project_encoder(encoder_output)

    results: list[list[int]] = []

    for b in range(batch_size):
        tokens: list[int] = []
        t = 0
        h_states, c_states = prediction_net.init_states()
        prev_token = blank_id

        # Get initial predictor output and pre-project it.
        pred_out, h_states, c_states = prediction_net(
            prev_token, h_states, c_states
        )
        pred_projected = joint_net.project_prednet(pred_out)

        while t < T:
            symbols_at_t = 0

            while symbols_at_t < max_symbols_per_step:
                # Joint uses pre-projected inputs: just add + ReLU + linear.
                logits = joint_net.joint_after_projection(
                    encoder_projected[b, t], pred_projected
                )

                token_logits = logits[:num_token_classes]
                dur_logits = logits[num_token_classes:]

                token = token_logits.argmax().item()
                dur_idx = dur_logits.argmax().item()
                duration = durations_t[dur_idx].item()

                if token == blank_id:
                    t += max(duration, 1)
                    break

                # Non-blank: emit token, update prediction network.
                tokens.append(token)
                prev_token = token
                pred_out, h_states, c_states = prediction_net(
                    prev_token, h_states, c_states
                )
                # Pre-project updated predictor output.
                pred_projected = joint_net.project_prednet(pred_out)
                symbols_at_t += 1

                if duration > 0:
                    t += duration
                    break
                # duration == 0: stay at same t (multiple emissions)
            else:
                # Safety: hit max_symbols_per_step, force advance.
                t += 1

        results.append(tokens)

    return results
