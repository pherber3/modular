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
"""TDT greedy decoding with the decoder step running on GPU.

The decode loop runs in Python (matching MAX's autoregressive serving
pattern), calling the compiled decoder step graph per iteration. LSTM
states stay on GPU between steps — only the argmax token and duration
(two int64 scalars, 8 bytes) cross PCIe each step.
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer, Device
from max.engine import Model


def tdt_greedy_decode_gpu(
    enc_projected_all: Buffer,
    decoder_model: Model,
    device: Device,
    durations: list[int],
    vocab_size: int,
    blank_id: int,
    pred_hidden: int,
    max_symbols_per_step: int = 10,
) -> list[list[int]]:
    """TDT greedy decode with GPU-accelerated decoder steps.

    Args:
        enc_projected_all: Pre-projected encoder output on GPU,
            shape ``(batch, T, joint_hidden)``.
        decoder_model: Compiled decoder step graph.
        device: GPU device for buffer allocation.
        durations: Duration values, e.g. ``[0, 1, 2, 3, 4]``.
        vocab_size: Number of vocabulary tokens (excluding blank).
        blank_id: Blank token ID (typically ``vocab_size``).
        pred_hidden: LSTM hidden dimension (for state initialization).
        max_symbols_per_step: Safety limit on tokens emitted per timestep.

    Returns:
        List of token ID sequences, one per batch element.
    """
    batch_size = enc_projected_all.shape[0]
    T = enc_projected_all.shape[1]
    results: list[list[int]] = []

    zeros = np.zeros((1, pred_hidden), dtype=np.float32)
    for _b in range(batch_size):
        tokens: list[int] = []
        t = 0

        # Initialize LSTM states on GPU
        h0 = Buffer.from_numpy(zeros.copy()).to(device)
        c0 = Buffer.from_numpy(zeros.copy()).to(device)
        h1 = Buffer.from_numpy(zeros.copy()).to(device)
        c1 = Buffer.from_numpy(zeros.copy()).to(device)
        label = Buffer.from_numpy(np.array([blank_id], dtype=np.int64)).to(
            device
        )

        while t < T:
            time_idx = Buffer.from_numpy(np.array([t], dtype=np.int64)).to(
                device
            )
            symbols_at_t = 0

            while symbols_at_t < max_symbols_per_step:
                outputs = decoder_model.execute(
                    label, h0, c0, h1, c1, enc_projected_all, time_idx
                )
                token_buf, dur_buf = outputs[0], outputs[1]
                h0, c0, h1, c1 = outputs[2], outputs[3], outputs[4], outputs[5]

                # 8 bytes GPU→CPU: two int64 scalars
                token = int(np.from_dlpack(token_buf))
                dur_idx = int(np.from_dlpack(dur_buf))
                duration = durations[dur_idx]

                if token == blank_id:
                    t += max(duration, 1)
                    break

                tokens.append(token)
                label = Buffer.from_numpy(np.array([token], dtype=np.int64)).to(
                    device
                )
                symbols_at_t += 1

                if duration > 0:
                    t += duration
                    break
            else:
                t += 1

        results.append(tokens)

    return results
