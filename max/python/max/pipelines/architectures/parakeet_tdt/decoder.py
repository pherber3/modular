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
"""TDT prediction network (LSTM) and joint network for numpy-based decoding.

These run outside the MAX graph — the encoder is compiled and executes in-graph,
then the LSTM + joint + decode loop runs in Python/numpy. The LSTM is tiny
(640-dim, 2 layers) so this adds negligible latency.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NDFloat = npt.NDArray[np.floating]


def _sigmoid(x: NDFloat) -> NDFloat:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class LSTMCell:
    """Single LSTM cell operating on numpy arrays.

    Implements the standard LSTM equations:
        gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
        i, f, g, o = split(gates, 4)
        c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
        h_new = sigmoid(o) * tanh(c_new)
    """

    def __init__(
        self,
        weight_ih: NDFloat,
        weight_hh: NDFloat,
        bias_ih: NDFloat,
        bias_hh: NDFloat,
    ) -> None:
        self.weight_ih = weight_ih  # (4*hidden, input)
        self.weight_hh = weight_hh  # (4*hidden, hidden)
        self.bias_ih = bias_ih  # (4*hidden,)
        self.bias_hh = bias_hh  # (4*hidden,)
        self.hidden_size = weight_hh.shape[1]

    def __call__(
        self, x: NDFloat, h: NDFloat, c: NDFloat
    ) -> tuple[NDFloat, NDFloat]:
        gates = (
            x @ self.weight_ih.T
            + self.bias_ih
            + h @ self.weight_hh.T
            + self.bias_hh
        )
        i, f, g, o = np.split(gates, 4, axis=-1)
        c_new = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h_new = _sigmoid(o) * np.tanh(c_new)
        return h_new, c_new


class PredictionNetwork:
    """LSTM-based prediction network for TDT.

    Takes previous token ID, returns predictor output and updated states.
    Architecture: Embedding → stacked LSTM layers.
    """

    def __init__(
        self,
        embedding_weight: NDFloat,
        lstm_cells: list[LSTMCell],
    ) -> None:
        self.embedding_weight = embedding_weight  # (vocab+1, pred_hidden)
        self.lstm_cells = lstm_cells
        self.pred_hidden = lstm_cells[0].hidden_size
        self.num_layers = len(lstm_cells)

    @classmethod
    def from_npz(cls, weights: dict[str, NDFloat]) -> PredictionNetwork:
        """Construct from decoder_joint.npz weight dict."""
        embedding = weights["decoder.prediction.embed.weight"]

        layer_indices = set()
        for key in weights:
            if "lstm.weight_ih_l" in key:
                idx = int(key.split("weight_ih_l")[1])
                layer_indices.add(idx)

        cells = []
        for i in sorted(layer_indices):
            prefix = "decoder.prediction.dec_rnn.lstm"
            cells.append(
                LSTMCell(
                    weight_ih=weights[f"{prefix}.weight_ih_l{i}"],
                    weight_hh=weights[f"{prefix}.weight_hh_l{i}"],
                    bias_ih=weights[f"{prefix}.bias_ih_l{i}"],
                    bias_hh=weights[f"{prefix}.bias_hh_l{i}"],
                )
            )

        return cls(embedding_weight=embedding, lstm_cells=cells)

    def init_states(self) -> tuple[list[NDFloat], list[NDFloat]]:
        """Initialize hidden and cell states to zeros."""
        h = [
            np.zeros(self.pred_hidden, dtype=np.float32)
            for _ in range(self.num_layers)
        ]
        c = [
            np.zeros(self.pred_hidden, dtype=np.float32)
            for _ in range(self.num_layers)
        ]
        return h, c

    def __call__(
        self,
        token_id: int,
        h_states: list[NDFloat],
        c_states: list[NDFloat],
    ) -> tuple[NDFloat, list[NDFloat], list[NDFloat]]:
        """Run one prediction step.

        Args:
            token_id: Previous token (or blank_id for start-of-sequence).
            h_states: Hidden states for each LSTM layer.
            c_states: Cell states for each LSTM layer.

        Returns:
            (predictor_output, updated_h_states, updated_c_states)
        """
        x = self.embedding_weight[token_id]  # (pred_hidden,)

        new_h = []
        new_c = []
        for i, cell in enumerate(self.lstm_cells):
            h_new, c_new = cell(x, h_states[i], c_states[i])
            new_h.append(h_new)
            new_c.append(c_new)
            x = h_new

        return x, new_h, new_c


class JointNetwork:
    """Joint network combining encoder and predictor outputs.

    Architecture: Linear(enc) + Linear(pred) → ReLU → Linear(output).
    Output size = vocab_size + 1 (blank) + num_durations.
    """

    def __init__(
        self,
        enc_weight: NDFloat,
        enc_bias: NDFloat,
        pred_weight: NDFloat,
        pred_bias: NDFloat,
        out_weight: NDFloat,
        out_bias: NDFloat,
    ) -> None:
        self.enc_weight = enc_weight  # (joint_hidden, encoder_hidden)
        self.enc_bias = enc_bias  # (joint_hidden,)
        self.pred_weight = pred_weight  # (joint_hidden, pred_hidden)
        self.pred_bias = pred_bias  # (joint_hidden,)
        self.out_weight = out_weight  # (vocab+1+durations, joint_hidden)
        self.out_bias = out_bias  # (vocab+1+durations,)

    @classmethod
    def from_npz(cls, weights: dict[str, NDFloat]) -> JointNetwork:
        """Construct from decoder_joint.npz weight dict."""
        return cls(
            enc_weight=weights["joint.enc.weight"],
            enc_bias=weights["joint.enc.bias"],
            pred_weight=weights["joint.pred.weight"],
            pred_bias=weights["joint.pred.bias"],
            out_weight=weights["joint.joint_net.2.weight"],
            out_bias=weights["joint.joint_net.2.bias"],
        )

    def __call__(self, encoder_out: NDFloat, predictor_out: NDFloat) -> NDFloat:
        """Compute joint logits.

        Args:
            encoder_out: Encoder output at a single timestep, shape ``(hidden,)``.
            predictor_out: Predictor output, shape ``(pred_hidden,)``.

        Returns:
            Logits of shape ``(vocab_size + 1 + num_durations,)``.
        """
        enc = encoder_out @ self.enc_weight.T + self.enc_bias
        pred = predictor_out @ self.pred_weight.T + self.pred_bias
        combined = np.maximum(0, enc + pred)
        return combined @ self.out_weight.T + self.out_bias
