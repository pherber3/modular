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
"""TDT prediction network (LSTM) and joint network — PyTorch GPU implementation.

Drop-in replacement for decoder.py (numpy). All weights are stored as CUDA
tensors and all computation runs on GPU, eliminating the GPU→CPU transfer
that was previously required after the encoder graph.

Follows NeMo's GreedyBatchedTDTLabelLoopingComputer optimization pattern:
  - Pre-project encoder output once before the decode loop
  - Pre-project decoder output after each prediction step
  - Joint network only does addition + ReLU + output linear per step

Reference: NeMo ``nemo.collections.asr.parts.submodules.transducer_decoding.tdt_label_looping``
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

NDFloat = npt.NDArray[np.floating]


class LSTMCell:
    """Single LSTM cell operating on CUDA tensors.

    Implements the standard LSTM equations with fused gate computation:
        gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
        i, f, g, o = split(gates, 4)
        c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
        h_new = sigmoid(o) * tanh(c_new)
    """

    def __init__(
        self,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> None:
        self.weight_ih = weight_ih  # (4*hidden, input)
        self.weight_hh = weight_hh  # (4*hidden, hidden)
        # Pre-add biases since they're always summed together.
        self.bias = bias_ih + bias_hh  # (4*hidden,)
        self.hidden_size = weight_hh.shape[1]

    def __call__(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = x @ self.weight_ih.T + h @ self.weight_hh.T + self.bias
        i, f, g, o = gates.chunk(4, dim=-1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new


class PredictionNetwork:
    """LSTM-based prediction network for TDT, running on GPU.

    Takes previous token ID, returns predictor output and updated states.
    Architecture: Embedding → stacked LSTM layers.
    """

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        lstm_cells: list[LSTMCell],
    ) -> None:
        self.embedding_weight = embedding_weight  # (vocab+1, pred_hidden)
        self.lstm_cells = lstm_cells
        self.pred_hidden = lstm_cells[0].hidden_size
        self.num_layers = len(lstm_cells)

    @classmethod
    def from_npz(
        cls, weights: dict[str, NDFloat], device: str = "cuda"
    ) -> PredictionNetwork:
        """Construct from decoder_joint.npz weight dict, placing on GPU."""

        def _t(arr: NDFloat) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(device)

        embedding = _t(weights["decoder.prediction.embed.weight"])

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
                    weight_ih=_t(weights[f"{prefix}.weight_ih_l{i}"]),
                    weight_hh=_t(weights[f"{prefix}.weight_hh_l{i}"]),
                    bias_ih=_t(weights[f"{prefix}.bias_ih_l{i}"]),
                    bias_hh=_t(weights[f"{prefix}.bias_hh_l{i}"]),
                )
            )

        return cls(embedding_weight=embedding, lstm_cells=cells)

    @property
    def device(self) -> torch.device:
        return self.embedding_weight.device

    def init_states(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Initialize hidden and cell states to zeros on the same device."""
        h = [
            torch.zeros(self.pred_hidden, device=self.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(self.pred_hidden, device=self.device)
            for _ in range(self.num_layers)
        ]
        return h, c

    def __call__(
        self,
        token_id: int,
        h_states: list[torch.Tensor],
        c_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
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
    """Joint network combining encoder and predictor outputs, on GPU.

    Architecture: Linear(enc) + Linear(pred) → ReLU → Linear(output).
    Output size = vocab_size + 1 (blank) + num_durations.

    Supports pre-projection: call ``project_encoder`` once before the
    decode loop to avoid redundant matmuls at every timestep.
    """

    def __init__(
        self,
        enc_weight: torch.Tensor,
        enc_bias: torch.Tensor,
        pred_weight: torch.Tensor,
        pred_bias: torch.Tensor,
        out_weight: torch.Tensor,
        out_bias: torch.Tensor,
    ) -> None:
        self.enc_weight = enc_weight  # (joint_hidden, encoder_hidden)
        self.enc_bias = enc_bias  # (joint_hidden,)
        self.pred_weight = pred_weight  # (joint_hidden, pred_hidden)
        self.pred_bias = pred_bias  # (joint_hidden,)
        self.out_weight = out_weight  # (vocab+1+durations, joint_hidden)
        self.out_bias = out_bias  # (vocab+1+durations,)

    @classmethod
    def from_npz(
        cls, weights: dict[str, NDFloat], device: str = "cuda"
    ) -> JointNetwork:
        """Construct from decoder_joint.npz weight dict, placing on GPU."""

        def _t(arr: NDFloat) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(device)

        return cls(
            enc_weight=_t(weights["joint.enc.weight"]),
            enc_bias=_t(weights["joint.enc.bias"]),
            pred_weight=_t(weights["joint.pred.weight"]),
            pred_bias=_t(weights["joint.pred.bias"]),
            out_weight=_t(weights["joint.joint_net.2.weight"]),
            out_bias=_t(weights["joint.joint_net.2.bias"]),
        )

    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Pre-project encoder output to joint hidden dimension.

        Call this once before the decode loop to avoid redundant (1024→640)
        matmul at every timestep. Shape: ``(..., encoder_hidden)`` → ``(..., joint_hidden)``.
        """
        return encoder_output @ self.enc_weight.T + self.enc_bias

    def project_prednet(self, pred_output: torch.Tensor) -> torch.Tensor:
        """Pre-project prediction network output to joint hidden dimension.

        Shape: ``(..., pred_hidden)`` → ``(..., joint_hidden)``.
        """
        return pred_output @ self.pred_weight.T + self.pred_bias

    def joint_after_projection(
        self, enc_projected: torch.Tensor, pred_projected: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint logits from pre-projected inputs.

        This is the fast path used inside the decode loop — only ReLU +
        one linear layer, no redundant encoder/predictor projections.

        Args:
            enc_projected: Pre-projected encoder output at a single timestep.
            pred_projected: Pre-projected predictor output.

        Returns:
            Logits of shape ``(vocab_size + 1 + num_durations,)``.
        """
        combined = torch.relu(enc_projected + pred_projected)
        return combined @ self.out_weight.T + self.out_bias

    def __call__(
        self, encoder_out: torch.Tensor, predictor_out: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint logits (non-pre-projected path).

        Args:
            encoder_out: Encoder output at a single timestep, shape ``(hidden,)``.
            predictor_out: Predictor output, shape ``(pred_hidden,)``.

        Returns:
            Logits of shape ``(vocab_size + 1 + num_durations,)``.
        """
        enc = encoder_out @ self.enc_weight.T + self.enc_bias
        pred = predictor_out @ self.pred_weight.T + self.pred_bias
        combined = torch.relu(enc + pred)
        return combined @ self.out_weight.T + self.out_bias
