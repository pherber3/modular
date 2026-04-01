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
"""TDT decoder step as a compiled MAX graph module.

Implements the LSTM prediction network + joint network as graph-level
operations so the decoder step runs on GPU. The greedy decode loop
stays in Python (matching the established autoregressive pattern in MAX),
calling ``model.execute()`` per step with loop-carried LSTM state.

This mirrors the TensorRT reference where each decoder step is a compiled
GPU kernel, with the loop control in Python.
"""

from __future__ import annotations

from collections.abc import Mapping

from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph.weights import WeightData
from max.nn import Embedding, Linear
from max.nn.layer import Module

from .model_config import TDTModelConfig


class GraphLSTMCell(Module):
    """Single LSTM cell as a MAX graph module.

    Computes the standard LSTM equations using graph-level ops so the
    cell runs on GPU when the graph is compiled for a GPU device.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        gate_size = 4 * hidden_size
        self.weight_ih = Weight(
            "weight_ih", dtype, shape=(gate_size, input_size), device=device
        )
        self.weight_hh = Weight(
            "weight_hh", dtype, shape=(gate_size, hidden_size), device=device
        )
        self.bias_ih = Weight(
            "bias_ih", dtype, shape=(gate_size,), device=device
        )
        self.bias_hh = Weight(
            "bias_hh", dtype, shape=(gate_size,), device=device
        )
        self.hidden_size = hidden_size

    def __call__(
        self, x: TensorValue, h: TensorValue, c: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Run one LSTM step.

        Args:
            x: Input tensor, shape ``(1, input_size)``.
            h: Hidden state, shape ``(1, hidden_size)``.
            c: Cell state, shape ``(1, hidden_size)``.

        Returns:
            ``(h_new, c_new)`` each shape ``(1, hidden_size)``.
        """
        w_ih = TensorValue(self.weight_ih)
        w_hh = TensorValue(self.weight_hh)
        b_ih = TensorValue(self.bias_ih)
        b_hh = TensorValue(self.bias_hh)

        gates = (
            ops.matmul(x, ops.transpose(w_ih, -1, -2))
            + b_ih
            + ops.matmul(h, ops.transpose(w_hh, -1, -2))
            + b_hh
        )
        i, f, g, o = ops.split(gates, 4, axis=-1)
        c_new = ops.sigmoid(f) * c + ops.sigmoid(i) * ops.tanh(g)
        h_new = ops.sigmoid(o) * ops.tanh(c_new)
        return h_new, c_new


class DecoderStepGraph(Module):
    """TDT decoder step: embedding → LSTM → predictor projection → joint.

    One call to this module runs a single decode step on GPU. The greedy
    decode loop in Python calls this repeatedly, passing updated LSTM
    states between steps (same pattern as LLM KV cache passing in MAX).

    The encoder projection is pre-computed once in the encoder graph
    (not here), so ``enc_projected`` is already the projected encoder
    frame at time ``t``.
    """

    def __init__(self, config: TDTModelConfig) -> None:
        super().__init__()
        dtype = config.dtype
        device = config.device
        pred_hidden = config.pred_hidden
        joint_hidden = config.joint_hidden
        vocab_size = config.vocab_size
        num_durations = len(config.tdt_durations)

        self.embed = Embedding(
            vocab_size=vocab_size + 1,  # +1 for blank token
            hidden_dim=pred_hidden,
            dtype=dtype,
            device=device,
            name="embed",
        )

        self.lstm_cell_0 = GraphLSTMCell(
            pred_hidden, pred_hidden, dtype, device
        )
        self.lstm_cell_1 = GraphLSTMCell(
            pred_hidden, pred_hidden, dtype, device
        )

        self.pred_proj = Linear(
            in_dim=pred_hidden,
            out_dim=joint_hidden,
            dtype=dtype,
            device=device,
            name="pred_proj",
        )

        self.joint_out = Linear(
            in_dim=joint_hidden,
            out_dim=vocab_size + 1 + num_durations,
            dtype=dtype,
            device=device,
            name="joint_out",
        )

        self.num_token_classes = vocab_size + 1
        self.device = device

    def __call__(
        self,
        token_id: TensorValue,
        h0: TensorValue,
        c0: TensorValue,
        h1: TensorValue,
        c1: TensorValue,
        enc_projected_all: TensorValue,
        time_idx: TensorValue,
    ) -> tuple[
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
    ]:
        """Run one decoder step.

        Args:
            token_id: Previous token index, shape ``(1,)`` int64.
            h0, c0: LSTM layer 0 hidden/cell state, shape ``(1, pred_hidden)``.
            h1, c1: LSTM layer 1 hidden/cell state, shape ``(1, pred_hidden)``.
            enc_projected_all: Pre-projected encoder output, shape
                ``(1, T, joint_hidden)``.
            time_idx: Current encoder timestep, shape ``(1,)`` int64.

        Returns:
            ``(token_argmax, dur_argmax, h0_new, c0_new, h1_new, c1_new)``
            where argmax values are ``(1,)`` int64 scalars.
        """
        # Embedding lookup
        emb = self.embed(token_id)  # (1, pred_hidden)

        # Stacked LSTM
        h0_new, c0_new = self.lstm_cell_0(emb, h0, c0)
        h1_new, c1_new = self.lstm_cell_1(h0_new, h1, c1)

        # Predictor projection
        pred = self.pred_proj(h1_new)  # (1, joint_hidden)

        # Index encoder projection at current timestep
        enc_frame = ops.gather(
            enc_projected_all, time_idx, axis=1
        )  # (1, 1, joint_hidden)
        enc_frame = ops.squeeze(enc_frame, 1)  # (1, joint_hidden)

        # Joint: ReLU(enc + pred) → output linear
        combined = ops.relu(enc_frame + pred)
        logits = self.joint_out(combined)  # (1, vocab+1+num_durations)

        # Argmax on GPU — only 2 int64 scalars cross PCIe per step
        n = self.num_token_classes
        token_logits = ops.slice_tensor(logits, [slice(None), slice(0, n)])
        dur_logits = ops.slice_tensor(logits, [slice(None), slice(n, None)])
        token_argmax = ops.argmax(token_logits, axis=-1)  # (1,) int64
        dur_argmax = ops.argmax(dur_logits, axis=-1)  # (1,) int64

        return token_argmax, dur_argmax, h0_new, c0_new, h1_new, c1_new


def build_decoder_step_graph(
    config: TDTModelConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
) -> Graph:
    """Build the decoder step computation graph.

    Takes token ID, LSTM states, pre-projected encoder output, and time
    index as inputs. Returns argmax token/duration and updated LSTM states.
    """
    dtype = config.dtype
    device = config.device
    pred_hidden = config.pred_hidden
    joint_hidden = config.joint_hidden

    input_types = [
        TensorType(DType.int64, shape=[1], device=device),  # token_id
        TensorType(dtype, shape=[1, pred_hidden], device=device),  # h0
        TensorType(dtype, shape=[1, pred_hidden], device=device),  # c0
        TensorType(dtype, shape=[1, pred_hidden], device=device),  # h1
        TensorType(dtype, shape=[1, pred_hidden], device=device),  # c1
        TensorType(
            dtype, shape=[1, "T", joint_hidden], device=device
        ),  # enc_projected_all
        TensorType(DType.int64, shape=[1], device=device),  # time_idx
    ]

    with Graph("tdt_decoder_step", input_types=input_types) as graph:
        decoder_step = DecoderStepGraph(config)
        decoder_step.load_state_dict(state_dict)

        inputs = [inp.tensor for inp in graph.inputs]
        token_id, h0, c0, h1, c1, enc_proj_all, time_idx = inputs

        token_argmax, dur_argmax, h0_new, c0_new, h1_new, c1_new = decoder_step(
            token_id, h0, c0, h1, c1, enc_proj_all, time_idx
        )
        graph.output(token_argmax, dur_argmax, h0_new, c0_new, h1_new, c1_new)

    return graph
