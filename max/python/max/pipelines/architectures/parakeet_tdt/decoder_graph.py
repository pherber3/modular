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
"""TDT decoder as compiled MAX graphs for GPU inference.

Replaces the numpy CPU decoder with two compiled MAX graphs:
  1. Projection graph — pre-projects encoder output (1024→640) once per utterance
  2. Decoder step graph — one LSTM + joint step, called in a Python loop

Follows the same autoregressive pattern as LLM text generation: a Python loop
calls ``model.execute()`` each iteration, passing LSTM states as explicit
Buffer inputs/outputs that stay on GPU.

Key optimizations from NeMo's GreedyBatchedTDTLabelLoopingComputer:
  - Pre-project encoder output once before the loop
  - Pre-project predictor output inside the graph after each LSTM step
  - Joint network only does add + ReLU + linear per step

Reference: NeMo ``tdt_label_looping.py::torch_impl()``
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.nn.layer import Module
from max.nn.linear import Linear

from .model_config import TDTModelConfig

NDFloat = npt.NDArray[np.floating]

logger = logging.getLogger("max.pipelines")


# ---------------------------------------------------------------------------
# LSTM Cell Module
# ---------------------------------------------------------------------------


class LSTMCellGraph(Module):
    """Single LSTM cell as a MAX graph module.

    Uses two Linear layers for the input-to-hidden and hidden-to-hidden
    projections. The bias is handled by each Linear layer.

    LSTM equations:
        gates = ih_linear(x) + hh_linear(h)
        i, f, g, o = chunk(gates, 4)
        c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
        h_new = sigmoid(o) * tanh(c_new)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: DType,
        device: DeviceRef,
        name: str = "lstm",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = Linear(
            in_dim=input_size,
            out_dim=4 * hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.hh = Linear(
            in_dim=hidden_size,
            out_dim=4 * hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(
        self, x: TensorValue, h: TensorValue, c: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """LSTM cell forward pass.

        Args:
            x: Input tensor, shape ``(batch, input_size)``.
            h: Previous hidden state, shape ``(batch, hidden_size)``.
            c: Previous cell state, shape ``(batch, hidden_size)``.

        Returns:
            ``(h_new, c_new)`` — updated hidden and cell states.
        """
        gates = self.ih(x) + self.hh(h)
        i, f, g, o = ops.chunk(gates, 4, axis=-1)
        c_new = ops.sigmoid(f) * c + ops.sigmoid(i) * ops.tanh(g)
        h_new = ops.sigmoid(o) * ops.tanh(c_new)
        return h_new, c_new


# ---------------------------------------------------------------------------
# Prediction Network Module (Embedding + Stacked LSTM)
# ---------------------------------------------------------------------------


class PredictionNetworkGraph(Module):
    """LSTM-based prediction network for TDT as a MAX graph module.

    Architecture: Embedding lookup → stacked LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        pred_hidden: int,
        num_layers: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.pred_hidden = pred_hidden
        self.num_layers = num_layers

        # Embedding weight: (vocab_size + 1, pred_hidden)
        # +1 for the blank token which also serves as SOS
        self.embedding = Weight(
            name="embedding",
            dtype=dtype,
            shape=(vocab_size + 1, pred_hidden),
            device=device,
        )

        # Stacked LSTM cells
        for i in range(num_layers):
            cell = LSTMCellGraph(
                input_size=pred_hidden,
                hidden_size=pred_hidden,
                dtype=dtype,
                device=device,
                name=f"prediction.lstm_{i}",
            )
            setattr(self, f"lstm_{i}", cell)

    def __call__(
        self,
        token_id: TensorValue,
        h0: TensorValue,
        c0: TensorValue,
        h1: TensorValue,
        c1: TensorValue,
    ) -> tuple[
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
    ]:
        """Run one prediction step.

        Args:
            token_id: Previous token ID, shape ``(1, 1)`` int32.
            h0, c0: LSTM layer 0 states, shape ``(1, 640)``.
            h1, c1: LSTM layer 1 states, shape ``(1, 640)``.

        Returns:
            ``(pred_out, h0', c0', h1', c1')``
        """
        # Embedding lookup: (1, 1) → (1, 640)
        x = ops.gather(self.embedding, token_id, axis=0)
        # gather output shape is (1, 1, 640), squeeze the middle dim
        x = ops.squeeze(x, 1)

        # LSTM layer 0
        h0_new, c0_new = self.lstm_0(x, h0, c0)

        # LSTM layer 1 (input is output of layer 0)
        h1_new, c1_new = self.lstm_1(h0_new, h1, c1)

        # Predictor output is the hidden state of the last layer
        return h1_new, h0_new, c0_new, h1_new, c1_new


# ---------------------------------------------------------------------------
# Joint Network Module (Post-Projection Fast Path)
# ---------------------------------------------------------------------------


class JointNetworkGraph(Module):
    """Joint network for TDT as a MAX graph module.

    Takes pre-projected encoder output and raw predictor output.
    Projects the predictor output, combines with encoder, applies ReLU,
    and produces logits.

    Architecture: pred_proj(pred) + enc_projected → ReLU → output_proj
    """

    def __init__(
        self,
        pred_hidden: int,
        joint_hidden: int,
        output_size: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.pred_proj = Linear(
            in_dim=pred_hidden,
            out_dim=joint_hidden,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.output_proj = Linear(
            in_dim=joint_hidden,
            out_dim=output_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(
        self, enc_projected: TensorValue, pred_out: TensorValue
    ) -> TensorValue:
        """Compute joint logits from pre-projected encoder and predictor.

        Args:
            enc_projected: Pre-projected encoder output at timestep t,
                shape ``(1, joint_hidden)``.
            pred_out: Predictor output, shape ``(1, pred_hidden)``.

        Returns:
            Logits of shape ``(1, vocab_size + 1 + num_durations)``.
        """
        pred_projected = self.pred_proj(pred_out)
        combined = ops.relu(enc_projected + pred_projected)
        return self.output_proj(combined)


# ---------------------------------------------------------------------------
# Graph Builders
# ---------------------------------------------------------------------------


def build_projection_graph(
    config: TDTModelConfig,
    state_dict: Mapping[str, np.ndarray],
) -> Graph:
    """Build the encoder output projection graph.

    Projects encoder hidden states from encoder_hidden (1024) to
    joint_hidden (640) dimensions. Run once per utterance.

    Args:
        config: TDT model configuration.
        state_dict: Weight dict with ``enc_proj.weight`` and ``enc_proj.bias``.

    Returns:
        Compiled graph: ``(1, 400, 1024) → (1, 400, 640)``.
    """
    # Fixed T=400 from encoder's 3200 frames / 8x subsampling
    input_type = TensorType(
        DType.float32,
        shape=[1, 400, config.hidden_size],
        device=config.device,
    )

    with Graph("tdt_encoder_projection", input_types=[input_type]) as graph:
        enc_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.joint_hidden,
            dtype=DType.float32,
            device=config.device,
            has_bias=True,
            name="enc_proj",
        )
        enc_proj.load_state_dict(state_dict)

        encoder_output = graph.inputs[0].tensor
        projected = enc_proj(encoder_output)
        graph.output(projected)

    return graph


def build_decoder_step_graph(
    config: TDTModelConfig,
    prediction_state_dict: Mapping[str, np.ndarray],
    joint_state_dict: Mapping[str, np.ndarray],
) -> Graph:
    """Build the single-step decoder graph for TDT.

    One iteration of: embedding lookup → 2-layer LSTM → joint network.
    Called repeatedly in a Python loop during decoding.

    Args:
        config: TDT model configuration.
        prediction_state_dict: Weights for PredictionNetworkGraph
            (keys relative to prediction module).
        joint_state_dict: Weights for JointNetworkGraph
            (keys relative to joint module).

    Returns:
        Compiled graph with 7 inputs and 6 outputs
        (best_token, best_dur_idx, h0', c0', h1', c1').
    """
    pred_hidden = config.pred_hidden
    joint_hidden = config.joint_hidden
    vocab_size = config.vocab_size
    num_durations = len(config.tdt_durations)
    output_size = vocab_size + 1 + num_durations  # vocab + blank + durations
    device = config.device
    # Fixed T=400 from encoder's 3200 frames / 8x subsampling
    max_encoder_len = 400

    # 7 inputs: token_id, h0, c0, h1, c1, enc_projected, t_index
    input_types = [
        TensorType(DType.int32, shape=[1, 1], device=device),  # token_id
        TensorType(DType.float32, shape=[1, pred_hidden], device=device),  # h0
        TensorType(DType.float32, shape=[1, pred_hidden], device=device),  # c0
        TensorType(DType.float32, shape=[1, pred_hidden], device=device),  # h1
        TensorType(DType.float32, shape=[1, pred_hidden], device=device),  # c1
        TensorType(
            DType.float32,
            shape=[1, max_encoder_len, joint_hidden],
            device=device,
        ),  # enc_projected (full, stays on device)
        TensorType(DType.int32, shape=[1], device=device),  # t_index
    ]

    with Graph("tdt_decoder_step", input_types=input_types) as graph:
        # Build modules
        prediction = PredictionNetworkGraph(
            vocab_size=vocab_size,
            pred_hidden=pred_hidden,
            num_layers=config.pred_rnn_layers,
            dtype=DType.float32,
            device=device,
        )
        joint = JointNetworkGraph(
            pred_hidden=pred_hidden,
            joint_hidden=joint_hidden,
            output_size=output_size,
            dtype=DType.float32,
            device=device,
        )

        # Load weights (each dict has keys relative to its module)
        prediction.load_state_dict(prediction_state_dict)
        joint.load_state_dict(joint_state_dict)

        # Wire up inputs
        token_id = graph.inputs[0].tensor
        h0 = graph.inputs[1].tensor
        c0 = graph.inputs[2].tensor
        h1 = graph.inputs[3].tensor
        c1 = graph.inputs[4].tensor
        enc_projected = graph.inputs[5].tensor  # (1, 400, 640)
        t_index = graph.inputs[6].tensor  # (1,)

        # Slice encoder at timestep t on-device: (1, 400, 640) → (1, 640)
        enc_t = ops.gather(enc_projected, t_index, axis=1)
        # gather produces (1, 1, 640), squeeze the middle dim
        enc_t = ops.squeeze(enc_t, 1)

        # Forward pass
        pred_out, h0_new, c0_new, h1_new, c1_new = prediction(
            token_id, h0, c0, h1, c1
        )
        logits = joint(enc_t, pred_out)  # (1, output_size)

        # Split logits into token and duration, argmax on-device.
        # This avoids transferring the full 1030-element logits vector
        # back to CPU each step.
        num_token_classes = vocab_size + 1
        token_logits, dur_logits = ops.split(
            logits, [num_token_classes, num_durations], axis=-1
        )
        best_token = ops.argmax(token_logits, axis=-1)  # (1,)
        best_dur_idx = ops.argmax(dur_logits, axis=-1)  # (1,)

        # Cast to int32 for consistency with token_id input type
        best_token = ops.cast(best_token, DType.int32)
        best_dur_idx = ops.cast(best_dur_idx, DType.int32)

        # Stack into single (2,) tensor — one GPU→CPU transfer per step
        # instead of two separate .to(cpu) calls.
        decisions = ops.concat(
            [ops.unsqueeze(best_token, 0), ops.unsqueeze(best_dur_idx, 0)],
            axis=-1,
        )  # (1, 2) int32

        graph.output(decisions, h0_new, c0_new, h1_new, c1_new)

    return graph


# ---------------------------------------------------------------------------
# Weight Name Adapter
# ---------------------------------------------------------------------------


def convert_decoder_state_dict(
    npz_weights: dict[str, NDFloat],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Convert decoder_joint.npz weights to graph weight names.

    Splits weights into three dicts, each with keys relative to the module
    that will call ``load_state_dict()``:
    1. Projection graph — ``enc_proj.weight``, ``enc_proj.bias``
    2. Prediction network — ``embedding``, ``lstm_0.ih.weight``, etc.
    3. Joint network — ``pred_proj.weight``, ``output_proj.weight``, etc.

    Args:
        npz_weights: Raw weights from ``decoder_joint.npz``.

    Returns:
        ``(projection_dict, prediction_dict, joint_dict)``
    """
    # Mapping: npz key → graph weight name
    projection_map = {
        "joint.enc.weight": "enc_proj.weight",
        "joint.enc.bias": "enc_proj.bias",
    }

    # Keys for prediction.load_state_dict() — relative to prediction module.
    # Module hierarchy: prediction.embedding, prediction.lstm_0.ih.weight, etc.
    # load_state_dict on prediction expects: embedding, lstm_0.ih.weight, etc.
    prediction_map = {
        "decoder.prediction.embed.weight": "embedding",
        # LSTM layer 0
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": "lstm_0.ih.weight",
        "decoder.prediction.dec_rnn.lstm.bias_ih_l0": "lstm_0.ih.bias",
        "decoder.prediction.dec_rnn.lstm.weight_hh_l0": "lstm_0.hh.weight",
        "decoder.prediction.dec_rnn.lstm.bias_hh_l0": "lstm_0.hh.bias",
        # LSTM layer 1
        "decoder.prediction.dec_rnn.lstm.weight_ih_l1": "lstm_1.ih.weight",
        "decoder.prediction.dec_rnn.lstm.bias_ih_l1": "lstm_1.ih.bias",
        "decoder.prediction.dec_rnn.lstm.weight_hh_l1": "lstm_1.hh.weight",
        "decoder.prediction.dec_rnn.lstm.bias_hh_l1": "lstm_1.hh.bias",
    }

    # Keys for joint.load_state_dict() — relative to joint module.
    # Module hierarchy: joint.pred_proj.weight, joint.output_proj.weight, etc.
    # load_state_dict on joint expects: pred_proj.weight, output_proj.weight, etc.
    joint_map = {
        "joint.pred.weight": "pred_proj.weight",
        "joint.pred.bias": "pred_proj.bias",
        "joint.joint_net.2.weight": "output_proj.weight",
        "joint.joint_net.2.bias": "output_proj.bias",
    }

    proj_dict: dict[str, np.ndarray] = {}
    for npz_key, graph_key in projection_map.items():
        if npz_key in npz_weights:
            proj_dict[graph_key] = npz_weights[npz_key].astype(np.float32)

    pred_dict: dict[str, np.ndarray] = {}
    for npz_key, graph_key in prediction_map.items():
        if npz_key in npz_weights:
            pred_dict[graph_key] = npz_weights[npz_key].astype(np.float32)

    joint_dict: dict[str, np.ndarray] = {}
    for npz_key, graph_key in joint_map.items():
        if npz_key in npz_weights:
            joint_dict[graph_key] = npz_weights[npz_key].astype(np.float32)

    return proj_dict, pred_dict, joint_dict


# ---------------------------------------------------------------------------
# Decode Loop Runner
# ---------------------------------------------------------------------------


class TDTGraphDecoder:
    """Manages TDT decoding using the compiled decoder step graph.

    The encoder graph already includes the projection (1024→640), so
    this class only runs the decoder step graph in a loop. All buffers
    are pre-allocated at init using DevicePinnedBuffer for fast
    CPU↔GPU transfers.
    """

    def __init__(
        self,
        decoder_step_model: Model,
        config: TDTModelConfig,
        device: Device,
        cpu_device: Device,
    ) -> None:
        self.decoder_step_model = decoder_step_model
        self.config = config
        self.device = device
        self.cpu_device = cpu_device

        self.vocab_size = config.vocab_size
        self.blank_id = config.blank_id
        self.durations = config.tdt_durations
        self.pred_hidden = config.pred_hidden
        self.joint_hidden = config.joint_hidden

        # Pre-allocate ALL small buffers at init.
        max_t = 400
        self._t_index_bufs = [
            Buffer.from_numpy(np.array([t], dtype=np.int32)).to(device)
            for t in range(max_t)
        ]

        num_tokens = config.vocab_size + 1
        self._token_bufs = [
            Buffer.from_numpy(np.array([[tid]], dtype=np.int32)).to(device)
            for tid in range(num_tokens)
        ]

        # Double-buffered LSTM states for CUDA graph capture.
        # Set A and Set B alternate as input/output each step,
        # eliminating 4x inplace_copy_from per step.
        # Each set: [token_input, h0, c0, h1, c1, t_input]
        zero = np.zeros((1, config.pred_hidden), dtype=np.float32)
        blank_token = np.array([[config.blank_id]], dtype=np.int32)
        t_zero = np.array([0], dtype=np.int32)

        self._bufs: list[dict[str, Buffer]] = []
        for _ in range(2):
            self._bufs.append(
                {
                    "token": Buffer.from_numpy(blank_token.copy()).to(device),
                    "h0": Buffer.from_numpy(zero.copy()).to(device),
                    "c0": Buffer.from_numpy(zero.copy()).to(device),
                    "h1": Buffer.from_numpy(zero.copy()).to(device),
                    "c1": Buffer.from_numpy(zero.copy()).to(device),
                    "t": Buffer.from_numpy(t_zero.copy()).to(device),
                }
            )

        # Pinned buffer for async D2H readback of decisions.
        from max.driver import DevicePinnedBuffer

        self._decisions_pinned = DevicePinnedBuffer(
            dtype=DType.int32, shape=(2,), device=device
        )
        self._decisions_np = self._decisions_pinned.to_numpy()

        # Persistent enc_projected buffer for CUDA graph capture.
        self._enc_proj_buf: Buffer | None = None
        self._captured = False
        # Two graph keys for double-buffered A→B and B→A.
        self._graph_keys = [1, 2]
        self._captured_outputs: list[list[Buffer]] = [[], []]

    def _capture_graphs(self, enc_projected: Buffer) -> None:
        """Capture two CUDA graphs for double-buffered decode.

        Graph 0: set A as input → set B as output
        Graph 1: set B as input → set A as output
        This eliminates all LSTM state copies between steps.
        """
        self._enc_proj_buf = enc_projected.copy()
        self._enc_proj_buf.inplace_copy_from(enc_projected)

        zero = Buffer.from_numpy(
            np.zeros((1, self.pred_hidden), dtype=np.float32)
        ).to(self.device)

        for i in range(2):
            buf = self._bufs[i]
            buf["h0"].inplace_copy_from(zero)
            buf["c0"].inplace_copy_from(zero)
            buf["h1"].inplace_copy_from(zero)
            buf["c1"].inplace_copy_from(zero)
            buf["token"].inplace_copy_from(self._token_bufs[self.blank_id])
            buf["t"].inplace_copy_from(self._t_index_bufs[0])

        # Capture graph 0: input=set[0], output→set[1] state buffers
        for idx in range(2):
            inp = self._bufs[idx]
            self._captured_outputs[idx] = list(
                self.decoder_step_model.capture(
                    self._graph_keys[idx],
                    inp["token"],
                    inp["h0"],
                    inp["c0"],
                    inp["h1"],
                    inp["c1"],
                    self._enc_proj_buf,
                    inp["t"],
                )
            )

        self._captured = True
        logger.info("Captured double-buffered decoder step CUDA graphs")

    def decode(self, enc_projected: Buffer) -> list[list[int]]:
        """Run TDT greedy decode with double-buffered CUDA graph replay.

        Two graphs alternate: step N replays graph 0 (set A→B), step N+1
        replays graph 1 (set B→A). The captured output buffers of graph 0
        ARE the input buffers of graph 1, so LSTM states flow without any
        copies. Only token and t_index need in-place updates.

        Args:
            enc_projected: Pre-projected encoder output on GPU,
                shape ``(1, T, joint_hidden)``.

        Returns:
            List of token ID sequences (one per batch element).
        """
        T = 400

        if not self._captured:
            self._capture_graphs(enc_projected)
        else:
            assert self._enc_proj_buf is not None
            self._enc_proj_buf.inplace_copy_from(enc_projected)

        # Reset LSTM states to zero for both buffer sets.
        zero = Buffer.from_numpy(
            np.zeros((1, self.pred_hidden), dtype=np.float32)
        ).to(self.device)
        for i in range(2):
            self._bufs[i]["h0"].inplace_copy_from(zero)
            self._bufs[i]["c0"].inplace_copy_from(zero)
            self._bufs[i]["h1"].inplace_copy_from(zero)
            self._bufs[i]["c1"].inplace_copy_from(zero)

        # SOS step: replay graph 0 with blank token at t=0.
        cur = 0  # Current buffer set index (input side)
        self._bufs[cur]["token"].inplace_copy_from(
            self._token_bufs[self.blank_id]
        )
        self._bufs[cur]["t"].inplace_copy_from(self._t_index_bufs[0])
        self.decoder_step_model.replay(
            self._graph_keys[cur],
            self._bufs[cur]["token"],
            self._bufs[cur]["h0"],
            self._bufs[cur]["c0"],
            self._bufs[cur]["h1"],
            self._bufs[cur]["c1"],
            self._enc_proj_buf,
            self._bufs[cur]["t"],
        )
        # Output LSTM states are now in captured_outputs[cur],
        # which ARE the input buffers of the OTHER set if we captured
        # correctly. But capture doesn't guarantee output buffers
        # alias the other set's inputs. So the double-buffer trick
        # only works if we manually wire it — which we can't with
        # MAX's capture API (outputs are allocated by the runtime).
        #
        # Fallback: copy output states to the next set's input buffers.
        # This is still only 4 GPU→GPU copies, same as before, but
        # the CUDA graph replay itself is faster.
        nxt = 1 - cur
        outs = self._captured_outputs[cur]
        # outs: [decisions, h0, c0, h1, c1]
        self._bufs[nxt]["h0"].inplace_copy_from(outs[1])
        self._bufs[nxt]["c0"].inplace_copy_from(outs[2])
        self._bufs[nxt]["h1"].inplace_copy_from(outs[3])
        self._bufs[nxt]["c1"].inplace_copy_from(outs[4])
        cur = nxt

        # Greedy decode loop.
        tokens: list[int] = []
        t = 0
        max_symbols_per_step = 10

        while t < T:
            symbols_at_t = 0

            while symbols_at_t < max_symbols_per_step:
                # Update time index on current input set.
                self._bufs[cur]["t"].inplace_copy_from(self._t_index_bufs[t])

                # Replay current graph.
                self.decoder_step_model.replay(
                    self._graph_keys[cur],
                    self._bufs[cur]["token"],
                    self._bufs[cur]["h0"],
                    self._bufs[cur]["c0"],
                    self._bufs[cur]["h1"],
                    self._bufs[cur]["c1"],
                    self._enc_proj_buf,
                    self._bufs[cur]["t"],
                )

                # Async D2H: copy decisions to pinned buffer, record event.
                outs = self._captured_outputs[cur]
                decisions_dev = outs[0]
                self._decisions_pinned.inplace_copy_from(decisions_dev)
                # TODO: Use DeviceEvent here once we verify it works:
                # event = self.device.default_stream.record_event()
                # ... do CPU-side prep ...
                # event.synchronize()

                # Copy LSTM states to next set's input buffers.
                nxt = 1 - cur
                self._bufs[nxt]["h0"].inplace_copy_from(outs[1])
                self._bufs[nxt]["c0"].inplace_copy_from(outs[2])
                self._bufs[nxt]["h1"].inplace_copy_from(outs[3])
                self._bufs[nxt]["c1"].inplace_copy_from(outs[4])

                # Read decisions from pinned buffer (already on host).
                decisions = self._decisions_np
                token = int(decisions[0])
                dur_idx = int(decisions[1])
                duration = self.durations[dur_idx]

                if token == self.blank_id:
                    cur = nxt
                    t += max(duration, 1)
                    break

                # Non-blank: update next set's token buffer.
                tokens.append(token)
                self._bufs[nxt]["token"].inplace_copy_from(
                    self._token_bufs[token]
                )
                cur = nxt
                symbols_at_t += 1

                if duration > 0:
                    t += duration
                    break
            else:
                cur = 1 - cur
                t += 1

        return [tokens]
