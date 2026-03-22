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
        # back to CPU each step — only two int32 scalars are returned.
        num_token_classes = vocab_size + 1
        token_logits, dur_logits = ops.split(
            logits, [num_token_classes, num_durations], axis=-1
        )
        best_token = ops.argmax(token_logits, axis=-1)  # (1,)
        best_dur_idx = ops.argmax(dur_logits, axis=-1)  # (1,)

        # Cast to int32 for consistency with token_id input type
        best_token = ops.cast(best_token, DType.int32)
        best_dur_idx = ops.cast(best_dur_idx, DType.int32)

        # 7 outputs: best_token, best_dur_idx, h0', c0', h1', c1', pred_out
        # pred_out is returned so we can feed it back for the next token
        # when a non-blank is emitted (need to update token_id input).
        graph.output(best_token, best_dur_idx, h0_new, c0_new, h1_new, c1_new)

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
    """Manages TDT decoding using compiled MAX graphs.

    Runs the projection graph once, then loops the decoder step graph
    with LSTM state Buffers flowing between iterations on GPU.
    """

    def __init__(
        self,
        projection_model: Model,
        decoder_step_model: Model,
        config: TDTModelConfig,
        device: Device,
        cpu_device: Device,
    ) -> None:
        self.projection_model = projection_model
        self.decoder_step_model = decoder_step_model
        self.config = config
        self.device = device
        self.cpu_device = cpu_device

        self.vocab_size = config.vocab_size
        self.blank_id = config.blank_id
        self.durations = config.tdt_durations
        self.pred_hidden = config.pred_hidden
        self.joint_hidden = config.joint_hidden

    def decode(self, encoder_output: Buffer) -> list[list[int]]:
        """Run full TDT decode: projection + greedy loop.

        Args:
            encoder_output: Encoder hidden states on GPU,
                shape ``(1, T, encoder_hidden)``.

        Returns:
            List of token ID sequences (one per batch element).
        """
        # Step 1: Pre-project encoder output on device (once per utterance).
        # projected_buf stays on device — no CPU transfer.
        proj_outputs = self.projection_model.execute(encoder_output)
        projected_buf = proj_outputs[0]  # (1, 400, joint_hidden) on device
        T = 400  # Fixed from encoder's 3200 frames / 8x subsampling

        # Step 2: Initialize LSTM states as zero Buffers on device.
        zero_state = np.zeros((1, self.pred_hidden), dtype=np.float32)
        h0 = Buffer.from_numpy(zero_state).to(self.device)
        c0 = Buffer.from_numpy(zero_state).to(self.device)
        h1 = Buffer.from_numpy(zero_state).to(self.device)
        c1 = Buffer.from_numpy(zero_state).to(self.device)

        # Initial token: blank (SOS)
        token_buf = Buffer.from_numpy(
            np.array([[self.blank_id]], dtype=np.int32)
        ).to(self.device)

        # Run initial prediction step to get SOS state.
        t_index_buf = Buffer.from_numpy(np.array([0], dtype=np.int32)).to(
            self.device
        )
        outputs = self.decoder_step_model.execute(
            token_buf, h0, c0, h1, c1, projected_buf, t_index_buf
        )
        _, _, h0, c0, h1, c1 = outputs

        # Step 3: Greedy decode loop.
        # Per step: t_index upload (4 bytes) + token/dur_idx readback (8 bytes).
        # All state Buffers and projected_buf stay on device throughout.
        tokens: list[int] = []
        t = 0
        max_symbols_per_step = 10

        while t < T:
            symbols_at_t = 0

            while symbols_at_t < max_symbols_per_step:
                # Upload timestep index (4 bytes)
                t_index_buf = Buffer.from_numpy(
                    np.array([t], dtype=np.int32)
                ).to(self.device)

                # Execute decoder step — all on device.
                # Argmax is done inside the graph; only two int32 scalars
                # are returned (best_token, best_dur_idx).
                outputs = self.decoder_step_model.execute(
                    token_buf, h0, c0, h1, c1, projected_buf, t_index_buf
                )
                token_buf_out, dur_idx_buf, h0, c0, h1, c1 = outputs

                # Read back greedy decisions (8 bytes total).
                # .to(cpu) is a no-op when already on CPU.
                token = int(
                    np.from_dlpack(token_buf_out.to(self.cpu_device))[0]
                )
                dur_idx = int(
                    np.from_dlpack(dur_idx_buf.to(self.cpu_device))[0]
                )
                duration = self.durations[dur_idx]

                if token == self.blank_id:
                    t += max(duration, 1)
                    break

                # Non-blank: emit token. The graph already output the
                # token as int32 — reshape it for next step's input.
                tokens.append(token)
                token_buf = Buffer.from_numpy(
                    np.array([[token]], dtype=np.int32)
                ).to(self.device)
                symbols_at_t += 1

                if duration > 0:
                    t += duration
                    break
                # duration == 0: stay at same t (multiple emissions)
            else:
                # Safety: hit max_symbols_per_step, force advance
                t += 1

        return [tokens]
