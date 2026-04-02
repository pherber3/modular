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
from max.driver import Buffer, Device, DevicePinnedBuffer
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
        lstm_state_packed: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Run one prediction step with packed LSTM states.

        Args:
            token_id: Previous token ID, shape ``(1, 1)`` int32.
            lstm_state_packed: All LSTM states packed as ``(1, 4*pred_hidden)``,
                order: ``[h0, c0, h1, c1]``.

        Returns:
            ``(pred_out, lstm_state_packed_new)``
        """
        # Unpack LSTM states: (1, 2560) → 4 x (1, 640)
        h0, c0, h1, c1 = ops.chunk(lstm_state_packed, 4, axis=-1)

        # Embedding lookup: (1, 1) → (1, 640)
        x = ops.gather(self.embedding, token_id, axis=0)
        # gather output shape is (1, 1, 640), squeeze the middle dim
        x = ops.squeeze(x, 1)

        # LSTM layer 0
        h0_new, c0_new = self.lstm_0(x, h0, c0)

        # LSTM layer 1 (input is output of layer 0)
        h1_new, c1_new = self.lstm_1(h0_new, h1, c1)

        # Repack LSTM states: 4 x (1, 640) → (1, 2560)
        lstm_state_packed_new = ops.concat(
            [h0_new, c0_new, h1_new, c1_new], axis=-1
        )

        # Predictor output is the hidden state of the last layer
        return h1_new, lstm_state_packed_new


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
        Compiled graph with 4 inputs and 2 outputs
        (decisions, lstm_state_packed).
    """
    pred_hidden = config.pred_hidden
    joint_hidden = config.joint_hidden
    vocab_size = config.vocab_size
    num_durations = len(config.tdt_durations)
    output_size = vocab_size + 1 + num_durations  # vocab + blank + durations
    device = config.device
    max_encoder_len = TDTGraphDecoder.MAX_ENCODER_FRAMES

    # 4 inputs: token_id, lstm_state_packed, enc_projected, t_index
    input_types = [
        TensorType(DType.int32, shape=[1, 1], device=device),  # token_id
        TensorType(
            DType.float32, shape=[1, 4 * pred_hidden], device=device
        ),  # lstm_state_packed [h0, c0, h1, c1]
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
        lstm_state_packed = graph.inputs[1].tensor
        enc_projected = graph.inputs[2].tensor  # (1, 400, 640)
        t_index = graph.inputs[3].tensor  # (1,)

        # Slice encoder at timestep t on-device: (1, 400, 640) → (1, 640)
        enc_t = ops.gather(enc_projected, t_index, axis=1)
        # gather produces (1, 1, 640), squeeze the middle dim
        enc_t = ops.squeeze(enc_t, 1)

        # Forward pass
        pred_out, lstm_state_packed_new = prediction(
            token_id, lstm_state_packed
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

        graph.output(decisions, lstm_state_packed_new)

    return graph


def build_mojo_decoder_step_graph(
    config: TDTModelConfig,
    prediction_state_dict: Mapping[str, np.ndarray],
    joint_state_dict: Mapping[str, np.ndarray],
) -> Graph:
    """Build decoder step graph using the fused Mojo GPU kernel.

    Same input/output signature as ``build_decoder_step_graph`` — the
    ``TDTGraphDecoder`` class works without any changes.

    The Mojo kernel fuses embedding + 2-layer LSTM + joint network + argmax
    into a single GPU kernel launch per step.

    Args:
        config: TDT model configuration.
        prediction_state_dict: Weights for prediction network
            (keys: embedding, lstm_0.ih.weight, etc.).
        joint_state_dict: Weights for joint network
            (keys: pred_proj.weight, output_proj.weight, etc.).

    Returns:
        Compiled graph with 4 inputs and 2 outputs
        (decisions, lstm_state_packed).
    """
    from pathlib import Path

    kernels_dir = Path(__file__).parent / "kernels"

    pred_hidden = config.pred_hidden
    joint_hidden = config.joint_hidden
    vocab_size = config.vocab_size
    num_durations = len(config.tdt_durations)
    output_size = vocab_size + 1 + num_durations
    gates_dim = 4 * pred_hidden
    device = config.device
    max_encoder_len = TDTGraphDecoder.MAX_ENCODER_FRAMES
    dtype = DType.float32

    # Same 4 runtime inputs as the existing graph.
    input_types = [
        TensorType(DType.int32, shape=[1, 1], device=device),
        TensorType(dtype, shape=[1, 4 * pred_hidden], device=device),
        TensorType(
            dtype, shape=[1, max_encoder_len, joint_hidden], device=device
        ),
        TensorType(DType.int32, shape=[1], device=device),
    ]

    with Graph(
        "tdt_mojo_decoder_step",
        input_types=input_types,
        custom_extensions=[kernels_dir],
    ) as graph:
        token_id = graph.inputs[0].tensor  # (1, 1) int32
        lstm_state_packed = graph.inputs[1].tensor  # (1, 4*pred_hidden)
        enc_projected = graph.inputs[2].tensor  # (1, T, joint_hidden)
        t_index = graph.inputs[3].tensor  # (1,)

        # Slice encoder at timestep t on-device (same as current graph).
        enc_t = ops.squeeze(
            ops.gather(enc_projected, t_index, axis=1), 1
        )  # (1, joint_hidden)

        # Reshape token_id: (1,1) → (1,) for the Mojo kernel.
        token_flat = ops.squeeze(token_id, 1)  # (1,)

        # Declare weights. The Mojo kernel expects W[j,k] layout (original,
        # not transposed). Weight matrices are (out_dim, in_dim).
        w = {}
        w["embedding"] = Weight(
            "embedding", dtype, (vocab_size + 1, pred_hidden), device
        )
        for layer in (0, 1):
            pfx = f"lstm_{layer}"
            w[f"{pfx}.ih.weight"] = Weight(
                f"{pfx}.ih.weight", dtype, (gates_dim, pred_hidden), device
            )
            w[f"{pfx}.ih.bias"] = Weight(
                f"{pfx}.ih.bias", dtype, (gates_dim,), device
            )
            w[f"{pfx}.hh.weight"] = Weight(
                f"{pfx}.hh.weight", dtype, (gates_dim, pred_hidden), device
            )
            w[f"{pfx}.hh.bias"] = Weight(
                f"{pfx}.hh.bias", dtype, (gates_dim,), device
            )
        w["pred_proj.weight"] = Weight(
            "pred_proj.weight", dtype, (joint_hidden, pred_hidden), device
        )
        w["pred_proj.bias"] = Weight(
            "pred_proj.bias", dtype, (joint_hidden,), device
        )
        w["output_proj.weight"] = Weight(
            "output_proj.weight", dtype, (output_size, joint_hidden), device
        )
        w["output_proj.bias"] = Weight(
            "output_proj.bias", dtype, (output_size,), device
        )

        decisions, lstm_state_new = ops.custom(
            name="tdt_decode_step",
            device=device,
            values=[
                enc_t,
                token_flat,
                lstm_state_packed,
                w["embedding"],
                w["lstm_0.ih.weight"],
                w["lstm_0.ih.bias"],
                w["lstm_0.hh.weight"],
                w["lstm_0.hh.bias"],
                w["lstm_1.ih.weight"],
                w["lstm_1.ih.bias"],
                w["lstm_1.hh.weight"],
                w["lstm_1.hh.bias"],
                w["pred_proj.weight"],
                w["pred_proj.bias"],
                w["output_proj.weight"],
                w["output_proj.bias"],
            ],
            out_types=[
                TensorType(DType.int32, shape=[1, 2], device=device),
                TensorType(dtype, shape=[1, 4 * pred_hidden], device=device),
            ],
        )

        graph.output(decisions.tensor, lstm_state_new.tensor)

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
    are pre-allocated at init for zero-allocation decode loops.
    """

    # Fixed input/encoder frame counts. 3200 mel frames ≈ 20s audio at
    # 16kHz with hop=160. 8x subsampling → 400 encoder timesteps.
    MAX_INPUT_FRAMES = 3200
    MAX_ENCODER_FRAMES = MAX_INPUT_FRAMES // 8  # 400

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
        max_t = self.MAX_ENCODER_FRAMES
        self._t_index_bufs = [
            Buffer.from_numpy(np.array([t], dtype=np.int32)).to(device)
            for t in range(max_t)
        ]

        num_tokens = config.vocab_size + 1
        self._token_bufs = [
            Buffer.from_numpy(np.array([[tid]], dtype=np.int32)).to(device)
            for tid in range(num_tokens)
        ]

        # Packed LSTM states: [h0, c0, h1, c1] as (1, 4*pred_hidden).
        # Single buffer set — output is copied back to input each step.
        packed_size = 4 * config.pred_hidden
        zero_packed = np.zeros((1, packed_size), dtype=np.float32)
        blank_token = np.array([[config.blank_id]], dtype=np.int32)
        t_zero = np.array([0], dtype=np.int32)

        self._token_buf = Buffer.from_numpy(blank_token).to(device)
        self._lstm_buf = Buffer.from_numpy(zero_packed.copy()).to(device)
        self._t_buf = Buffer.from_numpy(t_zero).to(device)

        # Pre-allocated zero buffer for resetting packed LSTM states.
        self._zero_buf = Buffer.from_numpy(zero_packed.copy()).to(device)

        # Pinned buffer for async D2H readback of decisions.
        self._decisions_pinned = DevicePinnedBuffer(
            dtype=DType.int32, shape=(2,), device=device
        )
        self._decisions_np = self._decisions_pinned.to_numpy()

        # Persistent enc_projected buffer for CUDA graph capture.
        self._enc_proj_buf: Buffer | None = None
        self._captured = False
        self._captured_outputs: list[Buffer] = []

    def _capture_graph(self, enc_projected: Buffer) -> None:
        """Capture a single CUDA graph for the decoder step."""
        self._enc_proj_buf = enc_projected.copy()
        self._enc_proj_buf.inplace_copy_from(enc_projected)

        self._lstm_buf.inplace_copy_from(self._zero_buf)
        self._token_buf.inplace_copy_from(self._token_bufs[self.blank_id])
        self._t_buf.inplace_copy_from(self._t_index_bufs[0])

        self._captured_outputs = list(
            self.decoder_step_model.capture(
                1,
                self._token_buf,
                self._lstm_buf,
                self._enc_proj_buf,
                self._t_buf,
            )
        )

        self._captured = True
        logger.info("Captured decoder step CUDA graph")

    def decode(self, enc_projected: Buffer) -> list[list[int]]:
        """Run TDT greedy decode with CUDA graph replay.

        Each step: replay captured graph → copy packed LSTM output back
        to input buffer → read decisions from pinned D2H buffer.

        Args:
            enc_projected: Pre-projected encoder output on GPU,
                shape ``(1, T, joint_hidden)``.

        Returns:
            List of token ID sequences (one per batch element).
        """
        T = self.MAX_ENCODER_FRAMES

        if not self._captured:
            self._capture_graph(enc_projected)
        else:
            assert self._enc_proj_buf is not None
            self._enc_proj_buf.inplace_copy_from(enc_projected)

        # Hoist attribute lookups to locals for hot-loop performance.
        token_buf = self._token_buf
        lstm_buf = self._lstm_buf
        t_buf = self._t_buf
        token_bufs = self._token_bufs
        t_index_bufs = self._t_index_bufs
        enc_proj_buf = self._enc_proj_buf
        outs = self._captured_outputs
        decisions_pinned = self._decisions_pinned
        decisions_np = self._decisions_np
        blank_id = self.blank_id
        durations = self.durations
        replay = self.decoder_step_model.replay

        # Reset packed LSTM states to zero.
        lstm_buf.inplace_copy_from(self._zero_buf)

        # SOS step: blank token at t=0.
        token_buf.inplace_copy_from(token_bufs[blank_id])
        t_buf.inplace_copy_from(t_index_bufs[0])
        replay(1, token_buf, lstm_buf, enc_proj_buf, t_buf)
        # outs: [decisions, lstm_state_packed]
        lstm_buf.inplace_copy_from(outs[1])

        # Greedy decode loop.
        tokens: list[int] = []
        t = 0
        max_symbols_per_step = 10

        while t < T:
            symbols_at_t = 0

            while symbols_at_t < max_symbols_per_step:
                t_buf.inplace_copy_from(t_index_bufs[t])

                replay(1, token_buf, lstm_buf, enc_proj_buf, t_buf)

                decisions_pinned.inplace_copy_from(outs[0])
                lstm_buf.inplace_copy_from(outs[1])

                token = int(decisions_np[0])
                dur_idx = int(decisions_np[1])
                duration = durations[dur_idx]

                if token == blank_id:
                    t += max(duration, 1)
                    break

                tokens.append(token)
                token_buf.inplace_copy_from(token_bufs[token])
                symbols_at_t += 1

                if duration > 0:
                    t += duration
                    break
            else:
                t += 1

        return [tokens]
