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
"""Correctness test for the fused TDT decode step Mojo kernel.

Compares the Mojo custom op output against the numpy reference decoder
(decoder.py LSTMCell + JointNetwork) using real model weights.

Usage:
  ./bazelw run //max/tests/integration/tools:test_tdt_decode_step
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops

NDFloat = npt.NDArray[np.floating]

# Model constants for parakeet-tdt-0.6b-v3.
MODEL_ID = "pherber3/parakeet-tdt-0.6b-v3"
NPZ_FILE = "decoder_joint.npz"
PRED_HIDDEN = 640
JOINT_HIDDEN = 640
VOCAB_SIZE = 8192
BLANK_ID = 8192  # = vocab_size
NUM_DURATIONS = 5
OUTPUT_SIZE = VOCAB_SIZE + 1 + NUM_DURATIONS  # 8198
GATES_DIM = 4 * PRED_HIDDEN  # 2560


# ---- Numpy reference (inline, no import dependency on parakeet_tdt) ----


def _sigmoid(x: NDFloat) -> NDFloat:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def numpy_decode_step(
    enc_t: NDFloat,
    token_id: int,
    h0: NDFloat,
    c0: NDFloat,
    h1: NDFloat,
    c1: NDFloat,
    weights: dict[str, NDFloat],
) -> tuple[int, int, NDFloat, NDFloat, NDFloat, NDFloat]:
    """One TDT decode step in numpy. Returns (best_token, best_dur, h0, c0, h1, c1)."""
    embed = weights["embedding"]
    x = embed[token_id]  # (640,)

    # LSTM Layer 0
    gates0 = (
        x @ weights["l0_ih_w"].T
        + weights["l0_ih_b"]
        + h0 @ weights["l0_hh_w"].T
        + weights["l0_hh_b"]
    )
    i0, f0, g0, o0 = np.split(gates0, 4, axis=-1)
    c0_new = _sigmoid(f0) * c0 + _sigmoid(i0) * np.tanh(g0)
    h0_new = _sigmoid(o0) * np.tanh(c0_new)

    # LSTM Layer 1
    gates1 = (
        h0_new @ weights["l1_ih_w"].T
        + weights["l1_ih_b"]
        + h1 @ weights["l1_hh_w"].T
        + weights["l1_hh_b"]
    )
    i1, f1, g1, o1 = np.split(gates1, 4, axis=-1)
    c1_new = _sigmoid(f1) * c1 + _sigmoid(i1) * np.tanh(g1)
    h1_new = _sigmoid(o1) * np.tanh(c1_new)

    # Joint network
    pred_proj = h1_new @ weights["pred_w"].T + weights["pred_b"]
    combined = np.maximum(0, enc_t + pred_proj)
    logits = combined @ weights["out_w"].T + weights["out_b"]

    best_token = int(np.argmax(logits[: VOCAB_SIZE + 1]))
    best_dur = int(np.argmax(logits[VOCAB_SIZE + 1 :]))

    return best_token, best_dur, h0_new, c0_new, h1_new, c1_new


# ---- Load and prepare weights ----


def load_weights() -> dict[str, NDFloat]:
    """Download and load TDT decoder weights from HuggingFace."""
    npz_path = hf_hub_download(repo_id=MODEL_ID, filename=NPZ_FILE)
    raw = dict(np.load(npz_path))

    # Map NeMo keys to our kernel's expected names.
    prefix_pred = "decoder.prediction.dec_rnn.lstm"
    return {
        "embedding": raw["decoder.prediction.embed.weight"],
        "l0_ih_w": raw[f"{prefix_pred}.weight_ih_l0"],
        "l0_ih_b": raw[f"{prefix_pred}.bias_ih_l0"],
        "l0_hh_w": raw[f"{prefix_pred}.weight_hh_l0"],
        "l0_hh_b": raw[f"{prefix_pred}.bias_hh_l0"],
        "l1_ih_w": raw[f"{prefix_pred}.weight_ih_l1"],
        "l1_ih_b": raw[f"{prefix_pred}.bias_ih_l1"],
        "l1_hh_w": raw[f"{prefix_pred}.weight_hh_l1"],
        "l1_hh_b": raw[f"{prefix_pred}.bias_hh_l1"],
        "pred_w": raw["joint.pred.weight"],
        "pred_b": raw["joint.pred.bias"],
        "out_w": raw["joint.joint_net.2.weight"],
        "out_b": raw["joint.joint_net.2.bias"],
        # Encoder projection (not needed for decode step test, but load anyway)
        "enc_w": raw["joint.enc.weight"],
        "enc_b": raw["joint.enc.bias"],
    }


def build_mojo_graph(device: Device) -> Graph:
    """Build the custom op graph for the fused decode step."""
    dev = DeviceRef.from_device(device)
    dtype = DType.float32
    kernels_dir = Path(__file__).resolve().parent / "parakeet_kernels"

    input_types = [
        TensorType(dtype, shape=[1, JOINT_HIDDEN], device=dev),  # enc_t
        TensorType(DType.int32, shape=[1], device=dev),  # token_id
        TensorType(dtype, shape=[1, 4 * PRED_HIDDEN], device=dev),  # lstm_state
        TensorType(
            dtype, shape=[VOCAB_SIZE + 1, PRED_HIDDEN], device=dev
        ),  # embedding
        # LSTM L0 weights (original layout: out_dim, in_dim)
        TensorType(
            dtype, shape=[GATES_DIM, PRED_HIDDEN], device=dev
        ),  # l0_ih_w
        TensorType(dtype, shape=[GATES_DIM], device=dev),  # l0_ih_b
        TensorType(
            dtype, shape=[GATES_DIM, PRED_HIDDEN], device=dev
        ),  # l0_hh_w
        TensorType(dtype, shape=[GATES_DIM], device=dev),  # l0_hh_b
        # LSTM L1 weights
        TensorType(
            dtype, shape=[GATES_DIM, PRED_HIDDEN], device=dev
        ),  # l1_ih_w
        TensorType(dtype, shape=[GATES_DIM], device=dev),  # l1_ih_b
        TensorType(
            dtype, shape=[GATES_DIM, PRED_HIDDEN], device=dev
        ),  # l1_hh_w
        TensorType(dtype, shape=[GATES_DIM], device=dev),  # l1_hh_b
        # Joint weights (original layout)
        TensorType(
            dtype, shape=[JOINT_HIDDEN, PRED_HIDDEN], device=dev
        ),  # pred_w
        TensorType(dtype, shape=[JOINT_HIDDEN], device=dev),  # pred_b
        TensorType(
            dtype, shape=[OUTPUT_SIZE, JOINT_HIDDEN], device=dev
        ),  # out_w
        TensorType(dtype, shape=[OUTPUT_SIZE], device=dev),  # out_b
    ]

    with Graph(
        "tdt_decode_step_test",
        input_types=input_types,
        custom_extensions=[kernels_dir],
    ) as graph:
        inputs = [
            inp.tensor if hasattr(inp, "tensor") else inp
            for inp in graph.inputs
        ]

        decisions, state_new = ops.custom(
            name="tdt_decode_step",
            device=dev,
            values=inputs,
            out_types=[
                TensorType(dtype=DType.int32, shape=[1, 2], device=dev),
                TensorType(dtype=dtype, shape=[1, 4 * PRED_HIDDEN], device=dev),
            ],
        )
        graph.output(decisions.tensor, state_new.tensor)

    return graph


def prepare_buffers(
    device: Device,
    enc_t: NDFloat,
    token_id: int,
    lstm_state: NDFloat,
    weights: dict[str, NDFloat],
) -> list[Buffer]:
    """Create GPU buffers for all inputs (weights in original layout)."""
    bufs = [
        Buffer.from_numpy(enc_t.reshape(1, JOINT_HIDDEN).astype(np.float32)),
        Buffer.from_numpy(np.array([token_id], dtype=np.int32)),
        Buffer.from_numpy(
            lstm_state.reshape(1, 4 * PRED_HIDDEN).astype(np.float32)
        ),
        Buffer.from_numpy(weights["embedding"].astype(np.float32)),
        Buffer.from_numpy(
            np.ascontiguousarray(weights["l0_ih_w"].astype(np.float32))
        ),
        Buffer.from_numpy(weights["l0_ih_b"].astype(np.float32)),
        Buffer.from_numpy(
            np.ascontiguousarray(weights["l0_hh_w"].astype(np.float32))
        ),
        Buffer.from_numpy(weights["l0_hh_b"].astype(np.float32)),
        Buffer.from_numpy(
            np.ascontiguousarray(weights["l1_ih_w"].astype(np.float32))
        ),
        Buffer.from_numpy(weights["l1_ih_b"].astype(np.float32)),
        Buffer.from_numpy(
            np.ascontiguousarray(weights["l1_hh_w"].astype(np.float32))
        ),
        Buffer.from_numpy(weights["l1_hh_b"].astype(np.float32)),
        Buffer.from_numpy(
            np.ascontiguousarray(weights["pred_w"].astype(np.float32))
        ),
        Buffer.from_numpy(weights["pred_b"].astype(np.float32)),
        Buffer.from_numpy(
            np.ascontiguousarray(weights["out_w"].astype(np.float32))
        ),
        Buffer.from_numpy(weights["out_b"].astype(np.float32)),
    ]
    return [b.to(device) for b in bufs]


def run_test(
    label: str,
    model: Model,
    device: Device,
    enc_t: NDFloat,
    token_id: int,
    h0: NDFloat,
    c0: NDFloat,
    h1: NDFloat,
    c1: NDFloat,
    weights: dict[str, NDFloat],
) -> bool:
    """Run one test case: compare Mojo kernel vs numpy reference."""
    print(f"\n--- {label} ---")

    # Numpy reference
    ref_token, ref_dur, ref_h0, ref_c0, ref_h1, ref_c1 = numpy_decode_step(
        enc_t, token_id, h0, c0, h1, c1, weights
    )
    ref_state = np.concatenate([ref_h0, ref_c0, ref_h1, ref_c1])

    # Mojo kernel
    lstm_state = np.concatenate([h0, c0, h1, c1]).reshape(1, -1)
    bufs = prepare_buffers(device, enc_t, token_id, lstm_state, weights)
    results = model.execute(*bufs)

    decisions = results[0]
    assert isinstance(decisions, Buffer)
    decisions_np = decisions.to(CPU()).to_numpy().flatten()
    mojo_token = int(decisions_np[0])
    mojo_dur = int(decisions_np[1])

    state_new = results[1]
    assert isinstance(state_new, Buffer)
    mojo_state = state_new.to(CPU()).to_numpy().flatten()

    # Compare decisions
    token_match = mojo_token == ref_token
    dur_match = mojo_dur == ref_dur
    print(
        f"  Token: mojo={mojo_token} ref={ref_token} {'PASS' if token_match else 'FAIL'}"
    )
    print(
        f"  Duration: mojo={mojo_dur} ref={ref_dur} {'PASS' if dur_match else 'FAIL'}"
    )

    # Compare LSTM states
    state_err = np.max(np.abs(mojo_state - ref_state))
    state_rel = state_err / (np.max(np.abs(ref_state)) + 1e-8)
    state_pass = state_rel < 1e-4
    print(
        f"  State: max_abs_err={state_err:.2e} rel_err={state_rel:.2e}"
        f" {'PASS' if state_pass else 'FAIL'}"
    )

    passed = token_match and dur_match and state_pass
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    if accelerator_count() == 0:
        print("ERROR: No GPU detected.")
        sys.exit(1)

    device = Accelerator()
    print(f"Device: {device}")
    print("Loading weights...")
    weights = load_weights()

    print("Compiling Mojo decode step graph...")
    t0 = time.perf_counter()
    graph = build_mojo_graph(device)
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    print(f"Compiled in {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(42)
    all_passed = True

    # Test 1: Zero state, blank token
    enc_t = rng.standard_normal(JOINT_HIDDEN).astype(np.float32)
    all_passed &= run_test(
        "Test 1: zero state, blank token",
        model,
        device,
        enc_t,
        token_id=BLANK_ID,
        h0=np.zeros(PRED_HIDDEN, dtype=np.float32),
        c0=np.zeros(PRED_HIDDEN, dtype=np.float32),
        h1=np.zeros(PRED_HIDDEN, dtype=np.float32),
        c1=np.zeros(PRED_HIDDEN, dtype=np.float32),
        weights=weights,
    )

    # Test 2: Zero state, non-blank token
    all_passed &= run_test(
        "Test 2: zero state, token=42",
        model,
        device,
        enc_t,
        token_id=42,
        h0=np.zeros(PRED_HIDDEN, dtype=np.float32),
        c0=np.zeros(PRED_HIDDEN, dtype=np.float32),
        h1=np.zeros(PRED_HIDDEN, dtype=np.float32),
        c1=np.zeros(PRED_HIDDEN, dtype=np.float32),
        weights=weights,
    )

    # Test 3: Random state, random token
    all_passed &= run_test(
        "Test 3: random state, token=500",
        model,
        device,
        enc_t,
        token_id=500,
        h0=rng.standard_normal(PRED_HIDDEN).astype(np.float32) * 0.1,
        c0=rng.standard_normal(PRED_HIDDEN).astype(np.float32) * 0.1,
        h1=rng.standard_normal(PRED_HIDDEN).astype(np.float32) * 0.1,
        c1=rng.standard_normal(PRED_HIDDEN).astype(np.float32) * 0.1,
        weights=weights,
    )

    # Test 4: Different encoder output
    enc_t2 = rng.standard_normal(JOINT_HIDDEN).astype(np.float32) * 2.0
    all_passed &= run_test(
        "Test 4: different enc_t, random state",
        model,
        device,
        enc_t2,
        token_id=100,
        h0=rng.standard_normal(PRED_HIDDEN).astype(np.float32) * 0.05,
        c0=rng.standard_normal(PRED_HIDDEN).astype(np.float32) * 0.3,
        h1=rng.standard_normal(PRED_HIDDEN).astype(np.float32) * 0.05,
        c1=rng.standard_normal(PRED_HIDDEN).astype(np.float32) * 0.3,
        weights=weights,
    )

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
