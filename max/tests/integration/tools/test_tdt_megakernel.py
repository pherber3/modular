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
"""Correctness test for the TDT persistent megakernel.

Runs the full TDT decode loop through both the numpy reference and the Mojo
megakernel, then compares the output token sequences.

Usage:
  ./bazelw run //max/tests/integration/tools:test_tdt_megakernel
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

MODEL_ID = "pherber3/parakeet-tdt-0.6b-v3"
NPZ_FILE = "decoder_joint.npz"
PRED_HIDDEN = 640
JOINT_HIDDEN = 640
VOCAB_SIZE = 8192
BLANK_ID = 8192
NUM_DURATIONS = 5
OUTPUT_SIZE = VOCAB_SIZE + 1 + NUM_DURATIONS  # 8198
GATES_DIM = 4 * PRED_HIDDEN  # 2560
MAX_ENCODER_FRAMES = 400
MAX_OUTPUT_TOKENS = 4096
TDT_DURATIONS = [0, 2, 4, 6, 8]


# ---- Numpy full decode reference ----


def _sigmoid(x: NDFloat) -> NDFloat:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _lstm_step(
    x: NDFloat,
    h: NDFloat,
    c: NDFloat,
    ih_w: NDFloat,
    ih_b: NDFloat,
    hh_w: NDFloat,
    hh_b: NDFloat,
) -> tuple[NDFloat, NDFloat]:
    gates = x @ ih_w.T + ih_b + h @ hh_w.T + hh_b
    i, f, g, o = np.split(gates, 4, axis=-1)
    c_new = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
    h_new = _sigmoid(o) * np.tanh(c_new)
    return h_new, c_new


def numpy_full_decode(
    enc_projected: NDFloat,
    weights: dict[str, NDFloat],
    durations: list[int],
) -> list[int]:
    """Full TDT greedy decode in numpy. Returns token list."""
    T = enc_projected.shape[0]  # (T, 640)

    h0 = np.zeros(PRED_HIDDEN, dtype=np.float32)
    c0 = np.zeros(PRED_HIDDEN, dtype=np.float32)
    h1 = np.zeros(PRED_HIDDEN, dtype=np.float32)
    c1 = np.zeros(PRED_HIDDEN, dtype=np.float32)
    current_token = BLANK_ID

    def decode_step(
        token_id: int,
        enc_t: NDFloat,
        h0: NDFloat,
        c0: NDFloat,
        h1: NDFloat,
        c1: NDFloat,
    ) -> tuple[int, int, NDFloat, NDFloat, NDFloat, NDFloat]:
        x = weights["embedding"][token_id]
        h0_new, c0_new = _lstm_step(
            x,
            h0,
            c0,
            weights["l0_ih_w"],
            weights["l0_ih_b"],
            weights["l0_hh_w"],
            weights["l0_hh_b"],
        )
        h1_new, c1_new = _lstm_step(
            h0_new,
            h1,
            c1,
            weights["l1_ih_w"],
            weights["l1_ih_b"],
            weights["l1_hh_w"],
            weights["l1_hh_b"],
        )
        pred_proj = h1_new @ weights["pred_w"].T + weights["pred_b"]
        combined = np.maximum(0, enc_t + pred_proj)
        logits = combined @ weights["out_w"].T + weights["out_b"]
        best_token = int(np.argmax(logits[: VOCAB_SIZE + 1]))
        best_dur = int(np.argmax(logits[VOCAB_SIZE + 1 :]))
        return best_token, best_dur, h0_new, c0_new, h1_new, c1_new

    # SOS step (matches decoder_graph.py:572-577)
    _, _, h0, c0, h1, c1 = decode_step(
        BLANK_ID, enc_projected[0], h0, c0, h1, c1
    )

    # Main decode loop
    tokens: list[int] = []
    t = 0
    max_symbols_per_step = 10

    while t < T:
        symbols_at_t = 0
        while symbols_at_t < max_symbols_per_step:
            best_token, best_dur, h0, c0, h1, c1 = decode_step(
                current_token, enc_projected[t], h0, c0, h1, c1
            )
            duration = durations[best_dur]

            if best_token == BLANK_ID:
                t += max(duration, 1)
                break

            tokens.append(best_token)
            current_token = best_token
            symbols_at_t += 1

            if duration > 0:
                t += duration
                break
        else:
            t += 1

    return tokens


# ---- Weight loading ----


def load_weights() -> dict[str, NDFloat]:
    npz_path = hf_hub_download(repo_id=MODEL_ID, filename=NPZ_FILE)
    raw = dict(np.load(npz_path))
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
        "enc_w": raw["joint.enc.weight"],
        "enc_b": raw["joint.enc.bias"],
    }


# ---- Graph construction ----


def build_megakernel_graph(
    device: Device, weights: dict[str, NDFloat]
) -> Graph:
    dev = DeviceRef.from_device(device)
    dtype = DType.float32
    kernels_dir = Path(__file__).resolve().parent / "parakeet_kernels"

    vocab_plus_blank = weights["embedding"].shape[0]
    output_size = weights["out_w"].shape[0]
    gates_dim = weights["l0_ih_w"].shape[0]

    input_types = [
        TensorType(
            dtype, shape=[1, MAX_ENCODER_FRAMES, JOINT_HIDDEN], device=dev
        ),
        TensorType(dtype, shape=[vocab_plus_blank, PRED_HIDDEN], device=dev),
        TensorType(DType.int32, shape=[NUM_DURATIONS], device=dev),
        TensorType(dtype, shape=[gates_dim, PRED_HIDDEN], device=dev),
        TensorType(dtype, shape=[gates_dim], device=dev),
        TensorType(dtype, shape=[gates_dim, PRED_HIDDEN], device=dev),
        TensorType(dtype, shape=[gates_dim], device=dev),
        TensorType(dtype, shape=[gates_dim, PRED_HIDDEN], device=dev),
        TensorType(dtype, shape=[gates_dim], device=dev),
        TensorType(dtype, shape=[gates_dim, PRED_HIDDEN], device=dev),
        TensorType(dtype, shape=[gates_dim], device=dev),
        TensorType(dtype, shape=[JOINT_HIDDEN, PRED_HIDDEN], device=dev),
        TensorType(dtype, shape=[JOINT_HIDDEN], device=dev),
        TensorType(dtype, shape=[output_size, JOINT_HIDDEN], device=dev),
        TensorType(dtype, shape=[output_size], device=dev),
    ]

    with Graph(
        "tdt_megakernel_test",
        input_types=input_types,
        custom_extensions=[kernels_dir],
    ) as graph:
        inputs = [
            inp.tensor if hasattr(inp, "tensor") else inp
            for inp in graph.inputs
        ]
        out_tokens, out_count = ops.custom(
            name="tdt_megakernel",
            device=dev,
            values=inputs,
            out_types=[
                TensorType(
                    dtype=DType.int32, shape=[MAX_OUTPUT_TOKENS], device=dev
                ),
                TensorType(dtype=DType.int32, shape=[1], device=dev),
            ],
        )
        graph.output(out_tokens.tensor, out_count.tensor)

    return graph


def prepare_buffers(
    device: Device,
    enc_projected: NDFloat,
    weights: dict[str, NDFloat],
) -> list[Buffer]:
    # Pad enc_projected to MAX_ENCODER_FRAMES
    T = enc_projected.shape[0]
    padded = np.zeros((1, MAX_ENCODER_FRAMES, JOINT_HIDDEN), dtype=np.float32)
    padded[0, :T, :] = enc_projected

    bufs = [
        Buffer.from_numpy(padded),
        Buffer.from_numpy(weights["embedding"].astype(np.float32)),
        Buffer.from_numpy(np.array(TDT_DURATIONS, dtype=np.int32)),
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


def make_synthetic_enc_projected(
    T: int, weights: dict[str, NDFloat], seed: int = 42
) -> NDFloat:
    """Generate synthetic pre-projected encoder output.

    Simulates the real pipeline: random encoder hidden states (1024-dim)
    projected through joint.enc.weight + joint.enc.bias to get (T, 640).
    This produces values in the right distribution to trigger non-blank tokens.
    """
    rng = np.random.default_rng(seed)
    encoder_hidden = 1024
    raw_enc = rng.standard_normal((T, encoder_hidden)).astype(np.float32)
    return (raw_enc @ weights["enc_w"].T + weights["enc_b"]).astype(np.float32)


def run_test(
    label: str,
    model: Model,
    device: Device,
    enc_projected: NDFloat,
    weights: dict[str, NDFloat],
) -> bool:
    print(f"\n--- {label} ---")

    # Numpy reference
    t0 = time.perf_counter()
    ref_tokens = numpy_full_decode(enc_projected, weights, TDT_DURATIONS)
    ref_time = (time.perf_counter() - t0) * 1e3
    print(f"  Numpy: {len(ref_tokens)} tokens in {ref_time:.1f}ms")

    # Megakernel
    bufs = prepare_buffers(device, enc_projected, weights)
    t0 = time.perf_counter()
    results = model.execute(*bufs)
    mojo_time = (time.perf_counter() - t0) * 1e3

    token_buf = results[0]
    assert isinstance(token_buf, Buffer)
    token_np = token_buf.to(CPU()).to_numpy()

    count_buf = results[1]
    assert isinstance(count_buf, Buffer)
    count = int(count_buf.to(CPU()).to_numpy()[0])

    mojo_tokens = token_np[:count].tolist()
    print(f"  Mojo:  {count} tokens in {mojo_time:.1f}ms")

    match = mojo_tokens == ref_tokens
    if match:
        print(f"  Token sequences MATCH ({len(ref_tokens)} tokens)")
        print("  PASS")
    else:
        print("  MISMATCH!")
        print(f"  Ref  ({len(ref_tokens)}): {ref_tokens[:20]}...")
        print(f"  Mojo ({count}): {mojo_tokens[:20]}...")
        # Find first divergence
        for i in range(min(len(ref_tokens), count)):
            if i < len(mojo_tokens) and ref_tokens[i] != mojo_tokens[i]:
                print(
                    f"  First difference at position {i}: ref={ref_tokens[i]} mojo={mojo_tokens[i]}"
                )
                break
        print("  FAIL")

    return match


def main() -> None:
    if accelerator_count() == 0:
        print("ERROR: No GPU detected.")
        sys.exit(1)

    device = Accelerator()
    print(f"Device: {device}")

    print("Loading weights...")
    weights = load_weights()

    print("Compiling megakernel graph...")
    t0 = time.perf_counter()
    graph = build_megakernel_graph(device, weights)
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    print(f"Compiled in {time.perf_counter() - t0:.1f}s")

    all_passed = True

    # Test 1: Synthetic encoder output, T=50 (short)
    enc_50 = make_synthetic_enc_projected(50, weights, seed=42)
    all_passed &= run_test(
        "Test 1: synthetic T=50", model, device, enc_50, weights
    )

    # Test 2: Synthetic encoder output, T=200 (medium)
    enc_200 = make_synthetic_enc_projected(200, weights, seed=123)
    all_passed &= run_test(
        "Test 2: synthetic T=200", model, device, enc_200, weights
    )

    # Test 3: Synthetic encoder output, T=400 (max)
    enc_400 = make_synthetic_enc_projected(400, weights, seed=456)
    all_passed &= run_test(
        "Test 3: synthetic T=400 (max)", model, device, enc_400, weights
    )

    # Test 4: Very short, T=1
    enc_1 = make_synthetic_enc_projected(1, weights, seed=789)
    all_passed &= run_test(
        "Test 4: synthetic T=1 (edge case)", model, device, enc_1, weights
    )

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
