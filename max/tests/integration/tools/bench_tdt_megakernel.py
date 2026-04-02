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
"""Benchmark for the TDT persistent megakernel.

Measures single-launch latency for full decode (~300 steps).

Usage:
  ./bazelw run //max/tests/integration/tools:bench_tdt_megakernel
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from max.driver import Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

NDFloat = npt.NDArray[np.floating]

MODEL_ID = "pherber3/parakeet-tdt-0.6b-v3"
NPZ_FILE = "decoder_joint.npz"
PRED_HIDDEN = 640
JOINT_HIDDEN = 640
NUM_DURATIONS = 5
MAX_ENCODER_FRAMES = 400
MAX_OUTPUT_TOKENS = 4096
TDT_DURATIONS = [0, 2, 4, 6, 8]
WARMUP = 20
ITERS = 100


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
    }


def build_graph(device: Device, weights: dict[str, NDFloat]) -> Graph:
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
        "tdt_megakernel_bench",
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
    device: Device, weights: dict[str, NDFloat]
) -> list[Buffer]:
    rng = np.random.default_rng(42)
    enc = (
        rng.standard_normal((1, MAX_ENCODER_FRAMES, JOINT_HIDDEN)).astype(
            np.float32
        )
        * 0.5
    )

    bufs = [
        Buffer.from_numpy(enc),
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
    graph = build_graph(device, weights)
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    print(f"Compiled in {time.perf_counter() - t0:.1f}s")

    bufs = prepare_buffers(device, weights)

    # Warmup
    print(f"Warming up ({WARMUP} iterations)...")
    for _ in range(WARMUP):
        model.execute(*bufs)

    # Benchmark
    times: list[float] = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        results = model.execute(*bufs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms

    # Get token count from last run
    from max.driver import CPU

    count_buf = results[1]
    assert isinstance(count_buf, Buffer)
    count = int(count_buf.to(CPU()).to_numpy()[0])

    median = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]
    mean = sum(times) / len(times)

    print()
    print("=" * 60)
    print("TDT Persistent Megakernel — Full Decode Benchmark")
    print("=" * 60)
    print(f"  Tokens decoded: {count}")
    print(
        f"  Single launch:  median={median:.2f}ms  mean={mean:.2f}ms  p95={p95:.2f}ms"
    )
    print()
    print(
        "  Reference (Phase 2 step x 300, L4):     14.10ms"
        "  (47µs/step, includes dispatch)"
    )
    print(
        "  Reference (current MAX graph, A100):     23ms"
        "  (80µs/step, blank-skip)"
    )
    print("  Reference (TRT, L40S):                   15ms")
    print("  Target:                                   8-12ms")


if __name__ == "__main__":
    main()
