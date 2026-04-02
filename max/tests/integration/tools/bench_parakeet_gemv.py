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
"""GEMV benchmark for TDT decoder megakernel go/no-go decision.

Compares:
  1. ops.matmul — MAX's built-in dispatch (multi-block GEVM kernel / cuBLAS)
  2. cooperative_gemv — single-block cooperative GEMV (megakernel primitive)

Shape: (1, 640) x (640, 2560) -> (1, 2560)  [LSTM gates, the bottleneck]

Metrics:
  - Single-call latency (µs)
  - 300x sequential calls (simulates one utterance decode loop)

Go/no-go: If single-block GEMV is within ~1.5x of ops.matmul, the megakernel
approach is viable — per-step dispatch overhead (~90µs) dwarfs any GEMV gap.
If >2x slower, compute portion doubles and kills the advantage.

Usage:
  ./bazelw run //max/tests/integration/tools:bench_parakeet_gemv
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops

# TDT decoder LSTM gate shapes.
M, K, N = 1, 640, 2560
WARMUP = 50
ITERS = 500
DECODE_STEPS = 300


def build_matmul_graph(device: Device, dtype: DType) -> Graph:
    """Graph using ops.matmul (MAX built-in dispatch)."""
    dev = DeviceRef.from_device(device)
    with Graph(
        "matmul_baseline",
        input_types=[
            TensorType(dtype, shape=[M, K], device=dev),
            TensorType(dtype, shape=[K, N], device=dev),
        ],
    ) as graph:
        x, w = graph.inputs
        y = ops.matmul(x.tensor, w.tensor)
        graph.output(y)
    return graph


def build_custom_gemv_graph(device: Device, dtype: DType) -> Graph:
    """Graph using single-block cooperative GEMV custom op."""
    dev = DeviceRef.from_device(device)
    kernels_dir = Path(__file__).resolve().parent / "parakeet_kernels"
    with Graph(
        "cooperative_gemv",
        input_types=[
            TensorType(dtype, shape=[M, K], device=dev),
            TensorType(dtype, shape=[K, N], device=dev),
        ],
        custom_extensions=[kernels_dir],
    ) as graph:
        x, w = graph.inputs
        y = ops.custom(
            name="cooperative_gemv",
            device=dev,
            values=[x, w],
            out_types=[TensorType(dtype=dtype, shape=[M, N], device=dev)],
        )[0].tensor
        graph.output(y)
    return graph


def bench_model(
    model: Model, x_buf: Buffer, w_buf: Buffer, label: str
) -> tuple[float, float]:
    """Benchmark a compiled model: warmup, then timed iterations."""
    for _ in range(WARMUP):
        model.execute(x_buf, w_buf)

    # Single-call latency.
    times: list[float] = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        model.execute(x_buf, w_buf)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # µs

    median = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]
    mean = sum(times) / len(times)

    # 300x sequential (simulates decode loop).
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        model.execute(x_buf, w_buf)
    t_300 = (time.perf_counter() - t0) * 1e3  # ms

    print(f"\n{label}")
    print(f"  Shape: ({M}, {K}) x ({K}, {N})")
    print(
        f"  Single call:  median={median:.1f}µs"
        f"  mean={mean:.1f}µs  p95={p95:.1f}µs"
    )
    print(
        f"  300x sequential: {t_300:.2f}ms"
        f"  ({t_300 / DECODE_STEPS * 1000:.1f}µs/call)"
    )

    return median, t_300


def verify_correctness(
    model: Model,
    x_np: npt.NDArray[np.float32],
    w_np: npt.NDArray[np.float32],
    x_buf: Buffer,
    w_buf: Buffer,
    label: str,
) -> bool:
    """Check that the model produces correct results."""
    result = model.execute(x_buf, w_buf)[0]
    assert isinstance(result, Buffer)
    result_np = result.to(CPU()).to_numpy().reshape(M, N)
    expected = x_np @ w_np
    max_err = np.max(np.abs(result_np - expected))
    rel_err = max_err / (np.max(np.abs(expected)) + 1e-8)
    status = "PASS" if rel_err < 1e-4 else "FAIL"
    print(
        f"  {label} correctness: max_abs_err={max_err:.2e}"
        f"  rel_err={rel_err:.2e}  [{status}]"
    )
    return status == "PASS"


def main() -> None:
    if accelerator_count() == 0:
        print("ERROR: No GPU detected. This benchmark requires a GPU.")
        return

    device = Accelerator()
    dtype = DType.float32
    print(f"Device: {device}")
    print(f"Shape: ({M}, {K}) x ({K}, {N}) -> ({M}, {N})")
    print(
        f"Warmup: {WARMUP}  Iterations: {ITERS}  Decode steps: {DECODE_STEPS}"
    )

    # Random inputs.
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((M, K)).astype(np.float32)
    w_np = rng.standard_normal((K, N)).astype(np.float32)
    x_buf = Buffer.from_numpy(x_np).to(device)
    w_buf = Buffer.from_numpy(w_np).to(device)

    session = InferenceSession(devices=[device])

    # --- ops.matmul baseline ---
    print("\nCompiling ops.matmul graph...")
    matmul_graph = build_matmul_graph(device, dtype)
    matmul_model = session.load(matmul_graph)

    correct_matmul = verify_correctness(
        matmul_model, x_np, w_np, x_buf, w_buf, "ops.matmul"
    )

    matmul_median, matmul_300 = bench_model(
        matmul_model, x_buf, w_buf, "ops.matmul (built-in dispatch)"
    )

    # --- Custom cooperative GEMV ---
    print("\nCompiling cooperative_gemv graph...")
    gemv_graph = build_custom_gemv_graph(device, dtype)
    gemv_model = session.load(gemv_graph)

    correct_gemv = verify_correctness(
        gemv_model, x_np, w_np, x_buf, w_buf, "cooperative_gemv"
    )

    gemv_median, gemv_300 = bench_model(
        gemv_model, x_buf, w_buf, "cooperative_gemv (single-block, 256 threads)"
    )

    # --- Summary ---
    ratio = gemv_median / matmul_median if matmul_median > 0 else float("inf")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"  ops.matmul:       {matmul_median:.1f}µs/call"
        f"  {matmul_300:.2f}ms/300calls"
    )
    print(
        f"  cooperative_gemv: {gemv_median:.1f}µs/call"
        f"  {gemv_300:.2f}ms/300calls"
    )
    print(f"  Ratio (custom/builtin): {ratio:.2f}x")
    print()

    if not (correct_matmul and correct_gemv):
        print("  VERDICT: FAIL — correctness errors detected")
    elif ratio <= 1.5:
        print(
            f"  VERDICT: GO — custom GEMV is {ratio:.2f}x of built-in"
            f" ({ratio:.2f}x <= 1.5x threshold)"
        )
        print(
            f"  At 6 GEMVs/step: ~{gemv_median * 6:.0f}µs compute"
            f" + ~6µs barriers = ~{gemv_median * 6 + 6:.0f}µs/step"
        )
        print(
            f"  300 steps: ~{(gemv_median * 6 + 6) * 300 / 1000:.1f}ms"
            " (vs current 41ms on A100)"
        )
    elif ratio <= 2.0:
        print(
            f"  VERDICT: MARGINAL — custom GEMV is {ratio:.2f}x of built-in"
            " (between 1.5x and 2.0x)"
        )
        print("  Megakernel may still win due to eliminated dispatch overhead.")
    else:
        print(
            f"  VERDICT: NO-GO — custom GEMV is {ratio:.2f}x of built-in"
            f" (>{2.0}x threshold)"
        )
        print(
            "  Single-block GEMV too slow. Megakernel compute would dominate."
        )


if __name__ == "__main__":
    main()
