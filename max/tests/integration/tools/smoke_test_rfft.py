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
"""Smoke test for ops.rfft on GPU — compare MAX vs numpy.fft.rfft.

Uses numpy for reference since Bazel bundles CPU-only torch.
"""

from __future__ import annotations

import sys

import max.driver as md
import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def numpy_rfft(
    x: np.ndarray, n: int | None, axis: int, normalization: str
) -> np.ndarray:
    """Compute rfft with numpy and return as interleaved real/imag."""
    result = np.fft.rfft(x, n=n, axis=axis, norm=normalization)  # type: ignore[arg-type]
    # Normalize axis, then insert complex dim (size 2) right after it.
    norm_axis = axis % x.ndim
    return np.stack(
        [result.real, result.imag], axis=norm_axis + 1
    ).astype(np.float32)


def test_rfft(
    session: InferenceSession,
    gpu: md.Device,
    input_shape: tuple[int, ...],
    n: int | None,
    axis: int,
    normalization: str,
) -> bool:
    x = np.random.randn(*input_shape).astype(np.float32)

    with Graph(
        "rfft",
        input_types=(
            TensorType(DType.float32, x.shape, DeviceRef.GPU()),
        ),
    ) as graph:
        out = ops.rfft(graph.inputs[0].tensor, n, axis, normalization)
        graph.output(out)

    model = session.load(graph)
    x_gpu = Buffer.from_numpy(x).to(gpu)
    result_gpu = model(x_gpu)[0]
    result_cpu = result_gpu.to(md.CPU())
    max_out = np.from_dlpack(result_cpu).copy()

    np_out = numpy_rfft(x, n, axis, normalization)

    max_diff = np.abs(max_out - np_out).max()
    match = max_diff < 1e-4

    label = f"shape={input_shape}, n={n}, axis={axis}, norm={normalization}"
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {label}  (max_diff={max_diff:.2e})")
    if not match:
        print(f"    MAX shape={max_out.shape}, numpy shape={np_out.shape}")
    return match


def main() -> None:
    print("ops.rfft smoke test")
    print("=" * 60)

    devices: list[md.Device] = []
    gpu = None
    for i in range(md.accelerator_count()):
        dev = md.Accelerator(i)
        devices.append(dev)
        if gpu is None:
            gpu = dev
    devices.append(md.CPU())
    session = InferenceSession(devices=devices)
    print(f"Devices: {devices}")

    if gpu is None:
        print("ERROR: No GPU available")
        sys.exit(1)
    print()

    cases = [
        ((5, 10, 15), 3, -1, "backward"),
        ((5, 10, 15), 20, 0, "ortho"),
        ((5, 10, 15), None, 1, "forward"),
        ((64,), None, -1, "backward"),
        ((2, 256), 128, -1, "ortho"),
        ((1, 480, 80), None, 1, "backward"),  # Parakeet-like shape
    ]

    all_pass = True
    for input_shape, n, axis, norm in cases:
        if not test_rfft(session, gpu, input_shape, n, axis, norm):
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
