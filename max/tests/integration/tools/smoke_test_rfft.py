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
"""Smoke test for ops.rfft on GPU — compare MAX vs torch.fft.rfft."""

from __future__ import annotations

import sys

import torch
import torch.utils.dlpack
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_rfft(
    session: InferenceSession,
    input_shape: tuple[int, ...],
    n: int | None,
    axis: int,
    normalization: str,
) -> bool:
    x = torch.randn(*input_shape, dtype=torch.float32).to("cuda")

    with Graph(
        "rfft",
        input_types=(
            TensorType(DType.float32, x.shape, DeviceRef.GPU()),
        ),
    ) as graph:
        out = ops.rfft(graph.inputs[0].tensor, n, axis, normalization)
        graph.output(out)

    model = session.load(graph)
    max_out = torch.utils.dlpack.from_dlpack(model(x)[0])

    torch_out = torch.view_as_real(
        torch.fft.rfft(x, n=n, dim=axis, norm=normalization)
    )

    match = torch.allclose(max_out, torch_out, rtol=1e-5, atol=1e-5)
    max_diff = (max_out - torch_out).abs().max().item()

    label = f"shape={input_shape}, n={n}, axis={axis}, norm={normalization}"
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {label}  (max_diff={max_diff:.2e})")
    return match


def main() -> None:
    print("ops.rfft smoke test")
    print("=" * 60)

    session = InferenceSession()

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
        if not test_rfft(session, input_shape, n, axis, norm):
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
