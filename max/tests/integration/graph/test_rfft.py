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

from __future__ import annotations

import max.driver as md
import pytest
import torch
import torch.utils.dlpack
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def max_rfft(
    session: InferenceSession,
    input_tensor: torch.Tensor,
    n: int | None,
    axis: int,
    normalization: str,
) -> torch.Tensor:
    with Graph(
        "rfft",
        input_types=(
            TensorType(DType.float32, input_tensor.shape, DeviceRef.GPU()),
        ),
    ) as graph:
        output = ops.rfft(graph.inputs[0].tensor, n, axis, normalization)
        graph.output(output)
    model = session.load(graph)
    output = model(input_tensor)
    output = torch.utils.dlpack.from_dlpack(output[0])
    return output


def torch_rfft(
    input_tensor: torch.Tensor, n: int | None, axis: int, normalization: str
) -> torch.Tensor:
    output = torch.fft.rfft(input_tensor, n=n, dim=axis, norm=normalization)
    return torch.view_as_real(output)


@pytest.mark.parametrize(
    "input_shape,n,axis,normalization",
    [
        ((5, 10, 15), 3, -1, "backward"),
        ((5, 10, 15), 20, 0, "ortho"),
        ((5, 10, 15), None, 1, "forward"),
        ((64,), None, -1, "backward"),
        ((2, 256), 128, -1, "ortho"),
    ],
)
def test_rfft(
    session: InferenceSession,
    input_shape: tuple[int, ...],
    n: int | None,
    axis: int,
    normalization: str,
) -> None:
    assert md.accelerator_count() > 0, "No GPU available"
    assert (
        md.accelerator_api() == "cuda"
    ), "NVIDIA GPUs are required for this test."
    input_tensor = torch.randn(*input_shape, dtype=torch.float32).to("cuda")
    max_out = max_rfft(session, input_tensor, n, axis, normalization)
    torch_out = torch_rfft(input_tensor, n, axis, normalization)

    torch.testing.assert_close(
        torch_out,
        max_out,
        rtol=1e-6,
        atol=2 * torch.finfo(torch.float32).eps,
    )
