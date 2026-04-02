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

"""Op implementation for rfft."""

from __future__ import annotations

from max.dtype import DType

from ..dim import Dim, StaticDim
from ..type import DeviceKind, TensorType
from ..value import StrongTensorValueLike, TensorValue
from .constant import constant
from .custom import custom
from .elementwise import sqrt
from .irfft import Normalization
from .pad import pad
from .transpose import transpose


def _process_input_signal(
    input_tensor: TensorValue, n: int, axis: int
) -> TensorValue:
    """Resizes input tensor to the required signal size along the FFT axis."""
    axis = axis % input_tensor.rank
    axis_dim = input_tensor.shape[axis]
    if not isinstance(axis_dim, StaticDim):
        raise ValueError(f"Axis dimension must be static, got {axis_dim}.")

    if axis_dim > n:
        # Slice the input tensor to length n.
        index = [slice(None)] * input_tensor.rank
        index[axis] = slice(0, n)
        input_tensor = input_tensor[tuple(index)]
    elif axis_dim < n:
        # Pad the input tensor with zeros to length n.
        paddings = [0] * 2 * input_tensor.rank
        paddings[axis * 2 + 1] = n - int(axis_dim)
        input_tensor = pad(input_tensor, paddings=paddings)
    return input_tensor


def rfft(  # noqa: ANN201
    input_tensor: StrongTensorValueLike,
    n: int | None = None,
    axis: int = -1,
    normalization: Normalization | str = Normalization.BACKWARD,
    buffer_size_mb: int = 512,
):
    """Compute the forward real FFT of the input tensor.

    Args:
        input_tensor: The real-valued input tensor to compute the FFT of.
        n: The signal length. The input tensor will be padded or truncated to
            length `n` along the specified axis. If None, uses the axis
            dimension size.
        axis: The axis to compute the FFT along.
        normalization: The normalization to apply to the output tensor.
            Can be "backward", "ortho", or "forward". When "backward", no
            normalization is applied. When "ortho", the output is divided by
            `sqrt(n)`. When "forward", the output is divided by `n`.
        buffer_size_mb: The estimated size of a persistent buffer to use for
            storage of intermediate results. Needs to be the same across multiple
            calls to `rfft` within the same graph. Otherwise, multiple buffers
            will be allocated.

    Returns:
        The forward real FFT of the input tensor. The output shape is the same
        as the input shape except the FFT axis is replaced by `n // 2 + 1`,
        and a trailing dimension of size 2 is added for the interleaved
        real and imaginary parts.
    """
    input_tensor = TensorValue(input_tensor)

    if not input_tensor.dtype == DType.float32:
        raise ValueError(
            f"Input tensor must be of type float32, got {input_tensor.dtype}."
        )
    if input_tensor.device.device_type != DeviceKind.GPU:
        raise ValueError("RFFT is currently only supported on GPU.")

    # Transpose the input tensor so that the FFT axis is the last axis.
    orig_axis = axis % input_tensor.rank
    axis = input_tensor.rank - 1
    if orig_axis != axis:
        input_tensor = transpose(input_tensor, orig_axis, axis)

    input_shape = list(input_tensor.shape)

    if not n:
        n = int(input_shape[-1])
    input_tensor = _process_input_signal(input_tensor, n, axis=axis)

    # Output shape for the custom kernel: interleaved complex on the last axis.
    kernel_output_shape = list(input_tensor.shape)
    kernel_output_shape[-1] = Dim((n // 2 + 1) * 2)

    rfft_out = custom(
        "rfft",
        input_tensor.device,
        [input_tensor],
        [
            TensorType(
                dtype=input_tensor.dtype,
                shape=kernel_output_shape,
                device=input_tensor.device,
            )
        ],
        {"n": n, "buffer_size_mb": buffer_size_mb},
    )[0].tensor

    # Normalization for forward FFT:
    # "backward" (default) = no normalization
    # "ortho" = divide by sqrt(n)
    # "forward" = divide by n
    if normalization == Normalization.BACKWARD:
        pass
    elif normalization == Normalization.ORTHO:
        rfft_out /= sqrt(constant(n, input_tensor.dtype, input_tensor.device))
    elif normalization == Normalization.FORWARD:
        rfft_out /= n
    else:
        raise ValueError(f"Invalid normalization: {normalization}")

    # Transpose back to original axis if needed (before reshaping to complex).
    if orig_axis != axis:
        rfft_out = transpose(rfft_out, axis, orig_axis)
        # After transpose, the interleaved complex dim is at orig_axis.
        complex_axis = orig_axis
    else:
        complex_axis = rfft_out.rank - 1

    # Reshape interleaved output (..., (n//2+1)*2, ...) to (..., n//2+1, 2)
    # by splitting the complex axis into two dimensions.
    new_shape = list(rfft_out.shape)
    new_shape[complex_axis] = Dim(n // 2 + 1)
    new_shape.insert(complex_axis + 1, Dim(2))
    rfft_out = rfft_out.reshape(tuple(new_shape))

    return rfft_out
