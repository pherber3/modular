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
from std.gpu.host import DeviceContext
from std.gpu.host.info import Vendor
from layout import TileTensor, coord, row_major
from nn.rfft import rfft
from std.testing import assert_almost_equal


comptime dtype = DType.float32


def test_rfft_basic[
    batch_size: Int,
    input_size: Int,  # Size of real input
    dtype: DType = DType.float32,
](ctx: DeviceContext) raises:
    """
    Basic RFFT test.

    The input is real Float32 values.

    The output is complex data stored as interleaved Float32 values:
    [real0, imag0, real1, imag1, ...]
    """
    comptime output_size = (input_size // 2 + 1) * 2

    print(
        "== test_rfft_basic: batch_size=",
        batch_size,
        ", input_size=",
        input_size,
        ", output_size=",
        output_size,
    )

    comptime input_shape = coord[batch_size, input_size]()
    comptime output_shape = coord[batch_size, output_size]()

    var input_runtime_layout = row_major(input_shape)
    var output_runtime_layout = row_major(output_shape)

    # Create device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](
        batch_size * input_size
    )
    var output_device = ctx.enqueue_create_buffer[dtype](
        batch_size * output_size
    )

    # Initialize input with all ones — a constant signal.
    # RFFT of a constant signal of length n (all 1.0):
    #   DC bin (index 0): real = n, imag = 0
    #   All other bins: real = 0, imag = 0
    with input_device.map_to_host() as input_host:
        var input_tensor = TileTensor(
            input_host, input_runtime_layout
        ).make_dynamic[DType.int64]()
        for b in range(batch_size):
            for i in range(input_size):
                input_tensor[b, i] = 1.0

    # Initialize output with zeros
    with output_device.map_to_host() as output_host:
        for i in range(len(output_host)):
            output_host[i] = 0

    # Execute RFFT
    rfft[dtype, dtype](
        TileTensor(input_device, input_runtime_layout).make_dynamic[
            DType.int64
        ](),
        TileTensor(output_device, output_runtime_layout).make_dynamic[
            DType.int64
        ](),
        input_size,
        128,  # buffer_size_mb
        ctx,
    )

    ctx.synchronize()

    # Verify results
    with output_device.map_to_host() as output_host:
        var output_tensor = TileTensor(
            output_host, output_runtime_layout
        ).make_dynamic[DType.int64]()

        for b in range(batch_size):
            # DC bin: real part should be input_size, imag part should be 0
            assert_almost_equal(
                output_tensor[b, 0],
                Float32(input_size),
                rtol=0.01,
                msg="DC real component should equal input_size",
            )
            assert_almost_equal(
                output_tensor[b, 1],
                Float32(0.0),
                rtol=0.01,
                atol=1e-5,
                msg="DC imaginary component should be zero",
            )

            # All other frequency bins should be zero
            for i in range(1, input_size // 2 + 1):
                assert_almost_equal(
                    output_tensor[b, 2 * i],
                    Float32(0.0),
                    rtol=0.01,
                    atol=1e-5,
                    msg="Non-DC real component should be zero",
                )
                assert_almost_equal(
                    output_tensor[b, 2 * i + 1],
                    Float32(0.0),
                    rtol=0.01,
                    atol=1e-5,
                    msg="Non-DC imaginary component should be zero",
                )

    print("Succeed")


def main() raises:
    with DeviceContext() as ctx:
        # Check if we're running on an NVIDIA GPU
        if ctx.default_device_info.vendor != Vendor.NVIDIA_GPU:
            print("Skipping cuFFT tests - not running on NVIDIA GPU")
            return

        # Basic tests with different sizes
        test_rfft_basic[batch_size=1, input_size=62](ctx=ctx)

        test_rfft_basic[batch_size=2, input_size=126](ctx=ctx)

        test_rfft_basic[batch_size=4, input_size=254](ctx=ctx)

    # Test with multiple device contexts consecutively
    print("\n== Testing with multiple device contexts ==")

    # First context - default device (GPU 0)
    print("Creating first device context (default device)...")
    with DeviceContext() as ctx1:
        if ctx1.default_device_info.vendor != Vendor.NVIDIA_GPU:
            print("Skipping cuFFT tests - not running on NVIDIA GPU")
            return

        test_rfft_basic[batch_size=1, input_size=62](ctx=ctx1)

    if DeviceContext.number_of_devices() >= 2:
        # Second context - device 1
        print("Creating second device context (device 1)...")
        with DeviceContext(device_id=1) as ctx2:
            if ctx2.default_device_info.vendor != Vendor.NVIDIA_GPU:
                print(
                    "Skipping cuFFT tests on device 1 - not running on NVIDIA"
                    " GPU"
                )
                return

            test_rfft_basic[batch_size=1, input_size=62](ctx=ctx2)

        print("Multiple device context test completed successfully!")
