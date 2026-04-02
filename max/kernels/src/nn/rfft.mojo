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
"""Forward real FFT kernel using cuFFT."""


from std.ffi import external_call, _get_global_or_null

from _cufft.cufft import (
    cufftCreate,
    cufftEstimate1d,
    cufftExecR2C,
    cufftGetSize,
    cufftHandle,
    cufftMakePlan1d,
    cufftSetAutoAllocation,
    cufftSetStream,
    cufftSetWorkArea,
)
from _cufft.types import Type
from _cufft.utils import check_error
from std.complex import ComplexFloat32
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA
from layout import TileTensor, coord_to_index_list


@always_inline
def global_cache_insert(key: String, value: OpaquePointer):
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(key),
        value,
    )


def _get_fft_workarea(
    buffer_size: Int, ctx: DeviceContext
) raises -> OpaquePointer[MutExternalOrigin]:
    # Include device ID in cache key to ensure per-device workspace buffers.
    var fft_buffer_key = String(
        "CUFFT_BUFFER_PTR_", buffer_size, "_DEV_", ctx.id()
    )

    if lookup := _get_global_or_null(fft_buffer_key):
        # we found the allocated device buffer
        return lookup.unsafe_value()

    # manually allocate the memory on the device, and cache the pointer
    var work_space = ctx.enqueue_create_buffer[DType.uint8](buffer_size)
    var device_ptr = work_space.take_ptr()

    global_cache_insert(
        fft_buffer_key,
        # bitcast the device pointer to a void * to cache it
        device_ptr.bitcast[NoneType](),
    )

    return device_ptr.bitcast[NoneType]().unsafe_origin_cast[
        MutExternalOrigin
    ]()


def _get_fft_plan[
    create_if_not_found: Bool = True
](
    n: Int,
    batch_size: Int,
    workspace_size: Int,
    ctx: DeviceContext,
) raises -> cufftHandle:
    # Include device ID in cache key to ensure per-device cuFFT plans.
    # Use R2C prefix to avoid collisions with irfft's C2R plan cache.
    var cached_plan_key = String(
        "CUFFT_R2C_PLAN_", n, ",", batch_size, "_DEV_", ctx.id()
    )

    if lookup := _get_global_or_null(cached_plan_key):
        # We found the plan in the cache, so just return it
        return cufftHandle(Int(lookup.unsafe_value()))

    comptime if not create_if_not_found:
        # a valid cufft handle is always non-zero
        return cufftHandle(0)

    var plan = cufftHandle(0)
    var mem_size: Int = 0
    check_error(cufftCreate(UnsafePointer(to=plan)))
    check_error(cufftSetAutoAllocation(plan, 0))
    check_error(
        cufftMakePlan1d(
            plan,
            Int32(n),
            Type.CUFFT_R2C,
            Int32(batch_size),
            UnsafePointer(to=mem_size),
        )
    )
    var work_size: Int = 0
    # Get the precise size of the plan, assert that it is less than the allocated size
    check_error(cufftGetSize(plan, UnsafePointer(to=work_size)))
    work_space_ptr = _get_fft_workarea(workspace_size, ctx)

    if work_size > workspace_size:
        raise Error(
            "Need "
            + String(work_size // 1024 // 1024)
            + " MB of buffer allocated for cuFFT."
        )

    check_error(cufftSetWorkArea(plan, work_space_ptr))

    # We want to cache the cuFFT plan to avoid calling high overhead cuda
    # calls each time the plan is created and destroyed
    global_cache_insert(
        cached_plan_key,
        # we are bitcasting the integer plan to a void * to cache it,
        # because that's what KGEN_CompilerRT_InsertGlobal expects.
        UnsafePointer[NoneType, MutExternalOrigin](
            unsafe_from_address=Int(plan)
        ),
    )

    return plan


def _rfft[
    input_type: DType,
    output_type: DType,
](
    input: TileTensor[
        input_type,
        address_space=AddressSpace.GENERIC,
        ...,
    ],
    output: TileTensor[
        mut=True,
        output_type,
        address_space=AddressSpace.GENERIC,
        ...,
    ],
    n: Int,
    buffer_size_mb: Int,
    ctx: DeviceContext,
) raises:
    comptime assert (
        input.rank == output.rank
    ), "Input and output must have the same rank"
    comptime assert (
        input_type == DType.float32
    ), "Only Float32 is supported for RFFT"
    comptime assert (
        output_type == DType.float32
    ), "Only Float32 is supported for RFFT"
    # we allocate 64 MB more than the buffer size because the estimation might
    # not be exact.
    EST_WORKSPACE_SIZE = buffer_size_mb * 1024 * 1024
    ALLOCATED_WORKSPACE_SIZE = (buffer_size_mb + 64) * 1024 * 1024

    axis = input.rank - 1
    cuda_stream = CUDA(ctx.stream())

    # Get input and output dimensions
    input_shape = coord_to_index_list(input.layout.shape_coord())
    # Input is real-valued, so input_size is the real signal length.
    input_size = input_shape[axis]
    # n is the real-domain signal length for cuFFT plan creation.
    signal_length = n if n > 0 else input_size

    # Verify output dimensions
    # Output is interleaved complex: (n // 2 + 1) * 2 floats
    output_shape = coord_to_index_list(output.layout.shape_coord())
    expected_output_size = (signal_length // 2 + 1) * 2
    if output_shape[axis] != expected_output_size:
        raise Error(
            "Output shape mismatch: got "
            + String(output_shape[axis])
            + " expected "
            + String(expected_output_size)
        )

    # Calculate batch size.
    var batch_size = 1
    for i in range(input.rank - 1):
        batch_size *= input_shape[i]

    # skip size estimations if the plan is already cached, as
    # the function call is expensive
    if plan := _get_fft_plan[create_if_not_found=False](
        signal_length, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
    ):
        check_error(cufftSetStream(plan, cuda_stream))
        var input_ptr = input.ptr.bitcast[Float32]()
        var output_ptr = output.ptr.bitcast[ComplexFloat32]()
        check_error(cufftExecR2C(plan, input_ptr, output_ptr))

        return

    var work_size: Int = 0
    check_error(
        cufftEstimate1d(
            Int32(signal_length),
            Type.CUFFT_R2C,
            Int32(batch_size),
            UnsafePointer(to=work_size),
        )
    )

    if work_size < EST_WORKSPACE_SIZE:
        # Create a single cuFFT plan if the workspace size is less than
        # the given buffer size.
        var plan = _get_fft_plan(
            signal_length, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
        )

        # Set up cuda stream.
        # Notice that we do not want to have this part of the cache
        # The stream is set every time the call is executed and we get the
        # stream from the context we are executing within
        check_error(cufftSetStream(plan, cuda_stream))

        var input_ptr = input.ptr.bitcast[Float32]()
        var output_ptr = output.ptr.bitcast[ComplexFloat32]()
        check_error(cufftExecR2C(plan, input_ptr, output_ptr))

    else:
        # If the workspace size is too large, we need to run multiple steps
        # try to find the largest batch size that fits in the workspace
        var reduced_batch_size = batch_size

        while reduced_batch_size > 0:
            reduced_batch_size //= 2
            try:
                check_error(
                    cufftEstimate1d(
                        Int32(signal_length),
                        Type.CUFFT_R2C,
                        Int32(reduced_batch_size),
                        UnsafePointer(to=work_size),
                    )
                )
                if work_size < EST_WORKSPACE_SIZE:
                    break
            except e:
                # Try the next work_size
                pass

        if reduced_batch_size == 0:
            raise Error(
                "FFT output signal size is too large, try to increase the"
                " buffer size."
            )

        # Create cuFFT plan
        var plan = _get_fft_plan(
            signal_length, reduced_batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
        )

        # Set up cuda stream.
        check_error(cufftSetStream(plan, cuda_stream))

        var input_ptr = input.ptr
        var output_ptr = output.ptr

        while batch_size >= reduced_batch_size:
            # Execute the cuFFT plan for the current batch size
            check_error(
                cufftExecR2C(
                    plan,
                    input_ptr.bitcast[Float32](),
                    output_ptr.bitcast[ComplexFloat32](),
                )
            )

            # Update the pointers for the next batch
            batch_size -= reduced_batch_size
            input_ptr += reduced_batch_size * input_shape[axis]
            output_ptr += reduced_batch_size * output_shape[axis]

        if batch_size > 0:
            # Create a new cuFFT plan for the remaining batch size
            # we reuse the allocated workspace, as it is already large enough
            plan = _get_fft_plan(
                signal_length, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
            )
            check_error(cufftSetStream(plan, cuda_stream))

            check_error(
                cufftExecR2C(
                    plan,
                    input_ptr.bitcast[Float32](),
                    output_ptr.bitcast[ComplexFloat32](),
                )
            )


def rfft[
    input_type: DType,
    output_type: DType,
](
    input: TileTensor[
        input_type,
        address_space=AddressSpace.GENERIC,
        ...,
    ],
    output: TileTensor[
        mut=True,
        output_type,
        address_space=AddressSpace.GENERIC,
        ...,
    ],
    n: Int,
    buffer_size_mb: Int,
    ctx: DeviceContext,
) raises:
    """Compute the forward real FFT of the input tensor.

    Currently, only applies it to the last dimension.

    Args:
        input: Real input tensor (TileTensor).
        output: Complex output tensor stored as interleaved Float32 (TileTensor).
        n: Input signal size (if <= 0, uses input.size(axis)).
        buffer_size_mb: Estimated buffer size in MB.
        ctx: Device context.
    """
    # Set `ctx`'s CUcontext as current to satisfy cuFFT's stateful API.
    with ctx.push_context():
        _rfft(input, output, n, buffer_size_mb, ctx)
