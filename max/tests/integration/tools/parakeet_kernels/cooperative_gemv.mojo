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
"""Single-block cooperative GEMV for TDT decoder megakernel feasibility test.

Computes y = x @ W where x is (1, K) and W is (K, N), using a single
thread block. This is the core primitive for the persistent megakernel:
the entire decode loop runs on one SM, so every GEMV must execute within
one block.

For TDT decoder shapes:
  - LSTM gates:  (1, 640) x (640, 2560) — the bottleneck, benchmarked here
  - pred_proj:   (1, 640) x (640, 640)
  - output_proj: (1, 640) x (640, 1030)

Strategy:
  - Load input vector x into shared memory (640 x 4B = 2.5 KB)
  - Each thread computes ceil(N/BLOCK_SIZE) output elements
  - Consecutive threads handle consecutive columns for coalesced W reads
  - No cross-block communication needed (single block)
"""

import compiler
from std.math import ceildiv
from std.gpu import (
    barrier,
    thread_idx_uint as thread_idx,
)
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from layout import Layout, LayoutTensor
from std.gpu.memory import AddressSpace
from tensor import InputTensor, OutputTensor

# TDT decoder LSTM gate shape constants.
comptime TDT_K = 640
comptime TDT_N = 2560
comptime BLOCK_SIZE = 256


def _cooperative_gemv_kernel[
    dtype: DType,
    x_layout: Layout,
    w_layout: Layout,
    y_layout: Layout,
](
    x: LayoutTensor[dtype, x_layout, MutAnyOrigin],
    w: LayoutTensor[dtype, w_layout, MutAnyOrigin],
    y: LayoutTensor[dtype, y_layout, MutAnyOrigin],
):
    """Single-block cooperative GEMV: y[1, N] = x[1, K] @ W[K, N].

    All 256 threads cooperate:
      1. Load x into shared memory (3 coalesced iterations for K=640)
      2. Each thread computes 10 output columns (N=2560 / 256 threads)
      3. For each output: dot product of 640 elements from shared x and global W
    """
    var tid = Int(thread_idx.x)

    # Shared memory for input vector broadcast.
    var x_shared = LayoutTensor[
        dtype,
        Layout(TDT_K),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Cooperative load: 640 elements / 256 threads = 2.5 iterations.
    for i in range(ceildiv(TDT_K, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < TDT_K:
            x_shared[idx] = x[0, idx]
    barrier()

    # Each thread computes ceil(2560/256) = 10 output columns.
    # Stride by BLOCK_SIZE so consecutive threads hit consecutive columns
    # → coalesced global memory reads of W.
    for j_off in range(ceildiv(TDT_N, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < TDT_N:
            var acc: w.element_type = 0
            for k in range(TDT_K):
                acc = acc + x_shared[k].cast[dtype]() * w[k, j]
            y[0, j] = acc


@compiler.register("cooperative_gemv")
struct CooperativeGEMV:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2, ...],
        x: InputTensor[dtype=output.dtype, rank=2, ...],
        w: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "gpu":
            var x_lt = x.to_layout_tensor()
            var w_lt = w.to_layout_tensor()
            var y_lt = output.to_layout_tensor()

            var gpu_ctx = ctx.get_device_context()

            comptime kernel = _cooperative_gemv_kernel[
                output.dtype,
                type_of(x_lt).layout,
                type_of(w_lt).layout,
                type_of(y_lt).layout,
            ]
            gpu_ctx.enqueue_function[kernel, kernel](
                x_lt,
                w_lt,
                y_lt,
                grid_dim=1,
                block_dim=BLOCK_SIZE,
            )
        else:
            raise Error(
                "cooperative_gemv is GPU-only (megakernel feasibility test)"
            )
