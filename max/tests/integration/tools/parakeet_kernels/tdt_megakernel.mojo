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
"""TDT persistent megakernel — full decode loop on-device.

Runs the entire TDT decode loop (~300 steps) in a single persistent GPU kernel.
One thread block (256 threads), one kernel launch, one D2H transfer of the
final token list. Eliminates all per-step Python dispatch overhead.

LSTM state persists in shared memory across iterations (h0_buf, c0_buf,
h1_buf, c1_buf). Gate activations write directly to these buffers — no
save/load copies between iterations.

The decode step body is extracted into _run_decode_step() to avoid code
duplication between SOS and main loop (prevents I-cache thrashing and
reduces register pressure).
"""

import compiler
from std.math import ceildiv
from std.gpu import (
    WARP_SIZE,
    barrier,
    thread_idx_uint as thread_idx,
)
import std.gpu.primitives.warp as warp
from std.gpu.memory import AddressSpace
from std.runtime.asyncrt import DeviceContextPtr
from layout import Layout, LayoutTensor
from tensor import InputTensor, OutputTensor

# TDT decoder dimensions (parakeet-tdt-0.6b-v3).
comptime PRED_HIDDEN = 640
comptime GATES_DIM = 2560  # 4 * PRED_HIDDEN
comptime JOINT_HIDDEN = 640
comptime NUM_TOKENS = 8193  # vocab_size (8192) + 1 (blank)
comptime BLANK_ID = 8192  # = vocab_size
comptime NUM_DURATIONS = 5
comptime OUTPUT_SIZE = 8198  # NUM_TOKENS + NUM_DURATIONS
comptime BLOCK_SIZE = 256
comptime MAX_ENCODER_FRAMES = 400
comptime MAX_OUTPUT_TOKENS = 4096
comptime MAX_SYMBOLS_PER_STEP = 10
comptime MAX_OUTER_ITERS = 10000  # safety limit


# ====================================================================
# Single decode step — called from both SOS and main loop.
# Writes best_token to ctrl[0] and best_dur_idx to ctrl[1].
# Updates h0_buf, c0_buf, h1_buf, c1_buf in-place.
# ====================================================================
@no_inline
def _run_decode_step[
    dtype: DType,
    enc_proj_layout: Layout,
    embedding_layout: Layout,
    w_2560_layout: Layout,
    b_2560_layout: Layout,
    w_pred_layout: Layout,
    b_pred_layout: Layout,
    w_out_layout: Layout,
    b_out_layout: Layout,
](
    tid: Int,
    tok: Int,
    t_val: Int,
    # Shared memory buffers
    x_buf: LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    gates_buf: LayoutTensor[
        dtype,
        Layout(GATES_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    h0_buf: LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    c0_buf: LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    h1_buf: LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    c1_buf: LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    joint_buf: LayoutTensor[
        dtype,
        Layout(JOINT_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    ctrl: LayoutTensor[
        DType.int32, Layout(8), MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    # Global memory inputs
    enc_projected: LayoutTensor[dtype, enc_proj_layout, MutAnyOrigin],
    embedding: LayoutTensor[dtype, embedding_layout, MutAnyOrigin],
    l0_ih_w: LayoutTensor[dtype, w_2560_layout, MutAnyOrigin],
    l0_ih_b: LayoutTensor[dtype, b_2560_layout, MutAnyOrigin],
    l0_hh_w: LayoutTensor[dtype, w_2560_layout, MutAnyOrigin],
    l0_hh_b: LayoutTensor[dtype, b_2560_layout, MutAnyOrigin],
    l1_ih_w: LayoutTensor[dtype, w_2560_layout, MutAnyOrigin],
    l1_ih_b: LayoutTensor[dtype, b_2560_layout, MutAnyOrigin],
    l1_hh_w: LayoutTensor[dtype, w_2560_layout, MutAnyOrigin],
    l1_hh_b: LayoutTensor[dtype, b_2560_layout, MutAnyOrigin],
    pred_w: LayoutTensor[dtype, w_pred_layout, MutAnyOrigin],
    pred_b: LayoutTensor[dtype, b_pred_layout, MutAnyOrigin],
    out_w: LayoutTensor[dtype, w_out_layout, MutAnyOrigin],
    out_b: LayoutTensor[dtype, b_out_layout, MutAnyOrigin],
):
    """Execute one full decode step. Updates h0/c0/h1/c1 in-place.
    Writes best_token to ctrl[0], best_dur_idx to ctrl[1]."""

    # ---- Embedding lookup ----
    for i in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            x_buf[idx] = embedding[tok, idx].cast[dtype]()
    barrier()

    # ---- LSTM Layer 0: ih GEMV ----
    for j_off in range(ceildiv(GATES_DIM, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < GATES_DIM:
            var acc = l0_ih_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = acc + x_buf[k].cast[dtype]() * l0_ih_w[j, k].cast[dtype]()
            gates_buf[j] = acc
    barrier()

    # ---- LSTM Layer 0: hh GEMV-add ----
    for j_off in range(ceildiv(GATES_DIM, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < GATES_DIM:
            var acc = gates_buf[j].cast[dtype]() + l0_hh_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = (
                    acc + h0_buf[k].cast[dtype]() * l0_hh_w[j, k].cast[dtype]()
                )
            gates_buf[j] = acc
    barrier()

    # ---- LSTM Layer 0: gate activations → h0_buf, c0_buf ----
    for i_off in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i_off * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            var gi = gates_buf[idx].cast[DType.float32]()
            var gf = gates_buf[PRED_HIDDEN + idx].cast[DType.float32]()
            var gg = gates_buf[2 * PRED_HIDDEN + idx].cast[DType.float32]()
            var go = gates_buf[3 * PRED_HIDDEN + idx].cast[DType.float32]()
            var one = Float32(1)
            var sig_i = one / (one + std.math.exp(-gi))
            var sig_f = one / (one + std.math.exp(-gf))
            var tanh_g = std.math.tanh(gg)
            var sig_o = one / (one + std.math.exp(-go))
            var c_old = c0_buf[idx].cast[DType.float32]()
            var c_new = sig_f * c_old + sig_i * tanh_g
            var h_new = sig_o * std.math.tanh(c_new)
            h0_buf[idx] = h_new.cast[dtype]()
            c0_buf[idx] = c_new.cast[dtype]()
    barrier()

    # ---- Prepare L1 input: x_buf ← h0_buf (h0_new) ----
    for i in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            x_buf[idx] = h0_buf[idx]
    barrier()

    # ---- LSTM Layer 1: ih GEMV ----
    for j_off in range(ceildiv(GATES_DIM, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < GATES_DIM:
            var acc = l1_ih_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = acc + x_buf[k].cast[dtype]() * l1_ih_w[j, k].cast[dtype]()
            gates_buf[j] = acc
    barrier()

    # ---- LSTM Layer 1: hh GEMV-add ----
    for j_off in range(ceildiv(GATES_DIM, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < GATES_DIM:
            var acc = gates_buf[j].cast[dtype]() + l1_hh_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = (
                    acc + h1_buf[k].cast[dtype]() * l1_hh_w[j, k].cast[dtype]()
                )
            gates_buf[j] = acc
    barrier()

    # ---- LSTM Layer 1: gate activations → h1_buf, c1_buf ----
    for i_off in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i_off * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            var gi = gates_buf[idx].cast[DType.float32]()
            var gf = gates_buf[PRED_HIDDEN + idx].cast[DType.float32]()
            var gg = gates_buf[2 * PRED_HIDDEN + idx].cast[DType.float32]()
            var go = gates_buf[3 * PRED_HIDDEN + idx].cast[DType.float32]()
            var one = Float32(1)
            var sig_i = one / (one + std.math.exp(-gi))
            var sig_f = one / (one + std.math.exp(-gf))
            var tanh_g = std.math.tanh(gg)
            var sig_o = one / (one + std.math.exp(-go))
            var c_old = c1_buf[idx].cast[DType.float32]()
            var c_new = sig_f * c_old + sig_i * tanh_g
            var h_new = sig_o * std.math.tanh(c_new)
            h1_buf[idx] = h_new.cast[dtype]()
            c1_buf[idx] = c_new.cast[dtype]()
    barrier()

    # ---- Joint: pred_proj(h1_buf) → joint_buf ----
    for j_off in range(ceildiv(JOINT_HIDDEN, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < JOINT_HIDDEN:
            var acc = pred_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = acc + h1_buf[k].cast[dtype]() * pred_w[j, k].cast[dtype]()
            joint_buf[j] = acc
    barrier()

    # ---- Joint: relu(enc_projected[0, t, :] + pred_projected) ----
    for i in range(ceildiv(JOINT_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < JOINT_HIDDEN:
            var val = (
                joint_buf[idx].cast[dtype]()
                + enc_projected[0, t_val, idx].cast[dtype]()
            )
            joint_buf[idx] = max(val, Scalar[dtype](0))
    barrier()

    # ---- Fused output_proj GEMV + argmax ----
    for i in range(ceildiv(JOINT_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < JOINT_HIDDEN:
            x_buf[idx] = joint_buf[idx]
    barrier()

    var local_token_max_val = Float32(-1e30)
    var local_token_max_idx = Int32(0)
    var local_dur_max_val = Float32(-1e30)
    var local_dur_max_idx = Int32(0)

    for j_off in range(ceildiv(OUTPUT_SIZE, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < OUTPUT_SIZE:
            var acc = rebind[Float32](out_b[j].cast[DType.float32]())
            for k in range(JOINT_HIDDEN):
                acc = acc + rebind[Float32](
                    x_buf[k].cast[DType.float32]()
                ) * rebind[Float32](out_w[j, k].cast[DType.float32]())
            if j < NUM_TOKENS:
                if acc > local_token_max_val:
                    local_token_max_val = acc
                    local_token_max_idx = Int32(j)
            else:
                if acc > local_dur_max_val:
                    local_dur_max_val = acc
                    local_dur_max_idx = Int32(j - NUM_TOKENS)

    # Warp-level reduction for token argmax
    for offset in range(5):
        var shuf_val = warp.shuffle_down(
            local_token_max_val, UInt32(1 << (4 - offset))
        )
        var shuf_idx = warp.shuffle_down(
            local_token_max_idx, UInt32(1 << (4 - offset))
        )
        if shuf_val > local_token_max_val:
            local_token_max_val = shuf_val
            local_token_max_idx = shuf_idx

    # Warp-level reduction for duration argmax
    for offset in range(5):
        var shuf_val = warp.shuffle_down(
            local_dur_max_val, UInt32(1 << (4 - offset))
        )
        var shuf_idx = warp.shuffle_down(
            local_dur_max_idx, UInt32(1 << (4 - offset))
        )
        if shuf_val > local_dur_max_val:
            local_dur_max_val = shuf_val
            local_dur_max_idx = shuf_idx

    # Lane 0 of each warp writes to gates_buf for cross-warp reduction
    if tid % WARP_SIZE == 0:
        var warp_slot = tid // Int(WARP_SIZE)
        gates_buf[warp_slot] = local_token_max_val.cast[dtype]()
        gates_buf[8 + warp_slot] = Float32(local_token_max_idx).cast[dtype]()
        gates_buf[16 + warp_slot] = local_dur_max_val.cast[dtype]()
        gates_buf[24 + warp_slot] = Float32(local_dur_max_idx).cast[dtype]()
    barrier()

    # Thread 0: final cross-warp argmax → ctrl[0], ctrl[1]
    if tid == 0:
        var best_val = Float32(-1e30)
        var best_idx = Int32(0)
        for w in range(ceildiv(BLOCK_SIZE, Int(WARP_SIZE))):
            var wval = rebind[Float32](gates_buf[w].cast[DType.float32]())
            if wval > best_val:
                best_val = wval
                best_idx = Int32(
                    Int(rebind[Float32](gates_buf[8 + w].cast[DType.float32]()))
                )

        var dur_best_val = Float32(-1e30)
        var dur_best_idx = Int32(0)
        for w in range(ceildiv(BLOCK_SIZE, Int(WARP_SIZE))):
            var dval = rebind[Float32](gates_buf[16 + w].cast[DType.float32]())
            if dval > dur_best_val:
                dur_best_val = dval
                dur_best_idx = Int32(
                    Int(
                        rebind[Float32](gates_buf[24 + w].cast[DType.float32]())
                    )
                )

        ctrl[0] = best_idx
        ctrl[1] = dur_best_idx
    barrier()


# ====================================================================
# Main megakernel function
# ====================================================================
def _tdt_megakernel[
    dtype: DType,
    enc_proj_layout: Layout,
    embedding_layout: Layout,
    durations_layout: Layout,
    w_2560_layout: Layout,
    b_2560_layout: Layout,
    w_pred_layout: Layout,
    b_pred_layout: Layout,
    w_out_layout: Layout,
    b_out_layout: Layout,
    output_tokens_layout: Layout,
    output_count_layout: Layout,
](
    # Outputs
    output_tokens: LayoutTensor[
        DType.int32, output_tokens_layout, MutAnyOrigin
    ],
    output_count: LayoutTensor[DType.int32, output_count_layout, MutAnyOrigin],
    # Inputs
    enc_projected: LayoutTensor[dtype, enc_proj_layout, MutAnyOrigin],
    embedding: LayoutTensor[dtype, embedding_layout, MutAnyOrigin],
    durations: LayoutTensor[DType.int32, durations_layout, MutAnyOrigin],
    l0_ih_w: LayoutTensor[dtype, w_2560_layout, MutAnyOrigin],
    l0_ih_b: LayoutTensor[dtype, b_2560_layout, MutAnyOrigin],
    l0_hh_w: LayoutTensor[dtype, w_2560_layout, MutAnyOrigin],
    l0_hh_b: LayoutTensor[dtype, b_2560_layout, MutAnyOrigin],
    l1_ih_w: LayoutTensor[dtype, w_2560_layout, MutAnyOrigin],
    l1_ih_b: LayoutTensor[dtype, b_2560_layout, MutAnyOrigin],
    l1_hh_w: LayoutTensor[dtype, w_2560_layout, MutAnyOrigin],
    l1_hh_b: LayoutTensor[dtype, b_2560_layout, MutAnyOrigin],
    pred_w: LayoutTensor[dtype, w_pred_layout, MutAnyOrigin],
    pred_b: LayoutTensor[dtype, b_pred_layout, MutAnyOrigin],
    out_w: LayoutTensor[dtype, w_out_layout, MutAnyOrigin],
    out_b: LayoutTensor[dtype, b_out_layout, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)

    # ---- Shared memory ----
    var x_buf = LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var gates_buf = LayoutTensor[
        dtype,
        Layout(GATES_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var h0_buf = LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var c0_buf = LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var h1_buf = LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var c1_buf = LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var joint_buf = LayoutTensor[
        dtype,
        Layout(JOINT_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var ctrl = LayoutTensor[
        DType.int32, Layout(8), MutAnyOrigin, address_space=AddressSpace.SHARED
    ].stack_allocation()

    # ---- Initialize ----
    for i in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            h0_buf[idx] = Scalar[dtype](0)
            c0_buf[idx] = Scalar[dtype](0)
            h1_buf[idx] = Scalar[dtype](0)
            c1_buf[idx] = Scalar[dtype](0)
    if tid == 0:
        ctrl[2] = Int32(0)  # t
        ctrl[3] = Int32(0)  # token_count
        ctrl[4] = Int32(BLANK_ID)  # current_token
    barrier()

    # ---- SOS step ----
    _run_decode_step(
        tid,
        BLANK_ID,
        0,
        x_buf,
        gates_buf,
        h0_buf,
        c0_buf,
        h1_buf,
        c1_buf,
        joint_buf,
        ctrl,
        enc_projected,
        embedding,
        l0_ih_w,
        l0_ih_b,
        l0_hh_w,
        l0_hh_b,
        l1_ih_w,
        l1_ih_b,
        l1_hh_w,
        l1_hh_b,
        pred_w,
        pred_b,
        out_w,
        out_b,
    )
    # SOS: state updated, don't emit or advance t.

    # ---- Main decode loop ----
    var outer_done = False
    var outer_iter = 0
    while outer_iter < MAX_OUTER_ITERS and not outer_done:
        outer_iter += 1
        var t_val = Int(ctrl[2].cast[DType.int32]())
        if t_val >= MAX_ENCODER_FRAMES:
            outer_done = True

        if not outer_done:
            if tid == 0:
                ctrl[5] = Int32(0)  # break_inner
                ctrl[6] = Int32(0)  # symbols_at_t
            barrier()

        var inner_broke = False
        var inner_iter = 0
        while (
            inner_iter < MAX_SYMBOLS_PER_STEP
            and not outer_done
            and not inner_broke
        ):
            _run_decode_step(
                tid,
                Int(ctrl[4].cast[DType.int32]()),
                t_val,
                x_buf,
                gates_buf,
                h0_buf,
                c0_buf,
                h1_buf,
                c1_buf,
                joint_buf,
                ctrl,
                enc_projected,
                embedding,
                l0_ih_w,
                l0_ih_b,
                l0_hh_w,
                l0_hh_b,
                l1_ih_w,
                l1_ih_b,
                l1_hh_w,
                l1_hh_b,
                pred_w,
                pred_b,
                out_w,
                out_b,
            )

            # Thread 0: branch logic
            if tid == 0:
                var token = Int(ctrl[0].cast[DType.int32]())
                var dur_idx = Int(ctrl[1].cast[DType.int32]())
                var duration = Int(durations[dur_idx].cast[DType.int32]())
                var tc = Int(ctrl[3].cast[DType.int32]())

                if token == BLANK_ID:
                    var advance = duration
                    if advance < 1:
                        advance = 1
                    ctrl[2] = Int32(t_val + advance)
                    ctrl[5] = Int32(1)
                else:
                    if tc < MAX_OUTPUT_TOKENS:
                        output_tokens[tc] = Int32(token)
                    ctrl[3] = Int32(tc + 1)
                    ctrl[4] = Int32(token)
                    ctrl[6] = Int32(Int(ctrl[6].cast[DType.int32]()) + 1)
                    if duration > 0:
                        ctrl[2] = Int32(t_val + duration)
                        ctrl[5] = Int32(1)
                    else:
                        ctrl[5] = Int32(0)
            barrier()

            if Int(ctrl[5].cast[DType.int32]()) == 1:
                inner_broke = True

            inner_iter += 1

        # max_symbols_per_step reached without break → advance t by 1
        if not inner_broke and not outer_done:
            if tid == 0:
                ctrl[2] = Int32(Int(ctrl[2].cast[DType.int32]()) + 1)
            barrier()

    # ---- Write output ----
    if tid == 0:
        output_count[0] = ctrl[3]


# ====================================================================
# Custom Op Registration
# ====================================================================
@compiler.register("tdt_megakernel")
struct TDTMegakernel:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output_tokens: OutputTensor[dtype=DType.int32, rank=1, ...],
        output_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        enc_projected: InputTensor[rank=3, ...],
        embedding: InputTensor[dtype=enc_projected.dtype, rank=2, ...],
        durations: InputTensor[dtype=DType.int32, rank=1, ...],
        l0_ih_w: InputTensor[dtype=enc_projected.dtype, rank=2, ...],
        l0_ih_b: InputTensor[dtype=enc_projected.dtype, rank=1, ...],
        l0_hh_w: InputTensor[dtype=enc_projected.dtype, rank=2, ...],
        l0_hh_b: InputTensor[dtype=enc_projected.dtype, rank=1, ...],
        l1_ih_w: InputTensor[dtype=enc_projected.dtype, rank=2, ...],
        l1_ih_b: InputTensor[dtype=enc_projected.dtype, rank=1, ...],
        l1_hh_w: InputTensor[dtype=enc_projected.dtype, rank=2, ...],
        l1_hh_b: InputTensor[dtype=enc_projected.dtype, rank=1, ...],
        pred_w: InputTensor[dtype=enc_projected.dtype, rank=2, ...],
        pred_b: InputTensor[dtype=enc_projected.dtype, rank=1, ...],
        out_w: InputTensor[dtype=enc_projected.dtype, rank=2, ...],
        out_b: InputTensor[dtype=enc_projected.dtype, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            comptime kernel = _tdt_megakernel[
                enc_projected.dtype,
                type_of(enc_projected.to_layout_tensor()).layout,
                type_of(embedding.to_layout_tensor()).layout,
                type_of(durations.to_layout_tensor()).layout,
                type_of(l0_ih_w.to_layout_tensor()).layout,
                type_of(l0_ih_b.to_layout_tensor()).layout,
                type_of(pred_w.to_layout_tensor()).layout,
                type_of(pred_b.to_layout_tensor()).layout,
                type_of(out_w.to_layout_tensor()).layout,
                type_of(out_b.to_layout_tensor()).layout,
                type_of(output_tokens.to_layout_tensor()).layout,
                type_of(output_count.to_layout_tensor()).layout,
            ]
            gpu_ctx.enqueue_function[kernel, kernel](
                output_tokens.to_layout_tensor(),
                output_count.to_layout_tensor(),
                enc_projected.to_layout_tensor(),
                embedding.to_layout_tensor(),
                durations.to_layout_tensor(),
                l0_ih_w.to_layout_tensor(),
                l0_ih_b.to_layout_tensor(),
                l0_hh_w.to_layout_tensor(),
                l0_hh_b.to_layout_tensor(),
                l1_ih_w.to_layout_tensor(),
                l1_ih_b.to_layout_tensor(),
                l1_hh_w.to_layout_tensor(),
                l1_hh_b.to_layout_tensor(),
                pred_w.to_layout_tensor(),
                pred_b.to_layout_tensor(),
                out_w.to_layout_tensor(),
                out_b.to_layout_tensor(),
                grid_dim=1,
                block_dim=BLOCK_SIZE,
            )
        else:
            raise Error("tdt_megakernel is GPU-only")
