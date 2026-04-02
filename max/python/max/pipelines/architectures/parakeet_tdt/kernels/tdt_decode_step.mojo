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
"""Single TDT decode step as one fused GPU kernel.

Executes the full decode step — embedding lookup, 2-layer LSTM, joint network,
and argmax — in a single kernel launch on one thread block (256 threads).
All intermediate state lives in shared memory (~22 KB).

This is Phase 2 of the TDT megakernel: verifies that the fused kernel produces
identical outputs to the existing MAX graph-based decoder step.

Inputs (16 tensors):
  enc_t:              (1, 640)       Pre-sliced encoder output at timestep t
  token_id:           (1,)           Previous token (int32)
  lstm_state_packed:  (1, 2560)      [h0, c0, h1, c1] packed
  embedding:          (8193, 640)    Token embedding table
  lstm_0_ih_weight:   (2560, 640)    LSTM L0 input-hidden (original layout)
  lstm_0_ih_bias:     (2560,)
  lstm_0_hh_weight:   (2560, 640)    LSTM L0 hidden-hidden
  lstm_0_hh_bias:     (2560,)
  lstm_1_ih_weight:   (2560, 640)    LSTM L1 input-hidden
  lstm_1_ih_bias:     (2560,)
  lstm_1_hh_weight:   (2560, 640)    LSTM L1 hidden-hidden
  lstm_1_hh_bias:     (2560,)
  pred_proj_weight:   (640, 640)     Joint pred projection
  pred_proj_bias:     (640,)
  output_proj_weight: (8198, 640)    Joint output projection
  output_proj_bias:   (8198,)

  NOTE: Weight matrices are in original (out_dim, in_dim) layout — NOT
  transposed. The kernel reads W[j, k] where j is the output element
  (per-thread) and k is the reduction dimension. This enables vec4-style
  loads along k (contiguous in memory) at the cost of non-coalesced
  cross-thread access, which is fine since weights are L2-resident.

Outputs (2 tensors):
  decisions:          (1, 2)         [best_token, best_dur_idx] (int32)
  lstm_state_new:     (1, 2560)      Updated packed LSTM states
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
comptime NUM_DURATIONS = 5
comptime OUTPUT_SIZE = 8198  # NUM_TOKENS + NUM_DURATIONS
comptime BLOCK_SIZE = 256


def _tdt_decode_step_kernel[
    dtype: DType,
    enc_t_layout: Layout,
    token_id_layout: Layout,
    lstm_state_layout: Layout,
    embedding_layout: Layout,
    w_2560_layout: Layout,  # (2560, 640) — LSTM weight matrices
    b_2560_layout: Layout,  # (2560,) — LSTM biases
    w_pred_layout: Layout,  # (640, 640) — pred_proj weight
    b_pred_layout: Layout,  # (640,) — pred_proj bias
    w_out_layout: Layout,  # (8198, 640) — output_proj weight
    b_out_layout: Layout,  # (8198,) — output_proj bias
    decisions_layout: Layout,
    state_out_layout: Layout,
](
    # Outputs
    decisions_out: LayoutTensor[DType.int32, decisions_layout, MutAnyOrigin],
    state_out: LayoutTensor[dtype, state_out_layout, MutAnyOrigin],
    # Inputs
    enc_t: LayoutTensor[dtype, enc_t_layout, MutAnyOrigin],
    token_id: LayoutTensor[DType.int32, token_id_layout, MutAnyOrigin],
    lstm_state: LayoutTensor[dtype, lstm_state_layout, MutAnyOrigin],
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
    var tid = Int(thread_idx.x)

    # ---- Shared memory allocations ----
    # x_buf: LSTM input vector (embedding or previous layer output)
    var x_buf = LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # gates_buf: LSTM gate values (reused for both layers)
    var gates_buf = LayoutTensor[
        dtype,
        Layout(GATES_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # h_buf / c_buf: current LSTM hidden and cell state
    var h_buf = LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var c_buf = LayoutTensor[
        dtype,
        Layout(PRED_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # joint_buf: pred_proj intermediate and combined relu output (640 floats)
    var joint_buf = LayoutTensor[
        dtype,
        Layout(JOINT_HIDDEN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # ================================================================
    # Step 1: Load h0, c0 into shared memory from packed state
    # ================================================================
    for i in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            h_buf[idx] = lstm_state[0, idx].cast[dtype]()
            c_buf[idx] = lstm_state[0, PRED_HIDDEN + idx].cast[dtype]()
    barrier()

    # ================================================================
    # Step 2: Embedding lookup → x_buf
    # ================================================================
    var tok = Int(token_id[0].cast[DType.int32]())
    for i in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            x_buf[idx] = embedding[tok, idx].cast[dtype]()
    barrier()

    # ================================================================
    # Step 3: LSTM Layer 0
    #   gates = x_buf @ l0_ih_w + l0_ih_b + h_buf @ l0_hh_w + l0_hh_b
    # ================================================================
    # GEMV: gates = x_buf @ l0_ih_w^T + l0_ih_b
    # Weights are (2560, 640) original layout: W[j, k]
    for j_off in range(ceildiv(GATES_DIM, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < GATES_DIM:
            var acc = l0_ih_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = acc + x_buf[k].cast[dtype]() * l0_ih_w[j, k].cast[dtype]()
            gates_buf[j] = acc
    barrier()

    # GEMV-add: gates += h_buf @ l0_hh_w^T + l0_hh_b
    for j_off in range(ceildiv(GATES_DIM, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < GATES_DIM:
            var acc = gates_buf[j].cast[dtype]() + l0_hh_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = acc + h_buf[k].cast[dtype]() * l0_hh_w[j, k].cast[dtype]()
            gates_buf[j] = acc
    barrier()

    # LSTM gate activations: i, f, g, o → update h_buf, c_buf
    # Gates layout: [i(640) | f(640) | g(640) | o(640)]
    for i_off in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i_off * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            var gi = gates_buf[idx].cast[DType.float32]()
            var gf = gates_buf[PRED_HIDDEN + idx].cast[DType.float32]()
            var gg = gates_buf[2 * PRED_HIDDEN + idx].cast[DType.float32]()
            var go = gates_buf[3 * PRED_HIDDEN + idx].cast[DType.float32]()

            # sigmoid(x) = 1 / (1 + exp(-x))
            var one = Float32(1)
            var sig_i = one / (one + std.math.exp(-gi))
            var sig_f = one / (one + std.math.exp(-gf))
            var tanh_g = std.math.tanh(gg)
            var sig_o = one / (one + std.math.exp(-go))

            var c_old = c_buf[idx].cast[DType.float32]()
            var c_new = sig_f * c_old + sig_i * tanh_g
            var h_new = sig_o * std.math.tanh(c_new)

            # Write h0_new, c0_new to global output immediately (final values)
            state_out[0, idx] = h_new.cast[dtype]()
            state_out[0, PRED_HIDDEN + idx] = c_new.cast[dtype]()

            # Also keep in shared for Layer 1 input (h0_new) and overwrite h/c
            h_buf[idx] = h_new.cast[dtype]()
            c_buf[idx] = c_new.cast[dtype]()
    barrier()

    # ================================================================
    # Step 4: LSTM Layer 1
    #   Input = h0_new (in h_buf), state = h1/c1 from packed
    # ================================================================
    # Save h0_new as L1 input into x_buf
    for i in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            x_buf[idx] = h_buf[idx]
    barrier()

    # Load h1, c1 from packed state
    for i in range(ceildiv(PRED_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < PRED_HIDDEN:
            h_buf[idx] = lstm_state[0, 2 * PRED_HIDDEN + idx].cast[dtype]()
            c_buf[idx] = lstm_state[0, 3 * PRED_HIDDEN + idx].cast[dtype]()
    barrier()

    # GEMV: gates = x_buf @ l1_ih_w^T + l1_ih_b
    for j_off in range(ceildiv(GATES_DIM, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < GATES_DIM:
            var acc = l1_ih_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = acc + x_buf[k].cast[dtype]() * l1_ih_w[j, k].cast[dtype]()
            gates_buf[j] = acc
    barrier()

    # GEMV-add: gates += h_buf @ l1_hh_w^T + l1_hh_b
    for j_off in range(ceildiv(GATES_DIM, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < GATES_DIM:
            var acc = gates_buf[j].cast[dtype]() + l1_hh_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = acc + h_buf[k].cast[dtype]() * l1_hh_w[j, k].cast[dtype]()
            gates_buf[j] = acc
    barrier()

    # LSTM L1 gate activations
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

            var c_old = c_buf[idx].cast[DType.float32]()
            var c_new = sig_f * c_old + sig_i * tanh_g
            var h_new = sig_o * std.math.tanh(c_new)

            # Write h1_new, c1_new to global output (final values)
            state_out[0, 2 * PRED_HIDDEN + idx] = h_new.cast[dtype]()
            state_out[0, 3 * PRED_HIDDEN + idx] = c_new.cast[dtype]()

            # Keep h1_new in h_buf for joint network
            h_buf[idx] = h_new.cast[dtype]()
    barrier()

    # ================================================================
    # Step 5: Joint — pred_proj(h1_new) → joint_buf[0:640]
    # pred_w is (640, 640) original layout: W[j, k]
    # ================================================================
    for j_off in range(ceildiv(JOINT_HIDDEN, BLOCK_SIZE)):
        var j = j_off * BLOCK_SIZE + tid
        if j < JOINT_HIDDEN:
            var acc = pred_b[j].cast[dtype]()
            for k in range(PRED_HIDDEN):
                acc = acc + h_buf[k].cast[dtype]() * pred_w[j, k].cast[dtype]()
            joint_buf[j] = acc
    barrier()

    # ================================================================
    # Step 6: Joint — relu(enc_t + pred_projected)
    # ================================================================
    for i in range(ceildiv(JOINT_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < JOINT_HIDDEN:
            var val = joint_buf[idx].cast[dtype]() + enc_t[0, idx].cast[dtype]()
            joint_buf[idx] = max(val, Scalar[dtype](0))  # ReLU
    barrier()

    # ================================================================
    # Steps 7+8 FUSED: output_proj GEMV + argmax
    # out_w is (8198, 640) original layout: W[j, k]
    # K-outer / J-inner loop: reads x_buf[k] once per k-iteration
    # instead of once per (j, k) pair. Each thread handles ~33 output
    # elements with register accumulators.
    # ================================================================
    for i in range(ceildiv(JOINT_HIDDEN, BLOCK_SIZE)):
        var idx = i * BLOCK_SIZE + tid
        if idx < JOINT_HIDDEN:
            x_buf[idx] = joint_buf[idx]
    barrier()

    # Number of output elements per thread.
    comptime OUTS_PER_THREAD = ceildiv(OUTPUT_SIZE, BLOCK_SIZE)  # 33

    # Initialize accumulators with bias.
    var accs = List[Float32](capacity=OUTS_PER_THREAD)
    for j_local in range(OUTS_PER_THREAD):
        var j = j_local * BLOCK_SIZE + tid
        if j < OUTPUT_SIZE:
            accs.append(rebind[Float32](out_b[j].cast[DType.float32]()))
        else:
            accs.append(Float32(0))

    # K-outer loop: one pass over x_buf, accumulate into all outputs.
    for k in range(JOINT_HIDDEN):
        var xk = rebind[Float32](x_buf[k].cast[DType.float32]())
        for j_local in range(OUTS_PER_THREAD):
            var j = j_local * BLOCK_SIZE + tid
            if j < OUTPUT_SIZE:
                accs[j_local] = accs[j_local] + xk * rebind[Float32](
                    out_w[j, k].cast[DType.float32]()
                )

    # Per-thread argmax over accumulated logits.
    var local_token_max_val = Float32(-1e30)
    var local_token_max_idx = Int32(0)
    var local_dur_max_val = Float32(-1e30)
    var local_dur_max_idx = Int32(0)

    for j_local in range(OUTS_PER_THREAD):
        var j = j_local * BLOCK_SIZE + tid
        if j < OUTPUT_SIZE:
            var val = accs[j_local]
            if j < NUM_TOKENS:
                if val > local_token_max_val:
                    local_token_max_val = val
                    local_token_max_idx = Int32(j)
            else:
                if val > local_dur_max_val:
                    local_dur_max_val = val
                    local_dur_max_idx = Int32(j - NUM_TOKENS)

    # Warp-level reduction for token argmax
    for offset in range(5):  # log2(32) = 5
        var shuf_val = warp.shuffle_down(
            local_token_max_val, UInt32(1 << (4 - offset))
        )
        var shuf_idx = warp.shuffle_down(
            local_token_max_idx, UInt32(1 << (4 - offset))
        )
        if shuf_val > local_token_max_val:
            local_token_max_val = shuf_val
            local_token_max_idx = shuf_idx

    # Warp-level reduction for duration argmax (register-only, no barrier needed)
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

    # Lane 0 of each warp writes both token and duration results, single barrier
    if tid % WARP_SIZE == 0:
        var warp_slot = tid // Int(WARP_SIZE)
        gates_buf[warp_slot] = local_token_max_val.cast[dtype]()
        gates_buf[8 + warp_slot] = Float32(local_token_max_idx).cast[dtype]()
        gates_buf[16 + warp_slot] = local_dur_max_val.cast[dtype]()
        gates_buf[24 + warp_slot] = Float32(local_dur_max_idx).cast[dtype]()
    barrier()

    if tid == 0:
        # Final token argmax across 8 warps
        var best_val = Float32(-1e30)
        var best_idx = Int32(0)
        for w in range(ceildiv(BLOCK_SIZE, Int(WARP_SIZE))):
            var wval = rebind[Float32](gates_buf[w].cast[DType.float32]())
            if wval > best_val:
                best_val = wval
                best_idx = Int32(
                    Int(rebind[Float32](gates_buf[8 + w].cast[DType.float32]()))
                )
        decisions_out[0, 0] = best_idx

        # Final duration argmax across 8 warps
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
        decisions_out[0, 1] = dur_best_idx


# ====================================================================
# Custom Op Registration
# ====================================================================


@compiler.register("tdt_decode_step")
struct TDTDecodeStep:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        # Outputs (listed first per custom op convention)
        decisions_out: OutputTensor[dtype=DType.int32, rank=2, ...],
        state_out: OutputTensor[rank=2, ...],
        # Inputs
        enc_t: InputTensor[dtype=state_out.dtype, rank=2, ...],
        token_id: InputTensor[dtype=DType.int32, rank=1, ...],
        lstm_state: InputTensor[dtype=state_out.dtype, rank=2, ...],
        embedding: InputTensor[dtype=state_out.dtype, rank=2, ...],
        l0_ih_w: InputTensor[dtype=state_out.dtype, rank=2, ...],
        l0_ih_b: InputTensor[dtype=state_out.dtype, rank=1, ...],
        l0_hh_w: InputTensor[dtype=state_out.dtype, rank=2, ...],
        l0_hh_b: InputTensor[dtype=state_out.dtype, rank=1, ...],
        l1_ih_w: InputTensor[dtype=state_out.dtype, rank=2, ...],
        l1_ih_b: InputTensor[dtype=state_out.dtype, rank=1, ...],
        l1_hh_w: InputTensor[dtype=state_out.dtype, rank=2, ...],
        l1_hh_b: InputTensor[dtype=state_out.dtype, rank=1, ...],
        pred_w: InputTensor[dtype=state_out.dtype, rank=2, ...],
        pred_b: InputTensor[dtype=state_out.dtype, rank=1, ...],
        out_w: InputTensor[dtype=state_out.dtype, rank=2, ...],
        out_b: InputTensor[dtype=state_out.dtype, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()

            comptime kernel = _tdt_decode_step_kernel[
                state_out.dtype,
                type_of(enc_t.to_layout_tensor()).layout,
                type_of(token_id.to_layout_tensor()).layout,
                type_of(lstm_state.to_layout_tensor()).layout,
                type_of(embedding.to_layout_tensor()).layout,
                type_of(l0_ih_w.to_layout_tensor()).layout,
                type_of(l0_ih_b.to_layout_tensor()).layout,
                type_of(pred_w.to_layout_tensor()).layout,
                type_of(pred_b.to_layout_tensor()).layout,
                type_of(out_w.to_layout_tensor()).layout,
                type_of(out_b.to_layout_tensor()).layout,
                type_of(decisions_out.to_layout_tensor()).layout,
                type_of(state_out.to_layout_tensor()).layout,
            ]
            gpu_ctx.enqueue_function[kernel, kernel](
                decisions_out.to_layout_tensor(),
                state_out.to_layout_tensor(),
                enc_t.to_layout_tensor(),
                token_id.to_layout_tensor(),
                lstm_state.to_layout_tensor(),
                embedding.to_layout_tensor(),
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
            raise Error(
                "tdt_decode_step is GPU-only (megakernel feasibility test)"
            )
