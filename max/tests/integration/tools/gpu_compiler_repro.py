"""Minimal reproduction of MAX GPU compiler issues.

Demonstrates two bugs that affect conformer-style ASR models:
1. Flash attention fusion crash with attention bias
2. Conv2d compilation failure when spatial dim is derived from prior conv

All tests PASS on CPU. Failures are GPU-only.

Usage:
    ./bazelw run //max/tests/integration/tools:gpu_compiler_repro
    ./bazelw run //max/tests/integration/tools:gpu_compiler_repro -- --cpu
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

import numpy as np
from max.driver import Buffer, Device, DeviceSpec, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import WeightData
from max.nn.conv import Conv2d


def _make_conv_weights(
    names_and_shapes: list[tuple[str, tuple[int, ...]]],
) -> dict[str, WeightData]:
    """Create random WeightData for named weights."""
    registry: dict[str, WeightData] = {}
    for wname, shape in names_and_shapes:
        arr = np.random.randn(*shape).astype(np.float32)
        registry[wname] = WeightData.from_numpy(arr, name=wname)
    return registry


def run_test(
    name: str,
    device: DeviceRef,
    devices: list[Device],
    build_fn: Any,
    input_shapes: list[tuple[int, ...]],
    compile_only: bool = False,
    weight_specs: list[tuple[str, tuple[int, ...]]] | None = None,
) -> bool:
    """Build, compile, and optionally execute a graph."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    try:
        graph = build_fn(device)
        weights = _make_conv_weights(weight_specs) if weight_specs else {}
        session = InferenceSession(devices=[*devices])
        model = session.load(
            graph, weights_registry=weights if weights else None
        )
        if not compile_only:
            inputs = [
                Buffer.from_numpy(np.random.randn(*s).astype(np.float32)).to(
                    devices[0]
                )
                for s in input_shapes
            ]
            model.execute(*inputs)
        print("  PASS")
        return True
    except Exception:
        print("  FAIL")
        traceback.print_exc()
        return False


# ============================================================================
# Issue 1: Flash attention fusion crashes with attention bias
#
# softmax(scores) @ V gets fused into custom__masked_flash_attention_gpu.
# When scores = Q@K^T + bias (additive attention bias), the fused kernel
# crashes with CUDA_ERROR_MISALIGNED_ADDRESS.
# ============================================================================


def test_attn_no_bias(device: DeviceRef) -> Graph:
    """softmax(Q @ K^T) @ V -- no bias. Should PASS."""
    B, H, S, D = 1, 8, 400, 128
    types = [
        TensorType(DType.float32, shape=[B, H, S, D], device=device),
        TensorType(DType.float32, shape=[B, H, D, S], device=device),
        TensorType(DType.float32, shape=[B, H, S, D], device=device),
    ]
    with Graph("attn_no_bias", input_types=types) as g:
        q, k, v = [i.tensor for i in g.inputs]
        g.output(ops.softmax(q @ k) @ v)
    return g


def test_attn_with_bias(device: DeviceRef) -> Graph:
    """softmax(Q @ K^T + bias) @ V -- additive bias before softmax.

    FAILS on GPU: CUDA_ERROR_MISALIGNED_ADDRESS from flash attention fusion.
    """
    B, H, S, D = 1, 8, 400, 128
    types = [
        TensorType(DType.float32, shape=[B, H, S, D], device=device),
        TensorType(DType.float32, shape=[B, H, D, S], device=device),
        TensorType(DType.float32, shape=[B, H, S, D], device=device),
        TensorType(DType.float32, shape=[B, H, S, S], device=device),
    ]
    with Graph("attn_with_bias", input_types=types) as g:
        q, k, v, bias = [i.tensor for i in g.inputs]
        scores = (q @ k) + bias
        g.output(ops.softmax(scores) @ v)
    return g


def test_attn_manual_softmax(device: DeviceRef) -> Graph:
    """Same as above but manual softmax avoids fusion. Should PASS."""
    B, H, S, D = 1, 8, 400, 128
    types = [
        TensorType(DType.float32, shape=[B, H, S, D], device=device),
        TensorType(DType.float32, shape=[B, H, D, S], device=device),
        TensorType(DType.float32, shape=[B, H, S, D], device=device),
        TensorType(DType.float32, shape=[B, H, S, S], device=device),
    ]
    with Graph("attn_manual", input_types=types) as g:
        q, k, v, bias = [i.tensor for i in g.inputs]
        scores = (q @ k) + bias
        m = ops.max(scores, axis=-1)
        m = ops.reshape(m, [B, H, S, 1])
        w = ops.exp(scores - m)
        s = ops.sum(w, axis=-1)
        s = ops.reshape(s, [B, H, S, 1])
        g.output((w / s) @ v)
    return g


# ============================================================================
# Issue 2: Conv2d fails when spatial dim is derived from prior conv
#
# Conv2d works fine when the spatial dimension is a direct symbolic input.
# But after a stride-2 conv subsamples the input, the derived spatial dim
# causes "All operation types must have the same shape" during compilation.
# ============================================================================


def test_conv_direct_symbolic(device: DeviceRef) -> Graph:
    """Single Conv2d on direct symbolic spatial dim. Should PASS."""
    inp = TensorType(
        DType.float32, shape=[1, "num_frames", 128, 1], device=device
    )
    with Graph("conv_direct", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        conv = Conv2d(
            name="conv1",
            kernel_size=3,
            in_channels=1,
            out_channels=64,
            dtype=DType.float32,
            stride=2,
            padding=1,
            device=device,
            has_bias=True,
            permute=False,
        )
        g.output(ops.relu(conv(x)))
    return g


def test_conv_derived_symbolic(device: DeviceRef) -> Graph:
    """Two Conv2d in sequence -- second operates on derived spatial dim.

    FAILS on GPU: 'All operation types must have the same shape'.
    The first conv subsamples (stride=2), creating a derived dimension.
    The second conv cannot compile on GPU with this derived dimension.
    """
    inp = TensorType(
        DType.float32, shape=[1, "num_frames", 128, 1], device=device
    )
    with Graph("conv_derived", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        conv1 = Conv2d(
            name="conv1",
            kernel_size=3,
            in_channels=1,
            out_channels=64,
            dtype=DType.float32,
            stride=2,
            padding=1,
            device=device,
            has_bias=True,
            permute=False,
        )
        x = ops.relu(conv1(x))
        conv2 = Conv2d(
            name="conv2",
            kernel_size=3,
            in_channels=64,
            out_channels=64,
            dtype=DType.float32,
            stride=1,
            padding=1,
            device=device,
            has_bias=True,
            permute=False,
        )
        g.output(conv2(x))
    return g


def test_conv_derived_fixed(device: DeviceRef) -> Graph:
    """Same as above but fixed input shape. Should PASS.

    Proves the issue is specifically with derived symbolic dimensions,
    not with the conv operation itself.
    """
    inp = TensorType(DType.float32, shape=[1, 800, 128, 1], device=device)
    with Graph("conv_derived_fixed", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        conv1 = Conv2d(
            name="conv1",
            kernel_size=3,
            in_channels=1,
            out_channels=64,
            dtype=DType.float32,
            stride=2,
            padding=1,
            device=device,
            has_bias=True,
            permute=False,
        )
        x = ops.relu(conv1(x))
        conv2 = Conv2d(
            name="conv2",
            kernel_size=3,
            in_channels=64,
            out_channels=64,
            dtype=DType.float32,
            stride=1,
            padding=1,
            device=device,
            has_bias=True,
            permute=False,
        )
        g.output(conv2(x))
    return g


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    use_gpu = "--cpu" not in sys.argv
    if use_gpu:
        try:
            devices = load_devices([DeviceSpec.accelerator()])
            device = DeviceRef.GPU()
            print("Running on GPU\n")
        except Exception:
            print("No GPU, falling back to CPU\n")
            devices = load_devices([DeviceSpec.cpu()])
            device = DeviceRef.CPU()
    else:
        devices = load_devices([DeviceSpec.cpu()])
        device = DeviceRef.CPU()
        print("Running on CPU\n")

    B, H, S, D = 1, 8, 400, 128

    # Conv2d weights: permute=False means RSCF format (R,S,C,F)
    conv1_weights = [
        ("conv1.weight", (3, 3, 1, 64)),
        ("conv1.bias", (64,)),
    ]
    conv2_weights = conv1_weights + [
        ("conv2.weight", (3, 3, 64, 64)),
        ("conv2.bias", (64,)),
    ]

    tests: list[
        tuple[
            str,
            Any,
            list[tuple[int, ...]],
            bool,
            list[tuple[str, tuple[int, ...]]] | None,
        ]
    ] = [
        # Flash attention
        (
            "Attention: no bias (should PASS)",
            test_attn_no_bias,
            [(B, H, S, D), (B, H, D, S), (B, H, S, D)],
            False,
            None,
        ),
        (
            "Attention: with bias (FAILS on GPU: CUDA misaligned addr)",
            test_attn_with_bias,
            [(B, H, S, D), (B, H, D, S), (B, H, S, D), (B, H, S, S)],
            False,
            None,
        ),
        (
            "Attention: manual softmax workaround (should PASS)",
            test_attn_manual_softmax,
            [(B, H, S, D), (B, H, D, S), (B, H, S, D), (B, H, S, S)],
            False,
            None,
        ),
        # Conv2d with derived symbolic dims (compile only)
        (
            "Conv2d: direct symbolic dim (should PASS)",
            test_conv_direct_symbolic,
            [(1, 800, 128, 1)],
            True,
            conv1_weights,
        ),
        (
            "Conv2d: derived symbolic dim (FAILS on GPU)",
            test_conv_derived_symbolic,
            [(1, 800, 128, 1)],
            True,
            conv2_weights,
        ),
        (
            "Conv2d: derived fixed dim (should PASS)",
            test_conv_derived_fixed,
            [(1, 800, 128, 1)],
            True,
            conv2_weights,
        ),
    ]

    results = []
    for name, fn, shapes, compile_only, wspec in tests:
        ok = run_test(name, device, devices, fn, shapes, compile_only, wspec)
        results.append((name, ok))

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    n_fail = sum(1 for _, ok in results if not ok)
    if n_fail:
        print(f"\n{n_fail} test(s) failed.")
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    main()
