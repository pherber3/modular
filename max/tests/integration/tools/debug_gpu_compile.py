"""Debug script to isolate GPU compilation failures in Parakeet encoder.

Builds and compiles subgraphs one at a time to find which op fails.

Usage:
    ./bazelw run //max/tests/integration/tools:debug_gpu_compile
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from max.driver import Device, DeviceSpec, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def try_compile(
    name: str,
    build_fn: Callable[[DeviceRef], Graph],
    device: DeviceRef,
    devices: list[Device],
) -> bool:
    """Try to build and compile a graph, report success/failure."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    try:
        graph = build_fn(device)
        session = InferenceSession(devices=[*devices])
        session.load(graph)
        print(f"  OK: {name} compiled successfully")
        return True
    except Exception:
        print(f"  FAIL: {name}")
        traceback.print_exc()
        return False


def test_conv1d_permute_false(device: DeviceRef) -> Graph:
    """Conv1D with permute=False, num_groups=1."""
    from max.nn.conv import Conv1D

    inp = TensorType(DType.float32, shape=[1, 100, 1024], device=device)
    with Graph("test_conv1d_no_permute", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        conv = Conv1D(
            kernel_size=1,
            in_channels=1024,
            out_channels=2048,
            dtype=DType.float32,
            stride=1,
            padding=0,
            device=device,
            has_bias=True,
            permute=False,
        )
        # Initialize with random weights
        g.output(conv(x))
    return g


def test_conv1d_grouped(device: DeviceRef) -> Graph:
    """Conv1D with permute=False, num_groups=channels (depthwise)."""
    from max.nn.conv import Conv1D

    inp = TensorType(DType.float32, shape=[1, 100, 1024], device=device)
    with Graph("test_conv1d_grouped", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        conv = Conv1D(
            kernel_size=9,
            in_channels=1024,
            out_channels=1024,
            dtype=DType.float32,
            stride=1,
            padding=4,
            num_groups=1024,
            device=device,
            has_bias=True,
            permute=False,
        )
        g.output(conv(x))
    return g


def test_conv1d_permute_true(device: DeviceRef) -> Graph:
    """Conv1D with permute=True, num_groups=1."""
    from max.nn.conv import Conv1D

    inp = TensorType(DType.float32, shape=[1, 1024, 100], device=device)
    with Graph("test_conv1d_permute", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        conv = Conv1D(
            kernel_size=1,
            in_channels=1024,
            out_channels=2048,
            dtype=DType.float32,
            stride=1,
            padding=0,
            device=device,
            has_bias=True,
            permute=True,
        )
        g.output(conv(x))
    return g


def test_conv1d_grouped_permute_true(device: DeviceRef) -> Graph:
    """Conv1D with permute=True, num_groups=channels (depthwise)."""
    from max.nn.conv import Conv1D

    inp = TensorType(DType.float32, shape=[1, 1024, 100], device=device)
    with Graph("test_conv1d_grouped_permute", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        conv = Conv1D(
            kernel_size=9,
            in_channels=1024,
            out_channels=1024,
            dtype=DType.float32,
            stride=1,
            padding=4,
            num_groups=1024,
            device=device,
            has_bias=True,
            permute=True,
        )
        g.output(conv(x))
    return g


def test_conv2d_grouped(device: DeviceRef) -> Graph:
    """Conv2d grouped (subsampling style), permute=False."""
    from max.nn.conv import Conv2d

    inp = TensorType(DType.float32, shape=[1, 100, 16, 256], device=device)
    with Graph("test_conv2d_grouped", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        conv = Conv2d(
            kernel_size=3,
            in_channels=256,
            out_channels=256,
            dtype=DType.float32,
            stride=2,
            padding=1,
            num_groups=256,
            device=device,
            has_bias=True,
            permute=False,
        )
        g.output(conv(x))
    return g


def test_batchnorm(device: DeviceRef) -> Graph:
    """BatchNorm1d."""
    from max.pipelines.architectures.parakeet.layers import BatchNorm1d

    inp = TensorType(DType.float32, shape=[1, 1024, 100], device=device)
    with Graph("test_batchnorm", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        bn = BatchNorm1d(1024, dtype=DType.float32, device=device)
        g.output(bn(x))
    return g


def test_positional_encoding(device: DeviceRef) -> Graph:
    """Relative positional encoding."""
    from max.pipelines.architectures.parakeet.positional import (
        compute_relative_position_encoding,
    )

    inp = TensorType(DType.float32, shape=[1, "seq_len", 1024], device=device)
    with Graph("test_positional", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        pos = compute_relative_position_encoding(
            x.shape[1], 1024, DType.float32, device  # type: ignore[arg-type]
        )
        g.output(pos)
    return g


def test_rel_shift(device: DeviceRef) -> Graph:
    """rel_shift operation."""
    from max.pipelines.architectures.parakeet.positional import rel_shift

    inp = TensorType(DType.float32, shape=[1, 8, "q", "p"], device=device)
    with Graph("test_rel_shift", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        g.output(rel_shift(x))
    return g


def _make_mock_config(device: DeviceRef) -> Any:
    """Create a mock config with the attributes ParakeetModelConfig reads."""
    from types import SimpleNamespace

    encoder_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=8,
        num_mel_bins=128,
        subsampling_conv_channels=256,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        subsampling_factor=8,
        conv_kernel_size=9,
        attention_bias=False,
        scale_input=True,
        max_position_embeddings=5000,
        convolution_bias=True,
    )
    hf_cfg = SimpleNamespace(
        encoder_config=encoder_cfg,
        architectures=["ParakeetForCTC"],
    )
    pipeline_cfg = SimpleNamespace()
    from max.pipelines.architectures.parakeet.model_config import (
        ParakeetModelConfig,
    )

    return ParakeetModelConfig(
        dtype=DType.float32,
        device=device,
        huggingface_config=hf_cfg,
        pipeline_config=pipeline_cfg,  # type: ignore[arg-type]
    )


def test_attention(device: DeviceRef) -> Graph:
    """Single attention layer."""
    from max.pipelines.architectures.parakeet.layers import ParakeetAttention

    config = _make_mock_config(device)
    inp_h = TensorType(DType.float32, shape=[1, "seq", 1024], device=device)
    inp_p = TensorType(DType.float32, shape=[1, "pos", 1024], device=device)
    with Graph("test_attention", input_types=[inp_h, inp_p]) as g:
        h = g.inputs[0].tensor
        p = g.inputs[1].tensor
        attn = ParakeetAttention(config, 0)
        g.output(attn(h, p))
    return g


def test_split_glu(device: DeviceRef) -> Graph:
    """Split + GLU (sigmoid gate)."""
    inp = TensorType(DType.float32, shape=[1, "seq", 2048], device=device)
    with Graph("test_split_glu", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        half = x.shape[2] // 2
        a, b = ops.split(x, [half, half], axis=2)
        g.output(a * ops.sigmoid(b))
    return g


def test_scalar_mul(device: DeviceRef) -> Graph:
    """Python float * device tensor."""
    inp = TensorType(DType.float32, shape=[1, "seq", 1024], device=device)
    with Graph("test_scalar_mul", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        g.output(x * 0.5)
    return g


def test_full_encoder(device: DeviceRef) -> Graph:
    """Full Parakeet encoder (1 layer only for speed)."""
    from max.pipelines.architectures.parakeet.encoder import ParakeetEncoder

    config = _make_mock_config(device)
    inp = TensorType(DType.float32, shape=[1, "frames", 128], device=device)
    with Graph("test_encoder_1layer", input_types=[inp]) as g:
        x = g.inputs[0].tensor
        enc = ParakeetEncoder(config)
        g.output(enc(x))
    return g


def main() -> None:
    devices = load_devices([DeviceSpec.accelerator()])
    device = DeviceRef.GPU()

    tests = [
        ("scalar_mul", test_scalar_mul),
        ("batchnorm", test_batchnorm),
        ("split_glu", test_split_glu),
        ("conv1d_permute_false", test_conv1d_permute_false),
        ("conv1d_grouped_permute_false", test_conv1d_grouped),
        ("conv1d_permute_true", test_conv1d_permute_true),
        ("conv1d_grouped_permute_true", test_conv1d_grouped_permute_true),
        ("conv2d_grouped", test_conv2d_grouped),
        ("positional_encoding", test_positional_encoding),
        ("rel_shift", test_rel_shift),
        ("attention", test_attention),
        ("full_encoder_1layer", test_full_encoder),
    ]

    results = []
    for name, fn in tests:
        ok = try_compile(name, fn, device, devices)
        results.append((name, ok))

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    main()
