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
"""Profile individual components of a TDT decoder step.

Isolates the per-step overhead to identify what's slow:
  1. Buffer.from_numpy().to(device) for time index
  2. model.execute() call
  3. np.from_dlpack(.to(cpu)) readback
  4. Python loop overhead (no-op baseline)

Usage:
    ./bazelw run //max/tests/integration/tools:profile_decoder_step -- --device gpu
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from max.driver import Buffer, Device, DeviceSpec, load_devices
from max.engine import InferenceSession

N_WARMUP = 10
N_TIMED = 300


@dataclass
class StepResults:
    """Profiling results for a single decoder step."""

    execute_only_mean: float
    execute_only_std: float
    full_step_mean: float
    full_step_std: float
    buf_create_mean: float
    buf_create_std: float
    readback_mean: float
    readback_std: float


def time_buffer_creation(device: Device, n: int) -> float:
    """Time creating a small int32 buffer and transferring to device."""
    total = 0.0
    for i in range(n):
        t0 = time.perf_counter()
        Buffer.from_numpy(np.array([i], dtype=np.int32)).to(device)
        t1 = time.perf_counter()
        total += t1 - t0
    return total / n * 1000


def time_readback(device: Device, n: int) -> float:
    """Time reading a small scalar buffer back from device."""
    cpu_device = load_devices([DeviceSpec.cpu()])[0]
    buf = Buffer.from_numpy(np.array([42], dtype=np.int32)).to(device)
    total = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        int(np.from_dlpack(buf.to(cpu_device))[0])
        t1 = time.perf_counter()
        total += t1 - t0
    return total / n * 1000


def time_model_execute(
    device: Device, n_warmup: int, n_timed: int
) -> StepResults:
    """Profile the actual TDT decoder step graph execution."""
    import huggingface_hub
    from max.graph import DeviceRef
    from max.pipelines.architectures.parakeet_tdt.decoder_graph import (
        build_decoder_step_graph,
        build_projection_graph,
        convert_decoder_state_dict,
    )
    from transformers import AutoConfig

    model_id = "pherber3/parakeet-tdt-0.6b-v3"
    print(f"  Loading TDT model ({model_id})...")

    hf_config = AutoConfig.from_pretrained(model_id)

    npz_path = huggingface_hub.hf_hub_download(model_id, "decoder_joint.npz")
    npz_weights = dict(np.load(npz_path))
    proj_dict, pred_dict, joint_dict = convert_decoder_state_dict(npz_weights)

    # Extract config values
    from types import SimpleNamespace

    decoder_config = hf_config.decoder_config
    if isinstance(decoder_config, dict):
        decoder_config = SimpleNamespace(**decoder_config)
    joint_config = hf_config.joint_config
    if isinstance(joint_config, dict):
        joint_config = SimpleNamespace(**joint_config)

    pred_hidden: int = decoder_config.pred_hidden
    joint_hidden: int = joint_config.joint_hidden
    vocab_size: int = hf_config.vocab_size
    durations: list[int] = hf_config.tdt_durations

    device_ref = DeviceRef(
        device_type="gpu" if not device.is_host else "cpu",
        id=0,
    )

    # Minimal config duck-type matching TDTModelConfig interface
    config = SimpleNamespace(
        pred_hidden=pred_hidden,
        joint_hidden=joint_hidden,
        hidden_size=1024,
        vocab_size=vocab_size,
        tdt_durations=durations,
        pred_rnn_layers=2,
        device=device_ref,
    )

    session = InferenceSession(devices=[device])

    print("  Compiling projection graph...")
    proj_graph = build_projection_graph(config, proj_dict)  # type: ignore[arg-type]
    session.load(proj_graph, weights_registry=proj_dict)

    print("  Compiling decoder step graph...")
    dec_graph = build_decoder_step_graph(config, pred_dict, joint_dict)  # type: ignore[arg-type]
    dec_weights = {**pred_dict, **joint_dict}
    dec_model = session.load(dec_graph, weights_registry=dec_weights)

    # Create inputs
    zero_state = np.zeros((1, pred_hidden), dtype=np.float32)
    h0 = Buffer.from_numpy(zero_state).to(device)
    c0 = Buffer.from_numpy(zero_state).to(device)
    h1 = Buffer.from_numpy(zero_state).to(device)
    c1 = Buffer.from_numpy(zero_state).to(device)
    token_buf = Buffer.from_numpy(np.array([[vocab_size]], dtype=np.int32)).to(
        device
    )

    fake_proj = np.random.randn(1, 400, joint_hidden).astype(np.float32)
    proj_buf = Buffer.from_numpy(fake_proj).to(device)
    t_index_buf = Buffer.from_numpy(np.array([0], dtype=np.int32)).to(device)

    cpu_device = load_devices([DeviceSpec.cpu()])[0]

    # Warmup
    print(f"  Warming up ({n_warmup} iterations)...")
    for _ in range(n_warmup):
        outputs = dec_model.execute(
            token_buf, h0, c0, h1, c1, proj_buf, t_index_buf
        )
        _, _, h0, c0, h1, c1 = outputs

    # Reset states
    h0 = Buffer.from_numpy(zero_state).to(device)
    c0 = Buffer.from_numpy(zero_state).to(device)
    h1 = Buffer.from_numpy(zero_state).to(device)
    c1 = Buffer.from_numpy(zero_state).to(device)

    # --- Time just model.execute() ---
    print(f"  Timing model.execute() only ({n_timed} iterations)...")
    exec_times: list[float] = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        outputs = dec_model.execute(
            token_buf, h0, c0, h1, c1, proj_buf, t_index_buf
        )
        t1 = time.perf_counter()
        _, _, h0, c0, h1, c1 = outputs
        exec_times.append((t1 - t0) * 1000)

    # Reset states
    h0 = Buffer.from_numpy(zero_state).to(device)
    c0 = Buffer.from_numpy(zero_state).to(device)
    h1 = Buffer.from_numpy(zero_state).to(device)
    c1 = Buffer.from_numpy(zero_state).to(device)

    # --- Time full step ---
    print(f"  Timing full step ({n_timed} iterations)...")
    full_times: list[float] = []
    readback_times: list[float] = []
    buf_create_times: list[float] = []
    for i in range(n_timed):
        t0 = time.perf_counter()
        t_index_buf = Buffer.from_numpy(np.array([i % 400], dtype=np.int32)).to(
            device
        )
        t1 = time.perf_counter()
        buf_create_times.append((t1 - t0) * 1000)

        outputs = dec_model.execute(
            token_buf, h0, c0, h1, c1, proj_buf, t_index_buf
        )
        token_out, dur_out = outputs[0], outputs[1]
        h0, c0, h1, c1 = outputs[2], outputs[3], outputs[4], outputs[5]

        t4 = time.perf_counter()
        int(np.from_dlpack(token_out.to(cpu_device))[0])
        int(np.from_dlpack(dur_out.to(cpu_device))[0])
        t5 = time.perf_counter()
        readback_times.append((t5 - t4) * 1000)

        full_times.append((t5 - t0) * 1000)

    return StepResults(
        execute_only_mean=float(np.mean(exec_times)),
        execute_only_std=float(np.std(exec_times)),
        full_step_mean=float(np.mean(full_times)),
        full_step_std=float(np.std(full_times)),
        buf_create_mean=float(np.mean(buf_create_times)),
        buf_create_std=float(np.std(buf_create_times)),
        readback_mean=float(np.mean(readback_times)),
        readback_std=float(np.std(readback_times)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile TDT decoder step overhead"
    )
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()

    if args.device == "gpu":
        devices = load_devices([DeviceSpec.accelerator(id=0)])
    else:
        devices = load_devices([DeviceSpec.cpu()])
    device = devices[0]

    print("=" * 60)
    print(f"TDT Decoder Step Profiling (device={args.device})")
    print("=" * 60)

    # Isolated micro-benchmarks
    py_overhead = 0.0
    t0 = time.perf_counter()
    for _ in range(10000):
        pass
    py_overhead = (time.perf_counter() - t0) / 10000 * 1000
    print(f"\nPython loop overhead:     {py_overhead:.4f} ms/iter")

    buf_create = time_buffer_creation(device, 1000)
    print(f"Buffer.from_numpy().to(): {buf_create:.4f} ms/call")

    readback = time_readback(device, 1000)
    print(f"Scalar readback to CPU:   {readback:.4f} ms/call")

    # Full model profiling
    print()
    results = time_model_execute(device, N_WARMUP, N_TIMED)

    print()
    print("=" * 60)
    print("RESULTS (per decoder step)")
    print("=" * 60)
    print(
        f"  model.execute() only:   "
        f"{results.execute_only_mean:.3f} "
        f"\u00b1 {results.execute_only_std:.3f} ms"
    )
    print(
        f"  Buffer creation:        "
        f"{results.buf_create_mean:.3f} "
        f"\u00b1 {results.buf_create_std:.3f} ms"
    )
    print(
        f"  Readback (2 scalars):   "
        f"{results.readback_mean:.3f} "
        f"\u00b1 {results.readback_std:.3f} ms"
    )
    print(
        f"  Full step (all above):  "
        f"{results.full_step_mean:.3f} "
        f"\u00b1 {results.full_step_std:.3f} ms"
    )

    total_300 = results.full_step_mean * 300
    exec_300 = results.execute_only_mean * 300
    overhead_300 = total_300 - exec_300
    print()
    print("  Projected 300-step decode:")
    print(f"    Execute only:   {exec_300:.1f} ms")
    print(f"    Overhead:       {overhead_300:.1f} ms")
    print(f"    Total:          {total_300:.1f} ms")
    print("    (Actual decode: ~258 ms from benchmark)")
    print("=" * 60)


if __name__ == "__main__":
    main()
