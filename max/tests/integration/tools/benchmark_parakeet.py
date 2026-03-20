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
"""Benchmarks for Parakeet ASR pipeline stages.

Measures latency for each stage of the pipeline as-is:
  1. read_wav
  2. extract_mel
  3. normalize_per_feature
  4. Buffer.from_numpy (CPU→device)
  5. Encoder graph execution
  6. np.from_dlpack().copy() (device→CPU)
  7. CTC decode / TDT decode
  8. End-to-end transcribe()

Usage (preprocessing + decode, no bazel):
    python benchmark_parakeet.py --stages preprocess,decode

Usage (full pipeline via bazel):
    ./bazelw run //max/tests:benchmark_parakeet -- --model ctc --device cpu

Test audio is downloaded from LibriSpeech test-clean if no WAV files
are found in the benchmark_audio/ directory.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tarfile
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

NDFloat = npt.NDArray[np.floating]

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = SCRIPT_DIR / "benchmark_audio"

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


# ---------------------------------------------------------------------------
# Audio download / generation
# ---------------------------------------------------------------------------


def ensure_audio(n_samples: int = 105) -> list[Path]:
    """Get test WAV files, downloading LibriSpeech test-clean if needed.

    Requires ffmpeg to convert FLAC to WAV.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(AUDIO_DIR.glob("*.wav"))
    if len(existing) >= n_samples:
        return existing[:n_samples]

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required. Install it and re-run.")

    print(f"Downloading LibriSpeech test-clean for {n_samples} samples...")
    print("(One-time ~350MB download)")

    tar_path = AUDIO_DIR / "test-clean.tar.gz"
    if not tar_path.exists():
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path)

    flac_paths: list[str] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".flac") and len(flac_paths) < n_samples:
                flac_paths.append(member.name)

        wav_paths: list[Path] = []
        for i, flac_name in enumerate(flac_paths):
            wav_path = AUDIO_DIR / f"sample_{i:04d}.wav"
            if wav_path.exists():
                wav_paths.append(wav_path)
                continue

            tar.extract(member=tar.getmember(flac_name), path=AUDIO_DIR)
            flac_full = AUDIO_DIR / flac_name
            os.system(
                f'ffmpeg -y -i "{flac_full}" -ar 16000 -ac 1 '
                f'"{wav_path}" -loglevel error'
            )
            if wav_path.exists():
                wav_paths.append(wav_path)
            flac_full.unlink(missing_ok=True)

            if (i + 1) % 20 == 0:
                print(f"  Converted {i + 1}/{len(flac_paths)}")

    # Clean up extracted FLAC directory structure
    libri_dir = AUDIO_DIR / "LibriSpeech"
    if libri_dir.exists():
        shutil.rmtree(libri_dir, ignore_errors=True)

    print(f"Ready: {len(wav_paths)} WAV files")
    return wav_paths


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.times_ms))

    @property
    def std(self) -> float:
        return float(np.std(self.times_ms))

    @property
    def min(self) -> float:
        return float(np.min(self.times_ms))

    @property
    def max(self) -> float:
        return float(np.max(self.times_ms))

    @property
    def median(self) -> float:
        return float(np.median(self.times_ms))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.times_ms, 95))

    def summary(self) -> str:
        return (
            f"{self.name:>30s}: "
            f"mean={self.mean:8.2f}ms  "
            f"std={self.std:6.2f}ms  "
            f"min={self.min:8.2f}ms  "
            f"max={self.max:8.2f}ms  "
            f"p95={self.p95:8.2f}ms  "
            f"n={len(self.times_ms)}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mean_ms": round(self.mean, 3),
            "std_ms": round(self.std, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "median_ms": round(self.median, 3),
            "p95_ms": round(self.p95, 3),
            "n": len(self.times_ms),
        }


# ---------------------------------------------------------------------------
# Preprocessing benchmarks
# ---------------------------------------------------------------------------


def bench_preprocess(
    audio_bytes_list: list[bytes],
    model_type: str,
    n_warmup: int,
    n_runs: int,
) -> list[TimingResult]:
    from max.pipelines.architectures.parakeet.audio import (
        extract_mel,
        normalize_per_feature,
        read_wav,
    )

    if model_type == "ctc":
        preemphasis = 0.97
        periodic = False
    else:
        preemphasis = 0.0
        periodic = True

    r_wav = TimingResult(f"read_wav ({model_type})")
    r_mel = TimingResult(f"extract_mel ({model_type})")
    r_norm = TimingResult(f"normalize_per_feature ({model_type})")

    for i in range(n_warmup + n_runs):
        ab = audio_bytes_list[i % len(audio_bytes_list)]
        record = i >= n_warmup

        t0 = time.perf_counter()
        audio, _ = read_wav(ab)
        t1 = time.perf_counter()
        if record:
            r_wav.times_ms.append((t1 - t0) * 1000)

        t0 = time.perf_counter()
        features = extract_mel(
            audio,
            n_mels=80,
            preemphasis=preemphasis,
            periodic_window=periodic,
        )
        t1 = time.perf_counter()
        if record:
            r_mel.times_ms.append((t1 - t0) * 1000)

        t0 = time.perf_counter()
        normalize_per_feature(features)
        t1 = time.perf_counter()
        if record:
            r_norm.times_ms.append((t1 - t0) * 1000)

    return [r_wav, r_mel, r_norm]


# ---------------------------------------------------------------------------
# Decode benchmarks (with mock encoder output)
# ---------------------------------------------------------------------------


def bench_ctc_decode(n_warmup: int, n_runs: int) -> TimingResult:
    from max.pipelines.architectures.parakeet.decode import ctc_greedy_decode

    class MockTokenizer:
        def decode(
            self, ids: list[int], skip_special_tokens: bool = True
        ) -> str:
            return " ".join(str(x) for x in ids)

    tokenizer = MockTokenizer()
    rng = np.random.default_rng(42)
    # Varying sequence lengths to simulate real data
    logits_list = [
        rng.standard_normal((1, 100 + i * 30, 1025)).astype(np.float32)
        for i in range(20)
    ]

    r = TimingResult("ctc_greedy_decode")
    for i in range(n_warmup + n_runs):
        logits = logits_list[i % len(logits_list)]
        t0 = time.perf_counter()
        ctc_greedy_decode(logits, tokenizer, blank_id=1024)
        t1 = time.perf_counter()
        if i >= n_warmup:
            r.times_ms.append((t1 - t0) * 1000)
    return r


def bench_tdt_decode(n_warmup: int, n_runs: int) -> list[TimingResult]:
    from max.pipelines.architectures.parakeet_tdt.decode import (
        tdt_greedy_decode,
    )
    from max.pipelines.architectures.parakeet_tdt.decoder import (
        JointNetwork,
        LSTMCell,
        PredictionNetwork,
    )

    rng = np.random.default_rng(42)
    pred_hidden = 640
    encoder_hidden = 1024
    joint_hidden = 640
    vocab_size = 1024

    cells = []
    for _ in range(2):
        cells.append(
            LSTMCell(
                weight_ih=rng.standard_normal(
                    (4 * pred_hidden, pred_hidden)
                ).astype(np.float32),
                weight_hh=rng.standard_normal(
                    (4 * pred_hidden, pred_hidden)
                ).astype(np.float32),
                bias_ih=rng.standard_normal(4 * pred_hidden).astype(np.float32),
                bias_hh=rng.standard_normal(4 * pred_hidden).astype(np.float32),
            )
        )
    pred_net = PredictionNetwork(
        embedding_weight=rng.standard_normal(
            (vocab_size + 1, pred_hidden)
        ).astype(np.float32),
        lstm_cells=cells,
    )
    joint_net = JointNetwork(
        enc_weight=rng.standard_normal((joint_hidden, encoder_hidden)).astype(
            np.float32
        ),
        enc_bias=rng.standard_normal(joint_hidden).astype(np.float32),
        pred_weight=rng.standard_normal((joint_hidden, pred_hidden)).astype(
            np.float32
        ),
        pred_bias=rng.standard_normal(joint_hidden).astype(np.float32),
        out_weight=rng.standard_normal(
            (vocab_size + 1 + 5, joint_hidden)
        ).astype(np.float32),
        out_bias=rng.standard_normal(vocab_size + 1 + 5).astype(np.float32),
    )

    # Individual LSTM step
    r_lstm = TimingResult("lstm_step (2 layers, 640-dim)")
    h_states, c_states = pred_net.init_states()
    for i in range(n_warmup + n_runs):
        t0 = time.perf_counter()
        _, h_states, c_states = pred_net(0, h_states, c_states)
        t1 = time.perf_counter()
        if i >= n_warmup:
            r_lstm.times_ms.append((t1 - t0) * 1000)

    # Individual joint step
    r_joint = TimingResult("joint_network_step")
    enc_out = rng.standard_normal(encoder_hidden).astype(np.float32)
    pred_out = rng.standard_normal(pred_hidden).astype(np.float32)
    for i in range(n_warmup + n_runs):
        t0 = time.perf_counter()
        joint_net(enc_out, pred_out)
        t1 = time.perf_counter()
        if i >= n_warmup:
            r_joint.times_ms.append((t1 - t0) * 1000)

    # Full decode loop
    r_loop = TimingResult("tdt_full_decode_loop")
    encoder_outputs = [
        rng.standard_normal((1, 100 + i * 30, encoder_hidden)).astype(
            np.float32
        )
        for i in range(10)
    ]
    for i in range(n_warmup + n_runs):
        enc = encoder_outputs[i % len(encoder_outputs)]
        t0 = time.perf_counter()
        tdt_greedy_decode(
            encoder_output=enc,
            prediction_net=pred_net,
            joint_net=joint_net,
            durations=[0, 1, 2, 3, 4],
            vocab_size=vocab_size,
            blank_id=vocab_size,
        )
        t1 = time.perf_counter()
        if i >= n_warmup:
            r_loop.times_ms.append((t1 - t0) * 1000)

    return [r_lstm, r_joint, r_loop]


# ---------------------------------------------------------------------------
# Full pipeline benchmark (requires model loaded via bazel)
# ---------------------------------------------------------------------------


def bench_full_pipeline(
    audio_bytes_list: list[bytes],
    model_type: str,
    device: str,
    encoding: str,
    n_warmup: int,
    n_runs: int,
) -> list[TimingResult]:
    """Benchmark the full transcribe() path including encoder.

    This requires the MAX model to be loadable (i.e., running via bazel
    or with max installed and model weights available).
    """
    from max.driver import Buffer
    from max.pipelines.architectures.parakeet.audio import (
        extract_mel,
        normalize_per_feature,
        read_wav,
    )
    from max.pipelines.architectures.parakeet.decode import ctc_greedy_decode
    from max.pipelines.architectures.parakeet_tdt.decode import (
        tdt_greedy_decode,
    )

    r_wav = TimingResult(f"read_wav ({model_type})")
    r_mel = TimingResult(f"extract_mel ({model_type})")
    r_norm = TimingResult(f"normalize ({model_type})")
    r_buffer = TimingResult(f"buffer_from_numpy ({model_type})")
    r_encoder = TimingResult(f"encoder_execute ({model_type})")
    r_transfer = TimingResult(f"output_transfer ({model_type})")
    r_decode = TimingResult(f"decode ({model_type})")
    r_e2e = TimingResult(f"end_to_end ({model_type})")

    if model_type == "ctc":
        model_id = "nvidia/parakeet-ctc-1.1b"
        preemphasis = 0.97
        periodic = False
    else:
        model_id = "pherber3/parakeet-tdt-0.6b-v3"
        preemphasis = 0.0
        periodic = True

    print(f"  Loading {model_type.upper()} model ({model_id}) on {device}...")

    try:
        from max.pipelines.architectures import register_all_models

        register_all_models()
        from max.driver import DeviceSpec, load_devices
        from max.engine import InferenceSession
        from max.graph.weights import (
            WeightsFormat,
            load_weights,
            weights_format,
        )
        from max.pipelines.lib import (
            KVCacheConfig,
            MAXModelConfig,
            PipelineConfig,
            PipelineRuntimeConfig,
        )
        from max.pipelines.lib.hf_utils import download_weight_files

        # Build pipeline config
        device_spec = (
            DeviceSpec.cpu() if device == "cpu" else DeviceSpec.accelerator()
        )
        model_config = MAXModelConfig(
            device_specs=[device_spec],
            quantization_encoding=encoding,  # type: ignore[arg-type]
            model_path=model_id,
            kv_cache=KVCacheConfig(),
        )
        config = PipelineConfig(
            model=model_config,
            runtime=PipelineRuntimeConfig(max_num_steps=1),
        )

        devices = load_devices(config.model.device_specs)
        session = InferenceSession(devices=[*devices])

        weight_paths = download_weight_files(
            huggingface_model_id=config.model.huggingface_weight_repo_id,
            filenames=[str(x) for x in config.model.weight_path],
            revision=config.model.huggingface_weight_revision,
            force_download=config.model.force_download,
        )
        weights = load_weights(weight_paths)
        wfmt = weights_format(weight_paths)

        if model_type == "ctc":
            from max.pipelines.architectures.parakeet.model import (
                ParakeetInputs,
                ParakeetPipelineModel,
            )
            from max.pipelines.architectures.parakeet.weight_adapters import (
                convert_safetensor_state_dict as ctc_adapter,
            )

            ctc_model = ParakeetPipelineModel(
                pipeline_config=config,
                session=session,
                devices=devices,
                kv_cache_config=config.model.kv_cache,
                weights=weights,
                adapter=(
                    ctc_adapter if wfmt == WeightsFormat.safetensors else None
                ),
            )
        else:
            from max.pipelines.architectures.parakeet_tdt.model import (
                ParakeetTDTInputs,
                ParakeetTDTPipelineModel,
            )
            from max.pipelines.architectures.parakeet_tdt.weight_adapters import (
                convert_safetensor_state_dict as tdt_adapter,
            )

            tdt_model = ParakeetTDTPipelineModel(
                pipeline_config=config,
                session=session,
                devices=devices,
                kv_cache_config=config.model.kv_cache,
                weights=weights,
                adapter=(
                    tdt_adapter if wfmt == WeightsFormat.safetensors else None
                ),
            )

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(
            f"  Model loaded. Running {n_warmup} warmup + {n_runs} timed iterations..."
        )

        if model_type == "ctc":
            n_mels = ctc_model.config.num_mel_bins
        else:
            n_mels = tdt_model.tdt_config.num_mel_bins

        for i in range(n_warmup + n_runs):
            ab = audio_bytes_list[i % len(audio_bytes_list)]
            record = i >= n_warmup

            # -- Single pipeline pass, timing each stage --
            e2e_start = time.perf_counter()

            t0 = time.perf_counter()
            audio, _ = read_wav(ab)
            t1 = time.perf_counter()
            if record:
                r_wav.times_ms.append((t1 - t0) * 1000)

            t0 = time.perf_counter()
            features = extract_mel(
                audio,
                n_mels=n_mels,
                preemphasis=preemphasis,
                periodic_window=periodic,
            )
            t1 = time.perf_counter()
            if record:
                r_mel.times_ms.append((t1 - t0) * 1000)

            t0 = time.perf_counter()
            features = normalize_per_feature(features).astype(np.float32)
            t1 = time.perf_counter()
            if record:
                r_norm.times_ms.append((t1 - t0) * 1000)

            t0 = time.perf_counter()
            buf = Buffer.from_numpy(features)
            t1 = time.perf_counter()
            if record:
                r_buffer.times_ms.append((t1 - t0) * 1000)

            t0 = time.perf_counter()
            if model_type == "ctc":
                outputs = ctc_model.execute(ParakeetInputs(input_features=buf))
            else:
                outputs = tdt_model.execute(
                    ParakeetTDTInputs(input_features=buf)
                )
            t1 = time.perf_counter()
            if record:
                r_encoder.times_ms.append((t1 - t0) * 1000)

            t0 = time.perf_counter()
            logits = np.from_dlpack(outputs.logits).copy()
            t1 = time.perf_counter()
            if record:
                r_transfer.times_ms.append((t1 - t0) * 1000)

            t0 = time.perf_counter()
            if model_type == "ctc":
                ctc_greedy_decode(logits, tokenizer, blank_id=1024)
            else:
                tdt_greedy_decode(
                    encoder_output=logits,
                    prediction_net=tdt_model.prediction_net,
                    joint_net=tdt_model.joint_net,
                    durations=tdt_model.tdt_config.tdt_durations,
                    vocab_size=tdt_model.tdt_config.vocab_size,
                    blank_id=tdt_model.tdt_config.blank_id,
                )
            t1 = time.perf_counter()
            if record:
                r_decode.times_ms.append((t1 - t0) * 1000)

            e2e_end = time.perf_counter()
            if record:
                r_e2e.times_ms.append((e2e_end - e2e_start) * 1000)

        return [
            r_wav,
            r_mel,
            r_norm,
            r_buffer,
            r_encoder,
            r_transfer,
            r_decode,
            r_e2e,
        ]

    except Exception as e:
        print(f"  Failed to load model: {e}")
        print("  Run via bazel for full pipeline benchmarks:")
        print(
            f"    ./bazelw run //max/tests:benchmark_parakeet -- "
            f"--model {model_type} --device {device} --stages full"
        )
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Parakeet ASR stages"
    )
    parser.add_argument(
        "--model",
        choices=["ctc", "tdt", "both"],
        default="both",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--encoding", default="float32", choices=["float32", "bfloat16"]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument(
        "--stages",
        default="preprocess,decode",
        help="Comma-separated: preprocess,decode,full,all",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    stages = set(args.stages.split(","))
    run_all = "all" in stages

    print("=" * 70)
    print("Parakeet ASR Pipeline Benchmark")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Device:   {args.device}")
    print(f"Encoding: {args.encoding}")
    print(f"Warmup:   {args.warmup}")
    print(f"Runs:     {args.runs}")
    print(f"Stages:   {args.stages}")
    print()

    n_samples = (
        args.n_samples
        if args.n_samples is not None
        else args.warmup + args.runs
    )
    audio_files = ensure_audio(n_samples)
    audio_bytes_list = [f.read_bytes() for f in audio_files]

    # Print audio duration stats
    durations = []
    for ab in audio_bytes_list:
        with wave.open(io.BytesIO(ab), "rb") as wf:
            durations.append(wf.getnframes() / wf.getframerate())
    print(
        f"Audio: {len(audio_files)} files, "
        f"durations: min={min(durations):.1f}s "
        f"max={max(durations):.1f}s mean={np.mean(durations):.1f}s"
    )
    print()

    results: list[TimingResult] = []
    models = ["ctc", "tdt"] if args.model == "both" else [args.model]

    if run_all or "preprocess" in stages:
        for m in models:
            print(f"Benchmarking preprocessing ({m.upper()})...")
            results.extend(
                bench_preprocess(
                    audio_bytes_list,
                    m,
                    args.warmup,
                    args.runs,
                )
            )

    if run_all or "decode" in stages:
        if "ctc" in models:
            print("Benchmarking CTC decode...")
            results.append(bench_ctc_decode(args.warmup, args.runs))
        if "tdt" in models:
            print("Benchmarking TDT decode...")
            results.extend(bench_tdt_decode(args.warmup, args.runs))

    if run_all or "full" in stages:
        for m in models:
            print(f"Benchmarking full pipeline ({m.upper()})...")
            results.extend(
                bench_full_pipeline(
                    audio_bytes_list,
                    m,
                    args.device,
                    args.encoding,
                    args.warmup,
                    args.runs,
                )
            )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    for r in results:
        if r.times_ms:
            print(r.summary())
    print("=" * 70)

    if args.output_json:
        data = {
            "config": {
                "model": args.model,
                "device": args.device,
                "warmup": args.warmup,
                "runs": args.runs,
                "n_audio_files": len(audio_files),
            },
            "results": [r.to_dict() for r in results if r.times_ms],
        }
        Path(args.output_json).write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
