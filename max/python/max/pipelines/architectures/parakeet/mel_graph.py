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
"""GPU mel spectrogram extraction as a MAX graph.

Builds a compiled graph that performs STFT + mel filterbank + log on GPU,
matching the numpy implementation in audio.py. The mel filterbank matrix
and Hann window are embedded as graph constants.
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

from .audio import _mel_filterbank

# Audio constants matching NeMo's AudioToMelSpectrogramPreprocessor defaults.
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
LOG_EPSILON = 2**-24

# Maximum mel frames the encoder accepts. Will be replaced by per-bucket
# values in Step 8 of the bucketing refactor; kept here for callers that
# still use the legacy fixed-shape mel graph.
MAX_MEL_FRAMES = 3200

# Maximum audio samples after center-padding, derived from MAX_MEL_FRAMES.
# Inverse of: n_frames = 1 + (audio_samples - N_FFT) // HOP_LENGTH
MAX_AUDIO_SAMPLES = (MAX_MEL_FRAMES - 1) * HOP_LENGTH + N_FFT


def _build_padded_window(periodic: bool) -> np.ndarray:
    """Build the zero-padded Hann window matching audio.py."""
    if periodic:
        window = np.hanning(WIN_LENGTH + 1)[:WIN_LENGTH].astype(np.float32)
    else:
        window = np.hanning(WIN_LENGTH).astype(np.float32)
    padded = np.zeros(N_FFT, dtype=np.float32)
    offset = (N_FFT - WIN_LENGTH) // 2
    padded[offset : offset + WIN_LENGTH] = window
    return padded


def _build_frame_indices(n_frames: int) -> np.ndarray:
    """Build gather indices for STFT framing.

    Returns shape (n_frames, N_FFT) int64 array where row i contains
    [i*HOP_LENGTH, i*HOP_LENGTH+1, ..., i*HOP_LENGTH+N_FFT-1].
    """
    starts = np.arange(n_frames) * HOP_LENGTH
    offsets = np.arange(N_FFT)
    return (starts[:, None] + offsets[None, :]).astype(np.int64)


def build_mel_graph(
    n_mels: int,
    max_audio_samples: int,
    periodic_window: bool,
    device: DeviceRef,
    normalize: bool = True,
) -> Graph:
    """Build a MAX graph for GPU mel spectrogram extraction.

    The graph expects center-padded, preemphasis-applied audio as input
    and outputs log-mel features.

    Args:
        n_mels: Number of mel frequency bins.
        max_audio_samples: Fixed audio length after center-padding.
            Determines n_frames at compile time.
        periodic_window: If True, periodic Hann window (NeMo/TDT).
            If False, symmetric Hann window (HF Parakeet CTC).
        device: Target device (must be GPU for rfft).

    Returns:
        Graph with input (1, max_audio_samples) -> output (1, n_frames, n_mels).
    """
    n_frames = 1 + (max_audio_samples - N_FFT) // HOP_LENGTH

    # Precompute constants in numpy.
    padded_window = _build_padded_window(periodic_window)
    frame_indices = _build_frame_indices(n_frames)
    mel_basis = _mel_filterbank(SAMPLE_RATE, N_FFT, n_mels).astype(np.float32)

    input_type = TensorType(
        DType.float32,
        shape=[1, max_audio_samples],
        device=device,
    )

    with Graph("parakeet_mel", input_types=[input_type]) as graph:
        # Input: (1, audio_length) — center-padded audio on GPU.
        audio = graph.inputs[0].tensor  # (1, audio_length)
        audio_1d = audio.reshape((max_audio_samples,))  # (audio_length,)

        # 1. Frame extraction via gather.
        # frame_indices: (n_frames, N_FFT) int64 constant
        idx = ops.constant(frame_indices, DType.int64, device)
        frames = ops.gather(audio_1d, idx, axis=0)  # (n_frames, N_FFT)

        # 2. Windowing: multiply each frame by the Hann window.
        window = ops.constant(padded_window, DType.float32, device)
        windowed = frames * window  # (n_frames, N_FFT) broadcast

        # 3. RFFT: forward real FFT on each frame.
        # Output: (n_frames, N_FFT//2+1, 2) interleaved [real, imag]
        rfft_out = ops.rfft(windowed, n=N_FFT, axis=-1)

        # 4. Power spectrum: re^2 + im^2
        real = rfft_out[:, :, 0]  # (n_frames, N_FFT//2+1)
        imag = rfft_out[:, :, 1]  # (n_frames, N_FFT//2+1)
        power = real * real + imag * imag  # (n_frames, N_FFT//2+1)

        # 5. Mel filterbank: mel_basis @ power^T -> (n_mels, n_frames)
        # Then transpose to (n_frames, n_mels).
        mel_basis_const = ops.constant(mel_basis, DType.float32, device)
        # power is (n_frames, 257), mel_basis is (n_mels, 257)
        # matmul: (n_mels, 257) @ (257, n_frames) -> (n_mels, n_frames)
        power_t = ops.transpose(power, 0, 1)  # (257, n_frames)
        mel = ops.matmul(mel_basis_const, power_t)  # (n_mels, n_frames)
        mel = ops.transpose(mel, 0, 1)  # (n_frames, n_mels)

        # 6. Log with epsilon for numerical stability.
        log_mel = ops.log(mel + LOG_EPSILON)

        # Reshape to (1, n_frames, n_mels) to match encoder input.
        log_mel = log_mel.reshape((1, n_frames, n_mels))

        if normalize:
            # Per-feature normalization on GPU: (x - mean) / (std + 1e-5)
            # ops.mean/sum keep dims, so mean is (1, 1, n_mels).
            mean = ops.mean(log_mel, axis=1)  # (1, 1, n_mels)
            diff = log_mel - mean
            var = ops.mean(diff * diff, axis=1)  # (1, 1, n_mels)
            log_mel = diff * ops.rsqrt(var + 1e-5)

        graph.output(log_mel)

    return graph
