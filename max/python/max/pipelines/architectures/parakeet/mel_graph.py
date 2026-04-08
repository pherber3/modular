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
from .bucket_spec import HOP_LENGTH, N_FFT, SAMPLE_RATE

# Audio constants matching NeMo's AudioToMelSpectrogramPreprocessor defaults.
# SAMPLE_RATE / N_FFT / HOP_LENGTH are imported from bucket_spec (the canonical
# source — bucket_spec has to duplicate them otherwise since it stays
# stdlib-only to avoid an import cycle).
WIN_LENGTH = 400
LOG_EPSILON = 2**-24


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
    plus a scalar ``valid_frames`` int32 telling normalization how many
    leading mel frames came from real audio (the rest are zero-padded).
    Outputs log-mel features with per-feature normalization computed
    over only the valid frames and the padded tail mean-filled so it
    doesn't pollute the encoder's self-attention.

    Args:
        n_mels: Number of mel frequency bins.
        max_audio_samples: Fixed audio length after center-padding.
            Determines n_frames at compile time.
        periodic_window: If True, periodic Hann window (NeMo/TDT).
            If False, symmetric Hann window (HF Parakeet CTC).
        device: Target device (must be GPU for rfft).

    Returns:
        Graph with inputs ``(1, max_audio_samples)`` audio + ``(1,)``
        valid_frames -> output ``(1, n_frames, n_mels)``.
    """
    n_frames = 1 + (max_audio_samples - N_FFT) // HOP_LENGTH

    # Precompute constants in numpy.
    padded_window = _build_padded_window(periodic_window)
    frame_indices = _build_frame_indices(n_frames)
    mel_basis = _mel_filterbank(SAMPLE_RATE, N_FFT, n_mels).astype(np.float32)
    frame_arange = np.arange(n_frames, dtype=np.int32)

    audio_input_type = TensorType(
        DType.float32,
        shape=[1, max_audio_samples],
        device=device,
    )
    valid_frames_input_type = TensorType(
        DType.int32,
        shape=[1],
        device=device,
    )

    with Graph(
        "parakeet_mel",
        input_types=[audio_input_type, valid_frames_input_type],
    ) as graph:
        # Input: (1, audio_length) — center-padded audio on GPU.
        audio = graph.inputs[0].tensor  # (1, audio_length)
        valid_frames = graph.inputs[1].tensor  # (1,) int32
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
            # Padding-aware per-feature normalization. The bucket pads
            # short utterances with zero audio whose log-mel is
            # ``log(2^-24) ≈ -16.6`` per bin — far below real speech
            # values. Including those frames in the mean/std skews the
            # statistics enough to break TDT decoding (CTC tolerated it
            # at ~2% WER but TDT collapsed to 98% WER on padded clips).
            # Match HF ParakeetFeatureExtractor and NeMo's
            # AudioToMelSpectrogramPreprocessor: compute mean/var over
            # only the valid leading frames.
            arange = ops.constant(frame_arange, DType.int32, device)
            valid_count = ops.cast(valid_frames, DType.int32)  # (1,)
            mask_bool = arange < valid_count  # (n_frames,) bool
            mask = ops.cast(mask_bool, DType.float32)  # (n_frames,)
            mask_3d = mask.reshape((1, n_frames, 1))  # (1, n_frames, 1)
            valid_count_f = ops.cast(valid_frames, DType.float32).reshape(
                (1, 1, 1)
            )

            masked = log_mel * mask_3d
            mean = ops.sum(masked, axis=1) / valid_count_f  # (1, 1, n_mels)
            diff = (log_mel - mean) * mask_3d
            var = ops.sum(diff * diff, axis=1) / valid_count_f
            log_mel_norm = diff * ops.rsqrt(var + 1e-5)

            # Mean-fill (i.e. zero after subtracting the mean) the padded
            # tail so it doesn't pollute the encoder's self-attention with
            # extreme values. ``diff`` is already zero where ``mask == 0``,
            # so the multiplication by ``rsqrt(var)`` keeps it zero.
            log_mel = log_mel_norm

        graph.output(log_mel)

    return graph
