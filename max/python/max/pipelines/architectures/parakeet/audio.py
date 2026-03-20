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
"""Audio preprocessing for Parakeet ASR models.

Pure numpy mel spectrogram extraction matching NeMo's
AudioToMelSpectrogramPreprocessor (STFT + slaney mel filterbank + log).
No torch, librosa, or soundfile dependencies.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import numpy.typing as npt

NDFloat = npt.NDArray[np.floating]


def read_wav(audio_bytes: bytes) -> tuple[NDFloat, int]:
    """Read WAV audio bytes into a float32 numpy array.

    Returns:
        (audio_data, sample_rate) where audio_data is mono float32 in [-1, 1].
    """
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= np.iinfo(dtype).max
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sample_rate


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> NDFloat:
    """Compute a slaney-normalized mel filterbank matrix.

    Equivalent to ``librosa.filters.mel(sr, n_fft, n_mels, norm="slaney")``.
    Uses the Slaney mel scale (linear below 1kHz, log above).
    Returns shape ``(n_mels, n_fft // 2 + 1)``.
    """
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(f: NDFloat) -> NDFloat:
        f = np.asarray(f, dtype=np.float64)
        return np.where(
            f < min_log_hz,
            f / f_sp,
            min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep,
        )

    def mel_to_hz(m: NDFloat) -> NDFloat:
        m = np.asarray(m, dtype=np.float64)
        return np.where(
            m < min_log_mel,
            m * f_sp,
            min_log_hz * np.exp(logstep * (m - min_log_mel)),
        )

    fmin_mel = hz_to_mel(np.float64(0.0))
    fmax_mel = hz_to_mel(np.float64(sr / 2.0))
    mels = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    freqs = mel_to_hz(mels)

    fft_freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)

    for i in range(n_mels):
        low, center, high = freqs[i], freqs[i + 1], freqs[i + 2]
        up = (fft_freqs - low) / (center - low)
        down = (high - fft_freqs) / (high - center)
        fb[i] = np.maximum(0, np.minimum(up, down))
        fb[i] *= 2.0 / (high - low)

    return fb.astype(np.float32)


def extract_mel(
    audio: NDFloat,
    n_mels: int = 128,
    preemphasis: float = 0.0,
    periodic_window: bool = True,
) -> NDFloat:
    """Mel spectrogram extraction in pure numpy.

    Optionally applies a pre-emphasis filter, then STFT -> power ->
    mel filterbank (slaney) -> log. Returns shape ``(1, T, n_mels)``.

    Args:
        audio: Mono float32 audio at 16kHz.
        n_mels: Number of mel bins.
        preemphasis: Pre-emphasis coefficient (0 to disable).
        periodic_window: If True, use periodic Hann window (NeMo/TDT).
            If False, use symmetric Hann window (HF Parakeet CTC).
    """
    sr = 16000
    n_fft = 512
    hop_length = 160
    win_length = 400

    if preemphasis > 0:
        audio = np.append(audio[0:1], audio[1:] - preemphasis * audio[:-1])

    # Center-pad (matching torch.stft center=True, pad_mode="constant")
    pad_length = n_fft // 2
    audio_padded = np.pad(audio, (pad_length, pad_length), mode="constant")

    if periodic_window:
        window = np.hanning(win_length + 1)[:win_length].astype(np.float32)
    else:
        window = np.hanning(win_length).astype(np.float32)

    # Zero-pad window to n_fft, centered (matching PyTorch STFT behavior)
    padded_window = np.zeros(n_fft, dtype=np.float32)
    offset = (n_fft - win_length) // 2
    padded_window[offset : offset + win_length] = window

    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio_padded[start : start + n_fft]
        stft[:, i] = np.fft.rfft(frame * padded_window)

    power = np.abs(stft) ** 2
    mel_basis = _mel_filterbank(sr, n_fft, n_mels)
    mel = np.log(mel_basis @ power + 2**-24)

    return mel.T[np.newaxis, :, :].astype(np.float32)


def normalize_per_feature(features: NDFloat) -> NDFloat:
    """Zero-mean unit-variance normalization per mel bin per utterance.

    Args:
        features: Shape ``(batch, num_frames, num_mel_bins)``.
    """
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return (features - mean) / (std + 1e-5)
