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
"""Bucket spec for variable-length Parakeet ASR audio.

The Parakeet encoder graphs compile against fixed input shapes (a workaround
for GPU compiler bugs with derived symbolic dims). To support variable-length
audio without padding every short clip up to the maximum, we compile a small
set of encoder graphs at different fixed lengths and route each utterance to
the smallest fitting bucket.

This module is pure stdlib — it must not import from ``max.*`` so it stays
trivially unit-testable and can be imported from ``mel_graph.py`` without
creating a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

# Audio constants matching ``mel_graph.py``. Duplicated here (instead of
# imported) to keep this module dependency-free; ``mel_graph.py`` imports
# from us, not the other way around.
SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FFT = 512


@dataclass(frozen=True)
class ParakeetBucket:
    """A single shape bucket for the Parakeet ASR pipeline.

    All four fields are derivable from ``duration_s`` but are precomputed and
    stored so call sites don't repeat the math (and can't get it wrong).

    Attributes:
        duration_s: Nominal audio duration this bucket covers, in seconds.
        audio_samples: Fixed audio length the mel graph for this bucket
            expects, in samples at ``SAMPLE_RATE``. This is what
            ``_prepare_audio_for_bucket`` pads/truncates to.
        mel_frames: Fixed number of mel frames the encoder graph for this
            bucket expects. Always a multiple of 8 so the post-subsampling
            encoder-frame count is exact (``mel_frames // 8``).
        encoder_frames: Number of encoder frames after 8x subsampling.
            Used as the ``actual_t`` bound for the TDT decode loop and the
            slice length for CTC logits.
    """

    duration_s: int
    audio_samples: int
    mel_frames: int
    encoder_frames: int


def _bucket_for_duration(duration_s: int) -> ParakeetBucket:
    """Compute the canonical bucket for a given duration.

    Picks ``mel_frames`` as the natural STFT frame count for ``duration_s``
    seconds of audio, rounded UP to the next multiple of 8. The audio sample
    count is then derived so that ``build_mel_graph`` produces exactly
    ``mel_frames`` frames.
    """
    natural_mel_frames = 1 + duration_s * (SAMPLE_RATE // HOP_LENGTH)
    mel_frames = ((natural_mel_frames + 7) // 8) * 8
    audio_samples = (mel_frames - 1) * HOP_LENGTH + N_FFT
    encoder_frames = mel_frames // 8
    return ParakeetBucket(
        duration_s=duration_s,
        audio_samples=audio_samples,
        mel_frames=mel_frames,
        encoder_frames=encoder_frames,
    )


# Default bucket set for TDT: 10s through 60s in 10-second increments.
# Covers the range of typical VAD-segmented production audio. Compile
# time is ~1.5-3min. TDT's 0.6B-param encoder × 6 copies = ~14GB on GPU,
# fits comfortably on L4 (24GB).
DEFAULT_BUCKET_DURATIONS_S: tuple[int, ...] = (10, 20, 30, 40, 50, 60)

# CTC-specific bucket set: fewer/coarser buckets than TDT because the
# CTC encoder is 1.1B params (vs. TDT's 0.6B). MAX's ``session.load``
# does not share weight buffers across loads on GPU, so each bucket
# uploads a fresh ~4.4GB copy of the CTC weights. 6 buckets = ~26GB and
# OOMs on L4. 4 buckets = ~17.6GB, fits with headroom. Same 60s ceiling.
# See commit 15e2262374 for the MAX weight-materialization behavior.
CTC_DEFAULT_BUCKET_DURATIONS_S: tuple[int, ...] = (15, 30, 45, 60)

DEFAULT_BUCKETS: tuple[ParakeetBucket, ...] = tuple(
    _bucket_for_duration(d) for d in DEFAULT_BUCKET_DURATIONS_S
)


def build_buckets(durations_s: tuple[int, ...]) -> tuple[ParakeetBucket, ...]:
    """Build a sorted bucket tuple from a list of durations (seconds).

    Sorts ascending so ``select_bucket`` can do a simple linear scan.
    """
    if not durations_s:
        raise ValueError("bucket_durations_s must not be empty")
    return tuple(
        sorted(
            (_bucket_for_duration(d) for d in durations_s),
            key=lambda b: b.duration_s,
        )
    )


def mel_frames_for_audio(num_samples: int) -> int:
    """STFT frame count for a given raw audio length.

    Mirrors the formula in ``mel_graph.py``: after center-padding by
    ``N_FFT // 2`` on each side, the frame count is
    ``1 + (samples + 2*pad - N_FFT) // HOP_LENGTH``. The ``+2*pad - N_FFT``
    cancels exactly (since ``N_FFT == 2 * (N_FFT // 2)``), so this reduces
    to ``1 + samples // HOP_LENGTH``.
    """
    return 1 + num_samples // HOP_LENGTH


def select_bucket(
    mel_frames: int,
    buckets: tuple[ParakeetBucket, ...],
) -> tuple[ParakeetBucket, bool]:
    """Pick the smallest bucket whose ``mel_frames >= mel_frames``.

    Args:
        mel_frames: True STFT frame count of the input audio.
        buckets: Sorted (ascending) tuple of available buckets.

    Returns:
        ``(bucket, was_truncated)``. ``was_truncated`` is True iff the input
        exceeds the largest bucket — in that case the largest bucket is
        returned and the caller is expected to truncate the audio.
    """
    for bucket in buckets:
        if bucket.mel_frames >= mel_frames:
            return bucket, False
    return buckets[-1], True
