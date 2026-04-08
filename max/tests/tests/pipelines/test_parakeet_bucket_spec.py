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
"""Unit tests for ``parakeet.bucket_spec``.

The bucket math is the pure-stdlib core of the Parakeet ASR bucketing
refactor: compile N encoder graphs at fixed mel-frame counts, route each
utterance to the smallest fitting bucket, pad/slice on the boundaries.
Boundary arithmetic here is easy to eyeball-verify but also easy to get
off-by-one wrong — a miscount would silently push short clips into the
wrong bucket or truncate audio that should fit. These tests pin the
canonical numbers and the ``select_bucket`` dispatch rule.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures.parakeet.bucket_spec import (
    CTC_DEFAULT_BUCKET_DURATIONS_S,
    DEFAULT_BUCKET_DURATIONS_S,
    DEFAULT_BUCKETS,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    ParakeetBucket,
    build_buckets,
    mel_frames_for_audio,
    select_bucket,
)

# ---------------------------------------------------------------------------
# Canonical bucket math — these numbers are load-bearing downstream.
# Any shift here will desync the encoder compile shape from the mel graph
# output shape, so pin them explicitly.
# ---------------------------------------------------------------------------


CANONICAL_TDT_BUCKETS = [
    # duration_s, mel_frames, encoder_frames, audio_samples
    (10, 1008, 126, 161632),
    (20, 2008, 251, 321632),
    (30, 3008, 376, 481632),
    (40, 4008, 501, 641632),
    (50, 5008, 626, 801632),
    (60, 6008, 751, 961632),
]


@pytest.mark.parametrize(
    "duration_s,mel_frames,encoder_frames,audio_samples",
    CANONICAL_TDT_BUCKETS,
)
def test_default_bucket_shapes(
    duration_s: int,
    mel_frames: int,
    encoder_frames: int,
    audio_samples: int,
) -> None:
    """Every default TDT bucket pins its documented shape.

    ``mel_frames`` must be a multiple of 8 so the post-subsampling
    encoder-frame count is exact. ``audio_samples`` is the inverse of
    the STFT formula used in ``mel_graph.py``.
    """
    bucket = next(b for b in DEFAULT_BUCKETS if b.duration_s == duration_s)
    assert bucket.mel_frames == mel_frames
    assert bucket.mel_frames % 8 == 0
    assert bucket.encoder_frames == encoder_frames
    assert bucket.encoder_frames == bucket.mel_frames // 8
    assert bucket.audio_samples == audio_samples
    # Inverse of `n_frames = 1 + (samples - N_FFT) // HOP_LENGTH` for
    # the post-center-pad audio length.
    assert 1 + (bucket.audio_samples - N_FFT) // HOP_LENGTH == bucket.mel_frames


def test_default_buckets_sorted_ascending() -> None:
    """`build_buckets` guarantees ascending order so `select_bucket`
    can do a simple linear scan."""
    durations = [b.duration_s for b in DEFAULT_BUCKETS]
    assert durations == sorted(durations)
    mel_frames = [b.mel_frames for b in DEFAULT_BUCKETS]
    assert mel_frames == sorted(mel_frames)


def test_default_durations_match_tdt_and_ctc() -> None:
    """Pin the default bucket sets. TDT covers 10-60s in 10s steps;
    CTC uses a coarser set because its 1.1B encoder doesn't fit 6
    copies on a 24GB L4 (see ``bucket_spec.CTC_DEFAULT_BUCKET_DURATIONS_S``).
    """
    assert DEFAULT_BUCKET_DURATIONS_S == (10, 20, 30, 40, 50, 60)
    assert CTC_DEFAULT_BUCKET_DURATIONS_S == (15, 30, 45, 60)


# ---------------------------------------------------------------------------
# mel_frames_for_audio: mirrors mel_graph.py's STFT frame-count formula.
# ---------------------------------------------------------------------------


def test_mel_frames_for_audio_zero() -> None:
    """Zero-length audio still produces one frame (the center-pad alone
    gives one full window)."""
    assert mel_frames_for_audio(0) == 1


def test_mel_frames_for_audio_exact_10_seconds() -> None:
    """Exactly 10s of 16kHz audio = 160000 samples → 1001 mel frames."""
    assert mel_frames_for_audio(SAMPLE_RATE * 10) == 1001


def test_mel_frames_for_audio_exact_60_seconds() -> None:
    """Exactly 60s of 16kHz audio = 960000 samples → 6001 mel frames."""
    assert mel_frames_for_audio(SAMPLE_RATE * 60) == 6001


def test_mel_frames_for_audio_matches_bucket_math() -> None:
    """For any default bucket, the maximum raw (pre-center-pad) audio
    length that fits in the bucket produces exactly ``bucket.mel_frames``
    when measured with ``mel_frames_for_audio``.

    ``bucket.audio_samples`` is the **post-center-pad** length that the
    compiled mel graph expects, so the raw pre-pad audio length is
    ``bucket.audio_samples - N_FFT`` (center-pad is ``N_FFT // 2`` on
    each side). This is the invariant ``_prepare_audio_for_bucket``
    relies on: raw audio that passes ``select_bucket`` via
    ``mel_frames_for_audio`` will land in a bucket that the mel graph
    can consume without exceeding the compiled mel-frame count.
    """
    for bucket in DEFAULT_BUCKETS:
        max_raw_audio = bucket.audio_samples - N_FFT
        assert mel_frames_for_audio(max_raw_audio) == bucket.mel_frames


# ---------------------------------------------------------------------------
# select_bucket: "smallest bucket >= mel_frames" with truncation flag.
# Off-by-one here routes short clips into the wrong bucket or drops audio
# that should fit — worth pinning each boundary.
# ---------------------------------------------------------------------------


def test_select_bucket_exact_boundary_hits_smallest() -> None:
    """mel_frames == bucket.mel_frames should hit that exact bucket,
    not spill to the next one."""
    bucket, truncated = select_bucket(1008, DEFAULT_BUCKETS)
    assert bucket.duration_s == 10
    assert not truncated


def test_select_bucket_one_over_boundary_spills() -> None:
    """mel_frames == bucket.mel_frames + 1 should spill into the next
    larger bucket."""
    bucket, truncated = select_bucket(1009, DEFAULT_BUCKETS)
    assert bucket.duration_s == 20
    assert not truncated


def test_select_bucket_short_audio_picks_smallest() -> None:
    """A single-frame input goes to the smallest bucket."""
    bucket, truncated = select_bucket(1, DEFAULT_BUCKETS)
    assert bucket.duration_s == 10
    assert not truncated


def test_select_bucket_largest_exact_fits() -> None:
    """mel_frames == largest bucket's mel_frames is not a truncation."""
    bucket, truncated = select_bucket(6008, DEFAULT_BUCKETS)
    assert bucket.duration_s == 60
    assert not truncated


def test_select_bucket_over_largest_truncates() -> None:
    """mel_frames > largest bucket returns that bucket with
    ``was_truncated=True`` so the caller can log/truncate."""
    bucket, truncated = select_bucket(6009, DEFAULT_BUCKETS)
    assert bucket.duration_s == 60
    assert truncated


def test_select_bucket_far_over_largest_still_returns_largest() -> None:
    """No matter how large the input is, the fallback is the largest
    bucket (never None / never raising)."""
    bucket, truncated = select_bucket(1_000_000, DEFAULT_BUCKETS)
    assert bucket.duration_s == 60
    assert truncated


# ---------------------------------------------------------------------------
# build_buckets: sort, validate, and produce ParakeetBucket instances.
# ---------------------------------------------------------------------------


def test_build_buckets_sorts_input() -> None:
    """Input order doesn't matter — ``select_bucket`` assumes ascending
    order so ``build_buckets`` sorts for us."""
    shuffled = build_buckets((60, 10, 30))
    assert [b.duration_s for b in shuffled] == [10, 30, 60]


def test_build_buckets_single_duration() -> None:
    """A 1-bucket configuration is valid and useful (degenerate
    non-bucketed encoder, e.g. for single-shape testing)."""
    buckets = build_buckets((20,))
    assert len(buckets) == 1
    assert buckets[0].duration_s == 20


def test_build_buckets_empty_raises() -> None:
    """An empty bucket set is an error, not silently a fallback."""
    with pytest.raises(ValueError, match="must not be empty"):
        build_buckets(())


def test_build_buckets_produces_frozen_dataclass() -> None:
    """Buckets are immutable — they're passed around as keys / look-up
    values, and mutation would be a footgun."""
    buckets = build_buckets((10,))
    with pytest.raises(AttributeError):
        buckets[0].duration_s = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# End-to-end sanity: a few realistic audio lengths route to the buckets
# the docstrings promise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration_s,expected_bucket_s",
    [
        (1.8, 10),  # shortest LibriSpeech clip
        (6.7, 10),  # LibriSpeech mean
        (10.0, 10),  # exactly at the smallest bucket's nominal duration
        (10.1, 20),  # one-tenth over → spill to 20s
        (22.3, 30),  # longest in the current LibriSpeech test set
        (55.0, 60),  # near the 60s ceiling
        (60.0, 60),  # exactly at the largest bucket's nominal duration
        (75.0, 60),  # beyond ceiling → truncate to 60s
    ],
)
def test_realistic_audio_routes_to_expected_bucket(
    duration_s: float, expected_bucket_s: int
) -> None:
    """Smoke-test the full ``mel_frames_for_audio`` -> ``select_bucket``
    chain against audio lengths that span the LibriSpeech range plus
    the over-ceiling truncation path."""
    num_samples = int(duration_s * SAMPLE_RATE)
    bucket, truncated = select_bucket(
        mel_frames_for_audio(num_samples), DEFAULT_BUCKETS
    )
    assert bucket.duration_s == expected_bucket_s
    assert truncated == (duration_s > 60)


def test_parakeet_bucket_is_hashable() -> None:
    """Buckets end up as dict keys (``self._encoder_models[bucket.mel_frames]``);
    being ``frozen=True`` already makes them hashable — this is a
    regression guard in case the dataclass decorator is ever changed."""
    bucket = ParakeetBucket(
        duration_s=10, audio_samples=161632, mel_frames=1008, encoder_frames=126
    )
    assert hash(bucket) == hash(bucket)
    assert {bucket: "ok"}[bucket] == "ok"
