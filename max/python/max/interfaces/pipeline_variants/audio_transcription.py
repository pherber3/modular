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
"""Interfaces and response structures for audio transcription in the MAX API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import msgspec
from max.interfaces.context import BaseContext
from max.interfaces.pipeline import PipelineInputs, PipelineOutput
from max.interfaces.request import RequestID


@runtime_checkable
class AudioTranscriptionContext(BaseContext, Protocol):
    """Protocol defining the interface for audio transcription contexts.

    An ``AudioTranscriptionContext`` represents model inputs for ASR pipelines,
    managing the audio data needed for speech-to-text transcription. This is a
    single-step operation (non-autoregressive at the serving level).
    """

    @property
    def audio_data(self) -> bytes:
        """The raw audio bytes to transcribe.

        Returns:
            Raw audio file content (wav, mp3, flac, etc.).
        """
        ...

    @property
    def model_name(self) -> str:
        """The name of the transcription model to use.

        Returns:
            A string identifying the specific ASR model for this request.
        """
        ...

    @property
    def language(self) -> str | None:
        """Optional ISO-639-1 language code for the input audio.

        Returns:
            Language code string, or None for auto-detection.
        """
        ...


AudioTranscriptionContextType = TypeVar(
    "AudioTranscriptionContextType", bound=AudioTranscriptionContext
)


@dataclass(frozen=True)
class AudioTranscriptionInputs(PipelineInputs):
    """Batched inputs for an audio transcription pipeline step."""

    batches: list[dict[RequestID, AudioTranscriptionContext]]

    @property
    def batch(self) -> dict[RequestID, AudioTranscriptionContext]:
        """Returns merged batches."""
        return {k: v for batch in self.batches for k, v in batch.items()}


class AudioTranscriptionOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """Response structure for audio transcription.

    Configuration:
        text: The transcribed text.
    """

    text: str
    """The transcribed text."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the transcription is complete.

        Returns:
            bool: Always True, as transcription is a single-step operation.
        """
        return True


def _check_transcription_output_implements_pipeline_output(
    x: AudioTranscriptionOutput,
) -> PipelineOutput:
    return x
