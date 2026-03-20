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

"""Pipeline for running audio transcription (ASR)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import numpy as np
import numpy.typing as npt
from max.driver import load_devices
from max.engine import InferenceSession
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import (
    AudioTranscriptionContext,
    AudioTranscriptionInputs,
    AudioTranscriptionOutput,
    BaseContextType,
    Pipeline,
    PipelineTokenizer,
    RequestID,
    TextGenerationRequest,
)
from max.nn.transformer import ReturnLogits
from max.profiler import traced

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .interfaces import PipelineModel

logger = logging.getLogger("max.pipelines")

AudioTranscriptionPipelineType = Pipeline[
    AudioTranscriptionInputs, AudioTranscriptionOutput
]


@final
class AudioTranscriptionPipeline(AudioTranscriptionPipelineType):
    """Worker-side pipeline for audio-to-text transcription (ASR).

    Delegates audio preprocessing and decoding to the architecture-specific
    PipelineModel via its ``transcribe()`` method. This keeps CTC vs TDT
    differences encapsulated in each model.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[AudioTranscriptionContext]],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            BaseContextType, npt.NDArray[np.integer[Any]], TextGenerationRequest
        ],
    ) -> None:
        self._pipeline_config = pipeline_config
        self._weight_adapters = weight_adapters
        self._tokenizer = tokenizer

        devices = load_devices(self._pipeline_config.model.device_specs)
        session = InferenceSession(devices=[*devices])
        self._pipeline_config.configure_session(session)

        if not self._pipeline_config.model.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        weight_paths = [
            (
                Path(p)
                if Path(p).is_absolute()
                else Path(self._pipeline_config.model.model_path) / p
            )
            for p in self._pipeline_config.model.weight_path
        ]

        if all(p.exists() for p in weight_paths):
            pass  # Local files, no download needed
        else:
            from .hf_utils import download_weight_files

            weight_paths = download_weight_files(
                huggingface_model_id=self._pipeline_config.model.huggingface_weight_repo_id,
                filenames=[
                    str(x) for x in self._pipeline_config.model.weight_path
                ],
                revision=self._pipeline_config.model.huggingface_weight_revision,
                force_download=self._pipeline_config.model.force_download,
            )

        weights = load_weights(weight_paths)
        huggingface_config = self._pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"Audio transcription pipeline requires a config for "
                f"'{self._pipeline_config.model.model_path}'."
            )

        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=devices,
            kv_cache_config=self._pipeline_config.model.kv_cache,
            weights=weights,
            adapter=self._weight_adapters.get(
                weights_format(weight_paths), None
            ),
            return_logits=ReturnLogits.ALL,
        )

    @traced
    def execute(
        self,
        inputs: AudioTranscriptionInputs,
    ) -> dict[RequestID, AudioTranscriptionOutput]:
        """Processes a batch of audio inputs and returns transcriptions."""
        res: dict[RequestID, AudioTranscriptionOutput] = {}
        for request_id, context in inputs.batch.items():
            text = self._pipeline_model.transcribe(
                context.audio_data, self._tokenizer.delegate
            )
            res[request_id] = AudioTranscriptionOutput(text=text)
        return res

    def release(self, request_id: RequestID) -> None:
        """Release resources for the request (no-op for transcription)."""
