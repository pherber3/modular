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
"""Defines the Parakeet-CTC pipeline model.

Implements the PipelineModel interface for non-autoregressive CTC inference:
mel spectrogram in, CTC logits out, no KV cache.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)
from transformers import AutoConfig, PreTrainedTokenizer

from .audio import extract_mel, normalize_per_feature, read_wav
from .decode import ctc_greedy_decode
from .graph import build_graph
from .model_config import ParakeetModelConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class ParakeetInputs(ModelInputs):
    """Model inputs for Parakeet-CTC inference."""

    input_features: Buffer  # (batch, num_frames, num_mel_bins)


class ParakeetPipelineModel(PipelineModel[TextContext]):
    """Pipeline model for Parakeet-CTC ASR inference."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        # ASR processes variable-length audio; return a large upper bound.
        encoder_config = getattr(
            huggingface_config, "encoder_config", huggingface_config
        )
        return getattr(encoder_config, "max_position_embeddings", 100000)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, ParakeetInputs)
        model_outputs = self.model.execute(model_inputs.input_features)
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])

    def decode(
        self, model_inputs: ModelInputs, tokenizer: PreTrainedTokenizer
    ) -> list[str]:
        """Run encoder + CTC greedy decode, returning transcribed text.

        Args:
            model_inputs: Mel spectrogram inputs.
            tokenizer: HuggingFace tokenizer for ID-to-text conversion.

        Returns:
            List of transcribed strings, one per batch element.
        """
        outputs = self.execute(model_inputs)
        assert outputs.logits is not None
        logits = np.from_dlpack(outputs.logits).copy()
        return ctc_greedy_decode(logits, tokenizer, blank_id=1024)

    def transcribe(
        self, audio_bytes: bytes, tokenizer: PreTrainedTokenizer
    ) -> str:
        """Full audio-to-text pipeline: mel extraction → encoder → CTC decode."""
        audio_data, sample_rate = read_wav(audio_bytes)
        if sample_rate != 16000:
            raise ValueError(
                f"Expected 16kHz audio, got {sample_rate}Hz. "
                "Please resample before sending."
            )

        config = ParakeetModelConfig.initialize(self.pipeline_config)
        features = extract_mel(
            audio_data,
            n_mels=config.num_mel_bins,
            preemphasis=0.97,
            periodic_window=False,
        )
        features = normalize_per_feature(features)

        model_inputs = ParakeetInputs(
            input_features=Buffer.from_numpy(features)
        )
        texts = self.decode(model_inputs, tokenizer)
        return texts[0]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ParakeetInputs:
        if len(replica_batches) > 1:
            raise ValueError("Parakeet model does not support DP>1")

        raise NotImplementedError(
            "Audio preprocessing (mel spectrogram extraction) is not yet "
            "wired in. prepare_initial_token_inputs cannot produce real "
            "model inputs."
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ParakeetInputs:
        raise NotImplementedError(
            "Parakeet-CTC is non-autoregressive and does not support "
            "next-token generation."
        )

    def load_model(self, session: InferenceSession) -> Model:
        timer = CompilationTimer("Parakeet-CTC")

        if self.adapter:
            state_dict = self.adapter(dict(self.weights.items()))
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        config = ParakeetModelConfig.initialize(self.pipeline_config)
        graph = build_graph(config, state_dict)
        timer.mark_build_complete()

        model = session.load(graph, weights_registry=state_dict)
        timer.done()
        return model
