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
from max.driver import Buffer, Device, DeviceSpec, load_devices
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs
from max.nn.transformer import ReturnLogits
from max.pipelines.core import ASRContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig, PreTrainedTokenizer

from .audio import extract_mel, normalize_per_feature, read_wav
from .decode import ctc_greedy_decode
from .graph import build_graph
from .mel_graph import MAX_AUDIO_SAMPLES, N_FFT, build_mel_graph
from .model_config import ParakeetModelConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class ParakeetInputs(ModelInputs):
    """Model inputs for Parakeet-CTC inference."""

    input_features: Buffer  # (batch, num_frames, num_mel_bins)


class ParakeetPipelineModel(PipelineModel[ASRContext]):
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
        self.config = ParakeetModelConfig.initialize(self.pipeline_config)
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
        return ctc_greedy_decode(
            logits, tokenizer, blank_id=self.config.blank_id
        )

    def _prepare_audio_for_mel_graph(
        self, audio: np.ndarray, preemphasis: float = 0.97
    ) -> np.ndarray:
        """Apply preemphasis, center-pad, and fit to MAX_AUDIO_SAMPLES.

        Returns audio as (1, MAX_AUDIO_SAMPLES) float32 numpy array,
        padded or truncated to match the compiled mel graph's fixed input.
        """
        if preemphasis > 0:
            audio = np.append(audio[0:1], audio[1:] - preemphasis * audio[:-1])
        pad_length = N_FFT // 2
        padded = np.pad(audio, (pad_length, pad_length), mode="constant")
        if len(padded) < MAX_AUDIO_SAMPLES:
            padded = np.pad(padded, (0, MAX_AUDIO_SAMPLES - len(padded)))
        elif len(padded) > MAX_AUDIO_SAMPLES:
            padded = padded[:MAX_AUDIO_SAMPLES]
        return padded.reshape(1, -1).astype(np.float32)

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

        if self._mel_model is not None:
            # GPU mel extraction path — features stay on device.
            padded_audio = self._prepare_audio_for_mel_graph(audio_data)
            audio_buf = Buffer.from_numpy(padded_audio).to(self.devices[0])
            mel_buf = self._mel_model.execute(audio_buf)[0]
            model_inputs = ParakeetInputs(input_features=mel_buf)
        else:
            # CPU fallback.
            features = extract_mel(
                audio_data,
                n_mels=self.config.num_mel_bins,
                preemphasis=0.97,
                periodic_window=False,
            )
            features = normalize_per_feature(features)
            model_inputs = ParakeetInputs(
                input_features=Buffer.from_numpy(features).to(self.devices[0])
            )

        texts = self.decode(model_inputs, tokenizer)
        return texts[0]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ASRContext]],
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
        with CompilationTimer("Parakeet-CTC") as timer:
            state_dict = parse_state_dict_from_weights(
                self.pipeline_config, self.weights, self.adapter
            )

            graph = build_graph(self.config, state_dict)
            timer.mark_build_complete()

            model = session.load(graph, weights_registry=state_dict)

        self._mel_model: Model | None = None
        self._cpu_device = load_devices([DeviceSpec.cpu()])[0]
        if self.config.device != DeviceRef.CPU():
            mel_graph = build_mel_graph(
                n_mels=self.config.num_mel_bins,
                max_audio_samples=MAX_AUDIO_SAMPLES,
                periodic_window=False,
                device=self.config.device,
            )
            self._mel_model = session.load(mel_graph)
            logger.info("Loaded GPU mel extraction graph")

        return model
