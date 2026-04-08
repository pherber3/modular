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
mel spectrogram in, on-device argmaxed int32 predicted token ids out, no
KV cache. Compiles one encoder + mel graph per bucket in
``config.bucket_durations_s`` and dispatches per-utterance to the smallest
fitting bucket.
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
from .bucket_spec import (
    N_FFT,
    SAMPLE_RATE,
    ParakeetBucket,
    build_buckets,
    mel_frames_for_audio,
    select_bucket,
)
from .decode import ctc_greedy_decode
from .graph import build_graph
from .mel_graph import build_mel_graph
from .model_config import ParakeetModelConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class ParakeetInputs(ModelInputs):
    """Model inputs for Parakeet-CTC inference.

    Attributes:
        input_features: Mel features sized for this bucket's encoder graph,
            shape ``(1, bucket.mel_frames, num_mel_bins)``.
        bucket_mel_frames: The mel-frame count of the bucket this input
            was prepared for. Used by ``execute()`` to dispatch to the
            matching encoder model in ``self._encoder_models``.
        bucket_encoder_frames: The bucket's true (un-padded) encoder-frame
            count. Used by ``decode()`` to slice the padded predicted_ids
            tail before dedup.
    """

    input_features: Buffer
    bucket_mel_frames: int
    bucket_encoder_frames: int


class ParakeetPipelineModel(PipelineModel[ASRContext]):
    """Pipeline model for Parakeet-CTC ASR inference.

    Compiles N encoder graphs (one per bucket in
    ``config.bucket_durations_s``) and N matching mel preprocessing
    graphs (when running on GPU). Each utterance is routed to the
    smallest bucket that fits its audio length. The encoder graphs
    perform on-device argmax over the vocab dimension, so the host
    side only receives ``(1, max_encoder_frames)`` int32 predicted
    token ids (mirrors the TDT on-device decision pattern).
    """

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

        # Build the bucket set from config and pre-compute the global
        # max encoder-frame count. Every bucket's encoder graph pads
        # its predicted_ids output to this size, so the host-side slice
        # always knows where to cut.
        self._buckets: tuple[ParakeetBucket, ...] = build_buckets(
            self.config.bucket_durations_s
        )
        self._max_encoder_frames: int = max(
            b.encoder_frames for b in self._buckets
        )

        self._encoder_models, self._mel_models = self._load_encoders(session)

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
        """Run the encoder graph for this input's bucket.

        Note: for CTC the ``ModelOutputs.logits`` field carries int32
        argmaxed predicted token ids ``(1, max_encoder_frames)``, **not**
        raw float32 logits. The on-device argmax lives inside the
        compiled encoder graph (see ``graph.py::build_graph``). Field
        name is kept to avoid churning the base ``ModelOutputs`` class.
        """
        assert isinstance(model_inputs, ParakeetInputs)
        encoder_model = self._encoder_models[model_inputs.bucket_mel_frames]
        model_outputs = encoder_model.execute(model_inputs.input_features)
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])

    def decode(
        self, model_inputs: ModelInputs, tokenizer: PreTrainedTokenizer
    ) -> list[str]:
        """Run encoder + CTC greedy decode, returning transcribed text.

        Host-side work is minimal: transfer the int32 predicted_ids
        tensor, slice off the bucket's zero-padded tail, dedup + strip
        blanks in ``ctc_greedy_decode``.
        """
        assert isinstance(model_inputs, ParakeetInputs)
        outputs = self.execute(model_inputs)
        assert outputs.logits is not None
        predicted_ids = np.from_dlpack(
            outputs.logits.to(self._cpu_device)
        ).copy()
        predicted_ids = predicted_ids[:, : model_inputs.bucket_encoder_frames]
        return ctc_greedy_decode(
            predicted_ids, tokenizer, blank_id=self.config.blank_id
        )

    def _select_bucket(self, num_audio_samples: int) -> ParakeetBucket:
        """Pick the smallest compiled bucket that fits this audio length.

        Logs a warning and truncates if the audio exceeds the largest
        bucket — current behavior, just at the bucket-set ceiling instead
        of the old hardcoded 20s.
        """
        bucket, was_truncated = select_bucket(
            mel_frames_for_audio(num_audio_samples), self._buckets
        )
        if was_truncated:
            logger.warning(
                "Audio length %.1fs exceeds largest bucket (%ds); truncating",
                num_audio_samples / SAMPLE_RATE,
                bucket.duration_s,
            )
        return bucket

    def _prepare_audio_for_bucket(
        self, audio: np.ndarray, bucket: ParakeetBucket
    ) -> np.ndarray:
        """Apply preemphasis, center-pad, and fit audio to ``bucket.audio_samples``.

        Returns ``(1, bucket.audio_samples)`` float32. The compiled mel
        graph for this bucket expects exactly this size.
        """
        # Preemphasis (HF Parakeet CTC uses 0.97).
        audio = np.append(audio[0:1], audio[1:] - 0.97 * audio[:-1])
        pad_length = N_FFT // 2
        padded = np.pad(audio, (pad_length, pad_length), mode="constant")
        if len(padded) < bucket.audio_samples:
            padded = np.pad(padded, (0, bucket.audio_samples - len(padded)))
        elif len(padded) > bucket.audio_samples:
            padded = padded[: bucket.audio_samples]
        return padded.reshape(1, -1).astype(np.float32)

    def _run_for_bucket(
        self,
        audio: np.ndarray,
        bucket: ParakeetBucket,
        tokenizer: PreTrainedTokenizer,
    ) -> list[str]:
        """End-to-end mel → encoder → CTC decode for a single bucket.

        CTC is non-autoregressive, so the tokenizer is threaded through
        to the decode step. Unlike the TDT equivalent which returns raw
        token IDs, this returns decoded text directly.
        """
        if self._mel_models is not None:
            mel_model = self._mel_models[bucket.mel_frames]
            padded_audio = self._prepare_audio_for_bucket(audio, bucket)
            audio_buf = Buffer.from_numpy(padded_audio).to(self.devices[0])
            mel_buf = mel_model.execute(audio_buf)[0]
            model_inputs = ParakeetInputs(
                input_features=mel_buf,
                bucket_mel_frames=bucket.mel_frames,
                bucket_encoder_frames=bucket.encoder_frames,
            )
        else:
            # CPU fallback: extract mel on host, pad/truncate to bucket.mel_frames
            features = extract_mel(
                audio,
                n_mels=self.config.num_mel_bins,
                preemphasis=0.97,
                periodic_window=False,
            )
            features = normalize_per_feature(features)
            features = features.astype(np.float32)
            if features.shape[1] < bucket.mel_frames:
                pad_w = [
                    (0, 0),
                    (0, bucket.mel_frames - features.shape[1]),
                    (0, 0),
                ]
                features = np.pad(features, pad_w)
            elif features.shape[1] > bucket.mel_frames:
                features = features[:, : bucket.mel_frames, :]
            model_inputs = ParakeetInputs(
                input_features=Buffer.from_numpy(features).to(self.devices[0]),
                bucket_mel_frames=bucket.mel_frames,
                bucket_encoder_frames=bucket.encoder_frames,
            )

        return self.decode(model_inputs, tokenizer)

    def transcribe(
        self, audio_bytes: bytes, tokenizer: PreTrainedTokenizer
    ) -> str:
        """Full audio-to-text pipeline: bucket select → mel → encoder → decode."""
        audio, sample_rate = read_wav(audio_bytes)
        if sample_rate != 16000:
            raise ValueError(
                f"Expected 16kHz audio, got {sample_rate}Hz. "
                "Please resample before sending."
            )
        bucket = self._select_bucket(len(audio))
        texts = self._run_for_bucket(audio, bucket, tokenizer)
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

    def _load_encoders(
        self,
        session: InferenceSession,
    ) -> tuple[dict[int, Model], dict[int, Model] | None]:
        """Compile one encoder + mel graph per bucket.

        Returns:
            ``(encoder_models, mel_models)`` keyed on ``bucket.mel_frames``.
            ``mel_models`` is ``None`` on CPU (the CPU path uses numpy
            mel extraction in ``transcribe``).
        """
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        encoder_models: dict[int, Model] = {}
        mel_models: dict[int, Model] | None = None
        on_gpu = self.config.device != DeviceRef.CPU()
        if on_gpu:
            mel_models = {}

        with CompilationTimer("Parakeet-CTC (all buckets)") as timer:
            for bucket in self._buckets:
                graph = build_graph(
                    self.config,
                    state_dict,
                    num_frames=bucket.mel_frames,
                    pad_to_encoder_frames=self._max_encoder_frames,
                )
                encoder_models[bucket.mel_frames] = session.load(
                    graph, weights_registry=state_dict
                )
                if mel_models is not None:
                    mel_graph = build_mel_graph(
                        n_mels=self.config.num_mel_bins,
                        max_audio_samples=bucket.audio_samples,
                        periodic_window=False,
                        device=self.config.device,
                    )
                    mel_models[bucket.mel_frames] = session.load(mel_graph)
                logger.info(
                    "Compiled CTC bucket: %ds (mel=%d, enc=%d)",
                    bucket.duration_s,
                    bucket.mel_frames,
                    bucket.encoder_frames,
                )
            timer.mark_build_complete()

        self._cpu_device = load_devices([DeviceSpec.cpu()])[0]
        return encoder_models, mel_models

    def load_model(self, session: InferenceSession) -> Model:
        # Convention only — not used by the base ``PipelineModel``.
        # Returns the largest bucket so any caller using ``self.model`` as
        # a sentinel still gets a valid encoder. Actual loading is done
        # in ``__init__`` via ``_load_encoders``.
        largest = self._buckets[-1]
        return self._encoder_models[largest.mel_frames]
