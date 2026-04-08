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
"""Defines the Parakeet-TDT pipeline model.

The encoder runs as a compiled MAX graph. The decoder (LSTM prediction
network + joint network + greedy decode loop) also runs as compiled MAX
graphs — a projection graph and a decoder step graph, both loaded into
the same InferenceSession. Encoder output stays as a Buffer on-device,
eliminating any device transfer overhead.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import huggingface_hub
import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DeviceSpec, DLPackArray, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn import Linear
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

from ..parakeet.audio import extract_mel, normalize_per_feature, read_wav
from ..parakeet.bucket_spec import (
    N_FFT,
    SAMPLE_RATE,
    ParakeetBucket,
    build_buckets,
    mel_frames_for_audio,
    select_bucket,
)
from ..parakeet.encoder import ParakeetEncoder
from ..parakeet.mel_graph import build_mel_graph
from .decoder_graph import (
    TDTGraphDecoder,
    build_decoder_step_graph,
    convert_decoder_state_dict,
)
from .model_config import TDTModelConfig

NDFloat = npt.NDArray[np.floating]

logger = logging.getLogger("max.pipelines")


@dataclass
class ParakeetTDTInputs(ModelInputs):
    """Model inputs for Parakeet-TDT inference.

    Attributes:
        input_features: Mel features sized for this bucket's encoder graph,
            shape ``(1, bucket.mel_frames, num_mel_bins)``.
        bucket_mel_frames: The mel-frame count of the bucket this input
            was prepared for. Used by ``execute()`` to dispatch to the
            matching encoder model in ``self._encoder_models``.
        bucket_encoder_frames: The bucket's true (un-padded) encoder-frame
            count. Passed to the decoder loop as ``actual_t`` so it stops
            at the right place instead of iterating the zero-padded tail.
    """

    input_features: Buffer
    bucket_mel_frames: int
    bucket_encoder_frames: int


def build_graph(
    config: TDTModelConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
    num_frames: int,
    pad_to_encoder_frames: int,
) -> Graph:
    """Build the encoder + joint-encoder-projection computation graph.

    The graph takes mel spectrogram input and returns the pre-projected
    encoder output ``(1, pad_to_encoder_frames, joint_hidden)``. Fusing
    the projection here eliminates a separate ``model.execute()`` call per
    utterance.

    Args:
        config: TDT model configuration.
        state_dict: Encoder weights.
        num_frames: Fixed mel-frame count this graph compiles against.
            Each bucket builds its own graph at its own ``num_frames``.
        pad_to_encoder_frames: The global maximum encoder-frame count
            (across all buckets). The output ``enc_projected`` is
            zero-padded along the time axis to this size so all bucket
            graphs return the same shape — the decoder step graph is
            compiled against this same constant, and the actual loop
            bound (``actual_t``) comes from the bucket the caller routed
            to so the padded tail is never iterated.
    """
    input_type = TensorType(
        DType.float32,
        shape=[1, num_frames, config.num_mel_bins],
        device=config.device,
    )

    with Graph("parakeet_tdt_encoder", input_types=[input_type]) as graph:
        encoder = ParakeetEncoder(config)
        encoder.load_state_dict(state_dict, strict=False)
        input_features = graph.inputs[0].tensor
        hidden_states = encoder(input_features)

        enc_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.joint_hidden,
            dtype=DType.float32,
            device=config.device,
            has_bias=True,
            name="enc_proj",
        )
        enc_proj.load_state_dict(state_dict, strict=False)
        enc_projected = enc_proj(hidden_states)

        # Zero-pad the time axis to the global max so every bucket's
        # encoder graph returns ``[1, pad_to_encoder_frames, joint_hidden]``
        # — required so a single decoder step graph + CUDA-graph capture
        # serves every audio length without a per-bucket graph rebuild.
        # Uses ``ops.pad`` so the compiler emits a single constant-pad
        # kernel rather than materializing a zero tensor + concat.
        bucket_encoder_frames = num_frames // 8
        pad_len = pad_to_encoder_frames - bucket_encoder_frames
        if pad_len > 0:
            # paddings: [before_batch,  after_batch,
            #            before_time,   after_time,
            #            before_hidden, after_hidden]
            enc_projected = ops.pad(enc_projected, [0, 0, 0, pad_len, 0, 0])

        graph.output(enc_projected)

    return graph


class ParakeetTDTPipelineModel(PipelineModel[ASRContext]):
    """Pipeline model for Parakeet-TDT ASR inference.

    Compiles N encoder graphs (one per bucket in
    ``tdt_config.bucket_durations_s``), N matching mel preprocessing
    graphs (when running on GPU), and a single decoder step graph
    compiled at the largest bucket's encoder-frame count. Each utterance
    is routed to the smallest bucket that fits its audio length.

    All graphs run on the same device (CPU or GPU). Encoder output flows
    between graphs as Buffers with no host transfer.
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
        self.tdt_config = TDTModelConfig.initialize(self.pipeline_config)

        # Build the bucket set from config and pre-compute the global
        # max encoder-frame count. The decoder step graph compiles once
        # at this size and every encoder bucket pads its output to match.
        self._buckets: tuple[ParakeetBucket, ...] = build_buckets(
            self.tdt_config.bucket_durations_s
        )
        self._max_encoder_frames: int = max(
            b.encoder_frames for b in self._buckets
        )

        # Load decoder/joint NPZ once, split into dicts for encoder
        # projection and decoder step graphs.
        npz_path = huggingface_hub.hf_hub_download(
            self.pipeline_config.model.model_path, "decoder_joint.npz"
        )
        npz_weights = dict(np.load(npz_path))
        proj_dict, pred_dict, joint_dict = convert_decoder_state_dict(
            npz_weights
        )

        self._encoder_models, self._mel_models = self._load_encoders(
            session, proj_dict
        )
        self._load_decoder(session, pred_dict, joint_dict)

    def _load_encoders(
        self,
        session: InferenceSession,
        proj_dict: dict[str, np.ndarray],
    ) -> tuple[dict[int, Model], dict[int, Model] | None]:
        """Compile one encoder + mel graph per bucket.

        Returns:
            ``(encoder_models, mel_models)`` keyed on
            ``bucket.mel_frames``. ``mel_models`` is ``None`` on CPU
            (the CPU path uses numpy mel extraction).
        """
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )
        for key, arr in proj_dict.items():
            state_dict[key] = WeightData.from_numpy(arr.astype(np.float32), key)

        encoder_models: dict[int, Model] = {}
        mel_models: dict[int, Model] | None = None
        on_gpu = self.tdt_config.device != DeviceRef.CPU()
        if on_gpu:
            mel_models = {}

        with CompilationTimer("Parakeet-TDT (all buckets)") as timer:
            for bucket in self._buckets:
                graph = build_graph(
                    self.tdt_config,
                    state_dict,
                    num_frames=bucket.mel_frames,
                    pad_to_encoder_frames=self._max_encoder_frames,
                )
                encoder_models[bucket.mel_frames] = session.load(
                    graph, weights_registry=state_dict
                )
                if mel_models is not None:
                    mel_graph = build_mel_graph(
                        n_mels=self.tdt_config.num_mel_bins,
                        max_audio_samples=bucket.audio_samples,
                        periodic_window=True,
                        device=self.tdt_config.device,
                    )
                    mel_models[bucket.mel_frames] = session.load(mel_graph)
                logger.info(
                    "Compiled TDT bucket: %ds (mel=%d, enc=%d)",
                    bucket.duration_s,
                    bucket.mel_frames,
                    bucket.encoder_frames,
                )
            timer.mark_build_complete()

        self._cpu_device = load_devices([DeviceSpec.cpu()])[0]
        return encoder_models, mel_models

    def _load_decoder(
        self,
        session: InferenceSession,
        pred_dict: dict[str, np.ndarray],
        joint_dict: dict[str, np.ndarray],
    ) -> None:
        """Compile the single decoder step graph at ``max_encoder_len``."""
        with CompilationTimer("TDT-DecoderStep") as timer:
            dec_graph = build_decoder_step_graph(
                self.tdt_config,
                pred_dict,
                joint_dict,
                max_encoder_len=self._max_encoder_frames,
            )
            timer.mark_build_complete()
            dec_weights = {**pred_dict, **joint_dict}
            decoder_step_model = session.load(
                dec_graph, weights_registry=dec_weights
            )

        cpu_device = load_devices([DeviceSpec.cpu()])[0]

        self.graph_decoder = TDTGraphDecoder(
            decoder_step_model=decoder_step_model,
            config=self.tdt_config,
            device=self.devices[0],
            cpu_device=cpu_device,
            max_encoder_len=self._max_encoder_frames,
        )
        logger.info(
            "Loaded TDT decoder (MAX graph, device=%s): "
            "decoder step graph compiled at max_encoder_len=%d",
            self.tdt_config.device,
            self._max_encoder_frames,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        encoder_config = getattr(
            huggingface_config, "encoder_config", huggingface_config
        )
        return getattr(encoder_config, "max_position_embeddings", 100000)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, ParakeetTDTInputs)
        encoder_model = self._encoder_models[model_inputs.bucket_mel_frames]
        model_outputs = encoder_model.execute(model_inputs.input_features)
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])

    def decode(self, model_inputs: ModelInputs) -> list[list[int]]:
        """Run encoder + TDT greedy decode, returning token ID sequences.

        Encoder output stays as a Buffer on-device, padded to
        ``max_encoder_len``. The decoder loop bound (``actual_t``) is the
        bucket's true encoder-frame count, so short clips don't iterate
        the zero-padded tail.
        """
        assert isinstance(model_inputs, ParakeetTDTInputs)
        outputs = self.execute(model_inputs)
        assert outputs.logits is not None
        return self.graph_decoder.decode(
            outputs.logits, actual_t=model_inputs.bucket_encoder_frames
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

    def prepare_mel_input(
        self, features: NDFloat, bucket: ParakeetBucket
    ) -> ParakeetTDTInputs:
        """Prepare CPU-extracted mel features for a specific bucket.

        Applies feature normalization (if configured), pads/truncates to
        ``bucket.mel_frames``, and wraps in a Buffer. Used by the CPU
        fallback path; GPU callers go through ``_run_for_bucket`` and the
        compiled mel graph.

        Args:
            features: Mel spectrogram, shape ``(batch, num_frames, num_mel_bins)``.
            bucket: The bucket this input is being prepared for.
        """
        if self.tdt_config.normalize_features == "per_feature":
            features = normalize_per_feature(features)
        features = features.astype(np.float32)
        if features.shape[1] < bucket.mel_frames:
            pad_width = [
                (0, 0),
                (0, bucket.mel_frames - features.shape[1]),
                (0, 0),
            ]
            features = np.pad(features, pad_width)
        elif features.shape[1] > bucket.mel_frames:
            features = features[:, : bucket.mel_frames, :]
        return ParakeetTDTInputs(
            input_features=Buffer.from_numpy(features).to(self.devices[0]),
            bucket_mel_frames=bucket.mel_frames,
            bucket_encoder_frames=bucket.encoder_frames,
        )

    def _prepare_audio_for_bucket(
        self, audio: np.ndarray, bucket: ParakeetBucket
    ) -> np.ndarray:
        """Center-pad and fit audio to ``bucket.audio_samples``.

        Returns ``(1, bucket.audio_samples)`` float32. The compiled mel
        graph for this bucket expects exactly this size.
        """
        pad_length = N_FFT // 2
        padded = np.pad(audio, (pad_length, pad_length), mode="constant")
        if len(padded) < bucket.audio_samples:
            padded = np.pad(padded, (0, bucket.audio_samples - len(padded)))
        elif len(padded) > bucket.audio_samples:
            padded = padded[: bucket.audio_samples]
        return padded.reshape(1, -1).astype(np.float32)

    def _run_for_bucket(
        self, audio: np.ndarray, bucket: ParakeetBucket
    ) -> list[list[int]]:
        """End-to-end mel → encoder → decode for a single bucket.

        Picks the bucket's encoder + mel models from the dicts, runs the
        full pipeline, and returns decoded token IDs.
        """
        if self._mel_models is not None:
            mel_model = self._mel_models[bucket.mel_frames]
            padded_audio = self._prepare_audio_for_bucket(audio, bucket)
            audio_buf = Buffer.from_numpy(padded_audio).to(self.devices[0])
            mel_buf = mel_model.execute(audio_buf)[0]
            model_inputs = ParakeetTDTInputs(
                input_features=mel_buf,
                bucket_mel_frames=bucket.mel_frames,
                bucket_encoder_frames=bucket.encoder_frames,
            )
        else:
            features = extract_mel(audio, n_mels=self.tdt_config.num_mel_bins)
            model_inputs = self.prepare_mel_input(features, bucket)

        return self.decode(model_inputs)

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
        token_ids_batch = self._run_for_bucket(audio, bucket)
        return tokenizer.decode(token_ids_batch[0], skip_special_tokens=True)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ASRContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ParakeetTDTInputs:
        if len(replica_batches) > 1:
            raise ValueError("Parakeet-TDT model does not support DP>1")

        raise NotImplementedError(
            "Audio preprocessing (mel spectrogram extraction) is not yet "
            "wired in. Use prepare_mel_input() with pre-extracted features."
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ParakeetTDTInputs:
        raise NotImplementedError(
            "Parakeet-TDT is non-autoregressive at the encoder level "
            "and does not support next-token generation."
        )

    def load_model(self, session: InferenceSession) -> Model:
        # Convention only — not used by the base ``PipelineModel``.
        # Returns the largest bucket so any caller using ``self.model`` as
        # a sentinel still gets a valid encoder. Actual loading is done
        # in ``__init__`` via ``_load_encoders``.
        largest = self._buckets[-1]
        return self._encoder_models[largest.mel_frames]
