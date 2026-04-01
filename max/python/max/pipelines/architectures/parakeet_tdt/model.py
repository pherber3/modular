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

The encoder + joint encoder projection run as a compiled MAX graph. On GPU
the decoder step (LSTM + joint) also runs as a compiled graph, called per
step from a Python loop (matching the autoregressive serving pattern used
by LLMs in MAX). On CPU the decoder falls back to the numpy implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import huggingface_hub
import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn import Linear
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
from transformers import AutoConfig

from ..parakeet.audio import extract_mel, normalize_per_feature, read_wav
from ..parakeet.encoder import ParakeetEncoder
from .decode import tdt_greedy_decode
from .decode_gpu import tdt_greedy_decode_gpu
from .decoder import JointNetwork, PredictionNetwork
from .decoder_graph import build_decoder_step_graph
from .model_config import TDTModelConfig
from .weight_adapters import _DECODER_JOINT_MAPPINGS, _ENC_PROJ_MAPPINGS

NDFloat = npt.NDArray[np.floating]

logger = logging.getLogger("max.pipelines")


@dataclass
class ParakeetTDTInputs(ModelInputs):
    """Model inputs for Parakeet-TDT inference."""

    input_features: Buffer  # (batch, num_frames, num_mel_bins)


def build_graph(
    config: TDTModelConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
) -> Graph:
    """Build the encoder + joint-encoder-projection computation graph.

    The graph takes mel spectrogram input and returns the pre-projected
    encoder output ``(batch, T, joint_hidden)``. The encoder projection
    (``joint.enc``) is applied here once over all T frames, eliminating
    ~300 redundant matmuls from the per-step decode loop.
    """
    input_type = TensorType(
        DType.float32,
        shape=["batch_size", "num_frames", config.num_mel_bins],
        device=config.device,
    )

    with Graph("parakeet_tdt_encoder", input_types=[input_type]) as graph:
        encoder = ParakeetEncoder(config)
        encoder.load_state_dict(state_dict)
        input_features = graph.inputs[0].tensor
        hidden_states = encoder(input_features)

        enc_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.joint_hidden,
            dtype=config.dtype,
            device=config.device,
            name="enc_proj",
        )
        enc_proj.load_state_dict(state_dict)
        enc_projected = enc_proj(hidden_states)

        graph.output(enc_projected)

    return graph


class ParakeetTDTPipelineModel(PipelineModel[TextContext]):
    """Pipeline model for Parakeet-TDT ASR inference.

    On GPU: both encoder and decoder step run as compiled graphs. The
    decode loop runs in Python, calling the decoder step graph per
    iteration with LSTM states kept on GPU.

    On CPU: encoder runs as a compiled graph, decoder falls back to
    numpy (CPU decode is fast on Apple Silicon / x86).
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
        self._use_gpu_decode = not self.devices[0].is_host
        self.encoder_model, self.decoder_model = self._load_models(session)

        if not self._use_gpu_decode:
            self._load_cpu_decoder_weights()

    def _get_state_dict(self) -> dict[str, WeightData]:
        if self.adapter:
            return self.adapter(self.weights)
        return {key: value.data() for key, value in self.weights.items()}

    def _merge_npz_weights(self, state_dict: dict[str, WeightData]) -> None:
        """Merge decoder/joint weights from npz into the state dict.

        The safetensors file only contains encoder weights. Decoder and
        joint weights live in a separate ``decoder_joint.npz``. This
        method loads the npz, maps the key names using the same mappings
        as the weight adapter, and injects them into ``state_dict`` so
        both the encoder and decoder graphs can find their weights.
        """
        npz_path = huggingface_hub.hf_hub_download(
            self.pipeline_config.model.model_path, "decoder_joint.npz"
        )
        npz_weights = dict(np.load(npz_path))
        all_mappings = {**_DECODER_JOINT_MAPPINGS, **_ENC_PROJ_MAPPINGS}

        for npz_key, max_name in all_mappings.items():
            if npz_key in npz_weights:
                arr = npz_weights[npz_key].astype(np.float32)
                state_dict[max_name] = WeightData.from_numpy(arr, max_name)

        logger.info(
            "Merged %d decoder/joint weights from npz into state dict",
            sum(1 for k in all_mappings if k in npz_weights),
        )

    def _load_models(
        self, session: InferenceSession
    ) -> tuple[Model, Model | None]:
        """Compile encoder graph and optionally decoder step graph."""
        state_dict = self._get_state_dict()
        self._merge_npz_weights(state_dict)

        timer = CompilationTimer("Parakeet-TDT encoder")
        encoder_graph = build_graph(self.tdt_config, state_dict)
        timer.mark_build_complete()
        encoder_model = session.load(encoder_graph, weights_registry=state_dict)
        timer.done()

        decoder_model = None
        if self._use_gpu_decode:
            timer = CompilationTimer("Parakeet-TDT decoder step")
            decoder_graph = build_decoder_step_graph(
                self.tdt_config, state_dict
            )
            timer.mark_build_complete()
            decoder_model = session.load(
                decoder_graph, weights_registry=state_dict
            )
            timer.done()

        return encoder_model, decoder_model

    def _load_cpu_decoder_weights(self) -> None:
        """Load LSTM prediction network and joint network from npz file."""
        npz_path = huggingface_hub.hf_hub_download(
            self.pipeline_config.model.model_path, "decoder_joint.npz"
        )
        weights = dict(np.load(npz_path))
        self.prediction_net = PredictionNetwork.from_npz(weights)
        self.joint_net = JointNetwork.from_npz(weights)
        logger.info(
            "Loaded TDT CPU decoder: %d LSTM layers, pred_hidden=%d",
            self.prediction_net.num_layers,
            self.prediction_net.pred_hidden,
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
        model_outputs = self.encoder_model.execute(model_inputs.input_features)
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])

    def decode(self, model_inputs: ModelInputs) -> list[list[int]]:
        """Run encoder + TDT greedy decode, returning token ID sequences."""
        outputs = self.execute(model_inputs)
        assert outputs.logits is not None

        if self._use_gpu_decode:
            assert self.decoder_model is not None
            return tdt_greedy_decode_gpu(
                enc_projected_all=outputs.logits,
                decoder_model=self.decoder_model,
                device=self.devices[0],
                durations=self.tdt_config.tdt_durations,
                vocab_size=self.tdt_config.vocab_size,
                blank_id=self.tdt_config.blank_id,
                pred_hidden=self.tdt_config.pred_hidden,
            )

        encoder_output = np.from_dlpack(outputs.logits).copy()
        return tdt_greedy_decode(
            encoder_output=encoder_output,
            prediction_net=self.prediction_net,
            joint_net=self.joint_net,
            durations=self.tdt_config.tdt_durations,
            vocab_size=self.tdt_config.vocab_size,
            blank_id=self.tdt_config.blank_id,
        )

    def prepare_mel_input(self, features: NDFloat) -> ParakeetTDTInputs:
        """Prepare mel features for model execution."""
        if self.tdt_config.normalize_features == "per_feature":
            features = normalize_per_feature(features)
        return ParakeetTDTInputs(
            input_features=Buffer.from_numpy(features.astype(np.float32)).to(
                self.devices[0]
            )
        )

    def transcribe(self, audio_bytes: bytes, tokenizer: object) -> str:
        """Full audio-to-text pipeline: mel extraction → encoder → TDT decode."""
        audio, sample_rate = read_wav(audio_bytes)
        if sample_rate != 16000:
            raise ValueError(
                f"Expected 16kHz audio, got {sample_rate}Hz. "
                "Please resample before sending."
            )

        features = extract_mel(audio, n_mels=self.tdt_config.num_mel_bins)
        model_inputs = self.prepare_mel_input(features)
        token_ids_batch = self.decode(model_inputs)
        return tokenizer.decode(token_ids_batch[0], skip_special_tokens=True)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
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
        # Kept for PipelineModel interface compatibility.
        # Actual loading happens in _load_models().
        return self.encoder_model
