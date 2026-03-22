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
from max.driver import Buffer, Device, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import WeightData, Weights, WeightsAdapter
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
from .decoder_graph import (
    TDTGraphDecoder,
    build_decoder_step_graph,
    build_projection_graph,
    convert_decoder_state_dict,
)
from .model_config import TDTModelConfig

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
    """Build the encoder-only computation graph for Parakeet-TDT.

    The graph takes mel spectrogram input and returns encoder hidden states.
    TDT decoding (LSTM + joint) runs in Python after this graph executes.
    """
    input_type = TensorType(
        DType.float32,
        # Use fixed num_frames to avoid GPU compiler issues with derived
        # symbolic dimensions from conv subsampling. 3200 frames ≈ 20s audio.
        # Shorter inputs are padded, longer inputs are truncated.
        shape=[1, 3200, config.num_mel_bins],
        device=config.device,
    )

    with Graph("parakeet_tdt_encoder", input_types=[input_type]) as graph:
        encoder = ParakeetEncoder(config)
        encoder.load_state_dict(state_dict)
        input_features = graph.inputs[0].tensor
        hidden_states = encoder(input_features)
        graph.output(hidden_states)

    return graph


class ParakeetTDTPipelineModel(PipelineModel[TextContext]):
    """Pipeline model for Parakeet-TDT ASR inference.

    Loads three compiled MAX graphs into a single InferenceSession:
    1. Encoder graph — FastConformer, mel → encoder hidden states
    2. Projection graph — pre-projects encoder output (1024→640) once
    3. Decoder step graph — one LSTM + joint step, called in a loop

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
        self.model = self.load_model(session)
        self._load_decoder(session)

    def _load_decoder(self, session: InferenceSession) -> None:
        """Load decoder weights and compile projection + step graphs.

        Builds two MAX graphs that run on the same device as the encoder:
        1. Projection graph — pre-projects encoder output (1024→640) once
        2. Decoder step graph — one LSTM + joint step, called in a loop
        """
        npz_path = huggingface_hub.hf_hub_download(
            self.pipeline_config.model.model_path, "decoder_joint.npz"
        )
        npz_weights = dict(np.load(npz_path))
        proj_dict, pred_dict, joint_dict = convert_decoder_state_dict(
            npz_weights
        )

        timer = CompilationTimer("TDT-Projection")
        proj_graph = build_projection_graph(self.tdt_config, proj_dict)
        timer.mark_build_complete()
        projection_model = session.load(proj_graph, weights_registry=proj_dict)
        timer.done()

        timer = CompilationTimer("TDT-DecoderStep")
        dec_graph = build_decoder_step_graph(
            self.tdt_config, pred_dict, joint_dict
        )
        timer.mark_build_complete()
        # Merge prediction + joint dicts for the weights_registry
        dec_weights = {**pred_dict, **joint_dict}
        decoder_step_model = session.load(
            dec_graph, weights_registry=dec_weights
        )
        timer.done()

        from max.driver import DeviceSpec, load_devices

        cpu_device = load_devices([DeviceSpec.cpu()])[0]

        self.graph_decoder = TDTGraphDecoder(
            projection_model=projection_model,
            decoder_step_model=decoder_step_model,
            config=self.tdt_config,
            device=self.devices[0],
            cpu_device=cpu_device,
        )
        logger.info(
            "Loaded TDT decoder (MAX graph, device=%s): "
            "projection + step graphs compiled",
            self.tdt_config.device,
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
        model_outputs = self.model.execute(model_inputs.input_features)
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])

    def decode(self, model_inputs: ModelInputs) -> list[list[int]]:
        """Run encoder + TDT greedy decode, returning token ID sequences.

        Encoder output stays as a Buffer on-device. The projection graph
        pre-projects it once, then the decoder step graph runs in a Python
        loop with LSTM states flowing as Buffers between iterations.
        """
        outputs = self.execute(model_inputs)
        assert outputs.logits is not None
        return self.graph_decoder.decode(outputs.logits)

    def prepare_mel_input(self, features: NDFloat) -> ParakeetTDTInputs:
        """Prepare mel features for model execution.

        Applies feature normalization (if configured) and wraps in a Buffer.
        This is the entry point for audio that has already been converted to
        mel spectrogram features.

        Args:
            features: Mel spectrogram, shape ``(batch, num_frames, num_mel_bins)``.

        Returns:
            Model inputs ready for :meth:`execute` or :meth:`decode`.
        """
        if self.tdt_config.normalize_features == "per_feature":
            features = normalize_per_feature(features)
        features = features.astype(np.float32)
        # Pad or truncate to fixed 3200 frames for GPU compatibility.
        max_frames = 3200
        if features.shape[1] < max_frames:
            pad_width = [(0, 0), (0, max_frames - features.shape[1]), (0, 0)]
            features = np.pad(features, pad_width)
        elif features.shape[1] > max_frames:
            features = features[:, :max_frames, :]
        return ParakeetTDTInputs(
            input_features=Buffer.from_numpy(features).to(self.devices[0])
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
        timer = CompilationTimer("Parakeet-TDT")

        if self.adapter:
            state_dict = self.adapter(self.weights)
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        graph = build_graph(self.tdt_config, state_dict)
        timer.mark_build_complete()

        model = session.load(graph, weights_registry=state_dict)
        timer.done()
        return model
