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

The encoder runs as a compiled MAX graph. The LSTM prediction network,
joint network, and TDT greedy decode run on GPU using PyTorch tensors,
eliminating the GPU→CPU transfer that was previously required.

When PyTorch+CUDA is available, the decoder uses pre-projected encoder
outputs and GPU tensor operations (decoder_torch / decode_torch).
Falls back to the numpy CPU implementation otherwise (decoder / decode).
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

# Always import numpy CPU fallback.
from .decode import tdt_greedy_decode
from .decoder import JointNetwork, PredictionNetwork
from .model_config import TDTModelConfig

# Prefer PyTorch GPU decoder when available.
_USE_TORCH_DECODER = False
try:
    import torch

    if torch.cuda.is_available():
        from .decode_torch import (
            tdt_greedy_decode as tdt_greedy_decode_torch,
        )
        from .decoder_torch import (
            JointNetwork as JointNetworkTorch,
        )
        from .decoder_torch import (
            PredictionNetwork as PredictionNetworkTorch,
        )

        _USE_TORCH_DECODER = True
    else:
        print("PyTorch available but CUDA not detected, using numpy decoder")
except ImportError as e:
    print(f"PyTorch GPU decoder not available: {e}")

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

    Runs the FastConformer encoder in a compiled graph, then performs
    TDT greedy decoding in Python/numpy using the LSTM prediction network
    and joint network.
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
        self._load_decoder_weights()

    def _load_decoder_weights(self) -> None:
        """Load LSTM prediction network and joint network from npz file.

        When PyTorch+CUDA is available, weights are placed on GPU as torch
        tensors for GPU-accelerated decoding. Otherwise falls back to numpy.
        """
        npz_path = huggingface_hub.hf_hub_download(
            self.pipeline_config.model.model_path, "decoder_joint.npz"
        )
        weights = dict(np.load(npz_path))

        if _USE_TORCH_DECODER:
            self.prediction_net = PredictionNetworkTorch.from_npz(weights)
            self.joint_net = JointNetworkTorch.from_npz(weights)
            self._use_torch_decoder = True
            logger.info(
                "Loaded TDT decoder (PyTorch GPU): %d LSTM layers, "
                "pred_hidden=%d",
                self.prediction_net.num_layers,
                self.prediction_net.pred_hidden,
            )
        else:
            self.prediction_net = PredictionNetwork.from_npz(weights)
            self.joint_net = JointNetwork.from_npz(weights)
            self._use_torch_decoder = False
            logger.info(
                "Loaded TDT decoder (numpy CPU): %d LSTM layers, "
                "pred_hidden=%d",
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
        model_outputs = self.model.execute(model_inputs.input_features)
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])

    def decode(self, model_inputs: ModelInputs) -> list[list[int]]:
        """Run encoder + TDT greedy decode, returning token ID sequences.

        This is the full TDT inference path: encoder graph produces hidden
        states, then the LSTM prediction network + joint network + TDT greedy
        decode loop produces token IDs.

        When PyTorch+CUDA is available, the encoder output stays on GPU as a
        torch tensor (via ``torch.from_dlpack``), and decoding runs entirely
        on GPU with pre-projected encoder/decoder outputs. This eliminates
        the ~47ms GPU→CPU transfer and runs the decode loop ~10-30x faster
        than the numpy CPU path.

        Falls back to numpy CPU decoding otherwise.
        """
        outputs = self.execute(model_inputs)
        assert outputs.logits is not None

        if self._use_torch_decoder:
            # Keep encoder output on GPU — no copy to CPU.
            encoder_output = torch.from_dlpack(outputs.logits)
            if encoder_output.dim() == 2:
                encoder_output = encoder_output.unsqueeze(0)
            return tdt_greedy_decode_torch(
                encoder_output=encoder_output,
                prediction_net=self.prediction_net,
                joint_net=self.joint_net,
                durations=self.tdt_config.tdt_durations,
                vocab_size=self.tdt_config.vocab_size,
                blank_id=self.tdt_config.blank_id,
            )
        else:
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
