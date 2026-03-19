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
joint network, and TDT greedy decode run in Python/numpy after the
encoder produces its output.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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

from ..parakeet.encoder import ParakeetEncoder
from .decode import tdt_greedy_decode
from .decoder import JointNetwork, PredictionNetwork
from .model_config import TDTModelConfig

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
        shape=["batch_size", "num_frames", config.num_mel_bins],
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
        self.model = self.load_model(session)

        config = TDTModelConfig.initialize(self.pipeline_config)
        self._load_decoder_weights(config)

    def _load_decoder_weights(self, config: TDTModelConfig) -> None:
        """Load LSTM prediction network and joint network from npz file."""
        model_path = self.pipeline_config.model.model_path
        npz_path = Path(model_path) / "decoder_joint.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Decoder weights not found at {npz_path}. "
                "Run scripts/convert_nemo.py first."
            )
        weights = dict(np.load(npz_path))
        self.prediction_net = PredictionNetwork.from_npz(weights)
        self.joint_net = JointNetwork.from_npz(weights)
        self.tdt_config = config
        logger.info(
            "Loaded TDT decoder: %d LSTM layers, pred_hidden=%d",
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
        decode loop runs in Python/numpy to produce token IDs.
        """
        outputs = self.execute(model_inputs)
        assert outputs.logits is not None
        encoder_output = np.from_dlpack(outputs.logits).copy()

        return tdt_greedy_decode(
            encoder_output=encoder_output,
            prediction_net=self.prediction_net,
            joint_net=self.joint_net,
            durations=self.tdt_config.tdt_durations,
            vocab_size=self.tdt_config.vocab_size,
            blank_id=self.tdt_config.blank_id,
        )

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
            "wired in. prepare_initial_token_inputs cannot produce real "
            "model inputs."
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

        config = TDTModelConfig.initialize(self.pipeline_config)
        graph = build_graph(config, state_dict)
        timer.mark_build_complete()

        model = session.load(graph, weights_registry=state_dict)
        timer.done()
        return model
