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
"""Configuration for Parakeet-CTC models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class ParakeetModelConfig(ArchConfig):
    """Configuration for Parakeet-CTC models.

    Parses the nested encoder_config from HuggingFace config.json.
    """

    dtype: DType
    device: DeviceRef
    huggingface_config: AutoConfig
    pipeline_config: PipelineConfig

    @property
    def encoder_config(self) -> AutoConfig:
        return self.huggingface_config.encoder_config

    @property
    def num_hidden_layers(self) -> int:
        return self.encoder_config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self.encoder_config.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.encoder_config.intermediate_size

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_config.num_attention_heads

    @property
    def num_mel_bins(self) -> int:
        return self.encoder_config.num_mel_bins

    @property
    def subsampling_conv_channels(self) -> int:
        return self.encoder_config.subsampling_conv_channels

    @property
    def subsampling_conv_kernel_size(self) -> int:
        return self.encoder_config.subsampling_conv_kernel_size

    @property
    def subsampling_conv_stride(self) -> int:
        return self.encoder_config.subsampling_conv_stride

    @property
    def conv_kernel_size(self) -> int:
        return self.encoder_config.conv_kernel_size

    @property
    def subsampling_factor(self) -> int:
        return getattr(self.encoder_config, "subsampling_factor", 8)

    @property
    def scale_input(self) -> bool:
        return getattr(self.encoder_config, "scale_input", True)

    @property
    def vocab_size(self) -> int:
        return self.huggingface_config.vocab_size

    @property
    def attention_bias(self) -> bool:
        return getattr(self.encoder_config, "attention_bias", True)

    @override
    def get_max_seq_len(self) -> int:
        return getattr(
            self.encoder_config, "max_position_embeddings", 5000
        )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        quantization_encoding = model_config.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        if len(model_config.device_specs) != 1:
            raise ValueError(
                "Parakeet model is only supported on a single device"
            )
        device_spec = model_config.device_specs[0]
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls(
            dtype=supported_encoding_dtype(quantization_encoding),
            device=DeviceRef(
                device_type=device_spec.device_type, id=device_spec.id
            ),
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
        )
