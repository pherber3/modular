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
"""Architecture registration for Parakeet-TDT ASR model."""

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture

from . import weight_adapters
from .model import ParakeetTDTPipelineModel
from .model_config import TDTModelConfig
from .tokenizer import ParakeetTDTTokenizer

parakeet_tdt_arch = SupportedArchitecture(
    name="ParakeetForTDT",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "nvidia/parakeet-tdt-0.6b-v3",
    ],
    default_encoding="float32",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    pipeline_model=ParakeetTDTPipelineModel,
    tokenizer=ParakeetTDTTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    required_arguments={"enable_prefix_caching": False},
    config=TDTModelConfig,
)
