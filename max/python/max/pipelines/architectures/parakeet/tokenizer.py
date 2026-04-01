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
"""Tokenizer for Parakeet ASR models.

Parakeet models take audio input, not text. These tokenizers provide the
vocabulary mapping for decoding model outputs (token IDs back to text).
The shared ``_BaseParakeetTokenizer`` handles audio context creation;
subclasses override ``eos`` for their blank token convention.
"""

from __future__ import annotations

from max.interfaces import RequestID, TextGenerationRequest
from max.pipelines.core import ASRContext
from max.pipelines.lib import TextTokenizer


class _BaseParakeetTokenizer(TextTokenizer):
    """Shared base for Parakeet CTC and TDT tokenizers."""

    async def new_context(self, request: TextGenerationRequest) -> ASRContext:
        """Create an ASR context carrying audio bytes from the request."""
        audio_data = request.prompt
        if not isinstance(audio_data, bytes):
            raise TypeError(
                f"Expected audio bytes in prompt, got {type(audio_data).__name__}"
            )
        return ASRContext(
            request_id=RequestID(str(request.request_id)),
            audio_data=audio_data,
            model_name=request.model_name,
        )


class ParakeetTokenizer(_BaseParakeetTokenizer):
    """Tokenizer for Parakeet-CTC. Blank token at vocab_size - 1 (1024)."""

    @property
    def eos(self) -> int:
        if self.delegate.eos_token_id is not None:
            return self.delegate.eos_token_id
        if self.delegate.pad_token_id is not None:
            return self.delegate.pad_token_id
        return 1024
