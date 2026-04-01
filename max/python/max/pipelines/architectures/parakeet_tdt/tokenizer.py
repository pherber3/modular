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
"""Tokenizer for Parakeet-TDT models.

Uses SentencePiece BPE loaded via HuggingFace AutoTokenizer. The blank
token ID is vocab_size (e.g. 8192 for TDT v3).
"""

from __future__ import annotations

from ..parakeet.tokenizer import _BaseParakeetTokenizer


class ParakeetTDTTokenizer(_BaseParakeetTokenizer):
    """Tokenizer for Parakeet-TDT. Blank token at vocab_size."""

    @property
    def eos(self) -> int:
        if self.delegate.eos_token_id is not None:
            return self.delegate.eos_token_id
        return self.delegate.vocab_size
