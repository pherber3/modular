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

Uses SentencePiece BPE loaded via HuggingFace AutoTokenizer from the
converted model directory. The blank token ID is vocab_size (8192).
"""

from __future__ import annotations

from max.pipelines.lib import TextTokenizer


class ParakeetTDTTokenizer(TextTokenizer):
    @property
    def eos(self) -> int:
        # TDT blank token is vocab_size (one past the last BPE token).
        if self.delegate.eos_token_id is not None:
            return self.delegate.eos_token_id
        # Default: blank = vocab_size
        return self.delegate.vocab_size
