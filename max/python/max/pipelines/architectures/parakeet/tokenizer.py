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
"""Tokenizer for Parakeet-CTC models.

Parakeet is an ASR model that takes audio input, not text. This tokenizer
provides the CTC vocabulary mapping for decoding model outputs (token IDs
back to text). The blank token (CTC) is at index vocab_size - 1 = 1024.
"""

from __future__ import annotations

from max.pipelines.lib import TextTokenizer


class ParakeetTokenizer(TextTokenizer):
    @property
    def eos(self) -> int:
        # CTC blank token is the last token in the vocabulary.
        if self.delegate.eos_token_id is not None:
            return self.delegate.eos_token_id
        if self.delegate.pad_token_id is not None:
            return self.delegate.pad_token_id
        return 1024
