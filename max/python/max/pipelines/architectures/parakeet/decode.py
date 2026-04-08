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
"""CTC greedy decoding for Parakeet-CTC models."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from transformers import PreTrainedTokenizer


def ctc_greedy_decode(
    logits: npt.NDArray[np.floating],
    tokenizer: PreTrainedTokenizer,
    blank_id: int = 1024,
) -> list[str]:
    """Decode CTC logits to text using greedy decoding.

    Applies argmax over the vocab dimension, removes consecutive
    duplicate tokens, strips CTC blank tokens, then maps remaining IDs
    to text via the tokenizer.

    On-device argmax is a tracked follow-up but is blocked by a
    compile-time blowup when combined with the bucketed encoder graphs
    on L4 (see ``parakeet/graph.py`` docstring). Host-side argmax is
    the working path for now.

    Args:
        logits: Model output of shape ``(batch, seq_len, vocab_size)``.
        tokenizer: HuggingFace tokenizer for ID-to-text conversion.
        blank_id: CTC blank token ID (default 1024, i.e. ``vocab_size - 1``).

    Returns:
        List of decoded text strings, one per batch element.
    """
    predicted_ids = np.argmax(logits, axis=-1)  # (batch, seq_len)
    results: list[str] = []
    for seq in predicted_ids:
        deduped = [int(seq[0])]
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                deduped.append(int(seq[i]))
        filtered = [tok for tok in deduped if tok != blank_id]
        text = tokenizer.decode(filtered, skip_special_tokens=True)
        results.append(text)
    return results
