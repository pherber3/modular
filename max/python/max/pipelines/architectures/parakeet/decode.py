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
    predicted_ids: npt.NDArray[np.integer],
    tokenizer: PreTrainedTokenizer,
    blank_id: int = 1024,
) -> list[str]:
    """Decode pre-argmaxed CTC predictions to text.

    The argmax over the vocab dimension happens inside the compiled
    encoder graph on-device (see ``parakeet/graph.py::build_graph``),
    so this function only handles dedup of consecutive duplicates,
    stripping CTC blank tokens, and mapping IDs to text via the tokenizer.

    Args:
        predicted_ids: Argmaxed predictions of shape
            ``(batch, seq_len)``. Int32 on the GPU path, any integer
            dtype accepted for portability.
        tokenizer: HuggingFace tokenizer for ID-to-text conversion.
        blank_id: CTC blank token ID (default 1024, i.e. ``vocab_size - 1``).

    Returns:
        List of decoded text strings, one per batch element.
    """
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
