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

from .config_base import ParakeetConfigBase


@dataclass(kw_only=True)
class ParakeetModelConfig(ParakeetConfigBase):
    """Configuration for Parakeet-CTC models.

    Inherits all encoder properties from ``ParakeetConfigBase``.
    CTC uses ``blank_id = vocab_size - 1`` (the default from base).
    """
