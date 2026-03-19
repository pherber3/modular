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
"""Parakeet-TDT ASR architecture for MAX.

This module implements NVIDIA's Parakeet-TDT (Token-and-Duration Transducer),
a multilingual speech recognition model based on the FastConformer encoder
with an LSTM prediction network and joint network for TDT decoding.

The encoder is shared with the Parakeet-CTC architecture.
"""

from .arch import parakeet_tdt_arch

ARCHITECTURES = [parakeet_tdt_arch]

__all__ = ["parakeet_tdt_arch"]
