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
"""Parakeet-CTC ASR architecture for MAX.

This module implements NVIDIA's Parakeet-CTC-1.1B, a non-autoregressive
speech recognition model based on the FastConformer encoder architecture.
"""

from .arch import parakeet_arch

ARCHITECTURES = [parakeet_arch]

__all__ = ["parakeet_arch"]
