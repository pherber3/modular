#!/usr/bin/env python3
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
"""Dump TDT decoder weights to flat binary files for CUDA benchmark.

Downloads the model from HuggingFace, extracts decoder/joint weights,
and writes each tensor as a raw float32 binary file. Also generates
synthetic enc_projected for benchmarking.

Usage:
    python3 dump_weights.py [--output-dir weights]
"""

import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download

MODEL_ID = "pherber3/parakeet-tdt-0.6b-v3"
NPZ_FILE = "decoder_joint.npz"


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "weights"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {MODEL_ID}/{NPZ_FILE}...")
    npz_path = hf_hub_download(repo_id=MODEL_ID, filename=NPZ_FILE)
    raw = dict(np.load(npz_path))

    prefix = "decoder.prediction.dec_rnn.lstm"
    weight_map = {
        "embedding": raw["decoder.prediction.embed.weight"],
        "l0_ih_w": raw[f"{prefix}.weight_ih_l0"],
        "l0_ih_b": raw[f"{prefix}.bias_ih_l0"],
        "l0_hh_w": raw[f"{prefix}.weight_hh_l0"],
        "l0_hh_b": raw[f"{prefix}.bias_hh_l0"],
        "l1_ih_w": raw[f"{prefix}.weight_ih_l1"],
        "l1_ih_b": raw[f"{prefix}.bias_ih_l1"],
        "l1_hh_w": raw[f"{prefix}.weight_hh_l1"],
        "l1_hh_b": raw[f"{prefix}.bias_hh_l1"],
        "pred_w": raw["joint.pred.weight"],
        "pred_b": raw["joint.pred.bias"],
        "out_w": raw["joint.joint_net.2.weight"],
        "out_b": raw["joint.joint_net.2.bias"],
        "enc_w": raw["joint.enc.weight"],
        "enc_b": raw["joint.enc.bias"],
    }

    total_bytes = 0
    for name, arr in weight_map.items():
        arr = arr.astype(np.float32)
        path = os.path.join(out_dir, f"{name}.bin")
        arr.tofile(path)
        total_bytes += arr.nbytes
        print(f"  {name}: {arr.shape} -> {path} ({arr.nbytes / 1e6:.1f} MB)")

    # Generate synthetic enc_projected using the encoder projection weights
    # Same as test_tdt_megakernel.py make_synthetic_enc_projected(400, weights, seed=456)
    rng = np.random.default_rng(456)
    encoder_hidden = 1024
    T = 400
    raw_enc = rng.standard_normal((T, encoder_hidden)).astype(np.float32)
    enc_projected = (
        raw_enc @ weight_map["enc_w"].T + weight_map["enc_b"]
    ).astype(np.float32)
    path = os.path.join(out_dir, "enc_projected.bin")
    enc_projected.tofile(path)
    total_bytes += enc_projected.nbytes
    print(f"  enc_projected: {enc_projected.shape} -> {path} ({enc_projected.nbytes / 1e6:.1f} MB)")

    # Write durations
    durations = np.array([0, 2, 4, 6, 8], dtype=np.int32)
    path = os.path.join(out_dir, "durations.bin")
    durations.tofile(path)
    print(f"  durations: {durations.shape} -> {path}")

    print(f"\nTotal: {total_bytes / 1e6:.1f} MB in {out_dir}/")


if __name__ == "__main__":
    main()
