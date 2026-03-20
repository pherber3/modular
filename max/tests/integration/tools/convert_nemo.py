#!/usr/bin/env python3
"""Convert a Parakeet-TDT .nemo model to MAX-compatible format.

Extracts a .nemo archive and produces:
- model.safetensors: Encoder weights (NeMo names remapped, Conv2d permuted FCRS→RSCF)
- decoder_joint.npz: Decoder LSTM + Joint network weights
- tokenizer.model: SentencePiece BPE model
- config.json: HuggingFace-style config for MAX pipeline discovery

Usage:
    python convert_nemo.py nvidia/parakeet-tdt-0.6b-v3 -o ./converted-tdt
    python convert_nemo.py /path/to/model.nemo -o ./converted-tdt
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import yaml
from safetensors.numpy import save_file as safetensors_save

# NeMo pre_encode.conv uses a ModuleList with interleaved ReLU activations,
# so the conv layers are at indices 0, 2, 3, 5, 6 rather than sequential.
_SUBSAMPLING_REMAP = {
    "encoder.pre_encode.conv.0": "encoder.subsampling.initial_conv",
    "encoder.pre_encode.conv.2": "encoder.subsampling.dw_pw_stages.0",
    "encoder.pre_encode.conv.3": "encoder.subsampling.dw_pw_stages.1",
    "encoder.pre_encode.conv.5": "encoder.subsampling.dw_pw_stages.2",
    "encoder.pre_encode.conv.6": "encoder.subsampling.dw_pw_stages.3",
    "encoder.pre_encode.out": "encoder.subsampling.linear",
}

_LAYER_REMAP = {
    "self_attn.linear_q": "self_attn.q_proj",
    "self_attn.linear_k": "self_attn.k_proj",
    "self_attn.linear_v": "self_attn.v_proj",
    "self_attn.linear_out": "self_attn.o_proj",
    "self_attn.linear_pos": "self_attn.relative_k_proj",
    "self_attn.pos_bias_u": "self_attn.bias_u",
    "self_attn.pos_bias_v": "self_attn.bias_v",
    "conv.batch_norm": "conv.norm",
}


def _remap_encoder_key(key: str) -> str | None:
    """Remap a NeMo encoder weight key to MAX naming.

    Returns None if the key should be skipped (e.g. num_batches_tracked).
    """
    if "num_batches_tracked" in key:
        return None

    for nemo_prefix, max_prefix in _SUBSAMPLING_REMAP.items():
        if key.startswith(nemo_prefix):
            return max_prefix + key[len(nemo_prefix) :]

    for nemo_name, max_name in _LAYER_REMAP.items():
        if nemo_name in key:
            return key.replace(nemo_name, max_name)

    return key


def _is_subsampling_conv_weight(key: str) -> bool:
    """Check if a key is a subsampling Conv2d weight needing FCRS→RSCF permute."""
    return key.endswith(".weight") and (
        "subsampling.initial_conv." in key or "subsampling.dw_pw_stages." in key
    )


class NemoContents(NamedTuple):
    yaml_config: dict
    state_dict: dict[str, np.ndarray]
    tokenizer_model: Path | None
    tokenizer_files: list[Path]


def extract_nemo(nemo_path: Path, tmpdir: Path) -> NemoContents:
    """Extract .nemo archive contents."""
    with tarfile.open(nemo_path, "r:*") as tar:
        tar.extractall(tmpdir)

    config_path = tmpdir / "model_config.yaml"
    if not config_path.exists():
        candidates = list(tmpdir.rglob("model_config.yaml"))
        if not candidates:
            raise FileNotFoundError("No model_config.yaml found in archive")
        config_path = candidates[0]

    with open(config_path, encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)

    ckpt_path = tmpdir / "model_weights.ckpt"
    if not ckpt_path.exists():
        candidates = list(tmpdir.rglob("model_weights.ckpt"))
        if not candidates:
            raise FileNotFoundError("No model_weights.ckpt found in archive")
        ckpt_path = candidates[0]

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {k: v.numpy() for k, v in state_dict.items()}

    tokenizer_model = None
    tokenizer_files = []
    for p in tmpdir.rglob("*"):
        if not p.is_file():
            continue
        is_tokenizer = "tokenizer" in p.name or "vocab" in p.name
        if is_tokenizer and p.suffix in (".model", ".vocab", ".txt"):
            if p.suffix == ".model":
                tokenizer_model = p
            tokenizer_files.append(p)

    return NemoContents(yaml_config, state_dict, tokenizer_model, tokenizer_files)


def convert_weights(
    state_dict: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split and remap weights into encoder (safetensors) and decoder/joint (npz)."""
    encoder_weights: dict[str, np.ndarray] = {}
    decoder_joint_weights: dict[str, np.ndarray] = {}

    for key, arr in state_dict.items():
        if key.startswith("encoder."):
            max_key = _remap_encoder_key(key)
            if max_key is None:
                continue

            if _is_subsampling_conv_weight(max_key):
                arr = np.ascontiguousarray(arr.transpose(2, 3, 1, 0))

            encoder_weights[max_key] = arr

        elif key.startswith("decoder.") or key.startswith("joint."):
            decoder_joint_weights[key] = arr

        elif key.startswith("preprocessor."):
            continue

        else:
            print(f"  Warning: unknown weight prefix, skipping: {key}")

    return encoder_weights, decoder_joint_weights


def generate_config_json(yaml_config: dict) -> dict:
    """Generate a HuggingFace-style config.json from NeMo YAML config."""
    enc = yaml_config.get("encoder", {})
    dec = yaml_config.get("decoder", {})
    joint = yaml_config.get("joint", {})
    defaults = yaml_config.get("model_defaults", {})
    prednet = dec.get("prednet", {})
    jointnet = joint.get("jointnet", {})
    prep = yaml_config.get("preprocessor", {})

    return {
        "architectures": ["ParakeetForTDT"],
        "model_type": "parakeet_tdt",
        "vocab_size": dec.get("vocab_size", 8192),
        "normalize_features": prep.get("normalize", None),
        "encoder_config": {
            "num_hidden_layers": enc.get("n_layers", 24),
            "hidden_size": enc.get("d_model", 1024),
            "intermediate_size": enc.get("d_model", 1024)
            * enc.get("ff_expansion_factor", 4),
            "num_attention_heads": enc.get("n_heads", 8),
            "num_mel_bins": enc.get("feat_in", 128),
            "attention_bias": enc.get("use_bias", False),
            "conv_kernel_size": enc.get("conv_kernel_size", 9),
            "subsampling_factor": enc.get("subsampling_factor", 8),
            "subsampling_conv_channels": enc.get("subsampling_conv_channels", 256),
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "scale_input": True,
            "max_position_embeddings": enc.get("pos_emb_max_len", 5000),
        },
        "decoder_config": {
            "pred_hidden": prednet.get("pred_hidden", 640),
            "pred_rnn_layers": prednet.get("pred_rnn_layers", 2),
        },
        "joint_config": {
            "joint_hidden": jointnet.get("joint_hidden", 640),
            "encoder_hidden": jointnet.get("encoder_hidden", 1024),
            "pred_hidden": jointnet.get("pred_hidden", 640),
            "num_extra_outputs": joint.get("num_extra_outputs", 5),
        },
        "tdt_durations": defaults.get("tdt_durations", [0, 1, 2, 3, 4]),
    }


def convert(nemo_source: str, output_dir: Path) -> None:
    """Convert a .nemo model to MAX-compatible directory."""
    nemo_path: Path
    if Path(nemo_source).exists():
        nemo_path = Path(nemo_source)
        print(f"Using local .nemo file: {nemo_path}")
    else:
        from huggingface_hub import hf_hub_download, list_repo_files

        print(f"Downloading from HuggingFace: {nemo_source}")
        files = list_repo_files(nemo_source)
        nemo_files = [f for f in files if f.endswith(".nemo")]
        if not nemo_files:
            raise FileNotFoundError(f"No .nemo files in {nemo_source}. Files: {files}")
        nemo_path = Path(hf_hub_download(nemo_source, nemo_files[0]))
        print(f"Downloaded: {nemo_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Extracting .nemo archive...")
        contents = extract_nemo(nemo_path, Path(tmpdir))
        print(f"  {len(contents.state_dict)} weight keys")

        print("Converting weights...")
        encoder_weights, decoder_joint_weights = convert_weights(contents.state_dict)
        print(f"  Encoder: {len(encoder_weights)} keys")
        print(f"  Decoder+Joint: {len(decoder_joint_weights)} keys")

        safetensors_save(encoder_weights, str(output_dir / "model.safetensors"))
        np.savez(output_dir / "decoder_joint.npz", **decoder_joint_weights)

        config = generate_config_json(contents.yaml_config)
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        if contents.tokenizer_model:
            shutil.copy2(contents.tokenizer_model, output_dir / "tokenizer.model")
        for tf in contents.tokenizer_files:
            if tf != contents.tokenizer_model:
                shutil.copy2(tf, output_dir / tf.name)

    print(f"\nConversion complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .nemo model to MAX-compatible format"
    )
    parser.add_argument(
        "source",
        help="HuggingFace repo ID or local .nemo file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory",
    )
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
