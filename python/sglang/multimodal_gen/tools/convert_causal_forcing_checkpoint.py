# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import pathlib
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_NAME = "diffusion_pytorch_model.safetensors"

DEFAULT_WAN_MODEL_INDEX = {
    "_class_name": "WanPipeline",
    "_diffusers_version": "0.33.0.dev0",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}

REQUIRED_CAUSAL_FORCING_MODEL_INDEX_ENTRIES = {
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "CausalWanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}

REQUIRED_BASE_WAN_COMPONENT_DIRS = ("scheduler", "text_encoder", "tokenizer", "vae")
REQUIRED_BASE_WAN_COMPONENT_FILE_GROUPS = {
    "scheduler": (("scheduler_config.json",),),
    "text_encoder": (
        ("config.json",),
        ("*.safetensors", "*.bin"),
    ),
    "tokenizer": (
        ("tokenizer.json", "spiece.model"),
    ),
    "vae": (
        ("config.json",),
        ("*.safetensors",),
    ),
}

REQUIRED_CAUSAL_FORCING_TENSOR_KEY_PREFIXES = (
    "patch_embedding.",
    "condition_embedder.time_embedder.linear_1.",
    "condition_embedder.text_embedder.linear_1.",
    "blocks.0.attn1.to_q.",
    "blocks.0.attn2.to_q.",
    "blocks.0.ffn.net.0.proj.",
    "blocks.0.norm2.",
    "proj_out.",
)

REQUIRED_CAUSAL_FORCING_BLOCK_KEY_PREFIX_FORMATS = (
    "blocks.{layer}.attn1.to_q.",
    "blocks.{layer}.attn2.to_q.",
    "blocks.{layer}.ffn.net.0.proj.",
    "blocks.{layer}.norm2.",
    "blocks.{layer}.scale_shift_table",
)

# Causal Forcing checkpoints are trained against the official Wan module names.
# SGLang's native Wan loader accepts Diffusers-style names and then maps them to
# the internal transformer implementation.
WAN_OFFICIAL_TO_DIFFUSERS_RENAMES = (
    ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
    ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
    ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
    ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
    ("time_projection.1", "condition_embedder.time_proj"),
    ("head.modulation", "scale_shift_table"),
    ("head.head", "proj_out"),
    ("modulation", "scale_shift_table"),
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
    ("norm2", "norm__placeholder"),
    ("norm3", "norm2"),
    ("norm__placeholder", "norm3"),
    ("self_attn.q", "attn1.to_q"),
    ("self_attn.k", "attn1.to_k"),
    ("self_attn.v", "attn1.to_v"),
    ("self_attn.o", "attn1.to_out.0"),
    ("self_attn.norm_q", "attn1.norm_q"),
    ("self_attn.norm_k", "attn1.norm_k"),
    ("cross_attn.q", "attn2.to_q"),
    ("cross_attn.k", "attn2.to_k"),
    ("cross_attn.v", "attn2.to_v"),
    ("cross_attn.o", "attn2.to_out.0"),
    ("cross_attn.norm_q", "attn2.norm_q"),
    ("cross_attn.norm_k", "attn2.norm_k"),
)

WRAPPER_PREFIXES = (
    "model._fsdp_wrapped_module.",
    "model.",
)


def select_generator_state_dict(
    checkpoint: Mapping[str, Any],
    state_key: str = "generator_ema",
    fallback_state_key: str | None = "generator",
) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")

    if state_key in checkpoint:
        selected = checkpoint[state_key]
    elif fallback_state_key and fallback_state_key in checkpoint:
        selected = checkpoint[fallback_state_key]
    else:
        keys = [state_key]
        if fallback_state_key:
            keys.append(fallback_state_key)
        raise KeyError(f"Checkpoint does not contain any of: {keys}")

    if not isinstance(selected, Mapping):
        raise TypeError(f"Checkpoint entry is not a state dict: {type(selected)!r}")
    return selected


def normalize_causal_forcing_weight_name(name: str) -> str:
    for prefix in WRAPPER_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    for source, target in WAN_OFFICIAL_TO_DIFFUSERS_RENAMES:
        name = name.replace(source, target)

    return name


def normalize_causal_forcing_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = {}
    for name, tensor in state_dict.items():
        normalized_name = normalize_causal_forcing_weight_name(name)
        if normalized_name in normalized:
            raise ValueError(
                f"Duplicate normalized tensor name: {normalized_name!r}"
            )
        normalized[normalized_name] = tensor
    return normalized


def build_causal_wan_transformer_config(base_config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(base_config)
    config["_class_name"] = "CausalWanTransformer3DModel"
    config["local_attn_size"] = -1
    config["sink_size"] = 0
    config["num_frames_per_block"] = 3
    config["sliding_window_num_frames"] = 21
    return config


def build_causal_forcing_model_index(
    base_model_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_index = dict(base_model_index or DEFAULT_WAN_MODEL_INDEX)
    model_index["_class_name"] = "WanCausalForcingPipeline"
    model_index.setdefault("_diffusers_version", "0.33.0.dev0")
    model_index["transformer"] = ["diffusers", "CausalWanTransformer3DModel"]
    return model_index


def validate_causal_forcing_component_metadata(
    *,
    transformer_config: Mapping[str, Any],
    tensor_keys: list[str],
    model_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if transformer_config.get("_class_name") != "CausalWanTransformer3DModel":
        raise ValueError(
            "Transformer config must use _class_name='CausalWanTransformer3DModel'"
        )

    if model_index is not None:
        if model_index.get("_class_name") != "WanCausalForcingPipeline":
            raise ValueError("model_index.json must use WanCausalForcingPipeline")
        for module_name, expected_entry in (
            REQUIRED_CAUSAL_FORCING_MODEL_INDEX_ENTRIES.items()
        ):
            if model_index.get(module_name) != expected_entry:
                raise ValueError(
                    f"model_index.json {module_name} must be {expected_entry}"
                )

    if not tensor_keys:
        raise ValueError("Converted safetensors file contains no tensors")

    has_patch_embedding = any(name.startswith("patch_embedding.") for name in tensor_keys)
    has_attn1_to_q = any(".attn1.to_q." in name for name in tensor_keys)
    has_wrapper_prefix = any(
        name.startswith("model.") or "_fsdp_wrapped_module" in name
        for name in tensor_keys
    )
    has_official_wan_key = any(
        ".self_attn." in name or ".cross_attn." in name for name in tensor_keys
    )
    missing_required_prefixes = [
        prefix
        for prefix in REQUIRED_CAUSAL_FORCING_TENSOR_KEY_PREFIXES
        if not any(name.startswith(prefix) for name in tensor_keys)
    ]
    num_layers = transformer_config.get("num_layers")
    if num_layers is None:
        num_layers = transformer_config.get("num_layers_per_block")
    missing_layer_prefixes: list[str] = []
    if num_layers is not None:
        for layer in range(int(num_layers)):
            for prefix_format in REQUIRED_CAUSAL_FORCING_BLOCK_KEY_PREFIX_FORMATS:
                prefix = prefix_format.format(layer=layer)
                if not any(name.startswith(prefix) for name in tensor_keys):
                    missing_layer_prefixes.append(prefix)
    has_required_key_surface = not missing_required_prefixes

    if has_wrapper_prefix:
        raise ValueError("Converted safetensors still contains wrapper prefixes")
    if has_official_wan_key:
        raise ValueError("Converted safetensors still contains official Wan key names")
    if not has_patch_embedding:
        raise ValueError("Converted safetensors is missing patch_embedding tensors")
    if not has_attn1_to_q:
        raise ValueError("Converted safetensors is missing attn1.to_q tensors")
    if missing_required_prefixes:
        raise ValueError(
            "Converted safetensors is missing required converted key prefix: "
            + ", ".join(missing_required_prefixes)
        )
    if missing_layer_prefixes:
        raise ValueError(
            "Converted safetensors is missing required per-layer key prefix: "
            + ", ".join(missing_layer_prefixes[:8])
        )

    return {
        "tensor_count": len(tensor_keys),
        "has_patch_embedding": has_patch_embedding,
        "has_attn1_to_q": has_attn1_to_q,
        "has_required_key_surface": has_required_key_surface,
        "has_wrapper_prefix": has_wrapper_prefix,
        "has_official_wan_key": has_official_wan_key,
    }


def validate_base_wan_component_dirs(model_dir: pathlib.Path) -> dict[str, str]:
    missing = [
        component
        for component in REQUIRED_BASE_WAN_COMPONENT_DIRS
        if not (model_dir / component).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Base Wan component directory is missing required launch component(s): "
            + ", ".join(missing)
        )

    missing_files = []
    for component, file_groups in REQUIRED_BASE_WAN_COMPONENT_FILE_GROUPS.items():
        component_dir = model_dir / component
        for file_group in file_groups:
            if not any(
                any(component_dir.glob(pattern)) for pattern in file_group
            ):
                missing_files.append(f"{component}/{' or '.join(file_group)}")
    if missing_files:
        raise FileNotFoundError(
            "Base Wan component directory is missing required launch file(s): "
            + ", ".join(missing_files)
        )

    return {
        component: str(model_dir / component)
        for component in REQUIRED_BASE_WAN_COMPONENT_DIRS
    }


def validate_causal_forcing_component(
    *,
    output_dir: pathlib.Path,
    model_dir: pathlib.Path | None = None,
    base_model_dir: pathlib.Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    from safetensors import safe_open

    config_path = output_dir / "config.json"
    weights_path = output_dir / output_name
    if not config_path.exists():
        raise FileNotFoundError(f"Transformer config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Transformer safetensors not found: {weights_path}")

    with open(config_path, encoding="utf-8") as f:
        transformer_config = json.load(f)

    model_index = None
    if model_dir is not None:
        model_index_path = model_dir / "model_index.json"
        if not model_index_path.exists():
            raise FileNotFoundError(f"model_index.json not found: {model_index_path}")
        with open(model_index_path, encoding="utf-8") as f:
            model_index = json.load(f)

    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        tensor_keys = list(f.keys())

    result = validate_causal_forcing_component_metadata(
        transformer_config=transformer_config,
        tensor_keys=tensor_keys,
        model_index=model_index,
    )
    if base_model_dir is not None:
        result["base_component_dirs"] = validate_base_wan_component_dirs(base_model_dir)
    return result


def convert_causal_forcing_checkpoint(
    *,
    checkpoint_path: pathlib.Path,
    base_transformer_config: pathlib.Path,
    base_model_index: pathlib.Path | None,
    output_dir: pathlib.Path,
    model_dir: pathlib.Path | None,
    state_key: str,
    fallback_state_key: str | None,
    output_name: str,
    overwrite: bool,
) -> None:
    import torch
    from safetensors.torch import save_file

    output_weights_path = output_dir / output_name
    output_config_path = output_dir / "config.json"
    output_model_index_path = model_dir / "model_index.json" if model_dir else None
    if not overwrite and (output_weights_path.exists() or output_config_path.exists()):
        raise FileExistsError(
            f"{output_dir} already contains output files; pass --overwrite to replace them"
        )
    if (
        not overwrite
        and output_model_index_path is not None
        and output_model_index_path.exists()
    ):
        raise FileExistsError(
            f"{output_model_index_path} already exists; pass --overwrite to replace it"
        )

    with open(base_transformer_config, encoding="utf-8") as f:
        base_config = json.load(f)

    logger.info("Loading Causal Forcing checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = select_generator_state_dict(
        checkpoint,
        state_key=state_key,
        fallback_state_key=fallback_state_key,
    )
    normalized = normalize_causal_forcing_state_dict(state_dict)

    non_tensor_names = [
        name for name, value in normalized.items() if not isinstance(value, torch.Tensor)
    ]
    if non_tensor_names:
        examples = ", ".join(non_tensor_names[:3])
        raise TypeError(f"State dict contains non-tensor values: {examples}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(normalized, output_weights_path)

    causal_config = build_causal_wan_transformer_config(base_config)
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(causal_config, f, indent=2, sort_keys=True)
        f.write("\n")

    if model_dir is not None:
        if base_model_index is not None:
            with open(base_model_index, encoding="utf-8") as f:
                source_model_index = json.load(f)
        else:
            source_model_index = DEFAULT_WAN_MODEL_INDEX

        model_dir.mkdir(parents=True, exist_ok=True)
        assert output_model_index_path is not None
        with open(output_model_index_path, "w", encoding="utf-8") as f:
            json.dump(
                build_causal_forcing_model_index(source_model_index),
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
        logger.info("Wrote %s", output_model_index_path)

    logger.info("Wrote %s", output_weights_path)
    logger.info("Wrote %s", output_config_path)
    validation = validate_causal_forcing_component(
        output_dir=output_dir,
        model_dir=model_dir,
        output_name=output_name,
    )
    logger.info("Validated converted component: %s", validation)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Causal Forcing Wan generator checkpoint into an SGLang "
            "native transformer component directory"
        )
    )
    parser.add_argument("--checkpoint-path", type=pathlib.Path)
    parser.add_argument("--base-transformer-config", type=pathlib.Path)
    parser.add_argument("--base-model-index", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        help="Optional model directory where a Causal Forcing model_index.json is written",
    )
    parser.add_argument(
        "--base-model-dir",
        type=pathlib.Path,
        help=(
            "Optional Wan Diffusers base directory to preflight for scheduler, "
            "text_encoder, tokenizer, and vae component overrides"
        ),
    )
    parser.add_argument("--state-key", default="generator_ema")
    parser.add_argument("--fallback-state-key", default="generator")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing converted component instead of converting a checkpoint",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.validate_only:
        result = validate_causal_forcing_component(
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            base_model_dir=args.base_model_dir,
            output_name=args.output_name,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if args.checkpoint_path is None or args.base_transformer_config is None:
            raise SystemExit(
                "--checkpoint-path and --base-transformer-config are required unless "
                "--validate-only is set"
            )
        convert_causal_forcing_checkpoint(
            checkpoint_path=args.checkpoint_path,
            base_transformer_config=args.base_transformer_config,
            base_model_index=args.base_model_index,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            state_key=args.state_key,
            fallback_state_key=args.fallback_state_key,
            output_name=args.output_name,
            overwrite=args.overwrite,
        )
