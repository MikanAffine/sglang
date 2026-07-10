# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[4]
CONVERTER_PATH = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "tools"
    / "convert_causal_forcing_checkpoint.py"
)
spec = importlib.util.spec_from_file_location(
    "convert_causal_forcing_checkpoint", CONVERTER_PATH
)
converter = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(converter)

build_causal_forcing_model_index = converter.build_causal_forcing_model_index
build_causal_wan_transformer_config = converter.build_causal_wan_transformer_config
normalize_causal_forcing_state_dict = converter.normalize_causal_forcing_state_dict
normalize_causal_forcing_weight_name = converter.normalize_causal_forcing_weight_name
select_generator_state_dict = converter.select_generator_state_dict
validate_causal_forcing_component_metadata = (
    converter.validate_causal_forcing_component_metadata
)
validate_base_wan_component_dirs = converter.validate_base_wan_component_dirs


REPRESENTATIVE_CONVERTED_TENSOR_KEYS = [
    "patch_embedding.weight",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.text_embedder.linear_1.weight",
    "blocks.0.attn1.to_q.weight",
    "blocks.0.attn2.to_q.weight",
    "blocks.0.ffn.net.0.proj.weight",
    "blocks.0.norm2.weight",
    "proj_out.weight",
]


def test_select_generator_state_dict_prefers_generator():
    generator = {"model.weight": object()}
    generator_ema = {"model.weight": object()}

    selected = select_generator_state_dict(
        {"generator": generator, "generator_ema": generator_ema}
    )

    assert selected is generator


def test_select_generator_state_dict_falls_back_to_generator_ema():
    generator_ema = {"model.weight": object()}

    selected = select_generator_state_dict({"generator_ema": generator_ema})

    assert selected is generator_ema


def test_select_generator_state_dict_requires_known_key():
    with pytest.raises(KeyError, match="generator"):
        select_generator_state_dict({"model.weight": object()})


def test_normalize_causal_forcing_weight_name_strips_wrapper_and_maps_wan_keys():
    name = normalize_causal_forcing_weight_name(
        "model._fsdp_wrapped_module.blocks.0.self_attn.q.weight"
    )

    assert name == "blocks.0.attn1.to_q.weight"


def test_normalize_causal_forcing_state_dict_rejects_collisions():
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_causal_forcing_state_dict(
            {
                "model.blocks.0.self_attn.q.weight": object(),
                "blocks.0.attn1.to_q.weight": object(),
            }
        )


def test_build_causal_wan_transformer_config_patches_class_and_causal_fields():
    config = build_causal_wan_transformer_config(
        {
            "_class_name": "WanTransformer3DModel",
            "num_layers": 30,
            "num_attention_heads": 12,
        }
    )

    assert config["_class_name"] == "CausalWanTransformer3DModel"
    assert config["num_layers"] == 30
    assert config["num_attention_heads"] == 12
    assert config["num_frames_per_block"] == 3
    assert config["sliding_window_num_frames"] == 21
    assert config["local_attn_size"] == -1
    assert config["sink_size"] == 0


def test_build_causal_wan_transformer_config_overwrites_stale_causal_fields():
    config = build_causal_wan_transformer_config(
        {
            "_class_name": "WanTransformer3DModel",
            "num_frames_per_block": 1,
            "sliding_window_num_frames": 5,
            "local_attn_size": 2,
            "sink_size": 1,
        }
    )

    assert config["num_frames_per_block"] == 3
    assert config["sliding_window_num_frames"] == 21
    assert config["local_attn_size"] == -1
    assert config["sink_size"] == 0


def test_build_causal_forcing_model_index_uses_causal_pipeline_and_transformer():
    model_index = build_causal_forcing_model_index(
        {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.33.0.dev0",
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "vae": ["diffusers", "AutoencoderKLWan"],
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
        }
    )

    assert model_index["_class_name"] == "WanCausalForcingPipeline"
    assert model_index["transformer"] == ["diffusers", "CausalWanTransformer3DModel"]
    assert model_index["text_encoder"] == ["transformers", "UMT5EncoderModel"]


def test_validate_causal_forcing_component_metadata_accepts_converted_keys():
    result = validate_causal_forcing_component_metadata(
        transformer_config={"_class_name": "CausalWanTransformer3DModel"},
        tensor_keys=REPRESENTATIVE_CONVERTED_TENSOR_KEYS,
        model_index={
            "_class_name": "WanCausalForcingPipeline",
            "transformer": ["diffusers", "CausalWanTransformer3DModel"],
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        },
    )

    assert result["tensor_count"] == len(REPRESENTATIVE_CONVERTED_TENSOR_KEYS)
    assert result["has_patch_embedding"] is True
    assert result["has_attn1_to_q"] is True
    assert result["has_required_key_surface"] is True


def test_validate_causal_forcing_component_metadata_rejects_incomplete_key_surface():
    with pytest.raises(ValueError, match="missing required converted key prefix"):
        validate_causal_forcing_component_metadata(
            transformer_config={"_class_name": "CausalWanTransformer3DModel"},
            tensor_keys=[
                "patch_embedding.weight",
                "blocks.0.attn1.to_q.weight",
            ],
        )


def test_validate_causal_forcing_component_metadata_rejects_missing_layer_surface():
    with pytest.raises(ValueError, match="blocks.1.attn1.to_q"):
        validate_causal_forcing_component_metadata(
            transformer_config={
                "_class_name": "CausalWanTransformer3DModel",
                "num_layers": 2,
            },
            tensor_keys=REPRESENTATIVE_CONVERTED_TENSOR_KEYS,
        )


def test_validate_causal_forcing_component_metadata_rejects_original_wan_keys():
    with pytest.raises(ValueError, match="official Wan key"):
        validate_causal_forcing_component_metadata(
            transformer_config={"_class_name": "CausalWanTransformer3DModel"},
            tensor_keys=[
                "patch_embedding.weight",
                "blocks.0.self_attn.q.weight",
            ],
        )


def test_validate_causal_forcing_component_metadata_rejects_incomplete_model_index():
    with pytest.raises(ValueError, match="scheduler"):
        validate_causal_forcing_component_metadata(
            transformer_config={"_class_name": "CausalWanTransformer3DModel"},
            tensor_keys=[
                "patch_embedding.weight",
                "blocks.0.attn1.to_q.weight",
            ],
            model_index={
                "_class_name": "WanCausalForcingPipeline",
                "transformer": ["diffusers", "CausalWanTransformer3DModel"],
            },
        )


def test_validate_base_wan_component_dirs_reports_missing_launch_components(tmp_path):
    base_model_dir = tmp_path / "wan-base"
    base_model_dir.mkdir()
    for component in ("scheduler", "tokenizer"):
        (base_model_dir / component).mkdir()

    with pytest.raises(FileNotFoundError, match="text_encoder"):
        validate_base_wan_component_dirs(base_model_dir)

    for component in ("text_encoder", "vae"):
        (base_model_dir / component).mkdir()

    with pytest.raises(FileNotFoundError, match="scheduler_config.json"):
        validate_base_wan_component_dirs(base_model_dir)

    (base_model_dir / "scheduler" / "scheduler_config.json").write_text("{}")
    (base_model_dir / "tokenizer" / "tokenizer_config.json").write_text("{}")
    (base_model_dir / "text_encoder" / "config.json").write_text("{}")
    (base_model_dir / "text_encoder" / "model.safetensors.index.json").write_text("{}")
    (base_model_dir / "vae" / "config.json").write_text("{}")
    (
        base_model_dir / "vae" / "diffusion_pytorch_model.safetensors.index.json"
    ).write_text("{}")
    (base_model_dir / "tokenizer" / "tokenizer.json").write_text("{}")

    with pytest.raises(
        FileNotFoundError, match=r"text_encoder/\*\.safetensors or \*\.bin"
    ):
        validate_base_wan_component_dirs(base_model_dir)

    (base_model_dir / "text_encoder" / "model-00001-of-00005.safetensors").write_text(
        ""
    )

    with pytest.raises(FileNotFoundError, match=r"vae/\*\.safetensors"):
        validate_base_wan_component_dirs(base_model_dir)

    (base_model_dir / "vae" / "diffusion_pytorch_model.safetensors").write_text("")

    (base_model_dir / "tokenizer" / "tokenizer.json").unlink()

    with pytest.raises(FileNotFoundError, match="tokenizer.json or spiece.model"):
        validate_base_wan_component_dirs(base_model_dir)

    (base_model_dir / "tokenizer" / "tokenizer.json").write_text("{}")

    result = validate_base_wan_component_dirs(base_model_dir)

    assert result == {
        "scheduler": str(base_model_dir / "scheduler"),
        "text_encoder": str(base_model_dir / "text_encoder"),
        "tokenizer": str(base_model_dir / "tokenizer"),
        "vae": str(base_model_dir / "vae"),
    }
