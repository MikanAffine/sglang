# SPDX-License-Identifier: Apache-2.0

import json

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    CausalForcingWanT2V480PConfig,
)
from sglang.multimodal_gen.configs.sample.wan import (
    CausalForcingWanT2V480PConfig as CausalForcingWanT2V480PSamplingParams,
)
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_non_diffusers_pipeline_name,
    get_model_info,
)
from sglang.utils import is_known_non_diffusers_diffusion_model


def test_causal_forcing_sampling_defaults_disable_cfg():
    sampling_params = CausalForcingWanT2V480PSamplingParams()

    assert sampling_params.guidance_scale == 1.0
    assert sampling_params.negative_prompt is None


def test_registry_resolves_causal_forcing_checkpoint_to_native_pipeline():
    get_model_info.cache_clear()
    _get_config_info.cache_clear()

    info = get_model_info("thu-ml/Causal-Forcing", backend="sglang")

    assert info is not None
    assert info.pipeline_cls.__name__ == "WanCausalForcingPipeline"
    assert info.pipeline_config_cls is CausalForcingWanT2V480PConfig
    assert info.sampling_param_cls is CausalForcingWanT2V480PSamplingParams


def test_registry_resolves_causal_forcing_short_model_id():
    _get_config_info.cache_clear()

    info = _get_config_info("local-overlay", model_id="Causal-Forcing")

    assert info is not None
    assert info.pipeline_config_cls is CausalForcingWanT2V480PConfig
    assert info.sampling_param_cls is CausalForcingWanT2V480PSamplingParams


def test_causal_forcing_local_overlay_is_detected_before_cli_generate_dispatch():
    model_path = "checkpoints/causal-forcing-wan"

    assert is_known_non_diffusers_diffusion_model(model_path)
    assert get_non_diffusers_pipeline_name(model_path) == "WanCausalForcingPipeline"


def test_registry_resolves_causal_forcing_overlay_from_model_index(tmp_path):
    get_model_info.cache_clear()
    _get_config_info.cache_clear()
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (tmp_path / component).mkdir()

    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanCausalForcingPipeline",
                "_diffusers_version": "0.33.0.dev0",
                "scheduler": ["diffusers", "UniPCMultistepScheduler"],
                "text_encoder": ["transformers", "UMT5EncoderModel"],
                "tokenizer": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "CausalWanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
            }
        ),
        encoding="utf-8",
    )

    info = get_model_info(str(tmp_path), backend="sglang")

    assert info is not None
    assert info.pipeline_cls.__name__ == "WanCausalForcingPipeline"
    assert info.pipeline_config_cls is CausalForcingWanT2V480PConfig
    assert info.sampling_param_cls is CausalForcingWanT2V480PSamplingParams
