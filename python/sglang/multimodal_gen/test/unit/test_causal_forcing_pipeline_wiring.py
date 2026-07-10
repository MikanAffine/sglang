# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[4]
WAN_CAUSAL_FORCING_PIPELINE = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "runtime"
    / "pipelines"
    / "wan_causal_forcing_pipeline.py"
)
CAUSAL_DENOISING_STAGE = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "runtime"
    / "pipelines_core"
    / "stages"
    / "causal_denoising.py"
)
CAUSAL_FORCING_DENOISING_STAGE = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "runtime"
    / "pipelines_core"
    / "stages"
    / "model_specific_stages"
    / "wan_causal_forcing.py"
)
CAUSAL_FORCING_SAMPLING_CONFIG = (
    PYTHON_ROOT / "sglang" / "multimodal_gen" / "configs" / "sample" / "wan.py"
)
CAUSAL_FORCING_PIPELINE_CONFIG = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "configs"
    / "pipeline_configs"
    / "wan.py"
)
CAUSAL_FORCING_CONVERTER = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "tools"
    / "convert_causal_forcing_checkpoint.py"
)
WANVIDEO_MODEL = (
    PYTHON_ROOT / "sglang" / "multimodal_gen" / "runtime" / "models" / "dits" / "wanvideo.py"
)
CAUSAL_WANVIDEO_MODEL = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "runtime"
    / "models"
    / "dits"
    / "causal_wanvideo.py"
)
TRANSFORMER_LOADER = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "runtime"
    / "loader"
    / "component_loaders"
    / "transformer_loader.py"
)
WANVIDEO_ARCH_CONFIG = (
    PYTHON_ROOT
    / "sglang"
    / "multimodal_gen"
    / "configs"
    / "models"
    / "dits"
    / "wanvideo.py"
)
REGISTRY = PYTHON_ROOT / "sglang" / "multimodal_gen" / "registry.py"
WARMUP_REQUEST_BUILDER = (
    PYTHON_ROOT / "sglang" / "multimodal_gen" / "runtime" / "warmup_request_builder.py"
)


def test_wan_causal_forcing_pipeline_prepares_dmd_timesteps_before_denoising():
    source = WAN_CAUSAL_FORCING_PIPELINE.read_text(encoding="utf-8")

    assert "DMDTimestepPreparationStage" in source
    assert source.index("self.add_stage(DMDTimestepPreparationStage") < source.index(
        "self.add_standard_latent_preparation_stage()"
    )
    assert source.index("self.add_stage(DMDTimestepPreparationStage") < source.index(
        "CausalForcingDMDDenoisingStage("
    )


def test_wan_causal_forcing_pipeline_uses_causal_forcing_specific_denoising_stage():
    pipeline_source = WAN_CAUSAL_FORCING_PIPELINE.read_text(encoding="utf-8")
    stage_source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "CausalForcingDMDDenoisingStage" in pipeline_source
    assert "CausalDMDDenoisingStage(" not in pipeline_source
    assert "class CausalForcingDMDDenoisingStage(CausalDMDDenoisingStage):" in stage_source
    assert "Causal Forcing-specific" in stage_source


def test_wan_causal_forcing_pipeline_defers_denoiser_construction_by_role():
    source = WAN_CAUSAL_FORCING_PIPELINE.read_text(encoding="utf-8")

    factory_start = source.index("        def create_denoising_stage()")
    factory_body = source[
        factory_start : source.index("        self.add_standard_decoding_stage()")
    ]

    assert "RoleType" in source
    assert "return CausalForcingDMDDenoisingStage(" in factory_body
    assert "self.add_stage_factory(" in factory_body
    assert "RoleType.DENOISER," in factory_body
    assert "create_denoising_stage," in factory_body
    assert '"CausalForcingDMDDenoisingStage",' in factory_body


def test_causal_forcing_denoising_uses_declared_transformer_lifecycle():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")
    forward_start = source.index("    def forward(")
    forward_body = source[forward_start:]

    assert "self._component_name_for_stage_module(" in forward_body
    assert "with self.use_declared_component(" in forward_body
    assert "component_name=component_name," in forward_body
    assert "module=self.transformer," in forward_body
    assert (
        "return self._forward_with_resident_transformer(batch, server_args)"
        in forward_body
    )


def test_causal_forcing_clones_latents_before_inplace_chunk_updates():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "latents = ctx.latents.clone()" in source


def test_wan_causal_forcing_pipeline_exposes_config_classes_for_discovery():
    source = WAN_CAUSAL_FORCING_PIPELINE.read_text(encoding="utf-8")
    class_start = source.index("class WanCausalForcingPipeline")
    class_body = source[class_start : source.index("    _required_config_modules", class_start)]

    assert "CausalForcingWanT2V480PConfig" in source
    assert "CausalForcingWanT2V480PSamplingParams" in source
    assert "pipeline_config_cls = CausalForcingWanT2V480PConfig" in class_body
    assert "sampling_params_cls = CausalForcingWanT2V480PSamplingParams" in class_body


def test_wan_causal_forcing_pipeline_initializes_upstream_flow_match_scheduler():
    source = WAN_CAUSAL_FORCING_PIPELINE.read_text(encoding="utf-8")

    assert '"scheduler",' in source
    assert "SelfForcingFlowMatchScheduler" in source
    assert 'self.modules["scheduler"] = SelfForcingFlowMatchScheduler(' in source
    assert "num_inference_steps=1000" in source
    assert "shift=server_args.pipeline_config.flow_shift" in source
    assert "sigma_min=0.0" in source
    assert "extra_one_step=True" in source


def test_causal_denoising_uses_prepared_batch_timesteps():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "if batch.timesteps is not None:" in source


def test_causal_denoising_fallback_matches_dmd_timestep_preparation_shape():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert 'getattr(scheduler, "num_train_timesteps", None)' in source
    assert "scheduler.set_timesteps(num_train_timesteps)" in source
    assert "scheduler_timesteps[num_train_timesteps - timesteps]" in source
    assert "scheduler_timesteps[1000 - timesteps]" not in source


def test_causal_denoising_uses_per_frame_timestep_shape_for_chunks():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "num_frames = latent_model_input.shape[2]" in source
    assert "timestep_2d = self._expand_timestep(" in source
    assert "num_frames," in source
    assert "timestep=timestep_2d," in source
    assert "timestep=timestep_2d.unsqueeze(1)" not in source


def test_causal_denoising_keeps_transformer_timesteps_2d_then_flattens_for_scheduler():
    denoising_source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")
    utils_source = (
        PYTHON_ROOT / "sglang" / "multimodal_gen" / "runtime" / "models" / "utils.py"
    ).read_text(encoding="utf-8")
    scheduler_source = (
        PYTHON_ROOT
        / "sglang"
        / "multimodal_gen"
        / "runtime"
        / "models"
        / "schedulers"
        / "scheduling_self_forcing_flow_match.py"
    ).read_text(encoding="utf-8")

    assert "timestep=timestep_2d," in denoising_source
    assert "pred_noise=pred_noise_btchw.flatten(0, 1)" in denoising_source
    assert "noise_input_latent=noise_latents_btchw.flatten(0, 1)" in denoising_source
    assert "timestep=timestep_2d," in denoising_source
    assert "if timestep.ndim == 2:" in utils_source
    assert "timestep = timestep.flatten(0, 1)" in utils_source
    assert "assert timestep.numel() == noise_input_latent.shape[0]" in utils_source
    assert "if timestep.ndim == 2:" in scheduler_source
    assert "timestep = timestep.flatten(0, 1)" in scheduler_source


def test_causal_denoising_applies_classifier_free_guidance():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "negative_prompt_embeds" in source
    assert "neg_cond_kwargs" in source
    assert "batch.do_classifier_free_guidance" in source
    assert "batch.guidance_scale" in source
    assert "pred_noise_uncond + batch.guidance_scale * (" in source
    assert "self._reset_crossattn_cache(crossattn_cache)" in source


def test_causal_denoising_uses_pipeline_config_prompt_embed_accessors():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "server_args.pipeline_config.get_pos_prompt_embeds(batch)" in source
    assert "server_args.pipeline_config.get_neg_prompt_embeds(" in source


def test_causal_denoising_casts_prompt_embeds_to_transformer_dtype():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    pos_start = source.index("def _prepare_causal_dmd_prompt_embeds(")
    pos_body = source[
        pos_start : source.index(
            "    def _prepare_causal_dmd_negative_prompt_embeds", pos_start
        )
    ]
    neg_start = source.index("def _prepare_causal_dmd_negative_prompt_embeds(")
    neg_body = source[
        neg_start : source.index(
            "    def _prepare_causal_dmd_forward_context", neg_start
        )
    ]

    assert "self._cast_prompt_embeds_to_dtype(prompt_embeds, target_dtype)" in pos_body
    assert "self._cast_prompt_embeds_to_dtype(" in neg_body
    assert "negative_prompt_embeds" in neg_body
    assert "target_dtype" in neg_body


def test_wan_cross_attention_accepts_causal_cache_argument():
    source = WANVIDEO_MODEL.read_text(encoding="utf-8")
    class_start = source.index("class WanT2VCrossAttention")
    class_body = source[class_start : source.index("class WanI2VCrossAttention")]

    assert "crossattn_cache" in class_body
    assert "crossattn_cache.store(k, v)" in class_body


def test_causal_denoising_feeds_renoised_latents_to_next_dmd_step():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")
    loop_start = source.index("for i, timestep in enumerate(timesteps):")
    loop_body = source[loop_start : source.index("        return current_latents")]

    assert "noise_latents_btchw = self._add_noise_for_next_timestep(" in loop_body
    assert "current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)" in loop_body
    assert "current_latents = x0_btchw.permute(0, 2, 1, 3, 4)" in loop_body


def test_causal_denoising_uses_absolute_start_frame_after_context_warmup():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")
    loop_start = source.index("for current_num_frames in block_sizes:")
    loop_body = source[
        loop_start : source.index("                start_index += current_num_frames", loop_start)
    ]

    assert "current_start_frame = pos_start_base + start_index" in loop_body
    assert "current_start_tokens = current_start_frame * self.num_token_per_frame" in loop_body
    assert "start_frame=current_start_frame," in loop_body


def test_causal_denoising_keeps_warmup_context_out_of_denoise_input():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")
    prepare_start = source.index("                def prepare_model_input(")
    prepare_body = source[
        prepare_start : source.index("                current_start_frame =", prepare_start)
    ]

    assert "return current_latents" in prepare_body
    assert "torch.cat(" not in prepare_body
    assert "batch.image_latent" not in prepare_body


def test_causal_context_cache_refresh_uses_per_frame_timestep_shape():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "(context_input.shape[0], context_input.shape[2])" in source


def test_causal_denoising_reinitializes_stage_caches_when_request_shape_changes():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "def _causal_caches_match(" in source
    assert "policy.expected_cache_tokens" in source
    assert "first_kv.k.shape[0] == batch_size" in source
    assert "first_cross.k.shape[1] == max_text_len" in source
    assert "not self._causal_caches_match(" in source


def test_realtime_causal_cache_reset_checks_full_request_shape():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")
    check_start = source.index("def _should_reset_realtime_causal_caches(")
    check_body = source[check_start : source.index("    def _prepare_realtime_causal_caches", check_start)]

    assert "causal_kv_cache[0].k.shape[0] != ctx.batch_size" in check_body
    assert "causal_kv_cache[0].k.dtype != ctx.target_dtype" in check_body
    assert "causal_kv_cache[0].k.device != ctx.device" in check_body
    assert "causal_kv_cache[0].attention_window_size" in check_body
    assert "crossattn_cache[0].k.shape[0] != ctx.batch_size" in check_body
    assert "crossattn_cache[0].k.shape[1] != max_text_len" in check_body
    assert "crossattn_cache[0].k.dtype != ctx.target_dtype" in check_body
    assert "crossattn_cache[0].k.device != ctx.device" in check_body


def test_causal_denoising_reads_causal_window_sizes_from_arch_config():
    source = CAUSAL_FORCING_DENOISING_STAGE.read_text(encoding="utf-8")

    init_start = source.index("def __init__(self, transformer, scheduler)")
    init_body = source[
        init_start : source.index(
            "    def _prepare_causal_dmd_timesteps", init_start
        )
    ]

    assert "arch_config = self.transformer.config.arch_config" in init_body
    assert "self.local_attn_size = int(getattr(arch_config, \"local_attn_size\", -1))" in init_body
    assert "self.sink_size = int(getattr(arch_config, \"sink_size\", 0))" in init_body
    assert "self.transformer.model" not in init_body


def test_shared_causal_dmd_stage_stays_close_to_original_surface():
    source = CAUSAL_DENOISING_STAGE.read_text(encoding="utf-8")

    assert "class CausalForcingDMDDenoisingStage" not in source
    assert "def _causal_caches_match(" not in source
    assert "server_args.pipeline_config.get_pos_prompt_embeds(batch)" not in source
    assert "pred_noise_uncond + batch.guidance_scale * (" not in source
    assert "scheduler_timesteps[num_train_timesteps - timesteps]" not in source


def test_causal_forcing_server_warmup_aligns_to_causal_latent_blocks():
    source = WARMUP_REQUEST_BUILDER.read_text(encoding="utf-8")

    assert "def _align_causal_warmup_num_frames(" in source
    assert "def _is_causal_forcing_pipeline(" in source
    assert "WanCausalForcingPipeline" in source
    assert "getattr(pipeline_config, \"is_causal\", False)" not in source
    assert "num_frames_per_block" in source
    assert "latent_num_frames = (num_frames - 1) // temporal_scale_factor + 1" in source
    assert "aligned_latent_frames = max(" in source
    assert "return (aligned_latent_frames - 1) * temporal_scale_factor + 1" in source
    assert "_align_causal_warmup_num_frames(" in source[
        source.index("def _resolve_warmup_num_frames") :
    ]


def test_causal_forcing_defaults_match_chunkwise_four_step_latent_shape():
    sampling_source = CAUSAL_FORCING_SAMPLING_CONFIG.read_text(encoding="utf-8")
    pipeline_source = CAUSAL_FORCING_PIPELINE_CONFIG.read_text(encoding="utf-8")
    converter_source = CAUSAL_FORCING_CONVERTER.read_text(encoding="utf-8")

    class_start = sampling_source.index("class CausalForcingWanT2V480PConfig")
    class_body = sampling_source[class_start:]
    assert "num_inference_steps: int = 4" in class_body
    assert "num_frames: int = 81" in class_body
    assert "adjust_frames: bool = False" in class_body
    assert "guidance_scale: float = 1.0" in class_body
    assert "negative_prompt: str | None = None" in class_body

    assert "[1000, 750, 500, 250]" in pipeline_source
    assert "context_noise: int = 0" in pipeline_source
    assert 'config["num_frames_per_block"] = 3' in converter_source
    assert 'config["sliding_window_num_frames"] = 21' in converter_source


def test_causal_forcing_transformer_loader_resolves_causal_wan_arch_config():
    converter_source = CAUSAL_FORCING_CONVERTER.read_text(encoding="utf-8")
    model_source = CAUSAL_WANVIDEO_MODEL.read_text(encoding="utf-8")
    loader_source = TRANSFORMER_LOADER.read_text(encoding="utf-8")
    arch_config_source = WANVIDEO_ARCH_CONFIG.read_text(encoding="utf-8")

    assert 'config["_class_name"] = "CausalWanTransformer3DModel"' in converter_source
    assert "EntryClass = CausalWanTransformer3DModel" in model_source
    assert "dit_config.update_model_arch(config)" in loader_source
    assert 'cls_name = config.pop("_class_name")' in loader_source
    assert "ModelRegistry.resolve_model_cls(cls_name)" in loader_source
    assert "num_frames_per_block: int = 3" in arch_config_source
    assert "sliding_window_num_frames: int = 21" in arch_config_source
    assert "local_attn_size: int = (" in arch_config_source
    assert "sink_size: int = (" in arch_config_source


def test_causal_forcing_detector_matches_model_index_pipeline_name():
    source = REGISTRY.read_text(encoding="utf-8")

    registration_start = source.index(
        "sampling_param_cls=CausalForcingWanT2V480PSamplingParams"
    )
    registration_body = source[registration_start : source.index("    # MOVA")]
    assert "wancausalforcingpipeline" in registration_body
