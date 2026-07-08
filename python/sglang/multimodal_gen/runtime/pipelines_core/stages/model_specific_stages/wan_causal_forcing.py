# SPDX-License-Identifier: Apache-2.0
"""Wan Causal Forcing-specific denoising stage."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.utils import pred_noise_to_pred_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
    CausalDMDCachePolicy,
    CausalDMDRealtimeCacheContext,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class CausalForcingDMDForwardContext:
    target_dtype: torch.dtype
    autocast_enabled: bool
    device: torch.device
    scheduler: Any
    timesteps: torch.Tensor
    latents: torch.Tensor
    prompt_embeds: Any
    negative_prompt_embeds: Any
    image_kwargs: dict[str, Any]
    pos_cond_kwargs: dict[str, Any]
    neg_cond_kwargs: dict[str, Any]
    batch_size: int
    channels: int
    num_frames: int
    height: int
    width: int


class CausalForcingDMDDenoisingStage(CausalDMDDenoisingStage):
    """Causal Forcing-specific entry point over the shared causal DMD machinery.

    Keep Causal Forcing-only behavior here so the shared
    ``CausalDMDDenoisingStage`` stays aligned with the SGLang base
    implementation.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        arch_config = self.transformer.config.arch_config
        self.local_attn_size = int(getattr(arch_config, "local_attn_size", -1))
        self.sink_size = int(getattr(arch_config, "sink_size", 0))
        self.sliding_window_num_frames = int(
            getattr(arch_config, "sliding_window_num_frames", 0)
        )

    def _prepare_causal_dmd_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        scheduler,
        device: torch.device,
    ) -> torch.Tensor:
        if batch.timesteps is not None:
            timesteps = batch.timesteps.to(device)
            logger.info("Using prepared DMD timesteps: %s", timesteps)
            return timesteps

        timesteps = torch.tensor(
            server_args.pipeline_config.dmd_denoising_steps, dtype=torch.long
        ).cpu()

        if server_args.pipeline_config.warp_denoising_step:
            logger.info("Warping timesteps...")
            num_train_timesteps = getattr(scheduler, "num_train_timesteps", None)
            if num_train_timesteps is None:
                num_train_timesteps = scheduler.config.num_train_timesteps
            num_train_timesteps = int(num_train_timesteps)
            scheduler.set_timesteps(num_train_timesteps)
            scheduler_timesteps = torch.cat(
                (scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            timesteps = scheduler_timesteps[num_train_timesteps - timesteps]
        timesteps = timesteps.to(device)
        logger.info("Using timesteps: %s", timesteps)
        return timesteps

    def _prepare_causal_dmd_neg_cond_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict[str, Any]:
        if not batch.do_classifier_free_guidance:
            return {}
        return self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_attention_mask": batch.negative_attention_mask,
            },
        )

    def _prepare_causal_dmd_prompt_embeds(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ):
        prompt_embeds = server_args.pipeline_config.get_pos_prompt_embeds(batch)
        prompt_embeds = self._cast_prompt_embeds_to_dtype(prompt_embeds, target_dtype)
        assert torch.isnan(prompt_embeds[0]).sum() == 0
        return prompt_embeds

    @staticmethod
    def _cast_prompt_embeds_to_dtype(prompt_embeds, target_dtype: torch.dtype):
        if isinstance(prompt_embeds, list):
            return [embed.to(target_dtype) for embed in prompt_embeds]
        return prompt_embeds.to(target_dtype)

    def _prepare_causal_dmd_negative_prompt_embeds(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ):
        if not batch.do_classifier_free_guidance:
            return None
        negative_prompt_embeds = server_args.pipeline_config.get_neg_prompt_embeds(
            batch
        )
        assert negative_prompt_embeds is not None
        negative_prompt_embeds = self._cast_prompt_embeds_to_dtype(
            negative_prompt_embeds, target_dtype
        )
        assert torch.isnan(negative_prompt_embeds[0]).sum() == 0
        return negative_prompt_embeds

    def _prepare_causal_dmd_forward_context(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> CausalForcingDMDForwardContext:
        target_dtype = self._target_dtype()
        autocast_enabled = self._autocast_enabled(target_dtype, server_args)
        device = get_local_torch_device()
        scheduler = self._get_causal_dmd_scheduler(batch, server_args)
        latents = self._get_causal_dmd_latents(batch)
        b, c, t, h, w = latents.shape
        self._prepare_frame_seq_length(h, w)
        timesteps = self._prepare_causal_dmd_timesteps(
            batch,
            server_args,
            scheduler,
            device,
        )
        image_kwargs = self._prepare_causal_dmd_image_kwargs(
            batch,
            server_args,
            target_dtype,
        )
        pos_cond_kwargs = self._prepare_causal_dmd_pos_cond_kwargs(
            batch,
            server_args,
            target_dtype,
        )
        neg_cond_kwargs = self._prepare_causal_dmd_neg_cond_kwargs(
            batch,
            server_args,
            target_dtype,
        )

        if self.attn_backend.get_enum() == AttentionBackendEnum.SLIDING_TILE_ATTN:
            self.prepare_sta_param(batch, server_args)

        prompt_embeds = self._prepare_causal_dmd_prompt_embeds(
            batch,
            server_args,
            target_dtype,
        )
        negative_prompt_embeds = self._prepare_causal_dmd_negative_prompt_embeds(
            batch,
            server_args,
            target_dtype,
        )
        return CausalForcingDMDForwardContext(
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            device=device,
            scheduler=scheduler,
            timesteps=timesteps,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            batch_size=b,
            channels=c,
            num_frames=t,
            height=h,
            width=w,
        )

    @staticmethod
    def _expand_timestep(
        timestep: torch.Tensor, batch_size: int, num_frames: int, device
    ) -> torch.Tensor:
        return timestep.reshape(1, 1).to(device=device).expand(batch_size, num_frames)

    def _predict_x0_btchw(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        latent_model_input: torch.Tensor,
        noise_latents_btchw: torch.Tensor,
        timestep: torch.Tensor,
        scheduler,
        prompt_embeds,
        negative_prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        neg_cond_kwargs: dict,
        attn_raw_latent_shape: tuple[int, int, int],
        current_timestep: int,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, object | None]:
        attn_metadata = self._build_causal_attn_metadata(
            batch,
            server_args,
            current_timestep=current_timestep,
            raw_latent_shape=attn_raw_latent_shape,
            device=device,
        )
        batch_size = latent_model_input.shape[0]
        num_frames = latent_model_input.shape[2]
        timestep_2d = self._expand_timestep(
            timestep, batch_size, num_frames, latent_model_input.device
        )
        if batch.do_classifier_free_guidance:
            self._reset_crossattn_cache(crossattn_cache)
        pred_noise = self._forward_causal_transformer(
            batch,
            latent_model_input=latent_model_input,
            prompt_embeds=prompt_embeds,
            timestep=timestep_2d,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            current_timestep=current_timestep,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )
        if batch.do_classifier_free_guidance:
            assert negative_prompt_embeds is not None
            self._reset_crossattn_cache(crossattn_cache)
            pred_noise_uncond = self._forward_causal_transformer(
                batch,
                latent_model_input=latent_model_input,
                prompt_embeds=negative_prompt_embeds,
                timestep=timestep_2d,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_tokens=current_start_tokens,
                start_frame=start_frame,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=neg_cond_kwargs,
                current_timestep=current_timestep,
                attn_metadata=attn_metadata,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )
            pred_noise = pred_noise_uncond + batch.guidance_scale * (
                pred_noise - pred_noise_uncond
            )
            self._reset_crossattn_cache(crossattn_cache)
        pred_noise_btchw = pred_noise.permute(0, 2, 1, 3, 4)
        x0_btchw = pred_noise_to_pred_video(
            pred_noise=pred_noise_btchw.flatten(0, 1),
            noise_input_latent=noise_latents_btchw.flatten(0, 1),
            timestep=timestep_2d,
            scheduler=scheduler,
        ).unflatten(0, pred_noise_btchw.shape[:2])
        return x0_btchw, attn_metadata

    def _add_noise_for_next_timestep(
        self,
        batch: Req,
        *,
        x0_btchw: torch.Tensor,
        raw_latent_shape: torch.Size,
        next_timestep: torch.Tensor,
        scheduler,
        device,
    ) -> torch.Tensor:
        noise = torch.randn(
            raw_latent_shape,
            dtype=x0_btchw.dtype,
            generator=self._single_generator(batch),
            device=device,
        )
        return scheduler.add_noise(
            x0_btchw.flatten(0, 1),
            noise.flatten(0, 1),
            next_timestep.reshape(1, 1)
            .to(device=x0_btchw.device)
            .expand(x0_btchw.shape[:2])
            .flatten(),
        ).unflatten(0, x0_btchw.shape[:2])

    def _denoise_causal_dmd_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        chunk_latents: torch.Tensor,
        scheduler,
        timesteps: torch.Tensor,
        prompt_embeds,
        negative_prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        neg_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
        attn_raw_latent_shape: tuple[int, int, int],
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor],
        progress_bar=None,
    ) -> tuple[torch.Tensor, object | None]:
        current_latents = chunk_latents
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        raw_latent_shape = noise_latents_btchw.shape
        attn_metadata = None

        for i, timestep in enumerate(timesteps):
            noise_latents = noise_latents_btchw
            latent_model_input = prepare_model_input(current_latents).to(target_dtype)
            x0_btchw, attn_metadata = self._predict_x0_btchw(
                batch,
                server_args,
                latent_model_input=latent_model_input,
                noise_latents_btchw=noise_latents,
                timestep=timestep,
                scheduler=scheduler,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_tokens=current_start_tokens,
                start_frame=start_frame,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=pos_cond_kwargs,
                neg_cond_kwargs=neg_cond_kwargs,
                attn_raw_latent_shape=attn_raw_latent_shape,
                current_timestep=i,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                device=device,
            )

            if i < len(timesteps) - 1:
                next_timestep = timesteps[i + 1 : i + 2]
                noise_latents_btchw = self._add_noise_for_next_timestep(
                    batch,
                    x0_btchw=x0_btchw,
                    raw_latent_shape=raw_latent_shape,
                    next_timestep=next_timestep,
                    scheduler=scheduler,
                    device=device,
                )
                current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
            else:
                current_latents = x0_btchw.permute(0, 2, 1, 3, 4)

            if progress_bar is not None:
                progress_bar.update()

        return current_latents, attn_metadata

    def _update_causal_context_cache(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        context_input: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        attn_metadata,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
        timestep = torch.full(
            (context_input.shape[0], context_input.shape[2]),
            int(context_noise),
            device=context_input.device,
            dtype=torch.long,
        )
        self._forward_causal_transformer(
            batch,
            latent_model_input=context_input.to(target_dtype),
            prompt_embeds=prompt_embeds,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            current_timestep=0,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )

    def _denoise_and_update_causal_block(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        chunk_latents: torch.Tensor,
        scheduler,
        timesteps: torch.Tensor,
        prompt_embeds,
        negative_prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        neg_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
        attn_raw_latent_shape: tuple[int, int, int],
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor],
        prepare_context_input: Callable[[torch.Tensor], torch.Tensor],
        progress_bar=None,
    ) -> torch.Tensor:
        current_latents, attn_metadata = self._denoise_causal_dmd_chunk(
            batch,
            server_args,
            chunk_latents=chunk_latents,
            scheduler=scheduler,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            device=device,
            attn_raw_latent_shape=attn_raw_latent_shape,
            prepare_model_input=prepare_model_input,
            progress_bar=progress_bar,
        )
        self._update_causal_context_cache(
            batch,
            server_args,
            context_input=prepare_context_input(current_latents),
            prompt_embeds=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )
        return current_latents

    def _causal_caches_match(
        self,
        *,
        batch_size: int,
        max_text_len: int,
        dtype: torch.dtype,
        device: torch.device,
        policy: CausalDMDCachePolicy,
    ) -> bool:
        causal_kv_cache = self.causal_kv_cache
        crossattn_cache = self.crossattn_cache
        if causal_kv_cache is None or crossattn_cache is None:
            return False
        if len(causal_kv_cache) != self.num_transformer_blocks:
            return False
        if len(crossattn_cache) != self.num_transformer_blocks:
            return False

        first_kv = causal_kv_cache[0]
        first_cross = crossattn_cache[0]
        return (
            first_kv.k.shape[0] == batch_size
            and first_kv.k.shape[1] == policy.expected_cache_tokens
            and first_kv.k.shape[2] == policy.num_attention_heads
            and first_kv.k.dtype == dtype
            and first_kv.k.device == device
            and first_kv.sink_tokens == policy.expected_sink_tokens
            and first_kv.attention_window_size
            == self._get_causal_attention_window_size(policy.expected_cache_tokens)
            and first_cross.k.shape[0] == batch_size
            and first_cross.k.shape[1] == max_text_len
            and first_cross.k.shape[2] == self.transformer.num_attention_heads
            and first_cross.k.dtype == dtype
            and first_cross.k.device == device
        )

    def _should_reset_realtime_causal_caches(
        self,
        batch: Req,
        *,
        cache_state,
        policy: CausalDMDCachePolicy,
        ctx: CausalForcingDMDForwardContext,
        max_text_len: int,
    ) -> bool:
        causal_kv_cache = cache_state.kv_cache
        crossattn_cache = cache_state.crossattn_cache
        return (
            batch.block_idx == 0
            or causal_kv_cache is None
            or crossattn_cache is None
            or len(causal_kv_cache) != self.num_transformer_blocks
            or len(crossattn_cache) != self.num_transformer_blocks
            or causal_kv_cache[0].k.shape[0] != ctx.batch_size
            or causal_kv_cache[0].k.shape[1] != policy.expected_cache_tokens
            or causal_kv_cache[0].k.shape[2] != policy.num_attention_heads
            or causal_kv_cache[0].k.dtype != ctx.target_dtype
            or causal_kv_cache[0].k.device != ctx.device
            or causal_kv_cache[0].sink_tokens != policy.expected_sink_tokens
            or causal_kv_cache[0].attention_window_size
            != self._get_causal_attention_window_size(policy.expected_cache_tokens)
            or crossattn_cache[0].k.shape[0] != ctx.batch_size
            or crossattn_cache[0].k.shape[1] != max_text_len
            or crossattn_cache[0].k.shape[2] != self.transformer.num_attention_heads
            or crossattn_cache[0].k.dtype != ctx.target_dtype
            or crossattn_cache[0].k.device != ctx.device
        )

    def _prepare_realtime_causal_caches(
        self,
        batch: Req,
        server_args: ServerArgs,
        ctx: CausalForcingDMDForwardContext,
    ) -> CausalDMDRealtimeCacheContext:
        policy = self._build_realtime_causal_cache_policy(batch, server_args)
        cache_state, persist_state = self._get_realtime_causal_cache_state(batch)
        max_text_len = self._get_max_text_len(server_args)

        if self._should_reset_realtime_causal_caches(
            batch,
            cache_state=cache_state,
            policy=policy,
            ctx=ctx,
            max_text_len=max_text_len,
        ):
            causal_kv_cache, crossattn_cache = self._initialize_causal_caches(
                batch_size=ctx.batch_size,
                max_text_len=max_text_len,
                dtype=ctx.target_dtype,
                device=ctx.device,
                kv_cache_kwargs=policy.kv_cache_kwargs,
            )
            cache_state.kv_cache = causal_kv_cache
            cache_state.crossattn_cache = crossattn_cache
            self._clear_stage_causal_cache_refs()
            cache_state.current_chunk_start_frame = 0
            cache_state.chunk_idx = 0
        else:
            causal_kv_cache = cache_state.kv_cache
            crossattn_cache = cache_state.crossattn_cache

        assert causal_kv_cache is not None
        assert crossattn_cache is not None
        return CausalDMDRealtimeCacheContext(
            cache_state=cache_state,
            persist_state=persist_state,
            kv_cache=causal_kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=cache_state.current_chunk_start_frame,
            chunk_idx=cache_state.chunk_idx,
        )

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        target_dtype = ctx.target_dtype
        autocast_enabled = ctx.autocast_enabled
        scheduler = ctx.scheduler
        device = ctx.device
        timesteps = ctx.timesteps
        image_kwargs = ctx.image_kwargs
        pos_cond_kwargs = ctx.pos_cond_kwargs
        neg_cond_kwargs = ctx.neg_cond_kwargs
        latents = ctx.latents
        prompt_embeds = ctx.prompt_embeds
        negative_prompt_embeds = ctx.negative_prompt_embeds
        t, h, w = ctx.num_frames, ctx.height, ctx.width

        independent_first_frame = self.transformer.independent_first_frame

        cache_policy = self._build_realtime_causal_cache_policy(batch, server_args)
        max_text_len = self._get_max_text_len(server_args)
        if not self._causal_caches_match(
            batch_size=latents.shape[0],
            max_text_len=max_text_len,
            dtype=target_dtype,
            device=latents.device,
            policy=cache_policy,
        ):
            self._initialize_causal_caches(
                batch_size=latents.shape[0],
                max_text_len=max_text_len,
                dtype=target_dtype,
                device=latents.device,
                kv_cache_kwargs=cache_policy.kv_cache_kwargs,
            )
        else:
            assert self.crossattn_cache is not None
            self._reset_causal_caches(
                kv_cache=self.causal_kv_cache,
                crossattn_cache=self.crossattn_cache,
            )

        current_start_frame = 0
        if getattr(batch, "image_latent", None) is not None:
            image_latent = batch.image_latent
            assert image_latent is not None
            input_frames = image_latent.shape[2]
            if independent_first_frame and input_frames >= 1:
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[:, :, :1, :, :],
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += 1
                remaining_frames = input_frames - 1
            else:
                remaining_frames = input_frames

            while remaining_frames > 0:
                block = min(self.num_frames_per_block, remaining_frames)
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[
                        :, :, current_start_frame : current_start_frame + block, :, :
                    ],
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += block
                remaining_frames -= block

        pos_start_base = current_start_frame

        if not independent_first_frame or (
            independent_first_frame and batch.image_latent is not None
        ):
            if t % self.num_frames_per_block != 0:
                raise ValueError(
                    "num_frames must be divisible by num_frames_per_block for causal DMD denoising"
                )
            num_blocks = t // self.num_frames_per_block
            block_sizes = [self.num_frames_per_block] * num_blocks
            start_index = 0
        else:
            if (t - 1) % self.num_frames_per_block != 0:
                raise ValueError(
                    "(num_frames - 1) must be divisible by num_frame_per_block when independent_first_frame=True"
                )
            num_blocks = (t - 1) // self.num_frames_per_block
            block_sizes = [1] + [self.num_frames_per_block] * num_blocks
            start_index = 0

        def prepare_context_input(current_latents):
            return current_latents

        with self.progress_bar(
            total=len(block_sizes) * len(timesteps), batch=batch
        ) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[
                    :, :, start_index : start_index + current_num_frames, :, :
                ]

                def prepare_model_input(current_latents):
                    return current_latents

                current_start_frame = pos_start_base + start_index
                current_start_tokens = current_start_frame * self.num_token_per_frame
                current_latents = self._denoise_and_update_causal_block(
                    batch,
                    server_args,
                    chunk_latents=current_latents,
                    scheduler=scheduler,
                    timesteps=timesteps,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_tokens=current_start_tokens,
                    start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    neg_cond_kwargs=neg_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                    device=device,
                    attn_raw_latent_shape=(current_num_frames, h, w),
                    prepare_model_input=prepare_model_input,
                    prepare_context_input=prepare_context_input,
                    progress_bar=progress_bar,
                )

                latents[:, :, start_index : start_index + current_num_frames, :, :] = (
                    current_latents
                )
                start_index += current_num_frames

        batch.latents = latents
        return batch


__all__ = ["CausalForcingDMDDenoisingStage"]
