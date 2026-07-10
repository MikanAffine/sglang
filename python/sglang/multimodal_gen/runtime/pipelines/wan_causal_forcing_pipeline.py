# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""Wan Causal Forcing pipeline implementation."""

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    CausalForcingWanT2V480PConfig,
)
from sglang.multimodal_gen.configs.sample.wan import (
    CausalForcingWanT2V480PConfig as CausalForcingWanT2V480PSamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_causal_forcing import (
    CausalForcingDMDDenoisingStage,
)

# isort: off
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DMDTimestepPreparationStage,
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# isort: on

logger = init_logger(__name__)


class WanCausalForcingPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanCausalForcingPipeline"
    pipeline_config_cls = CausalForcingWanT2V480PConfig
    sampling_params_cls = CausalForcingWanT2V480PSamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_stage(DMDTimestepPreparationStage(self.get_module("scheduler")))
        self.add_standard_latent_preparation_stage()

        def create_denoising_stage():
            return CausalForcingDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            )

        self.add_stage_factory(
            RoleType.DENOISER,
            create_denoising_stage,
            "CausalForcingDMDDenoisingStage",
        )

        self.add_standard_decoding_stage()


EntryClass = WanCausalForcingPipeline
