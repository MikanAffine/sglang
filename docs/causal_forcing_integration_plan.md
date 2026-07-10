# Causal Forcing Integration Plan and Log

## Goal

Integrate Causal Forcing into SGLang-Diffusion inference without copying the
reference pipeline wholesale. The first milestone is a naive native SGLang
pipeline for chunkwise 4-step Causal Forcing. After that works, implement a
disaggregated path that moves VAE decoding to an async worker process.

## References Consulted

- Causal Forcing paper: arXiv 2602.02214, v5, 2026-06-01.
- Causal Forcing++ paper: arXiv 2605.15141, submitted 2026-05-14.
- `thu-ml/Causal-Forcing` GitHub repository.
- Causal Forcing `configs/causal_forcing_dmd_chunkwise.yaml`.
- SGLang-Diffusion documentation: overview, support-new-models, and
  disaggregation pages.
- Local SGLang-Diffusion runtime: `WanCausalForcingPipeline`,
  `CausalForcingDMDDenoisingStage`, Wan config/sample registration, and model
  resolver.

## Reference Facts Used So Far

- Local validation scope is intentionally limited. The Windows/4060 machine is
  for checkpoint conversion and metadata/static loader checks only; it should
  not download the full Wan base component set or run full Wan generation unless
  explicitly requested.
- Any full Wan component download for H100 validation must use the mirror:
  `HF_ENDPOINT=https://hf-mirror.com`.
- The Causal Forcing chunkwise reference config uses 4 denoising steps:
  `[1000, 750, 500, 250]`.
- The same config enables `warp_denoising_step: true`, `timestep_shift: 5.0`,
  `guidance_scale: 3.0`, `num_frame_per_block: 3`, and latent shape
  `[1, 21, 16, 60, 104]`.
- SGLang-Diffusion resolves native pipelines by model family config plus
  pipeline name; non-diffusers checkpoints can be handled by explicit resolver
  patterns.
- The main checkout uses `checkpoints/`, not `checkpoint/`. The local Causal
  Forcing checkpoint found there is
  `checkpoints/chunkwise/causal_forcing.pt`.
- The local base Wan transformer config under
  `checkpoints/wan2.1-t2v-1.3b-diffusers/transformer/config.json` is a 1.3B
  shape (`num_layers: 30`, `num_attention_heads: 12`, `ffn_dim: 8960`).

## Implementation Log

### 2026-07-08: Naive Pipeline Resolver Slice

Implemented a minimal Causal Forcing model-family entrypoint:

- Added explicit `CausalForcingWanT2V480PConfig` pipeline config inheriting the
  existing causal Wan 4-step config.
- Added explicit `CausalForcingWanT2V480PConfig` sampling params alias.
- Registered `thu-ml/Causal-Forcing` with the native diffusion registry.
- Routed non-diffusers paths containing `causal-forcing` to
  `WanCausalForcingPipeline`.
- Added a focused registry unit test for resolving `thu-ml/Causal-Forcing` to
  `WanCausalForcingPipeline` and the Causal Forcing config classes.

Verification:

- `py_compile` passed for the modified Python files and the new test.
- Full pytest verification could not run locally on Windows. The checked-in root
  `.venv` has no `pip`, `pytest`, or core runtime deps. `uv run --project python`
  was attempted with cache access allowed, but dependency resolution failed
  because `sgl-deep-gemm==0.1.4` has no `win_amd64` wheel.

Remote/Linux validation needed:

- Run `uv run --project python python -m pytest
  python/sglang/multimodal_gen/test/unit/test_causal_forcing_registry.py -q`.
- Then run a resolver smoke test on Linux with the actual checkpoint/model path.

### 2026-07-08: Checkpoint Conversion Utility Slice

Added a minimal conversion path for the reference `.pt` generator checkpoint:

- Added `tools/convert_causal_forcing_checkpoint.py`.
- The tool selects `generator` by default, matching the official chunk-wise
  inference command, and falls back to `generator_ema`.
- It normalizes reference wrapper keys from `model._fsdp_wrapped_module.*` or
  `model.*` to component-local names.
- It converts official Wan module names to the Diffusers-style names expected by
  SGLang's existing Wan safetensors loader.
- It writes a transformer component directory containing
  `diffusion_pytorch_model.safetensors` and a patched `config.json` with
  `_class_name: CausalWanTransformer3DModel`.
- Optionally, it writes a lightweight overlay `model_index.json` with
  `_class_name: WanCausalForcingPipeline` and transformer architecture
  `CausalWanTransformer3DModel`; base Wan components can be supplied at launch
  through SGLang's `--component-paths.<component>` overrides.
- It also supports `--validate-only` to inspect an existing converted
  transformer component without loading the full model or downloading base Wan
  components.
- The validator checks that the overlay declares the required causal Wan
  components: `scheduler`, `text_encoder`, `tokenizer`, `transformer`, and
  `vae`.
- Added unit coverage for state-key selection, key normalization collision
  detection, causal config patching, overlay model-index patching, and converted
  component metadata validation.

Example conversion command from the main checkout:

```powershell
uv run --project python python -m sglang.multimodal_gen.tools.convert_causal_forcing_checkpoint `
  --checkpoint-path C:\Projects\sglang\checkpoints\chunkwise\causal_forcing.pt `
  --base-transformer-config C:\Projects\sglang\checkpoints\wan2.1-t2v-1.3b-diffusers\transformer\config.json `
  --output-dir C:\Projects\sglang\checkpoints\causal-forcing-wan\transformer `
  --model-dir C:\Projects\sglang\checkpoints\causal-forcing-wan
```

On Windows, where importing the package as `-m sglang...` currently hits the
Unix-only `resource` import, the script can be invoked directly:

```powershell
C:\Projects\sglang\.venv\Scripts\python.exe python\sglang\multimodal_gen\tools\convert_causal_forcing_checkpoint.py `
  --checkpoint-path C:\Projects\sglang\checkpoints\chunkwise\causal_forcing.pt `
  --base-transformer-config C:\Projects\sglang\checkpoints\wan2.1-t2v-1.3b-diffusers\transformer\config.json `
  --output-dir C:\Projects\sglang\checkpoints\causal-forcing-wan\transformer `
  --model-dir C:\Projects\sglang\checkpoints\causal-forcing-wan
```

Expected naive launch shape after conversion, assuming a complete base Wan
Diffusers directory is available on the validation machine. Prefer
`--model-id Causal-Forcing` for local overlay paths so config resolution uses
the Causal Forcing registry entry; SGLang matches this value against the HF repo
short name. The full HF id `thu-ml/Causal-Forcing` is the registered model path,
but it is not the explicit `--model-id` value. `model_index.json` supplies
`WanCausalForcingPipeline`, so `--pipeline-class-name` is not required.

```powershell
uv run --project python python -m sglang.launch_server `
  --model-path C:\Projects\sglang\checkpoints\causal-forcing-wan `
  --model-id Causal-Forcing `
  --backend sglang `
  --component-paths.text_encoder C:\path\to\Wan2.1-T2V-1.3B-Diffusers\text_encoder `
  --component-paths.tokenizer C:\path\to\Wan2.1-T2V-1.3B-Diffusers\tokenizer `
  --component-paths.vae C:\path\to\Wan2.1-T2V-1.3B-Diffusers\vae `
  --component-paths.scheduler C:\path\to\Wan2.1-T2V-1.3B-Diffusers\scheduler
```

Verification:

- `py_compile` passed for the converter, resolver changes, and focused tests.
- A direct file-level converter helper smoke test passed for wrapper key
  normalization and causal config patching.
- A direct file-level model-index helper smoke test passed for patching
  `WanPipeline`/`WanTransformer3DModel` to
  `WanCausalForcingPipeline`/`CausalWanTransformer3DModel`.
- `git diff --check` passed. Git reported CRLF warnings for existing tracked
  files on this Windows checkout.
- Focused pytest could not run in the checked-in `.venv`: `No module named
  pytest`.
- Direct helper import on Windows also fails before reaching the converter
  because `sglang.__init__` imports the Unix-only `resource` module.

### 2026-07-08: Real Local Checkpoint Conversion

Used the main checkout virtual environment at `C:\Projects\sglang\.venv`, which
already contains `torch`, `safetensors`, and `pytest`, to run the converter
against the actual downloaded checkpoint.

Command:

```powershell
C:\Projects\sglang\.venv\Scripts\python.exe `
  C:\Users\MikanAffine\.codex\worktrees\a3ef\sglang\python\sglang\multimodal_gen\tools\convert_causal_forcing_checkpoint.py `
  --checkpoint-path C:\Projects\sglang\checkpoints\chunkwise\causal_forcing.pt `
  --base-transformer-config C:\Projects\sglang\checkpoints\wan2.1-t2v-1.3b-diffusers\transformer\config.json `
  --output-dir C:\Projects\sglang\checkpoints\causal-forcing-wan\transformer `
  --model-dir C:\Projects\sglang\checkpoints\causal-forcing-wan
```

Generated files:

- `C:\Projects\sglang\checkpoints\causal-forcing-wan\model_index.json`
  (`WanCausalForcingPipeline`).
- `C:\Projects\sglang\checkpoints\causal-forcing-wan\transformer\config.json`
  (`CausalWanTransformer3DModel`, 30 layers, 12 heads, 4-step causal defaults).
- `C:\Projects\sglang\checkpoints\causal-forcing-wan\transformer\diffusion_pytorch_model.safetensors`
  (5,676,070,784 bytes).

Safetensors inspection:

- `tensor_count`: 825.
- `patch_embedding` keys are present.
- `blocks.*.attn1.to_q.*` keys are present.
- Old official Wan `self_attn.q` keys are absent.
- Reference wrapper `model.*` prefixes are absent.

Validated the converted component with:

```powershell
C:\Projects\sglang\.venv\Scripts\python.exe `
  C:\Users\MikanAffine\.codex\worktrees\a3ef\sglang\python\sglang\multimodal_gen\tools\convert_causal_forcing_checkpoint.py `
  --validate-only `
  --output-dir C:\Projects\sglang\checkpoints\causal-forcing-wan\transformer `
  --model-dir C:\Projects\sglang\checkpoints\causal-forcing-wan
```

Validator output:

```json
{
  "has_attn1_to_q": true,
  "has_official_wan_key": false,
  "has_patch_embedding": true,
  "has_required_key_surface": true,
  "has_wrapper_prefix": false,
  "tensor_count": 825
}
```

Static local loader-name check:

- `CausalWanTransformer3DModel` is registered by
  `runtime/models/dits/causal_wanvideo.py`.
- `UMT5EncoderModel` is registered by `runtime/models/encoders/t5.py`.
- `AutoencoderKLWan` is registered by `runtime/models/vaes/wanvae.py`.
- `UniPCMultistepScheduler` is registered by
  `runtime/models/schedulers/scheduling_unipc_multistep.py`.
- Upstream `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/model_index.json`, fetched as a
  small metadata file from `hf-mirror.com`, declares
  `scheduler: ["diffusers", "UniPCMultistepScheduler"]` and
  `tokenizer: ["transformers", "T5TokenizerFast"]`. The generated
  `checkpoints/causal-forcing-wan/model_index.json` was refreshed to match those
  upstream component entries while replacing only the pipeline and transformer
  class names with the causal SGLang classes.
- `WanCausalForcingPipeline` requires `text_encoder`, `tokenizer`, `vae`,
  `transformer`, and `scheduler`. `ComposedPipelineBase._resolve_component_path`
  applies `server_args.component_paths` per module before loading, so the H100
  launch can keep only the converted causal transformer under
  `checkpoints/causal-forcing-wan/transformer` and supply the other Wan
  components through `--component-paths.*`.
- `TransformerLoader` resolves the transformer implementation from
  `transformer/config.json` (`_class_name`), not from the architecture string in
  `model_index.json`. Therefore the converted transformer config must keep
  `_class_name: CausalWanTransformer3DModel`.
- The SGLang Wan loader maps Diffusers-style checkpoint keys such as
  `blocks.0.attn1.to_q.weight` to internal module names such as
  `blocks.0.to_q.weight`. The converted safetensors should keep Diffusers-style
  keys, not internal SGLang keys.
- The converter validator now checks a broader representative key surface:
  patch embedding, time/text embedding, self-attention, cross-attention, FFN,
  norm2, and output head prefixes.

Local pytest status:

- Running focused tests with `C:\Projects\sglang\.venv` and this worktree on
  `PYTHONPATH` fails before test collection because Windows lacks the Unix
  `resource` module.
- A temporary test-only `resource` shim moved import past that point, but
  collection then failed on missing `orjson`. The shim was removed.
- Therefore pytest is still a local environment/import-surface blocker, not a
  known Causal Forcing test assertion failure.

Additional local preflight after tightening validator:

- `py_compile` passed for
  `tools/convert_causal_forcing_checkpoint.py` and
  `test/unit/test_causal_forcing_conversion.py`.
- Direct import smoke checks passed for the broader converted key-surface
  validator and for rejecting an incomplete converted key surface.
- `--validate-only` against
  `C:\Projects\sglang\checkpoints\causal-forcing-wan\transformer` passed with
  `has_required_key_surface: true`.
- `git diff --check` passed with only CRLF warnings on this Windows checkout.
- Focused pytest with `PYTHONPATH` set still fails before collection on
  `ModuleNotFoundError: No module named 'resource'`.

### 2026-07-08: Naive Pipeline Timestep Wiring Fix

Found one local pipeline wiring issue before H100 execution:

- `CausalDMDDenoisingStage` maps DMD step ids like `[1000, 750, 500, 250]`
  through the scheduler's 1000-step training schedule when
  `warp_denoising_step` is enabled.
- The existing `DMDTimestepPreparationStage` already prepares this schedule by
  calling `scheduler.set_timesteps(num_train_timesteps)` and storing the warped
  DMD timesteps on `batch.timesteps`.
- `WanCausalForcingPipeline` did not include `DMDTimestepPreparationStage`, unlike
  the existing LingBotWorld causal DMD pipeline. This could leave the causal
  denoising stage dependent on its fallback timestep preparation instead of the
  modular SGLang stage chain.

Implemented the minimal wiring fix:

- Added `WanCausalForcingPipeline.initialize_pipeline` to override the loaded
  scheduler with `SelfForcingFlowMatchScheduler(num_inference_steps=1000,
  shift=flow_shift, sigma_min=0.0, extra_one_step=True)`. This matches the
  upstream Causal Forcing `WanDiffusionWrapper`, which uses a shifted
  flow-match scheduler with `sigma_min=0.0`, `extra_one_step=True`, and a
  1000-step training schedule before warping `[1000, 750, 500, 250]`.
- Rechecked the SGLang pipeline lifecycle after adding the scheduler override:
  `ComposedPipelineBase.__init__` calls `load_modules(...)` before
  `initialize_pipeline(...)`. `SchedulerLoader` therefore still needs the
  `scheduler` component directory during generic component loading, even though
  `WanCausalForcingPipeline.initialize_pipeline` replaces it before stage creation.
  The H100 runbook should keep downloading/preflighting `scheduler/*`.
- Added `DMDTimestepPreparationStage(self.get_module("scheduler"))` before
  latent preparation in `WanCausalForcingPipeline`.
- Updated `CausalDMDDenoisingStage._prepare_causal_dmd_timesteps` to reuse
  `batch.timesteps` when present, falling back to the old local preparation only
  if the stage was not run.
- Aligned that fallback preparation with `DMDTimestepPreparationStage`: it now
  reads `scheduler.num_train_timesteps` or `scheduler.config.num_train_timesteps`,
  calls `scheduler.set_timesteps(num_train_timesteps)`, and indexes
  `scheduler_timesteps[num_train_timesteps - timesteps]` instead of hard-coding
  `1000 - timesteps`.
- Added a focused static wiring test that does not import the SGLang package on
  Windows: it checks that Wan causal pipeline prepares DMD timesteps before
  constructing the causal denoising stage, initializes the upstream-compatible
  flow-match scheduler, and that causal denoising consumes prepared batch
  timesteps. The same test also locks the fallback preparation to the
  `num_train_timesteps`-based schedule shape.

Local reasoning checks:

- Wan default `num_frames=81` becomes 21 latent frames through Wan VAE temporal
  compression: `(81 - 1) // 4 + 1 = 21`, matching the reference latent shape and
  divisible by `num_frames_per_block=3`.
- The causal denoising loop then processes 7 causal blocks of 3 latent frames,
  with 4 DMD timesteps per block.

Verification:

- `py_compile` passed for `wan_causal_forcing_pipeline.py`,
  `causal_denoising.py`, and the new wiring test.
- Direct execution of the static wiring test functions passed on Windows,
  including the upstream-compatible flow-match scheduler initialization and the
  fallback `num_train_timesteps` schedule invariant.
- Converted checkpoint `--validate-only` still passed with
  `has_required_key_surface: true` and `tensor_count: 825`.
- `git diff --check` passed with only CRLF warnings.
- Focused pytest with `PYTHONPATH` set still fails before collection on
  `ModuleNotFoundError: No module named 'resource'`.

Additional local preflight on 2026-07-08 after tightening the H100 launch
command:

- Re-inspected `registry._get_config_info`: explicit `--model-id` is matched
  against `get_model_short_name(registered_hf_id)`, so the H100 command should
  use `--model-id Causal-Forcing` rather than `--model-id
  thu-ml/Causal-Forcing`.
- Re-inspected the pipeline/config resolver split. Native pipeline selection
  reads `model_index.json` and uses `_class_name: WanCausalForcingPipeline`, while
  config selection uses path/model-id detectors. The Causal Forcing detector now
  also matches `wancausalforcingpipeline`, so a valid local overlay can resolve the
  Causal Forcing config from its `model_index.json` pipeline name even if the
  directory is renamed and `--model-id` is omitted.
- Added a focused registry test for `_get_config_info("local-overlay",
  model_id="Causal-Forcing")` resolving to the Causal Forcing config classes.
- Re-inspected the `sglang generate` CLI path:
  `sglang.cli.main` forwards unknown args to `sglang.cli.generate`,
  `add_multimodal_gen_generate_args` registers the common server and sampling
  args, `ServerArgs._extract_component_paths` accepts
  `--component-paths.<component>`, and `DiffGenerator.from_pretrained(...,
  local_mode=True)` launches the local diffusion scheduler/worker. This means
  the H100 smoke command can use one `sglang generate` command without a
  separate `launch_server` step.
- Added a focused registry test for the CLI pre-dispatch condition: a local
  path containing `causal-forcing` is recognized as a known non-diffusers
  diffusion model and maps to `WanCausalForcingPipeline`.
- Added static local coverage for the model-index detector condition because
  normal pytest collection is still blocked on Windows by the Unix-only
  `resource` import.
- Added a Linux/H100 registry test with a temporary local directory containing
  only a Causal Forcing `model_index.json`. This verifies that
  `_class_name: WanCausalForcingPipeline` is enough for native pipeline resolution
  and Causal Forcing config selection even if the overlay directory name does
  not contain `causal-forcing`.
- `py_compile` passed for all modified Python files and the new tests.
- Converted checkpoint `--validate-only` still passed with
  `has_required_key_surface: true` and `tensor_count: 825`.
- Direct execution of the conversion and static wiring test functions passed on
  Windows through file-level imports; 21 direct Causal Forcing test functions
  passed after the model-index detector update.
- `git diff --check` passed with only CRLF warnings.
- Focused pytest for the three Causal Forcing tests still fails before
  collection on Windows because `sglang.__init__` imports the Unix-only
  `resource` module.

Additional algorithm-equivalence audit on 2026-07-08:

- Rechecked the upstream chunkwise Causal Forcing inference loop. For each
  causal block, upstream builds `timestep` with shape
  `[batch_size, current_num_frames]`, calls the generator with that shape, adds
  noise for the next denoising step with a flattened
  `[batch_size * current_num_frames]` timestep vector, and refreshes the KV
  cache with `torch.ones_like(timestep) * context_noise`.
- Rechecked SGLang's `CausalWanTransformer3DModel`: it derives the internal
  per-frame modulation layout from `timestep.shape`. Passing `(B, 1)` for a
  3-frame chunk makes the block look like one large frame for timestep
  modulation rather than three causal frames.
- Fixed `CausalDMDDenoisingStage` so DMD chunk denoising passes timestep shape
  `(B, F)` into the causal transformer, converts predicted flow/noise to x0
  with the same per-frame timestep shape, sends `add_noise` a flattened
  `(B * F)` next-timestep vector, and refreshes the context cache with
  `(B, F)` context timesteps.
- Added static wiring checks for these timestep-shape invariants because full
  package import is still blocked on Windows.
- Local verification after this change:

  ```text
  wiring tests passed: test_causal_context_cache_refresh_uses_per_frame_timestep_shape,
  test_causal_denoising_fallback_matches_dmd_timestep_preparation_shape,
  test_causal_denoising_uses_per_frame_timestep_shape_for_chunks,
  test_causal_denoising_uses_prepared_batch_timesteps,
  test_wan_causal_pipeline_prepares_dmd_timesteps_before_denoising
  ```

- `py_compile` passed for the edited causal denoising stage and static wiring
  test file.

Additional local preflight after making the H100 runbook reproducible:

- The H100 sequence now includes conversion from
  `checkpoints/chunkwise/causal_forcing.pt` into
  `checkpoints/causal-forcing-wan`, instead of assuming the converted overlay
  already exists on the validation machine.
- The H100 download command still uses `HF_ENDPOINT=https://hf-mirror.com` and
  only adds `transformer/config.json`; it does not download Wan transformer
  weights.
- The converter unit test now imports the standalone converter script by path,
  so conversion logic can be exercised locally on Windows without importing the
  full `sglang` package.
- Direct execution of the conversion unit test functions passed locally:
  state selection, key normalization, duplicate-key rejection, config/model
  index patching, and metadata validation all passed.
- Fresh local checks passed: converted overlay `--validate-only`,
  `py_compile` for all modified Python files and Causal Forcing tests, and
  `git diff --check` with only CRLF warnings.

Additional converter-surface audit:

- Compared the converted tensor key families against SGLang's Wan loader
  mapping. Diffusers-style `blocks.N.attn1.*` keys are mapped by
  `WanVideoArchConfig.param_names_mapping` to the internal causal Wan
  self-attention modules; `blocks.N.attn2.*` keys correspond directly to the
  internal cross-attention module names.
- Inspected the real converted safetensors file: it has 825 tensors and no
  leftover official key families such as `self_attn`, `cross_attn`,
  `time_embedding`, `text_embedding`, or `head`.
- Tightened converter validation so, when `transformer/config.json` declares
  `num_layers`, every layer must expose the required converted key surface
  (`attn1.to_q`, `attn2.to_q`, ffn input projection, residual norm, and
  scale-shift table). This catches truncated or partially converted
  multi-layer checkpoints before H100 model load.
- Added a regression test that fails if a two-layer config is accepted with
  only block-0 keys.
- Direct conversion unit functions, `--validate-only` on the current 30-layer
  overlay, and `py_compile` for the converter and test passed after this
  change.

Additional sampling/defaults audit:

- Confirmed the Causal Forcing sampling class should expose the checkpoint as
  a 4-step model, not inherit base Wan's 50-step default.
- Set `CausalForcingWanT2V480PConfig.num_inference_steps = 4` and restated
  `num_frames = 81`. With Wan temporal compression this corresponds to 21
  latent frames, matching the reference chunkwise latent shape and the
  converter's `sliding_window_num_frames = 21`.
- Set `adjust_frames = False` for this sampling class so the 81-frame request is
  not silently changed before the H100 smoke test.
- Added static wiring coverage that checks the Causal Forcing sampling default
  is 4 steps/81 frames, the pipeline config keeps `[1000, 750, 500, 250]`, and
  the converter writes 3-frame causal blocks with a 21-frame sliding window.
- Added converter validate-only preflight for base Wan component directories.

Additional causal DMD trajectory audit:

- Traced the chunk denoising data flow through
  `CausalDMDDenoisingStage._denoise_causal_dmd_chunk`. Each non-final DMD step
  predicts clean `x0`, re-noises it at the next DMD timestep, converts that
  tensor back to `[B, C, T, H, W]`, and feeds it into the next transformer call.
  The final DMD step writes clean `x0` back to the chunk. This matches the
  existing non-causal DMD stage's scheduler trajectory shape.
- Rechecked the latent layout at the stage boundaries: standard latent
  preparation, `CausalWanTransformer3DModel.forward`, and standard VAE decoding
  all use `[B, C, T, H, W]`, while only the x0/noise conversion helper uses the
  temporary `[B, T, C, H, W]` layout before flattening frame batches.
- Added a static regression check that locks this re-noise handoff invariant
  without importing the full `sglang` package on Windows.

Additional CFG/cache audit:

- The official chunkwise Causal Forcing config uses `guidance_scale: 3.0`, and
  the SGLang Wan 1.3B sampling defaults also set `guidance_scale = 3.0`.
  `Req` enables classifier-free guidance from that value, so the causal DMD
  denoising stage must run both positive and negative text branches.
- Found that `CausalDMDDenoisingStage` validated CFG fields but only ran the
  positive branch. Added local CFG prediction:
  `uncond + guidance_scale * (text - uncond)` using the existing negative prompt
  embeddings and negative attention mask.
- Found a launch-time compatibility issue in the same path:
  `CausalWanTransformerBlock` passed `crossattn_cache=` to
  `WanT2VCrossAttention`, but that forward method did not accept the argument.
  Added optional cross-attention cache support to avoid the TypeError and reuse
  text K/V when the context is unchanged.
- Because CFG switches text conditions, the causal denoising stage resets
  cross-attention caches before the positive branch, before the negative branch,
  and after combining CFG output. This prevents positive and negative text K/V
  from being reused across branches, while the later clean-latent context refresh
  rebuilds the positive cache for future causal context.
- Added static regression checks for CFG combination, cross-attention cache
  compatibility, and preserving the Causal Forcing sampling class as a CFG
  path rather than silently changing it to `guidance_scale=1.0`.
  Passing `--base-model-dir` checks that `scheduler`, `text_encoder`,
  `tokenizer`, and `vae` directories exist before the H100 generation command
  tries to load them through `--component-paths.*`.
- Tightened the base Wan preflight so empty component directories from an
  interrupted download do not pass. It now also checks for minimal launch files:
  scheduler config, `tokenizer.json` or `spiece.model`, text encoder
  config plus real `.safetensors`/`.bin` shards, and VAE config plus real
  `.safetensors` shards.
- Local preflight with the incomplete local base directory fails as intended:
  `FileNotFoundError: ... missing required launch component(s): vae`.
  This is the expected local state because full base Wan components are not being
  downloaded on the 4060 machine.

Additional source-level equivalence audit:

- Rechecked the upstream `thu-ml/Causal-Forcing` repository on 2026-07-08. The
  README still documents chunk-wise 4-step Causal Forcing as the Wan1.3B AR
  checkpoint path and notes that the base method is trained for 81-frame videos,
  matching the local `num_frames=81` smoke target.
- Rechecked upstream `configs/causal_forcing_dmd_chunkwise.yaml`: the chunkwise
  config uses `denoising_step_list: [1000, 750, 500, 250]`,
  `warp_denoising_step: true`, `timestep_shift: 5.0`,
  `guidance_scale: 3.0`, latent shape `[1, 21, 16, 60, 104]`, and
  `num_frame_per_block: 3`. It does not define a separate
  `denoising_step_list_first_chunk`, so using the same 4-step schedule for
  every chunk is equivalent for this checkpoint.
- Rechecked SGLang's `CausalWanTransformer3DModel`: it hardcodes
  `independent_first_frame = False`, so the naive T2V path uses seven
  3-latent-frame chunks for the 21-latent-frame target rather than the framewise
  `[1, 3, 3, ...]` branch.

Additional local no-download preflight on 2026-07-08:

- Re-audited the H100 smoke command against the local CLI definitions. The
  command-line flags in the runbook are registered locally:
  `--component-paths.<component>`, `--model-id`, `--backend`,
  `--text-encoder-cpu-offload`, `--pin-cpu-memory`, `--output-path`, and
  `--output-file-name`.
- Reproduced the base Wan preflight gap with a failing test: empty component
  directories were previously accepted. Tightened
  `validate_base_wan_component_dirs` to require minimal launch files as well as
  directories.
- Re-audited the preflight against the component loaders. Tokenizer
  `tokenizer_config.json` alone is not enough for `AutoTokenizer`, and weight
  index files alone are not loadable by the text encoder/VAE loaders. The
  preflight now rejects those incomplete download states.
- Direct execution of conversion and wiring test functions passed locally
  (`20` functions) without importing the full SGLang package or downloading
  Wan components.
- Converted overlay `--validate-only` still passed with
  `has_required_key_surface: true` and `tensor_count: 825`.
- Local `--base-model-dir` preflight still fails as intended because the local
  Wan base directory is incomplete. This is expected and avoids treating the
  4060 machine as a full generation target.

Additional transformer-loader audit:

- Rechecked the native transformer load path without downloading Wan locally.
  `TransformerLoader` reads `transformer/config.json`, updates the pipeline
  `dit_config.arch_config` with that config, then resolves the implementation
  from `_class_name` through `ModelRegistry.resolve_model_cls`.
- `runtime/models/dits/causal_wanvideo.py` exposes
  `EntryClass = CausalWanTransformer3DModel`, so the converted
  `_class_name: CausalWanTransformer3DModel` should select the causal Wan model
  rather than the standard Wan transformer.
- `WanVideoArchConfig` already owns the causal fields
  `num_frames_per_block`, `sliding_window_num_frames`, `local_attn_size`, and
  `sink_size`. Because `update_model_arch` copies config entries directly onto
  the arch config and calls `__post_init__`, the converted transformer config is
  the right place to carry the 3-frame chunk and 21-latent-frame window values.
- Added a static regression check for this loader chain so future edits do not
  accidentally make the converted overlay resolve to the standard Wan
  transformer.
- Local no-download verification after this audit passed:
  `py_compile` for the Causal Forcing files, `25` direct conversion/static
  wiring test functions, converted overlay `--validate-only` with
  `has_required_key_surface: true` and `tensor_count: 825`, and
  `git diff --check` with only CRLF warnings.

Additional causal cache reuse audit:

- Rechecked `DMDTimestepPreparationStage` and `CausalDMDDenoisingStage`: the
  DMD preparation stage stores both `batch.timesteps` and request-local
  `batch.scheduler`, and causal denoising consumes those prepared values before
  falling back to local timestep preparation. This keeps the modular SGLang
  stage chain as the primary path.
- Rechecked `LatentPreparationStage`: it prepares initial latents and scales by
  `scheduler.init_noise_sigma` when available, but does not call
  `scheduler.set_timesteps` or overwrite `batch.timesteps`. Therefore the Wan
  causal pipeline stage order of DMD timestep preparation before latent
  preparation preserves the intended 4-step DMD schedule.
- Rechecked the causal Wan cross-attention call chain. `CausalWanTransformerBlock`
  calls `WanT2VCrossAttention` with both `context_lens=None` and
  `crossattn_cache=...`, and `WanT2VCrossAttention.forward` accepts those
  arguments, so the previous cache-argument compatibility issue is covered.
- Rechecked the DMD timestep shape through `pred_noise_to_pred_video` and
  `SelfForcingFlowMatchScheduler.add_noise`: the causal transformer receives
  per-frame timesteps as `(batch, frames)`, while both scheduler-facing helpers
  explicitly flatten 2-D timesteps to match flattened `(batch * frames, ...)`
  latents. This preserves upstream Causal Forcing's per-frame modulation while
  keeping scheduler conversion shapes consistent.
- Found a normal-path cache reuse risk: `CausalDMDDenoisingStage.forward`
  reused stage-local self-attention and cross-attention caches across requests
  after only resetting cursors. If the next request changed batch size, latent
  spatial size, text length, dtype, device, or causal window size, stale cache
  tensors could be reused with incompatible shapes.
- Added a focused cache-shape guard for the normal causal path. It now
  reinitializes caches when the stored cache metadata no longer matches the
  current request and only resets caches when the shapes still match. This does
  not change the DMD denoising trajectory; it only prevents invalid state reuse
  across requests.
- Found and fixed a causal window config-source issue: the generic causal
  denoising stage tried to read `local_attn_size` from `self.transformer.model`,
  which is not the normal loaded Causal Wan model surface. It now reads
  `local_attn_size`, `sink_size`, `sliding_window_num_frames`, and causal block
  size from `self.transformer.config.arch_config`, matching the converted
  `transformer/config.json` source of truth.
- Found and fixed a prompt-embedding hook divergence: the shared denoising
  stage obtains text embeddings through
  `pipeline_config.get_pos_prompt_embeds/get_neg_prompt_embeds`, while causal
  denoising read `batch.prompt_embeds` and `batch.negative_prompt_embeds`
  directly. The causal path now uses the same pipeline-config accessors, keeping
  model-specific prompt selection centralized while preserving the existing Wan
  one-text-encoder behavior.
- Found and fixed a realtime causal cache reuse gap: the realtime/session cache
  reset predicate only checked block count and causal cache token/head layout,
  while the normal path also guarded against request batch size, text length,
  dtype, device, and attention-window changes. The realtime path now checks the
  same request-shape/device properties before reusing cached self-attention and
  cross-attention tensors.
- Found and fixed a server-warmup launch risk: SGLang server warmup caps video
  requests to 17 frames by default. For Wan's temporal compression ratio 4, that
  becomes 5 latent frames, which violates Causal Forcing's 3-latent-frame
  causal block requirement. Server warmup now aligns causal video warmup frame
  counts to a valid latent-block multiple, so Causal Forcing warmup uses 9
  frames (3 latent frames) instead of the invalid 17-frame shape.
- Local verification after the cache guard passed: `py_compile`, `26` direct
  conversion/static wiring test functions, converted overlay `--validate-only`
  with `tensor_count: 825`, `git diff --check` with only CRLF warnings, and no
  workspace temp/pytest residue.
- Follow-up local verification after adding the stage-order regression check
  also passed: `py_compile`, `26` direct conversion/static wiring test
  functions, converted overlay `--validate-only`, `git diff --check` with only
  CRLF warnings, and no workspace temp/pytest residue.
- Follow-up local verification after the timestep-shape helper audit passed:
  `py_compile`, `27` direct conversion/static wiring test functions, converted
  overlay `--validate-only`, `git diff --check` with only CRLF warnings, and no
  workspace temp/pytest residue.
- Follow-up local verification after the causal window config-source fix
  passed: `py_compile`, `28` direct conversion/static wiring test functions,
  converted overlay `--validate-only`, `git diff --check` with only CRLF
  warnings, and no workspace temp/pytest residue.
- Follow-up local verification after the prompt-embedding accessor fix passed:
  `py_compile`, `29` direct conversion/static wiring test functions, converted
  overlay `--validate-only`, `git diff --check` with only CRLF warnings, and no
  workspace temp/pytest residue.
- Follow-up local verification after the realtime cache reuse guard passed:
  `py_compile`, `30` direct conversion/static wiring test functions, converted
  overlay `--validate-only` with `tensor_count: 825`, `git diff --check` with
  only CRLF warnings, and no workspace temp/pytest residue.
- Follow-up local verification after the causal server-warmup frame alignment
  fix passed: `py_compile`, `31` direct conversion/static wiring test
  functions, converted overlay `--validate-only` with `tensor_count: 825`,
  `git diff --check` with only CRLF warnings, and no workspace temp/pytest
  residue.

Remaining local blocker for full naive generation:

- `C:\Projects\sglang\checkpoints\wan2.1-t2v-1.3b-diffusers` currently contains
  `transformer/config.json` plus incomplete local component directories from an
  aborted download attempt. It is missing required launch files/components,
  including `vae`. The overlay can load only once the full base Wan components
  are available on the H100 validation machine or supplied there via
  `--component-paths.*`.
- Hugging Face metadata for `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` shows the needed
  non-transformer files total 23,252,743,450 bytes:
  five `text_encoder/*.safetensors` shards, tokenizer files, scheduler config,
  and `vae/diffusion_pytorch_model.safetensors`.
- The local 4060 8GB machine is not a good target for full Wan end-to-end
  generation. Do not download the full base Wan component set locally unless
  explicitly requested. The attempted local component download was only for
  assembling a launchable overlay layout, but that is better deferred to H100.
  It was aborted and left only empty `scheduler`, `text_encoder`, and
  `tokenizer` directories.

Remote/Linux validation needed:

- Download the missing non-transformer Wan components on the H100 validation
  machine through the mirror:

  ```bash
  HF_ENDPOINT=https://hf-mirror.com hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --local-dir checkpoints/wan2.1-t2v-1.3b-diffusers \
    --include model_index.json 'scheduler/*' 'text_encoder/*' 'tokenizer/*' 'vae/*' \
      transformer/config.json
  ```

- Convert the raw Causal Forcing checkpoint on the H100 validation machine if
  `checkpoints/causal-forcing-wan` has not already been generated there. The
  observed main checkout path is `checkpoints/chunkwise/causal_forcing.pt`.
  Add `--overwrite` only when intentionally regenerating an existing overlay.
- Run `uv run --project python python -m pytest
  python/sglang/multimodal_gen/test/unit/test_causal_forcing_registry.py
  python/sglang/multimodal_gen/test/unit/test_causal_forcing_conversion.py
  python/sglang/multimodal_gen/test/unit/test_causal_forcing_pipeline_wiring.py
  -q`.
- Verify the converted transformer loads with `WanCausalForcingPipeline` using a
  complete Wan 2.1 T2V 1.3B Diffusers component set before running a full
  generation.
- Run a short generation smoke test with the converted
  `checkpoints/causal-forcing-wan` overlay.

Concrete H100 smoke-test sequence from the repo root:

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir checkpoints/wan2.1-t2v-1.3b-diffusers \
  --include model_index.json 'scheduler/*' 'text_encoder/*' 'tokenizer/*' 'vae/*' \
    transformer/config.json

uv run --project python python python/sglang/multimodal_gen/tools/convert_causal_forcing_checkpoint.py \
  --checkpoint-path checkpoints/chunkwise/causal_forcing.pt \
  --base-transformer-config checkpoints/wan2.1-t2v-1.3b-diffusers/transformer/config.json \
  --base-model-index checkpoints/wan2.1-t2v-1.3b-diffusers/model_index.json \
  --output-dir checkpoints/causal-forcing-wan/transformer \
  --model-dir checkpoints/causal-forcing-wan

uv run --project python python python/sglang/multimodal_gen/tools/convert_causal_forcing_checkpoint.py \
  --validate-only \
  --output-dir checkpoints/causal-forcing-wan/transformer \
  --model-dir checkpoints/causal-forcing-wan \
  --base-model-dir checkpoints/wan2.1-t2v-1.3b-diffusers

uv run --project python sglang generate \
  --model-path checkpoints/causal-forcing-wan \
  --model-id Causal-Forcing \
  --backend sglang \
  --component-paths.text_encoder checkpoints/wan2.1-t2v-1.3b-diffusers/text_encoder \
  --component-paths.tokenizer checkpoints/wan2.1-t2v-1.3b-diffusers/tokenizer \
  --component-paths.vae checkpoints/wan2.1-t2v-1.3b-diffusers/vae \
  --component-paths.scheduler checkpoints/wan2.1-t2v-1.3b-diffusers/scheduler \
  --prompt "A quiet city street after rain, cinematic lighting" \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --guidance-scale 3.0 \
  --num-inference-steps 4 \
  --seed 42 \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --save-output \
  --output-path outputs/causal-forcing-smoke \
  --output-file-name causal-forcing-smoke.mp4
```

Expected smoke-test evidence:

- The `--validate-only` command prints `has_required_key_surface: true` and
  `tensor_count: 825`.
- The generation command loads `WanCausalForcingPipeline` and
  `CausalWanTransformer3DModel` without missing component errors.
- The output file `outputs/causal-forcing-smoke/causal-forcing-smoke.mp4` is
  written.

## Current Isolation Audit

- `CausalDMDDenoisingStage` is back to the repository version. Causal
  Forcing-specific timestep mapping, CFG, per-frame timestep shape, cache shape
  validation, and prompt-embed accessors now live in
  `CausalForcingDMDDenoisingStage`.
- `WanCausalDMDPipeline` is back to the repository version. Causal Forcing uses
  a separate `WanCausalForcingPipeline` and model-index `_class_name`.
- The remaining shared Wan model change is the optional `crossattn_cache`
  argument on `WanT2VCrossAttention.forward`; the default `None` path keeps the
  original key/value projection behavior.
- The shared warmup builder now routes through a helper, but the frame-count
  realignment is guarded to `WanCausalForcingPipeline` or
  `CausalForcingWanT2V480PConfig`, not every causal DMD model.

## Algorithm-Correctness Notes

- Rechecked the official chunkwise inference path. The naive SGLang path now
  preserves these reference invariants: 81 output pixel frames map to 21 latent
  frames, latent blocks are 3 frames each, DMD timesteps are
  `[1000, 750, 500, 250]` warped through the shifted flow-match scheduler,
  `timestep_shift` is 5.0, `context_noise` is 0, and default CFG uses
  `guidance_scale` 3.0 with the Wan negative prompt.
- The Causal Forcing stage reuses the prepared DMD timesteps instead of
  re-warping them, keeps transformer timesteps per-frame as `[batch, frames]`,
  and flattens only for scheduler math.
- The DMD loop updates KV cache at the same `current_start` for repeated
  denoising steps, then refreshes the same cache range with the clean context
  latent after the block is finalized, matching the reference loop structure.
- Prompt and negative-prompt embeddings are cast to the Causal Wan transformer
  target dtype inside `CausalForcingDMDDenoisingStage`, preventing fp32 text
  encoder outputs from tripping the bf16 DiT dtype assertion.
- `CausalForcingWanT2V480PConfig` now carries the official `context_noise = 0`
  setting explicitly, instead of relying only on the denoising stage fallback.
- The Causal Forcing stage now passes the absolute generated frame offset
  (`pos_start_base + start_index`) to both KV-cache token positioning and Wan
  RoPE `start_frame`, so context warm-up does not leave later blocks with a
  relative frame index. Plain T2V remains unchanged because `pos_start_base = 0`.
- Denoise-time model input now stays limited to the current generated chunk.
  Provided context latents are still cached through the warm-up path, but are no
  longer concatenated into the first denoise chunk, matching the upstream
  Causal Forcing inference loop.
- `WanCausalForcingPipeline` now exposes `pipeline_config_cls` and
  `sampling_params_cls`, so SGLang's pipeline discovery can register the
  Causal Forcing config/sampling defaults from `pipeline_class_name` alone.
- The conversion script now overwrites Causal Forcing arch fields
  (`num_frames_per_block = 3`, `sliding_window_num_frames = 21`,
  `local_attn_size = -1`, `sink_size = 0`) instead of using `setdefault`.
  This prevents stale base or overlay configs from silently changing the
  chunkwise Causal Forcing attention/window structure.

## Next Steps

1. On H100, download the missing base Wan components with
   `HF_ENDPOINT=https://hf-mirror.com` and validate that the converted
   transformer loads with `CausalWanTransformer3DModel`.
2. Run focused pytest on Linux/H100 for the three Causal Forcing tests.
3. Run a short H100 generation smoke test against the naive pipeline.
4. After the naive pipeline loads and generates, design the disaggregated VAE
   decoder worker path.
