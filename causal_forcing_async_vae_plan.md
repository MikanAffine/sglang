# Causal Forcing Whole-Request Async VAE Decoding Plan

## Summary

Reuse `WanCausalForcingPipeline` for both monolithic and disaggregated
deployments. The default `disagg_role=monolithic` path remains unchanged. With
`encoder`, `denoiser`, `decoder`, and `server` roles, SGLang's existing
Disaggregated Diffusion Pipeline splits the same stage graph across processes.

Phase 1 transfers the complete denoised latent tensor so request A's VAE decode
can overlap request B's denoising. Block-level transfer and causal chunk decode
remain Phase 2 work.

## Implementation Changes

- Register `CausalForcingDMDDenoisingStage` through
  `add_stage_factory(RoleType.DENOISER, ...)` so only monolithic and denoiser
  workers construct the transformer-dependent stage.
- Preserve the current stage name, ordering, monolithic numerical path, and
  standard `DecodingStage`.
- Reuse the existing whole-request Denoiser-to-Decoder P2P transfer, background
  RDMA push, and decoder pool without changing the transfer protocol.
- Do not add a pipeline class, mode field, or CLI flag. Continue using
  `WanCausalForcingPipeline`; deployment topology is selected with the existing
  disaggregation role and URL settings.
- Document a four-process `thu-ml/Causal-Forcing` deployment and clarify that
  Phase 1 overlaps complete requests rather than latent blocks.

## Compatibility

- Public APIs remain unchanged.
- Monolithic stage order, model configuration, output format, and failure
  behavior remain unchanged.
- Encoder workers run validation, text encoding, DMD timestep preparation, and
  latent preparation; denoiser workers run only Causal Forcing denoising;
  decoder workers run only standard VAE decoding.
- Transfer, role-compute, or decode failures fail the complete request using
  the existing disaggregation cleanup and timeout behavior.

## Test Plan

- Verify encoder and decoder roles do not construct the causal denoiser.
- Verify the denoiser role constructs it exactly once with the configured
  transformer and scheduler.
- Verify monolithic stage ordering is unchanged and role-specific stage/module
  filtering is correct.
- Exercise the existing disaggregation transfer path and run the focused Causal
  Forcing and disaggregation unit tests.
- Where the required GPUs and model checkpoint are available, run a two-GPU
  end-to-end deployment with encoder and decoder sharing GPU 0 and denoiser on
  GPU 1, compare monolithic/disaggregated outputs for a fixed seed, and confirm
  overlapping decoder/denoiser spans for two concurrent requests.

## Phase 2 Boundary

Phase 1 does not add block identifiers, chunk-completion messages, per-request
causal VAE cache state, or streaming client responses. Those belong to a
separate block-level Async VAE Decoding design.
