from __future__ import annotations

from functools import wraps

import torch


_PATCHED = False


def apply_mlu_patches() -> None:
    """Apply compatibility fixes required by the pinned Diffusers revision."""
    global _PATCHED
    if _PATCHED:
        return

    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

    original_build_row_timesteps = MiniMaxH3SetTimestepsStep.build_row_timesteps

    @wraps(original_build_row_timesteps)
    def cpu_row_timesteps(video_indices, audio_indices, *args, **kwargs):
        return original_build_row_timesteps(
            video_indices.cpu(),
            audio_indices.cpu(),
            *args,
            **kwargs,
        )

    original_call = MiniMaxH3VideoDecodeStep.__call__

    @wraps(original_call)
    @torch.no_grad()
    def mlu_video_decode(self, components, state):
        block_state = self.get_block_state(state)
        device = components._execution_device

        if block_state.output_type not in ("pil", "np", "pt"):
            raise ValueError(
                "`output_type` must be one of 'pil', 'np' or 'pt', "
                f"got {block_state.output_type!r}."
            )

        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = block_state.latents * latents_std + latents_mean

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type in ("cuda", "mlu"),
        ):
            video = components.vae.decode(latents, return_dict=False)[0]

        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
        block_state.videos = components.video_processor.postprocess_video(
            video,
            output_type=block_state.output_type,
        )

        self.set_block_state(state, block_state)
        return components, state

    MiniMaxH3SetTimestepsStep.build_row_timesteps = staticmethod(cpu_row_timesteps)
    MiniMaxH3VideoDecodeStep.__call__ = mlu_video_decode
    _PATCHED = True


def set_attention_backend(pipe, backend: str) -> None:
    if backend == "default":
        return

    for name in ("transformer", "transformer_ref", "vae"):
        component = getattr(pipe, name, None)
        if component is not None and hasattr(component, "set_attention_backend"):
            component.set_attention_backend(backend)
