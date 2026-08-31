from __future__ import annotations

import argparse
import time

import torch
import torch_mlu
from diffusers import AutoencoderKLMiniMaxH3, ComponentsManager, MiniMaxH3Transformer3DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_utils import ModelMixin


DEVICE = torch.device("mlu:0")


class ToyModel(ModelMixin):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 64),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_states)


def synchronize() -> None:
    torch.mlu.synchronize(DEVICE)


def test_flash_attention(sequence_length: int) -> None:
    shape = (1, sequence_length, 56, 128)
    query = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    key = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    value = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)

    torch.mlu.reset_peak_memory_stats(DEVICE)
    started = time.perf_counter()
    output = dispatch_attention_fn(query, key, value, backend="flash")
    synchronize()
    elapsed = time.perf_counter() - started

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    peak_mib = torch.mlu.max_memory_allocated(DEVICE) / 1024**2
    print(f"flash_attention: shape={tuple(output.shape)}, elapsed={elapsed:.3f}s, peak={peak_mib:.0f}MiB")
    del query, key, value, output
    torch.mlu.empty_cache()


def test_transformer() -> None:
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=4,
        attention_head_dim=16,
        hidden_size=64,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=128,
        in_channels=4,
        audio_in_channels=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        freq_dim=16,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
        rope_freq_dim=2,
    ).to(device=DEVICE, dtype=torch.bfloat16).eval()
    model.set_attention_backend("flash")

    num_text, num_audio, num_video = 4, 4, 8
    sequence_length = num_text + num_audio + num_video
    text_indices = torch.arange(num_text, device=DEVICE)
    audio_indices = torch.arange(num_text, num_text + num_audio, device=DEVICE)
    video_indices = torch.arange(num_text + num_audio, sequence_length, device=DEVICE)
    token_tags = torch.empty(sequence_length, dtype=torch.long, device=DEVICE)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long, device=DEVICE)
    timestep_indices[audio_indices] = 1
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32, device=DEVICE)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        output = model(
            hidden_states=torch.randn(1, num_video, 16, device=DEVICE, dtype=torch.bfloat16),
            audio_hidden_states=torch.randn(1, num_audio, 8, device=DEVICE, dtype=torch.bfloat16),
            encoder_hidden_states=torch.randn(1, num_text, 32, device=DEVICE, dtype=torch.bfloat16),
            timestep=torch.tensor([0.7, 0.3], device=DEVICE),
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
        )
    synchronize()

    assert output.sample.shape == (1, num_video, 16)
    assert output.audio_sample.shape == (1, num_audio, 8)
    assert torch.isfinite(output.sample).all()
    assert torch.isfinite(output.audio_sample).all()
    print(
        "transformer: "
        f"video={tuple(output.sample.shape)}, audio={tuple(output.audio_sample.shape)}, dtype={output.sample.dtype}"
    )
    del model, output
    torch.mlu.empty_cache()


def test_video_vae() -> None:
    vae = AutoencoderKLMiniMaxH3(
        latent_channels=4,
        block_out_channels=(8, 8, 8),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2),
        temporal_downsample_factors=(1, 2, 2),
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        clip_length=17,
        token_drop=3,
    ).to(DEVICE).eval()
    vae.disable_tiling()
    vae.set_attention_backend("flash")
    latents = torch.randn(1, 4, 37, 4, 4, device=DEVICE)

    with torch.inference_mode(), torch.autocast(device_type="mlu", dtype=torch.float16):
        video = vae.decode(latents, return_dict=False)[0]
    synchronize()

    assert video.shape == (1, 3, 124, 32, 32)
    assert torch.isfinite(video).all()
    print(f"video_vae: output={tuple(video.shape)}, dtype={video.dtype}")
    del vae, latents, video
    torch.mlu.empty_cache()


def test_auto_offload() -> None:
    model = ToyModel().eval()
    manager = ComponentsManager()
    manager.add("toy", model)
    manager.enable_auto_cpu_offload(device=DEVICE, memory_reserve_margin="1GB")

    with torch.inference_mode():
        output = model(torch.randn(2, 64, device=DEVICE))
    synchronize()

    assert output.device.type == "mlu"
    assert torch.isfinite(output).all()
    print(f"auto_offload: output_device={output.device}, hooks={len(manager.model_hooks)}")
    del manager, model, output
    torch.mlu.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniMax-H3 MLU operator smoke tests.")
    parser.add_argument("--attention-sequence-length", type=int, default=40_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.mlu.is_available():
        raise SystemExit("No MLU device is available")

    print(f"device: {torch.mlu.get_device_name(0)}")
    test_flash_attention(args.attention_sequence_length)
    test_transformer()
    test_video_vae()
    test_auto_offload()
    print("MiniMax-H3 MLU smoke tests: PASS")


if __name__ == "__main__":
    main()
