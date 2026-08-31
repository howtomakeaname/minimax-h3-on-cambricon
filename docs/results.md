# Validation Results

## Test platform

- Accelerator: one Cambricon MLU590 with 80 GiB memory
- Host memory: approximately 1.9 TiB
- Precision: official BF16 checkpoint
- Runtime: eager mode with MLU Flash Attention
- Memory strategy: Diffusers automatic CPU offload, 12 GB reserve margin
- Model revision: `42ed227ee7df40d41602854ae760620d6eb651fe`
- Diffusers revision: `1b98ae1060b765f2efe22540f52691b8c00a83f1`

The model snapshot contains 32 safetensors files, 3,486 tensor headers, and 134.126 GiB of weights.

## Operator coverage

| Test | Result |
| --- | --- |
| H3 packed multimodal transformer | Pass |
| 40k-token, 56-head, 128-dim Flash Attention | Pass, 0.495 s |
| 124-frame video VAE decode | Pass |
| Diffusers `ComponentsManager` automatic CPU offload | Pass |
| Video VAE FP16 autocast on MLU | Pass with project compatibility hook |

The Flash Attention microbenchmark used about 6.68 GiB peak allocated memory, including its Q/K/V inputs and output.

## End-to-end T2VA

All runs generated 124 frames at 24fps and 32kHz stereo audio.

| Canvas | Scheduler points | Model evaluations | End-to-end | Peak MLU |
| --- | ---: | ---: | ---: | ---: |
| 32x32 | 2 | 1 | 103.9 s | 62.16 GiB |
| 960x544 | 2 | 1 | 126.0 s | 65.65 GiB |
| 960x544 | 5 | 4 | 150.9 s | 65.66 GiB |
| 1344x768 | 5 | 4 | 237.6 s | 69.26 GiB |
| 1344x768 | 50 | 49 | 1,553.4 s | 69.27 GiB |
| 1344x768 + Turbo v4 EMA | 9 | 8 | 363.9 s | 71.07 GiB |
| 960x544, 345 frames + Turbo v4 EMA | 9 | 8 | 557.2 s | 74.50 GiB |

At 1344x768 and 50 scheduler points, the first denoising iteration took 75.2 seconds including the transformer
transfer. The remaining iterations stabilized at approximately 29.27 seconds each.

The final 1344x768 file contains:

- H.264 video at 24fps
- AAC stereo audio at 32kHz
- 5.175 seconds duration
- Non-silent audio, measured at -37.7 dB mean and -24.0 dB maximum

The 8-evaluation Turbo run was 4.27x faster end to end than the 49-evaluation base run. Three sampled frames retained
the requested subject, environment, and continuous walking motion. Its generated audio was also non-silent. Turbo is
not bit-identical to the base trajectory; the v4 model card recommends 6 to 8 evaluations and notes that audio and
very fast motion remain the main quality caveats.

The longest valid native VAE grid below the 15-second ceiling is 345 frames, or 14.375 seconds at 24fps. The
960x544 long-video run produced 345 distinct frame hashes and non-silent stereo audio. It nearly filled the card, so
1344x768 at this duration is not recommended with the full BF16 transformer on an 80 GiB device.

## Rejected optimizations

Destructive QKV fusion was numerically valid but slower on this MLU stack. At 960x544 and 5 scheduler points,
end-to-end inference increased from 150.9 seconds to 156.7 seconds, plus 53 seconds of one-time CPU-side fusion.
It is therefore not exposed by the runtime.

The AdaLN-pruned BF16 checkpoint was not adopted. It uses ComfyUI's curve-table architecture rather than the current
Diffusers model layout, and public reports show unresolved black-output failures on some plain-BF16 ComfyUI paths.
Spectrum-style step forecasting was also not adopted because it changes the trajectory and documents possible motion,
detail, and audio differences. Turbo LoRA delivered the larger measured speedup through the already-supported
Diffusers LoRA path.

## FL2VA

A first-frame-conditioned 32x32, 124-frame, 2-point request completed in 110.4 seconds with 62.24 GiB peak MLU
memory. The current MLU runtime falls back to CPU for `reflection_pad3d` during keyframe VAE encoding. This does not
affect correctness, but may add overhead for full-resolution keyframes.

The same FL2VA path also passed with the Turbo LoRA at 4 model evaluations.

## Compatibility changes

- Build row-to-timestep mappings on CPU before copying them to MLU. This avoids mixed-device `index_put`.
- Enable FP16 autocast for the H3 video VAE decoder on MLU, matching the upstream CUDA decode recipe.
- Keep the compatibility changes in `minimax_h3_mlu/compat.py`; no installed package files are modified.

## Known environment warnings

The base image reports vendor package-version metadata warnings for PyTorch/MLU-OPS. The tested eager inference,
Flash Attention, CPU offload, video decode, and audio decode paths all completed despite those warnings. Use a
vendor-matched base image before treating this as a production deployment.
