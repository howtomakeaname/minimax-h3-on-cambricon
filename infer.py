from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_mlu
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video
from PIL import Image

from minimax_h3_mlu.compat import apply_mlu_patches, set_attention_backend
from minimax_h3_mlu.offload import DEFAULT_PINNED_COMPONENTS, OFFLOAD_MODES, enable_cpu_master_offload


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "MiniMax-H3-diffusers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniMax-H3 on one Cambricon MLU with BF16 CPU offload.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--workflow", choices=("t2va", "fl2va"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--last-image", type=Path)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mlu:0")
    parser.add_argument("--offload-margin", default="12GB")
    parser.add_argument(
        "--offload-mode",
        choices=OFFLOAD_MODES,
        default="cpu-master",
        help="Weight offload policy; pinned-master favors a persistent single-task service.",
    )
    parser.add_argument(
        "--pinned-components",
        default=",".join(DEFAULT_PINNED_COMPONENTS),
        help="Comma-separated components pinned by --offload-mode pinned-master.",
    )
    parser.add_argument("--attention", choices=("flash", "default"), default="flash")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/minimax-h3.mp4"))
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-step", type=int, default=1)
    parser.add_argument("--raw-output-dir", type=Path)
    parser.add_argument("--compare-raw-dir", type=Path)
    parser.add_argument("--allow-remote", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str:
    if not args.model.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model}. Run ./download.sh first.")
    if args.height % 32 or args.width % 32:
        raise SystemExit("--height and --width must both be multiples of 32")
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2")
    aligned_num_frames = args.num_frames
    while aligned_num_frames % 17 != 5:
        aligned_num_frames += 1
    if aligned_num_frames < 120 or aligned_num_frames > 360:
        raise SystemExit(
            "--num-frames must align to a 17n+5 value covering 5 to 15 seconds at 24fps; "
            f"{args.num_frames} would align to {aligned_num_frames}"
        )
    for image_path in (args.image, args.last_image):
        if image_path is not None and not image_path.is_file():
            raise SystemExit(f"Input image does not exist: {image_path}")
    if args.lora is not None and not args.lora.is_file():
        raise SystemExit(f"LoRA file does not exist: {args.lora}")
    if args.lora_scale <= 0:
        raise SystemExit("--lora-scale must be positive")
    if args.profile_step < 0:
        raise SystemExit("--profile-step must not be negative")
    if args.profile_dir is not None and args.profile_step >= args.steps - 1:
        raise SystemExit("--profile-step must select one of the model evaluations (0 to --steps - 2)")
    if args.offload_mode == "pinned-master" and not args.pinned_components.strip():
        raise SystemExit("--pinned-components must not be empty with --offload-mode pinned-master")
    if args.compare_raw_dir is not None:
        for name in ("video.npy", "audio.npy"):
            if not (args.compare_raw_dir / name).is_file():
                raise SystemExit(f"Baseline raw output is missing: {args.compare_raw_dir / name}")

    workflow = args.workflow or ("fl2va" if args.image or args.last_image else "t2va")
    if workflow == "t2va" and (args.image or args.last_image):
        raise SystemExit("Keyframes require --workflow fl2va")
    return workflow


def load_pipeline(args: argparse.Namespace, workflow: str):
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(
        str(args.model),
        workflow=workflow,
        components_manager=manager,
        local_files_only=not args.allow_remote,
    )
    pipe.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=str(args.model),
        local_files_only=not args.allow_remote,
        low_cpu_mem_usage=True,
    )

    missing = [
        name
        for name in pipe.pretrained_component_names
        if hasattr(pipe, name) and getattr(pipe, name) is None
    ]
    if missing:
        raise RuntimeError(f"Failed to load required pipeline components: {', '.join(sorted(missing))}")

    if args.lora is not None:
        pipe.load_lora_weights(
            str(args.lora),
            adapter_name="acceleration",
            low_cpu_mem_usage=True,
        )
        pipe.set_adapters("acceleration", adapter_weights=args.lora_scale)

    set_attention_backend(pipe, args.attention)
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None and hasattr(text_encoder, "set_attn_implementation"):
        text_encoder.set_attn_implementation("sdpa")

    manager.enable_auto_cpu_offload(
        device=args.device,
        memory_reserve_margin=args.offload_margin,
    )
    pinned_components = tuple(name.strip() for name in args.pinned_components.split(",") if name.strip())
    offload_stats = enable_cpu_master_offload(
        manager,
        mode=args.offload_mode,
        pinned_components=pinned_components,
    )
    return pipe, manager, offload_stats


def open_image(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    byte_view = memoryview(contiguous).cast("B")
    digest = hashlib.sha256()
    chunk_bytes = 64 * 1024 * 1024
    for offset in range(0, len(byte_view), chunk_bytes):
        digest.update(byte_view[offset : offset + chunk_bytes])
    return digest.hexdigest()


def to_numpy_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def array_metadata(array: np.ndarray) -> dict:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": array_sha256(array),
    }


def compare_array(current: np.ndarray, baseline_path: Path) -> dict:
    baseline = np.load(baseline_path, mmap_mode="r", allow_pickle=False)
    if current.shape != baseline.shape or current.dtype != baseline.dtype:
        return {
            "equal": False,
            "current_shape": list(current.shape),
            "baseline_shape": list(baseline.shape),
            "current_dtype": str(current.dtype),
            "baseline_dtype": str(baseline.dtype),
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    equal = np.array_equal(current, baseline)
    if equal:
        return {"equal": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0}

    max_abs_diff = 0.0
    total_abs_diff = 0.0
    total_values = 0
    chunk_length = max(1, min(current.shape[0], 8)) if current.ndim else 1
    current_view = current.reshape((1,)) if current.ndim == 0 else current
    baseline_view = baseline.reshape((1,)) if baseline.ndim == 0 else baseline
    for start in range(0, current_view.shape[0], chunk_length):
        current_chunk = np.asarray(current_view[start : start + chunk_length], dtype=np.float64)
        baseline_chunk = np.asarray(baseline_view[start : start + chunk_length], dtype=np.float64)
        difference = np.abs(current_chunk - baseline_chunk)
        max_abs_diff = max(max_abs_diff, float(difference.max(initial=0.0)))
        total_abs_diff += float(difference.sum())
        total_values += difference.size
    return {
        "equal": False,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": total_abs_diff / total_values,
    }


def save_raw_outputs(output_dir: Path, video: np.ndarray, audio: np.ndarray, sampling_rate: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "video.npy", video, allow_pickle=False)
    np.save(output_dir / "audio.npy", audio, allow_pickle=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "video": array_metadata(video),
                "audio": array_metadata(audio),
                "audio_sample_rate": sampling_rate,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    workflow = validate_args(args)
    if not torch.mlu.is_available():
        raise SystemExit("No MLU device is available")

    device = torch.device(args.device)
    torch.mlu.set_device(device)
    apply_mlu_patches()

    step_diagnostics = None
    if args.metrics_json is not None or args.profile_dir is not None:
        from minimax_h3_mlu.profiling import install_step_diagnostics

        step_diagnostics = install_step_diagnostics(
            profile_dir=args.profile_dir,
            profile_step=args.profile_step,
        )

    print(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "workflow": workflow,
                "device": str(device),
                "dtype": "bfloat16",
                "attention": args.attention,
                "offload_margin": args.offload_margin,
                "offload_mode": args.offload_mode,
                "size": [args.width, args.height],
                "num_frames": args.num_frames,
                "steps": args.steps,
                "seed": args.seed,
                "lora": str(args.lora.resolve()) if args.lora else None,
                "lora_scale": args.lora_scale if args.lora else None,
                "profile_dir": str(args.profile_dir.resolve()) if args.profile_dir else None,
                "profile_step": args.profile_step if args.profile_dir else None,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    load_started = time.perf_counter()
    pipe, manager, offload_stats = load_pipeline(args, workflow)
    load_elapsed = time.perf_counter() - load_started
    managed = len(manager.components)
    print(
        f"Loaded {managed} managed components in {load_elapsed:.1f}s; "
        f"CPU master={offload_stats.bytes / 1024**3:.2f} GiB, "
        f"pinned={offload_stats.pinned_bytes / 1024**3:.2f} GiB",
        flush=True,
    )

    request = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "generator": torch.Generator("cpu").manual_seed(args.seed),
        "output_type": "np",
        "output": ["videos", "audio", "sampling_rate"],
    }
    if workflow == "fl2va":
        request["image"] = open_image(args.image)
        request["last_image"] = open_image(args.last_image)

    torch.mlu.reset_peak_memory_stats(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        results = pipe(**request)
    torch.mlu.synchronize(device)
    inference_elapsed = time.perf_counter() - inference_started
    peak_gib = torch.mlu.max_memory_allocated(device) / 1024**3

    videos = results.get("videos")
    audio = results.get("audio")
    sampling_rate = results.get("sampling_rate")
    if videos is None or audio is None or sampling_rate is None:
        raise RuntimeError("Pipeline did not return video, audio, and sampling rate")

    video_array = to_numpy_array(videos[0])
    audio_array = to_numpy_array(audio[0])
    output_diagnostics = {
        "video": array_metadata(video_array),
        "audio": array_metadata(audio_array),
    }
    if args.compare_raw_dir is not None:
        output_diagnostics["comparison"] = {
            "video": compare_array(video_array, args.compare_raw_dir / "video.npy"),
            "audio": compare_array(audio_array, args.compare_raw_dir / "audio.npy"),
        }
    if args.raw_output_dir is not None:
        save_raw_outputs(args.raw_output_dir, video_array, audio_array, sampling_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        videos[0],
        fps=24,
        output_path=str(args.output),
        audio=audio[0],
        audio_sample_rate=sampling_rate,
    )
    metrics = {
        "output": str(args.output.resolve()),
        "inference_seconds": round(inference_elapsed, 3),
        "peak_mlu_gib": round(peak_gib, 3),
        "offload_mode": args.offload_mode,
        "cpu_master_gib": round(offload_stats.bytes / 1024**3, 3),
        "pinned_master_gib": round(offload_stats.pinned_bytes / 1024**3, 3),
        "frames": len(videos[0]),
        "fps": 24,
        "audio_sample_rate": sampling_rate,
        "output_diagnostics": output_diagnostics,
    }
    if step_diagnostics is not None:
        metrics["step_timing"] = step_diagnostics.summary()
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(metrics, ensure_ascii=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
