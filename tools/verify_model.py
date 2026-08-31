from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "MiniMax-H3-diffusers"

INDEX_FILES = (
    "text_encoder/model.safetensors.index.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
)
SINGLE_FILES = ("audio_vae/diffusion_pytorch_model.safetensors",)
REQUIRED_FILES = (
    "model_index.json",
    "modular_model_index.json",
    "processor/tokenizer.json",
    "tokenizer/tokenizer.json",
    "scheduler/scheduler_config.json",
    "audio_scheduler/scheduler_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a local MiniMax-H3 Diffusers FL2VA snapshot.")
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    model_dir = parse_args().model.resolve()
    missing = [path for path in REQUIRED_FILES if not (model_dir / path).is_file()]
    shard_paths: set[Path] = set()
    tensor_count = 0

    for index_name in INDEX_FILES:
        index_path = model_dir / index_name
        if not index_path.is_file():
            missing.append(index_name)
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        tensor_count += len(index["weight_map"])
        shard_paths.update(index_path.parent / name for name in set(index["weight_map"].values()))

    shard_paths.update(model_dir / name for name in SINGLE_FILES)
    missing.extend(str(path.relative_to(model_dir)) for path in shard_paths if not path.is_file())

    partials = list((model_dir / ".cache").rglob("*.incomplete"))
    if missing or partials:
        if missing:
            print("Missing files:")
            print("\n".join(f"  {name}" for name in sorted(set(missing))))
        if partials:
            print("Incomplete downloads:")
            print("\n".join(f"  {path}" for path in sorted(partials)))
        raise SystemExit(1)

    header_keys = 0
    total_bytes = 0
    for shard_path in sorted(shard_paths):
        with safe_open(shard_path, framework="numpy", device="cpu") as handle:
            header_keys += len(handle.keys())
        total_bytes += shard_path.stat().st_size

    print(
        json.dumps(
            {
                "model": str(model_dir),
                "weight_files": len(shard_paths),
                "indexed_tensors": tensor_count,
                "header_tensors": header_keys,
                "weight_gib": round(total_bytes / 1024**3, 3),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
