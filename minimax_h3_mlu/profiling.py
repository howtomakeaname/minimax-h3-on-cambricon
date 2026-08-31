from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class StepTiming:
    index: int
    denoiser_seconds: float = 0.0
    scheduler_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.denoiser_seconds + self.scheduler_seconds


class StepDiagnostics:
    def __init__(self, profile_dir: Path | None, profile_step: int):
        self.profile_dir = profile_dir
        self.profile_step = profile_step
        self.timings: dict[int, StepTiming] = {}
        self.events: dict[tuple[int, str], tuple[torch.mlu.Event, torch.mlu.Event]] = {}
        if self.profile_dir is not None:
            self.profile_dir.mkdir(parents=True, exist_ok=True)

    def timing(self, index: int) -> StepTiming:
        return self.timings.setdefault(index, StepTiming(index=index))

    def profile_denoiser(self, fn, device: torch.device, index: int):
        if self.profile_dir is None or index != self.profile_step:
            result = self.record_events(fn, device, index, "denoiser")
        else:
            with torch.mlu.device(device):
                started = torch.mlu.Event(enable_timing=True)
                finished = torch.mlu.Event(enable_timing=True)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.MLU,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as profiler:
                    started.record()
                    result = fn()
                    finished.record()
                    finished.synchronize()
            self.events[index, "denoiser"] = (started, finished)
            profiler.export_chrome_trace(str(self.profile_dir / f"denoiser-step-{index:03d}.json"))
            table = profiler.key_averages().table(sort_by="self_device_time_total", row_limit=100)
            (self.profile_dir / f"denoiser-step-{index:03d}.txt").write_text(table, encoding="utf-8")

        return result

    def time_scheduler(self, fn, device: torch.device, index: int):
        return self.record_events(fn, device, index, "scheduler")

    def record_events(self, fn, device: torch.device, index: int, section: str):
        with torch.mlu.device(device):
            started = torch.mlu.Event(enable_timing=True)
            finished = torch.mlu.Event(enable_timing=True)
            started.record()
            result = fn()
            finished.record()
        self.events[index, section] = (started, finished)
        return result

    def finalize_timings(self) -> None:
        if self.events:
            next(reversed(self.events.values()))[1].synchronize()
        for (index, section), (started, finished) in self.events.items():
            seconds = started.elapsed_time(finished) / 1000.0
            setattr(self.timing(index), f"{section}_seconds", seconds)

    def summary(self) -> dict:
        self.finalize_timings()
        rows = []
        for timing in sorted(self.timings.values(), key=lambda item: item.index):
            row = asdict(timing)
            row["total_seconds"] = timing.total_seconds
            rows.append(row)

        steady = [row["total_seconds"] for row in rows[1:]]
        return {
            "steps": rows,
            "first_step_seconds": rows[0]["total_seconds"] if rows else None,
            "steady_step_p50_seconds": statistics.median(steady) if steady else None,
            "steady_step_min_seconds": min(steady) if steady else None,
            "steady_step_max_seconds": max(steady) if steady else None,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8")


def install_step_diagnostics(profile_dir: Path | None = None, profile_step: int = 1) -> StepDiagnostics:
    from diffusers.modular_pipelines.minimax_h3.denoise import (
        MiniMaxH3LoopDenoiser,
        MiniMaxH3LoopSchedulerStep,
    )

    diagnostics = StepDiagnostics(profile_dir=profile_dir, profile_step=profile_step)
    original_denoiser = MiniMaxH3LoopDenoiser.__call__
    original_scheduler = MiniMaxH3LoopSchedulerStep.__call__

    def timed_denoiser(self, components, block_state, i, t):
        device = components._execution_device
        return diagnostics.profile_denoiser(
            lambda: original_denoiser(self, components, block_state, i, t),
            device,
            i,
        )

    def timed_scheduler(self, components, block_state, i, t):
        device = components._execution_device
        return diagnostics.time_scheduler(
            lambda: original_scheduler(self, components, block_state, i, t),
            device,
            i,
        )

    MiniMaxH3LoopDenoiser.__call__ = timed_denoiser
    MiniMaxH3LoopSchedulerStep.__call__ = timed_scheduler
    return diagnostics
