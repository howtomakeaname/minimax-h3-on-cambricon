# Per-step optimization results

## M0 baseline

Workload:

- T2VA
- 960x544
- 124 frames
- 5 scheduler points / 4 model evaluations
- BF16 base model
- MLU Flash Attention
- Seed 271828

Two independent runs produced identical raw video and audio tensors:

- Video SHA256: `d326880677f73dc6df7579802c747dec4824a9f41c52404f83115ec07560ca98`
- Audio SHA256: `281bec7546f9f101fff984de386b88d9e8a6f88b96a8f0f37ab89dd0be8a158d`
- Video and audio `max_abs_diff`: `0`

Across six steady steps, mean latency was 9.7941 seconds with a 0.0161% coefficient of variation.
Peak allocated MLU memory was 65.681 GiB in both runs.

## MLU profiler

The second model evaluation was profiled with CPU and MLU activities enabled.

| Operator group | Self MLU time |
| --- | ---: |
| Flash Attention | 57.86% |
| Matrix multiplication | 34.55% |
| Elementwise multiply | 2.36% |
| Elementwise add | 1.27% |
| Concatenation | 1.11% |
| SiLU | 1.02% |
| RMSNorm | 0.68% |
| `index_select` | 0.44% |
| Device idle | 0.02% |

Flash Attention and matrix multiplication account for 92.41% of steady-step MLU time. The trace contains 4,249
MLU computation kernels. Each step performs 304 `index_select` calls, but their aggregate MLU time is only 42.7 ms.

The same profile was repeated at 960x544 with 243 frames:

| Operator group | Self MLU time |
| --- | ---: |
| Flash Attention | 72.77% |
| Matrix multiplication | 22.42% |
| Elementwise multiply | 1.50% |
| Elementwise add | 0.81% |
| `index_select` | 0.27% |
| Device idle | 0.01% |

Attention and matrix multiplication account for 95.19% of the longer sequence's steady-step MLU time. The
non-Attention/GEMM optimization ceiling decreases as sequence length grows.

Raw profiler traces and generated tensors are intentionally excluded from the repository because they are large
runtime artifacts. The tables below preserve the measurements needed to evaluate each experiment.

## Rejected experiments

### Request-local invariant cache

Caching RoPE, `context_embedder`, and `token_refiner` outputs preserved exact video and audio tensors. All three
caches recorded three hits and one miss in a four-evaluation run. Steady-step latency improved by only 0.03%, below
the 3% acceptance threshold, so the implementation was removed.

### Local AdaLN TorchInductor compilation

A real-shape microbenchmark used sequence length 19,450 and hidden size 5,376. The compiled affine/gather fragment
improved from 1.9982 ms to 1.9551 ms, but changed BF16 outputs:

- First output max absolute difference: 0.125
- Second output max absolute difference: 0.0625

The local fragment is less than 0.1% of the full step after applying its measured speedup, and it fails strict
equality. It is not being integrated.

### Default attention backend

The default attention backend produced a 10.0223-second steady-step median, 2.33% slower than the Flash baseline.
It also changed raw video and audio tensors, so MLU Flash Attention remains the default.

### Flash Attention variants

Real-shape attention microbenchmarks compared fixed-length Flash Attention with varlen, native Flash, deterministic,
and implicit-scale variants:

- At 19,450 tokens, fixed Flash took 113.345 ms. Varlen and native Flash took about 118.28 ms and differed by
  `0.000244`.
- At 37,701 tokens, fixed Flash took 426.111 ms. Varlen and native Flash took about 444.22 ms with the same maximum
  difference.
- Deterministic mode was bitwise equal but did not improve latency.
- Omitting the explicit softmax scale was bitwise equal and had no measurable speed benefit.

The existing fixed-length Flash Attention call remains the fastest tested backend.

### GEMM layout and grouping

- `torch_mlu::grouped_gemm` produced bitwise-identical Q/K/V outputs but was 0.73% slower than three independent
  GEMMs at shape `[19450, 5376] x [5376, 7168]`.
- Storing the weight as contiguous `[in, out]` rather than using the normal transposed linear-weight view was
  bitwise equal and only 0.06% faster.
- Enabling the MLU TF32 matmul flag for BF16 input was bitwise equal and changed latency by only 0.015%.
- Enabling `torch.backends.cnnl.benchmark` emitted a runtime warning that the option is unavailable on MLU; the
  observed 0.06% timing difference is measurement noise.

None meets the project-level performance threshold.

### RoPE variants

An exact split implementation avoided one intermediate rotation concatenation but was 13.9% slower. A locally
compiled version was 1.78x faster for the RoPE fragment, but its maximum absolute difference was `0.125`. Even if
accepted, the local saving would be roughly 1% of a complete step; it fails the L0 gate.

## Decision

Single-card project-side optimizations outside Attention and GEMM have a measured upper bound below 8% and the
low-risk cache path did not meet the acceptance threshold. Further work should focus on:

1. A vendor-matched runtime with newer Attention and GEMM kernels.
2. Multi-card Context Parallel when more than one MLU is available.
3. A targeted MLU kernel only if profiling a longer workload changes the operator distribution materially.

The tested package source did not expose a newer compatible MLU-OPS candidate. Runtime A/B therefore requires a
separate, vendor-matched environment rather than an in-place package upgrade.

## Lossless component-transfer optimization

The original automatic offload hook copies a resident component back to CPU before loading the next one. In inference
the parameters are read-only, so the optimized path preserves their original CPU storage, copies it to MLU when
needed, and restores the CPU storage reference when evicting the MLU copy. This removes the redundant device-to-host
copy without changing any model operation. Tests used the same 960x544, 124-frame, 5-point workload as M0.

| Mode | Pipeline inference | First model evaluation | Steady-step P50 | Peak MLU | Raw outputs |
| --- | ---: | ---: | ---: | ---: | --- |
| Diffusers copy-back | 149.479 s | 56.697 s | 9.7193 s | 65.650 GiB | Baseline |
| CPU master | 79.894 s | 21.649 s | 9.7306 s | 65.650 GiB | Bitwise equal |
| CPU master + local NUMA | 71.778 s | 20.297 s | 9.7171 s | 65.650 GiB | Bitwise equal |
| Pinned CPU master + local NUMA | 53.404 s | 11.597 s | 9.7177 s | 65.650 GiB | Bitwise equal |

CPU master is the default because it gives the best one-shot wall time. Pinned master reduced the transfer portion
further, but pinning the selected weights increased component setup from about 1.7 seconds to 39.2 seconds; it is most
useful in a persistent process that amortizes setup over multiple requests. A 20-point CPU-master run completed with a
9.7166-second steady-step median, confirming that the optimization does not introduce step-wise transfer churn.

The trade-off is host memory: CPU master keeps the original checkpoint storage while the selected component is also
resident on MLU. It must only be installed after LoRA or other parameter mutations. If a parameter version changes
after the snapshot, the implementation falls back to the original copy-back path.

Step diagnostics now use MLU events for ordinary timing. They preserve the queued execution order without inserting
the four full-device synchronization barriers per step used by the original diagnostics; only the explicitly selected
profiler step synchronizes before exporting its trace.
