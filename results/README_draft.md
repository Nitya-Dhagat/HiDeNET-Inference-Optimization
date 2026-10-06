# HiDeNET Inference Optimization

PyTorch re-implementation of HiDeNET (DeBERTa-v3-base + Conv1D multilevel classifier, IEEE SCEECS 2026) retrained on the same stratified split, then benchmarked under quantization, compilation and ONNX Runtime. The original TensorFlow checkpoint was not saved, so this repo does **not** optimize the published weights; metrics are within the table below of the paper's.

TODO: one-sentence headline result from the table.

## Setup
- Hardware: CPU `Intel(R) Xeon(R) CPU @ 2.20GHz` (12 logical cores, 4 threads used), GPU `NVIDIA A100-SXM4-40GB`
- Versions: torch 2.11.0+cu130, transformers 5.18.0, onnxruntime 1.30.0
- Model tag: `deberta-v3-base_frozen` (backbone frozen: True), max length 256, fixed padding
- Protocol: same inputs for every variant; 10 warmup runs; timed runs: GPU 100, CPU batch 1 = 100, CPU batch 16 = 20; p50/p95; batch sizes 1 (latency) and 16 (throughput)
- Accuracy / agreement measured on a fixed random subset of 500 held-out examples (same for all variants); agreement = % identical argmax vs FP32 CPU baseline

## Re-implementation vs paper (full test split, n=1853)
| Level | Paper Hamming loss | Paper accuracy (1 - Hamming) | This re-implementation accuracy | This re-implementation macro-F1 |
|---|---|---|---|---|
| 1.0 | 0.2078 | 0.7922 | 0.8046 | 0.763 |
| 2.0 | 0.116 | 0.884 | 0.8834 | 0.41 |
| 3.0 | 0.1522 | 0.8478 | 0.8332 | 0.643 |

## Results
| Variant | Size (MB) | bs1 p50 / p95 (ms) | bs16 p50 / p95 (ms) | Throughput bs16 (samples/s) | Acc L1 / L2 / L3 | Macro-F1 L1 / L2 / L3 | Agreement vs FP32 CPU (%) L1 / L2 / L3 | Notes |
|---|---|---|---|---|---|---|---|---|
| FP32 baseline (CPU) | 703.4 | 241.8 / 316.4 | 4157.9 / 4241.6 | 3.85 | 0.780 / 0.880 / 0.864 | 0.747 / 0.407 / 0.693 | 100.0 / 100.0 / 100.0 | reference; agreement/maxdiff columns are measured against this row |
| FP32 baseline (GPU) | 703.4 | 22.3 / 23.9 | 69.4 / 69.5 | 230.54 | 0.780 / 0.880 / 0.864 | 0.747 / 0.407 / 0.693 | 100.0 / 100.0 / 100.0 | NVIDIA A100-SXM4-40GB |
| Dynamic INT8 (CPU) | 459.7 | 158.5 / 165.0 | 3080.7 / 3111.4 | 5.19 | 0.396 / 0.872 / 0.660 | 0.307 / 0.310 / 0.217 | 42.0 / 98.6 / 69.6 | torch.ao quantize_dynamic, 76 nn.Linear -> int8; Conv1d + embeddings stay FP32 |
| torch.compile (CPU) | 703.4 | 252.9 / 260.6 | 3160.0 / 3198.1 | 5.06 | 0.780 / 0.880 / 0.864 | 0.747 / 0.407 / 0.693 | 100.0 / 100.0 / 100.0 | torch.compile default (inductor); compile time excluded from latency |
| torch.compile (GPU) | 703.4 | 8.8 / 10.1 | 57.4 / 57.6 | 278.8 | 0.780 / 0.880 / 0.864 | 0.747 / 0.407 / 0.693 | 100.0 / 100.0 / 100.0 | torch.compile default (inductor); compile time excluded from latency |
| fp16 autocast (GPU) | 703.4 | 33.9 / 36.5 | 34.2 / 34.8 | 468.08 | 0.780 / 0.880 / 0.864 | 0.747 / 0.407 / 0.693 | 100.0 / 100.0 / 100.0 | torch.autocast fp16; weights stay FP32 on disk, so size is unchanged |
| ONNX Runtime FP32 (CPU) | 704.2 | 199.3 / 213.6 | 3404.2 / 3444.6 | 4.7 | 0.780 / 0.880 / 0.864 | 0.747 / 0.407 / 0.693 | 100.0 / 100.0 / 100.0 | opset 17, ORT_ENABLE_ALL, 4 intra-op threads; dynamic-shape check (3x128) max prob diff 1.85e-06 |
| ONNX Runtime INT8 (CPU) | 232.6 | 139.5 / 144.4 | 2390.2 / 2403.9 | 6.69 | 0.264 / 0.872 / 0.652 | 0.199 / 0.310 / 0.212 | 28.8 / 98.6 / 68.2 | ORT dynamic quantization, QInt8 weights; dynamic-shape check (3x128) max prob diff 7.13e-01 |

![latency and size](results/deberta-v3-base_frozen/latency_size.png)

## Profiling findings

**profile_cpu_fp32_bs16**

| op | calls | self_cpu_us | pct_of_host_time |
|---|---|---|---|
| aten::addmm | 500 | 7203123.699999983 | 41.7 |
| aten::bmm | 240 | 3466390.551999984 | 20.1 |
| aten::gather | 120 | 1436304.5739999902 | 8.3 |
| aten::copy_ | 1565 | 1349626.573000025 | 7.8 |
| aten::add | 485 | 1024173.1299999708 | 5.9 |
| aten::div | 190 | 843515.2480000083 | 4.9 |
| aten::_softmax | 60 | 630619.091999998 | 3.7 |
| aten::gelu | 60 | 529544.3359999965 | 3.1 |
| aten::add_ | 60 | 261926.41100000337 | 1.5 |
| aten::masked_fill_ | 60 | 222628.69999998764 | 1.3 |
| aten::native_layer_norm | 130 | 96240.2790000199 | 0.6 |
| aten::mkldnn_convolution | 5 | 31672.291999998502 | 0.2 |

**profile_cpu_fp32_bs1**

| op | calls | self_cpu_us | pct_of_host_time |
|---|---|---|---|
| aten::addmm | 500 | 733564.1550000014 | 60.7 |
| aten::bmm | 240 | 134423.0229999984 | 11.1 |
| aten::copy_ | 1565 | 59864.9770000022 | 5.0 |
| aten::gather | 120 | 58812.881999999656 | 4.9 |
| aten::add | 485 | 34512.8070000012 | 2.9 |
| aten::add_ | 60 | 16412.925000000283 | 1.4 |
| aten::masked_fill_ | 60 | 15629.788999999844 | 1.3 |
| aten::div | 190 | 15563.335000000525 | 1.3 |
| aten::_softmax | 60 | 15189.576999999988 | 1.3 |
| aten::gelu | 60 | 12302.962999999849 | 1.0 |
| aten::linear | 500 | 10992.223999999953 | 0.9 |
| aten::native_layer_norm | 130 | 9126.082000000808 | 0.8 |

Backbone share of CPU runtime: {"bs1": {"total_ms": 236.55, "backbone_ms": 238.01, "backbone_share_pct": 100.6}, "bs16": {"total_ms": 2583.03, "backbone_ms": 2580.94, "backbone_share_pct": 99.9}}

TODO: what dominated and whether it matched expectations.

## What worked / what didn't
Failed or skipped variants:
- none

TODO: honest trade-offs (e.g. embeddings are ~half the parameters and are not touched by dynamic quantization; Conv1D is not quantized; small subset makes minority-class F1 noisy).

## Reproduce
Run `hidenet_inference_optimization.ipynb` top to bottom on a Colab T4 (Parts A to D). Dataset rows are not included.
