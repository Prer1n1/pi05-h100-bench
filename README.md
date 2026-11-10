# pi0.5 H100 denoise-loop benchmark

Cross-GPU reference numbers for the pi0.5 denoise loop on **NVIDIA H100 SXM 80GB (HBM3)**, contributed to [huggingface/lerobot#5148](https://github.com/huggingface/lerobot/issues/5148).

The thread already had H200 (@arun-elr) and A100 (@molbap) numbers. This repo fills the H100 gap and validates both of arun-elr's optimization PRs on a third GPU generation.

| config                               | per-step p50 | per-chunk p50 | per-chunk p95 |
|--------------------------------------|-------------:|--------------:|--------------:|
| fp32 shipped (main)                  |     27.8 ms  |     277.9 ms  |     287.7 ms  |
| bf16 autocast (main)                 |     22.8 ms  |     227.5 ms  |     238.9 ms  |
| bf16 + #5152 + #5153 eager           |     18.8 ms  |     188.3 ms  |     211.6 ms  |
| bf16 + #5152 + #5153 cuda-graph      |      9.1 ms  |      90.5 ms  |      91.5 ms  |

**Both PRs apply cleanly on H100** — `#5152` alone gives **1.21×** over main bf16, stacking `#5153` on top gives **2.51×**. The cuda-graph row's `p95/p50 = 1.01` is the deterministic-replay signature that proves the graph is actually capturing and replaying.

Posted comment on the thread: [issuecomment-6780847297](https://github.com/huggingface/lerobot/issues/5148#issuecomment-6780847297)

## What pi0.5 is and why this matters

[π₀.₅](https://huggingface.co/lerobot/pi05_base) is a Vision-Language-Action (VLA) model from Physical Intelligence, co-trained on robot demonstrations and multimodal data for long-horizon manipulation. Inference uses **flow matching**: the policy iteratively denoises a noisy action chunk over 10 Euler steps to produce 50 actions.

For real-time robotics each action chunk's inference cost is user-facing latency — the robot is waiting. arun-elr's [issue #5148](https://github.com/huggingface/lerobot/issues/5148) identified the denoise loop as 86% of a bf16 chunk and traced the cost to a per-step clone of a 968-token prefix KV cache. Two stacked PRs fix it:

- **[#5152](https://github.com/huggingface/lerobot/pull/5152)** — share one read-only prefix cache across the 10 denoise steps instead of cloning per step.
- **[#5153](https://github.com/huggingface/lerobot/pull/5153)** — opt-in `denoise_cuda_graph` that captures the denoise step into a CUDA graph for launch-overhead-free replay.

## Launch-bound observation

After the CUDA graph kicks in, H100 lands at 90.5 ms/chunk — within 7% of @molbap's A100 cuda-graph number (85 ms). On a compute-bound workload, H100 would be roughly **3× faster** than A100 (bf16 tensor cores: ~989 vs ~312 TFLOPS). That they're neck-and-neck suggests the remaining per-chunk cost after graph replay is dominated by the one-time prefix encoding through PaliGemma, not the denoise loop itself.

This also squares with the H200→H100 bf16 gap being larger than pure compute predicts (151 → 227 ms, roughly matching the HBM3e→HBM3 bandwidth ratio for that prefix pass).

## Methodology

- **Hardware:** 1× NVIDIA H100 SXM 80GB (HBM3), driver 595.91.07, CUDA runtime 12.x
- **Pod:** RunPod community cloud, AP-IN-1
- **Image:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04` (uv venv Python 3.12)
- **Model:** `lerobot/pi05_base`
- **Input:** batch size 1, three 3×224×224 RGB views, 32-dim proprio state, 200-token PaliGemma prompt
- **Timing:** `torch.cuda.Event` + `torch.cuda.synchronize()`, 20 warmup + 200 timed iterations per config
- **Measurement scope:** full `policy.predict_action_chunk(batch)` call — preprocessing + 10 denoise steps

Per-chunk statistics are p50 / p95 / mean / min across 200 timed iterations. Per-step derivations divide by `num_inference_steps = 10`.

## Caveats

1. **bf16 rows use `torch.autocast(bfloat16)`**, model weights kept in fp32, not a hard `.to(bfloat16)` cast on the policy. A direct cast hit a dtype mismatch between the flow-matching integrator's fp32 noise init and the bf16 Linear weights in `action_in_proj`; autocast was the clean workaround.
2. **Patched venv resolved to `torch 2.11.0+cu128`, main venv to `torch 2.15.0+cu132`** (branch-specific pin differences in lerobot's `pyproject.toml`). Kernel-level impact for this workload should be minor but worth stating.
3. Inputs are **dummy tensors** (`torch.randn` for images, filler tokens), not a real LIBERO rollout observation. For a latency benchmark the GPU work is the same; for a correctness comparison (e.g. action deltas vs main) use arun-elr's LIBERO reproduction in the thread.

## Reproducing

### 1. Rent a GPU

Any H100 SXM 80GB works. These results were collected on RunPod. Budget: ~$4–5 of pod time if nothing goes wrong, 2× that if dependency resolution or license acceptance causes rework.

### 2. Install

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# cache + token
export HF_HOME=/root/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login   # accept Gemma + Paligemma licenses on HF first

# main venv
cd /root && git clone https://github.com/huggingface/lerobot.git lerobot-main
cd lerobot-main && uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[pi]" hf_transfer
python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/pi05_base')"

# patched venv — stacked PR #5153 includes #5152
cd /root && git clone https://github.com/huggingface/lerobot.git lerobot-patched
cd lerobot-patched
git fetch origin pull/5153/head:pr5153 && git checkout pr5153
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[pi]"
```

### 3. Run

```bash
cd /root/lerobot-main && source .venv/bin/activate
python bench_pi05_denoise.py fp32
python bench_pi05_denoise.py bf16

cd /root/lerobot-patched && source .venv/bin/activate
python bench_pi05_denoise.py bf16_patched
python bench_pi05_denoise.py bf16_patched_cudagraph

cat /root/bench_results.jsonl
```

Each config appends one JSON line to `/root/bench_results.jsonl`. See [`results.jsonl`](./results.jsonl) in this repo for the raw numbers that produced the table above.

## Files

- [`bench_pi05_denoise.py`](./bench_pi05_denoise.py) — the benchmark script (CLI: `python bench_pi05_denoise.py {fp32|bf16|bf16_patched|bf16_patched_cudagraph}`)
- [`results.jsonl`](./results.jsonl) — raw per-config statistics (one JSON object per line)
- [`README.md`](./README.md) — this file

## Credits

- [@arun-elr](https://github.com/arun-elr) for identifying the denoise-loop hotspot and shipping both optimization PRs
- [@molbap](https://github.com/molbap) for the A100 reference numbers and the `transformers`-migration context
