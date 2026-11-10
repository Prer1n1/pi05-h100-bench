"""
H100 benchmark for pi0.5 denoise_step / sample_actions.
Run under both /root/lerobot-main and /root/lerobot-patched envs.
Pass the config name as the first arg; it only labels output.

Usage:
    python bench_pi05_denoise.py fp32
    python bench_pi05_denoise.py bf16
    python bench_pi05_denoise.py bf16_patched
    python bench_pi05_denoise.py bf16_patched_cudagraph
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def build_dummy_batch(policy, device, dtype):
    """Build a batch matching pi05_base's expected input spec.

    pi05_base expects multi-view RGB, proprio state, and PRE-TOKENIZED
    language tokens + attention mask. `predict_action_chunk` does not
    tokenize the raw task string itself — the dataloader normally would.
    Shapes are read from policy.config so we don't hardcode them.
    """
    from transformers import AutoTokenizer

    cfg = policy.config
    batch = {}
    for key, feat in cfg.input_features.items():
        if "image" in key:
            batch[key] = torch.randn(1, *feat.shape, device=device, dtype=dtype)
        elif "state" in key:
            batch[key] = torch.randn(1, *feat.shape, device=device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer_name)
    enc = tokenizer(
        ["pick up the red block and place it in the bowl"],
        padding="max_length",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    batch["observation.language.tokens"] = enc["input_ids"].to(device)
    batch["observation.language.attention_mask"] = enc["attention_mask"].to(device, dtype=torch.bool)
    return batch


def time_one(fn, warmup=20, iters=200):
    """Time `fn` with cuda events; return list of ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def stats(xs):
    xs_sorted = sorted(xs)
    return {
        "p50": statistics.median(xs_sorted),
        "p95": xs_sorted[int(0.95 * len(xs_sorted))],
        "mean": statistics.mean(xs_sorted),
        "min": xs_sorted[0],
        "n": len(xs_sorted),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="config label (fp32|bf16|bf16_patched|bf16_patched_cudagraph)")
    ap.add_argument("--model", default="lerobot/pi05_base")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default="/root/bench_results.jsonl")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float32 if args.label == "fp32" else torch.bfloat16
    # Keep weights in fp32 and use autocast for bf16 — avoids dtype mismatch
    # between fp32 noise tensors (from the flow-matching integrator) and
    # bf16 weights. Standard PyTorch mixed-precision inference pattern.
    use_autocast = dtype == torch.bfloat16
    load_dtype = torch.float32

    print(f"[{args.label}] loading {args.model} in {load_dtype} (compute: {dtype})")
    policy = PI05Policy.from_pretrained(args.model)
    policy = policy.to(device=device, dtype=load_dtype).eval()

    # Flip the CUDA-graph flag for the stacked-PR eval only.
    if args.label == "bf16_patched_cudagraph":
        if not hasattr(policy.config, "denoise_cuda_graph"):
            sys.exit("denoise_cuda_graph flag missing — are you in /root/lerobot-patched env?")
        policy.config.denoise_cuda_graph = True

    # Build inputs in fp32; autocast casts on the fly for bf16 runs.
    batch = build_dummy_batch(policy, device, load_dtype)
    print(f"[{args.label}] input shapes:")
    for k, v in batch.items():
        print(f"    {k}: {tuple(v.shape) if torch.is_tensor(v) else v}")

    # --- Full-chunk timing (10 denoise steps -> one action chunk) ---
    @torch.no_grad()
    def full_chunk():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
            _ = policy.predict_action_chunk(batch)

    chunk_times = time_one(full_chunk, warmup=args.warmup, iters=args.iters)
    chunk_stats = stats(chunk_times)
    per_step_stats = {k: (v / policy.config.num_inference_steps if k != "n" else v)
                      for k, v in chunk_stats.items()}

    result = {
        "label": args.label,
        "model": args.model,
        "dtype": str(dtype),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "num_denoise_steps": policy.config.num_inference_steps,
        "per_chunk_ms": chunk_stats,
        "per_step_ms": per_step_stats,
    }
    print(json.dumps(result, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
