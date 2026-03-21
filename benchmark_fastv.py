"""
Benchmark FastV on Qwen2-VL: compare speed, memory, and output quality
with FastV enabled vs disabled.

Usage:
    python benchmark_fastv.py --model Qwen/Qwen2-VL-2B-Instruct --cache_dir ./checkpoints
"""

import argparse
import time
import torch
import gc
import json
from PIL import Image
import requests
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# ─── Test images ───
TEST_IMAGES = [
    {
        "url": "https://www.ilankelman.org/stopsigns/australia.jpg",
        "prompt": "Describe this image in detail.",
        "name": "stop_sign",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        "prompt": "What objects do you see in this image?",
        "name": "dice",
    },
]


def load_image(url):
    response = requests.get(url, stream=True)
    return Image.open(BytesIO(response.content)).convert("RGB")


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def run_inference(model, processor, image, prompt, max_new_tokens=200):
    """Run a single inference and return output text, time, and peak memory."""
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    num_input_tokens = inputs["input_ids"].shape[1]

    # Warmup run (first run has extra overhead)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)

    torch.cuda.reset_peak_memory_stats()

    # Timed run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    peak_mem = get_gpu_memory_mb()
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    tokens_per_sec = num_output_tokens / elapsed if elapsed > 0 else 0

    output_text = processor.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt portion from the decoded text
    # The output typically contains the full conversation; grab the assistant part
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n")[-1].strip()

    return {
        "text": output_text,
        "time_s": round(elapsed, 3),
        "peak_memory_mb": round(peak_mem, 1),
        "input_tokens": num_input_tokens,
        "output_tokens": num_output_tokens,
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--cache_dir", type=str, default="./checkpoints")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--fastv_k", type=int, default=2, help="Layer to prune at")
    parser.add_argument("--fastv_r", type=float, default=0.5, help="Ratio of vision tokens to keep")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=args.cache_dir,
    )
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)

    # Load test images
    images = {}
    for item in TEST_IMAGES:
        print(f"  Loading image: {item['name']}...")
        try:
            images[item["name"]] = load_image(item["url"])
        except Exception as e:
            print(f"  Failed to load {item['name']}: {e}")

    results = {}

    for config_name, fastv_r in [("baseline (no FastV)", 1.0), (f"FastV (K={args.fastv_k}, R={args.fastv_r})", args.fastv_r)]:
        print(f"\n{'='*60}")
        print(f"  Config: {config_name}")
        print(f"{'='*60}")

        # Set FastV parameters
        # R=1.0 means keep all tokens = effectively disabled
        model.config.fastv_k = args.fastv_k
        model.config.fastv_r = fastv_r

        config_results = {}

        for item in TEST_IMAGES:
            if item["name"] not in images:
                continue

            print(f"\n  Image: {item['name']}")
            print(f"  Prompt: {item['prompt']}")

            result = run_inference(
                model, processor,
                images[item["name"]], item["prompt"],
                max_new_tokens=args.max_new_tokens,
            )

            config_results[item["name"]] = result

            print(f"  Input tokens:  {result['input_tokens']}")
            print(f"  Output tokens: {result['output_tokens']}")
            print(f"  Time:          {result['time_s']}s")
            print(f"  Tokens/sec:    {result['tokens_per_sec']}")
            print(f"  Peak memory:   {result['peak_memory_mb']} MB")
            print(f"  Output:        {result['text'][:200]}...")

        results[config_name] = config_results

    # ─── Summary comparison ───
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    config_names = list(results.keys())
    if len(config_names) == 2:
        baseline_name, fastv_name = config_names
        for img_name in images:
            if img_name not in results[baseline_name] or img_name not in results[fastv_name]:
                continue
            b = results[baseline_name][img_name]
            f = results[fastv_name][img_name]
            speedup = b["time_s"] / f["time_s"] if f["time_s"] > 0 else 0
            mem_reduction = (1 - f["peak_memory_mb"] / b["peak_memory_mb"]) * 100 if b["peak_memory_mb"] > 0 else 0

            print(f"\n  Image: {img_name}")
            print(f"  {'Metric':<20} {'Baseline':>12} {'FastV':>12} {'Change':>12}")
            print(f"  {'-'*56}")
            print(f"  {'Time (s)':<20} {b['time_s']:>12.3f} {f['time_s']:>12.3f} {speedup:>11.2f}x")
            print(f"  {'Peak Mem (MB)':<20} {b['peak_memory_mb']:>12.1f} {f['peak_memory_mb']:>12.1f} {mem_reduction:>10.1f}% less")
            print(f"  {'Tokens/sec':<20} {b['tokens_per_sec']:>12.1f} {f['tokens_per_sec']:>12.1f}")
            print(f"  {'Output tokens':<20} {b['output_tokens']:>12} {f['output_tokens']:>12}")

    # ─── Side-by-side output comparison ───
    print(f"\n\n{'='*60}")
    print("  OUTPUT COMPARISON (check quality manually)")
    print(f"{'='*60}")
    for img_name in images:
        for cname in config_names:
            if img_name in results[cname]:
                print(f"\n  [{cname}] {img_name}:")
                print(f"  {results[cname][img_name]['text']}")

    # Save results to JSON
    output_path = "fastv_benchmark_results.json"
    serializable = {}
    for cname, cresults in results.items():
        serializable[cname] = cresults
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()