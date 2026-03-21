#!/usr/bin/env python3
"""
FastV Experiment Suite for LLaVA-1.5 and Qwen2-VL
Reproduces key experiments from "An Image is Worth 1/2 Tokens After Layer 2"

Usage:
  # Experiment 1+3: K/R Sweep on A-OKVQA and/or MMMU
  python fastv_experiments.py sweep --model_type qwen2vl --model_path Qwen/Qwen2-VL-2B-Instruct --benchmarks aokvqa mmmu --num_samples 500
  python fastv_experiments.py sweep --model_type llava   --model_path llava-hf/llava-1.5-7b-hf    --benchmarks aokvqa --num_samples 500

  # Experiment 2: Latency & Memory Measurement
  python fastv_experiments.py latency --model_type qwen2vl --model_path Qwen/Qwen2-VL-2B-Instruct --num_samples 200

  # Experiment 4: Attention Pattern Analysis
  python fastv_experiments.py attention --model_type qwen2vl --model_path Qwen/Qwen2-VL-2B-Instruct --num_samples 100

  # Then plot everything:
  python plot_fastv_results.py --results_dir ./fastv_results
"""

import os
import json
import time
import gc
import re
import ast
import math
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ════════════════════════════════════════════════════════════════

# FastV configs matching the paper's Table 1.
# R_prune = ratio of vision tokens to REMOVE (paper convention).
#   R_prune=0.0 → baseline (keep all).
#   R_prune=0.5 → remove 50%, keep 50%.
#   R_prune=0.75 → remove 75%, keep 25%.
SWEEP_CONFIGS = [
    {"K": 0,  "R_prune": 0.0,  "label": "Baseline"},
    {"K": 2,  "R_prune": 0.5,  "label": "K=2,R=50%"},
    {"K": 2,  "R_prune": 0.75, "label": "K=2,R=75%"},
    {"K": 2,  "R_prune": 0.9,  "label": "K=2,R=90%"},
    {"K": 3,  "R_prune": 0.5,  "label": "K=3,R=50%"},
    {"K": 3,  "R_prune": 0.75, "label": "K=3,R=75%"},
    {"K": 3,  "R_prune": 0.9,  "label": "K=3,R=90%"},
    {"K": 5,  "R_prune": 0.5,  "label": "K=5,R=50%"},
    {"K": 5,  "R_prune": 0.75, "label": "K=5,R=75%"},
    {"K": 5,  "R_prune": 0.9,  "label": "K=5,R=90%"},
]

MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
    "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
    "Clinical_Medicine", "Computer_Science", "Design",
    "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature",
    "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]


# ════════════════════════════════════════════════════════════════
# 2. MODEL MANAGEMENT
# ════════════════════════════════════════════════════════════════

def load_model_and_processor(model_type, model_path, cache_dir="./checkpoints",
                             revision=None, max_pixels=401408):
    """Load model + processor.  Always uses eager attention (required for
    FastV attention-weight extraction and for the attention analysis)."""
    print(f"Loading {model_type} model: {model_path}")

    if model_type == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        # Cap image resolution to avoid OOM in the vision encoder.
        # With eager attention the vision encoder materializes the full
        # [num_patches × num_patches × num_heads] attention matrix in float32.
        # Large MMMU diagrams can produce 27K+ patches → 45 GB just for softmax.
        #
        # max_pixels=401408 ≈ 634×634 image → ~4096 total patches (with
        # temporal=2) → attention ~1 GB per vision block. Safe on 80 GB GPU.
        # Lower this further if you still OOM (try 256*28*14*14 = 140672).
        processor = AutoProcessor.from_pretrained(
            model_path, cache_dir=cache_dir,
            min_pixels=4 * 28 * 28,
            max_pixels=max_pixels,
        )

    elif model_type == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        kwargs = {}
        if revision:
            kwargs["revision"] = revision
        # Load model; fastv_config defaults to None in LlavaConfig.
        # We set it dynamically via set_fastv_params before each config.
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
            **kwargs,
        )
        # The HF processor config for llava-1.5 now contains kwargs
        # (image_token, patch_size, etc.) that older transformers doesn't
        # understand.  Fall back to building it from tokenizer + image_processor.
        try:
            processor = AutoProcessor.from_pretrained(
                model_path, cache_dir=cache_dir, **kwargs,
            )
        except TypeError:
            from transformers import AutoTokenizer, AutoImageProcessor, LlavaProcessor
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, cache_dir=cache_dir, **kwargs,
            )
            image_processor = AutoImageProcessor.from_pretrained(
                model_path, cache_dir=cache_dir, **kwargs,
            )
            processor = LlavaProcessor(
                tokenizer=tokenizer, image_processor=image_processor,
            )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    print(f"  Loaded on {model.device}, dtype={model.dtype}")
    return model, processor


def set_fastv_params(model, model_type, K, R_prune, random_pruning=False):
    """Configure FastV at runtime.

    K               – layer index at which to prune (paper convention)
    R_prune         – fraction of vision tokens to PRUNE (paper convention).
                      0.0 = baseline, 0.5 = prune half, 0.9 = prune 90%.
    random_pruning  – if True, prune randomly instead of by attention rank
                      (for ablation study, Paper Table 7 row c).
    """
    if model_type == "qwen2vl":
        # Our Qwen2-VL implementation uses keep-ratio
        model.config.fastv_k = K
        model.config.fastv_r = 1.0 - R_prune          # keep ratio
        model.config.fastv_random_pruning = random_pruning

    elif model_type == "llava":
        if R_prune == 0.0:
            # Baseline: set to None so model uses regular forward path
            model.config.fastv_config = None
        else:
            # LLaVA FastV uses prune-ratio directly (matching paper)
            model.config.fastv_config = {
                "use_fastv": True,
                "fastv_k": K,
                "fastv_r": R_prune,
                "image_token_start_index": 5,     # updated per-sample below
                "image_token_length": 576,         # updated per-sample below
                "random_pruning": random_pruning,
            }


# def _update_llava_image_range(model, input_ids, processor):
#     """Dynamically set image_token_start_index and image_token_length for LLaVA
#     by scanning the actual input_ids."""
#     if not getattr(model.config, "fastv_config", None):
#         return
#     img_tok_id = getattr(model.config, "image_token_index",
#                          processor.tokenizer.convert_tokens_to_ids("<image>"))
#     positions = (input_ids[0] == img_tok_id).nonzero(as_tuple=True)[0]
#     if len(positions) > 0:
#         model.config.fastv_config["image_token_start_index"] = positions[0].item()
#         model.config.fastv_config["image_token_length"] = len(positions)
def _update_llava_image_range(model, input_ids, processor):
    if not getattr(model.config, "fastv_config", None):
        return
    img_tok_id = getattr(model.config, "image_token_index",
                         processor.tokenizer.convert_tokens_to_ids("<image>"))
    positions = (input_ids[0] == img_tok_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        num_placeholders = len(positions)
        first_pos = positions[0].item()
        # LLaVA-1.5: each <image> placeholder expands to 576 patch embeddings
        # (336px / 14px_patch = 24 patches per side, 24*24 = 576)
        num_image_patches = 576
        model.config.fastv_config["image_token_start_index"] = first_pos
        model.config.fastv_config["image_token_length"] = num_placeholders * num_image_patches

# ════════════════════════════════════════════════════════════════
# 3. DATASET LOADING
# ════════════════════════════════════════════════════════════════

def load_aokvqa(num_samples=None, cache_dir="./data"):
    """Load A-OKVQA validation set (multiple-choice)."""
    from datasets import load_dataset
    print("Loading A-OKVQA …")
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation",
                      cache_dir=cache_dir)
    samples = []
    for item in ds:
        choices = item["choices"]
        correct_idx = item["correct_choice_idx"]
        answer_letter = chr(ord("A") + correct_idx)
        options_str = "  ".join(
            f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
        )
        samples.append({
            "image": item["image"].convert("RGB"),
            "question": item["question"],
            "options_str": options_str,
            "answer": answer_letter,
            "benchmark": "aokvqa",
        })
    if num_samples and num_samples < len(samples):
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(samples), num_samples, replace=False)
        samples = [samples[i] for i in idxs]
    print(f"  Loaded {len(samples)} A-OKVQA samples")
    return samples


def load_mmmu(num_samples=None, cache_dir="./data"):
    """Load MMMU validation set (multiple-choice only, first image)."""
    from datasets import load_dataset, concatenate_datasets
    print("Loading MMMU …")
    all_ds = []
    for subj in MMMU_SUBJECTS:
        try:
            ds = load_dataset("MMMU/MMMU", subj, split="validation",
                              cache_dir=cache_dir)
            all_ds.append(ds)
        except Exception:
            try:
                ds = load_dataset("MMMU/MMMU", subj, split="validation",
                                  cache_dir=cache_dir, trust_remote_code=True)
                all_ds.append(ds)
            except Exception as e:
                print(f"  Warning: failed to load MMMU/{subj}: {e}")
    if not all_ds:
        print("  ERROR: Could not load any MMMU subjects!")
        return []
    combined = concatenate_datasets(all_ds)

    samples = []
    for item in combined:
        if item.get("question_type") != "multiple-choice":
            continue
        image = item.get("image_1")
        if image is None:
            continue
        # Parse options: stored as string repr of list
        try:
            opts = ast.literal_eval(item["options"])
        except Exception:
            try:
                opts = json.loads(item["options"])
            except Exception:
                continue
        options_str = "  ".join(
            f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts)
        )
        samples.append({
            "image": image.convert("RGB"),
            "question": item["question"],
            "options_str": options_str,
            "answer": item["answer"],        # letter: A, B, C, D
            "benchmark": "mmmu",
        })
    if num_samples and num_samples < len(samples):
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(samples), num_samples, replace=False)
        samples = [samples[i] for i in idxs]
    print(f"  Loaded {len(samples)} MMMU samples (MC only)")
    return samples


def load_sciqa(num_samples=None, cache_dir="./data"):
    """Load ScienceQA image-subset validation set (multiple-choice).
    Corresponds to SciQA-IMG in the FastV paper (Table 2)."""
    from datasets import load_dataset
    print("Loading ScienceQA (image subset) …")
    ds = load_dataset("derek-thomas/ScienceQA", split="validation",
                      cache_dir=cache_dir)
    samples = []
    for item in ds:
        # Only keep samples that have an image
        image = item.get("image")
        if image is None:
            continue
        choices = item["choices"]
        correct_idx = item["answer"]
        answer_letter = chr(ord("A") + correct_idx)
        options_str = "  ".join(
            f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
        )
        q = item["question"]
        # Prepend the lecture/hint if present for richer context
        hint = item.get("hint", "")
        if hint:
            q = f"{hint}\n{q}"
        samples.append({
            "image": image.convert("RGB"),
            "question": q,
            "options_str": options_str,
            "answer": answer_letter,
            "benchmark": "sciqa",
        })
    if num_samples and num_samples < len(samples):
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(samples), num_samples, replace=False)
        samples = [samples[i] for i in idxs]
    print(f"  Loaded {len(samples)} ScienceQA-IMG samples")
    return samples


# ════════════════════════════════════════════════════════════════
# 4. PROMPT FORMATTING & INFERENCE
# ════════════════════════════════════════════════════════════════

MC_TEMPLATE = (
    "Analyse the image and choose the best answer for the following question: "
    "{question}\nOptions: {options}\nOutput the letter of the correct answer."
)


def prepare_inputs(model, processor, model_type, image, question, options_str):
    """Build model inputs for a single MC sample.
    Returns (inputs_dict, prompt_text)."""
    mc_text = MC_TEMPLATE.format(question=question, options=options_str)

    if model_type == "qwen2vl":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": mc_text},
        ]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt], images=[image], return_tensors="pt"
        )
    elif model_type == "llava":
        prompt = f"USER: <image>\n{mc_text}\nASSISTANT:"
        inputs = processor(
            text=prompt, images=image, return_tensors="pt"
        )
    else:
        raise ValueError(model_type)

    inputs = {k: v.to(model.device) for k, v in inputs.items()
              if isinstance(v, torch.Tensor)}
    return inputs, prompt


def extract_answer(text):
    """Extract first A/B/C/D letter from generated text."""
    text = text.strip()
    # Check first character
    if text and text[0] in "ABCD":
        return text[0]
    # Search for "X." or "(X)" patterns
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    return text[:1].upper() if text else ""


@torch.no_grad()
def run_inference(model, processor, model_type, image, question, options_str,
                  max_new_tokens=10):
    """Run inference on one MC sample, return predicted letter."""
    inputs, _ = prepare_inputs(model, processor, model_type, image,
                               question, options_str)
    # For LLaVA: dynamically set image token range
    if model_type == "llava" and "input_ids" in inputs:
        _update_llava_image_range(model, inputs["input_ids"], processor)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False)
    # Decode only the NEW tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return extract_answer(text), text


# ════════════════════════════════════════════════════════════════
# 5. EVALUATION LOOP
# ════════════════════════════════════════════════════════════════

def evaluate(model, processor, model_type, samples, max_new_tokens=10,
             desc="Evaluating"):
    """Evaluate MC accuracy on a list of samples. Returns accuracy [0,1]."""
    correct = 0
    total = 0
    for s in tqdm(samples, desc=desc, leave=False):
        pred, _ = run_inference(model, processor, model_type,
                                s["image"], s["question"], s["options_str"],
                                max_new_tokens=max_new_tokens)
        if pred == s["answer"]:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    return round(acc * 100, 2)


# ════════════════════════════════════════════════════════════════
# 6. FLOPs CALCULATION  (paper Eq.5)
# ════════════════════════════════════════════════════════════════

def compute_flops(model_config, n_total, n_img, K, R_prune):
    """Return (flops_baseline, flops_fastv, flops_ratio).
    All in units of FLOPs (not GFLOPs)."""
    # For composite configs (LLaVA), the LLM params live in text_config
    cfg = getattr(model_config, "text_config", model_config)
    d = cfg.hidden_size
    m = cfg.intermediate_size
    T = cfg.num_hidden_layers

    def layer_flops(n):
        return 4 * n * d**2 + 2 * n**2 * d + 2 * n * d * m

    n_text = n_total - n_img
    n_hat = n_text + int(n_img * (1.0 - R_prune))

    baseline = T * layer_flops(n_total)
    if R_prune == 0:
        fastv = baseline
    else:
        fastv = K * layer_flops(n_total) + (T - K) * layer_flops(n_hat)

    ratio = fastv / baseline if baseline > 0 else 1.0
    return baseline, fastv, round(ratio, 4)


def estimate_token_counts(model, processor, model_type, samples,
                          max_count=50):
    """Estimate average (n_total, n_img) from a few samples.
    For LLaVA, account for the fact that the tokenizer produces a single
    <image> placeholder that gets expanded to 576 tokens inside the model."""
    n_totals, n_imgs = [], []
    for s in samples[:max_count]:
        inputs, _ = prepare_inputs(model, processor, model_type,
                                   s["image"], s["question"], s["options_str"])
        ids = inputs["input_ids"][0]

        if model_type == "qwen2vl":
            # Qwen2-VL processor already expands image tokens
            img_id = model.config.image_token_id
            n_img = (ids == img_id).sum().item()
            n_totals.append(len(ids))
            n_imgs.append(n_img)
        elif model_type == "llava":
            # LLaVA tokenizer keeps <image> as 1 token; the model expands it
            # to 576 image embeddings internally. Adjust counts accordingly.
            img_id = getattr(model.config, "image_token_index",
                             processor.tokenizer.convert_tokens_to_ids("<image>"))
            num_placeholders = (ids == img_id).sum().item()
            llava_img_tokens = 576  # standard LLaVA-1.5 with 336×336 → 576 tokens
            n_img = num_placeholders * llava_img_tokens
            # Total = original - placeholders + expanded
            n_total = len(ids) - num_placeholders + n_img
            n_totals.append(n_total)
            n_imgs.append(n_img)

    return int(np.mean(n_totals)), int(np.mean(n_imgs))


# ════════════════════════════════════════════════════════════════
# 7. EXPERIMENT: K/R SWEEP  (Paper Table 1 + Figure 1)
# ════════════════════════════════════════════════════════════════

def run_sweep(args):
    model, processor = load_model_and_processor(
        args.model_type, args.model_path, args.cache_dir,
        getattr(args, "revision", None), getattr(args, "max_pixels", 401408),
    )
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench_name in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"  SWEEP: {bench_name.upper()} on {args.model_path}")
        print(f"{'='*60}")

        if bench_name == "aokvqa":
            samples = load_aokvqa(args.num_samples, args.data_dir)
        elif bench_name == "mmmu":
            samples = load_mmmu(args.num_samples, args.data_dir)
        elif bench_name == "sciqa":
            samples = load_sciqa(args.num_samples, args.data_dir)
        else:
            print(f"Unknown benchmark: {bench_name}"); continue

        if not samples:
            print(f"  No samples loaded for {bench_name}, skipping."); continue

        # Estimate token counts for FLOPs
        n_total, n_img = estimate_token_counts(
            model, processor, args.model_type, samples
        )
        print(f"  Avg tokens: total={n_total}, image={n_img}")

        results = []
        for cfg in SWEEP_CONFIGS:
            K, R_prune, label = cfg["K"], cfg["R_prune"], cfg["label"]
            set_fastv_params(model, args.model_type, K, R_prune)

            print(f"\n  Config: {label}")
            acc = evaluate(model, processor, args.model_type, samples,
                           desc=label)

            _, _, flops_ratio = compute_flops(
                model.config, n_total, n_img, K, R_prune
            )

            results.append({
                "label": label, "K": K, "R_prune": R_prune,
                "accuracy": acc, "flops_ratio": flops_ratio,
                "flops_reduction": round(1.0 - flops_ratio, 4),
                "num_samples": len(samples),
            })
            print(f"    Accuracy: {acc}%  |  FLOPs ratio: {flops_ratio:.2%}")

            gc.collect(); torch.cuda.empty_cache()

        # Save
        out_path = out_dir / f"sweep_{args.model_type}_{bench_name}.json"
        payload = {
            "model": args.model_path,
            "model_type": args.model_type,
            "benchmark": bench_name,
            "n_total": n_total,
            "n_img": n_img,
            "configs": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Saved → {out_path}")

        # Print summary table
        print(f"\n  {'Label':<18} {'Acc':>7} {'FLOPs%':>8} {'FLOPs↓':>8}")
        print(f"  {'-'*43}")
        for r in results:
            print(f"  {r['label']:<18} {r['accuracy']:>6.1f}% "
                  f"{r['flops_ratio']:>7.1%} {r['flops_reduction']:>7.1%}")


# ════════════════════════════════════════════════════════════════
# 8. EXPERIMENT: LATENCY & MEMORY  (Paper Table 4)
# ════════════════════════════════════════════════════════════════

def run_latency(args):
    model, processor = load_model_and_processor(
        args.model_type, args.model_path, args.cache_dir,
        getattr(args, "revision", None), getattr(args, "max_pixels", 401408),
    )
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_aokvqa(args.num_samples, args.data_dir)
    if not samples:
        print("No samples!"); return

    configs_to_test = [
        {"K": 0,  "R_prune": 0.0, "label": "Baseline"},
        {"K": 2,  "R_prune": 0.5, "label": "FastV K=2,R=50%"},
        {"K": 2,  "R_prune": 0.75, "label": "FastV K=2,R=75%"},
    ]

    results = []
    for cfg in configs_to_test:
        K, R_prune, label = cfg["K"], cfg["R_prune"], cfg["label"]
        set_fastv_params(model, args.model_type, K, R_prune)

        print(f"\n  Latency test: {label}")

        # Warmup (3 samples)
        for s in samples[:3]:
            run_inference(model, processor, args.model_type,
                          s["image"], s["question"], s["options_str"])

        torch.cuda.reset_peak_memory_stats()
        gc.collect(); torch.cuda.empty_cache()

        correct = 0
        total_time = 0.0
        times = []

        for s in tqdm(samples, desc=label, leave=False):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred, _ = run_inference(model, processor, args.model_type,
                                    s["image"], s["question"], s["options_str"])

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            total_time += elapsed
            times.append(elapsed)
            if pred == s["answer"]:
                correct += 1

        acc = round(correct / len(samples) * 100, 2)
        avg_latency = round(total_time / len(samples), 4)
        peak_mem = round(torch.cuda.max_memory_allocated() / 1024**2, 1) \
            if torch.cuda.is_available() else 0

        results.append({
            "label": label, "K": K, "R_prune": R_prune,
            "accuracy": acc,
            "total_time_s": round(total_time, 2),
            "avg_latency_s": avg_latency,
            "peak_memory_mb": peak_mem,
            "num_samples": len(samples),
        })
        print(f"    Acc: {acc}% | Total: {total_time:.1f}s | "
              f"Avg: {avg_latency:.4f}s/sample | Mem: {peak_mem:.0f}MB")

        gc.collect(); torch.cuda.empty_cache()

    # Save
    out_path = out_dir / f"latency_{args.model_type}.json"
    payload = {"model": args.model_path, "model_type": args.model_type,
               "configs": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # Summary table
    print(f"\n  {'Config':<24} {'Acc':>7} {'Total':>8} {'Avg/ex':>10} {'Mem MB':>8}")
    print(f"  {'-'*59}")
    for r in results:
        print(f"  {r['label']:<24} {r['accuracy']:>6.1f}% "
              f"{r['total_time_s']:>7.1f}s {r['avg_latency_s']:>9.4f}s "
              f"{r['peak_memory_mb']:>7.0f}")


# ════════════════════════════════════════════════════════════════
# 9. EXPERIMENT: ATTENTION PATTERN ANALYSIS  (Paper §3, Fig 3-4)
# ════════════════════════════════════════════════════════════════

def _get_token_categories(input_ids, model_type, model, processor,
                          expanded_seq_len=None):
    """Return dict mapping category name → list of token indices.
    Categories: 'pre_image', 'image', 'post_image'.

    For LLaVA, the tokenizer produces 1 <image> placeholder, but the model
    internally expands it to 576 tokens. When expanded_seq_len is provided
    (from attention output shapes), we compute categories in expanded space.
    """
    ids = input_ids[0]  # [seq_len]

    if model_type == "qwen2vl":
        # Qwen2-VL processor already has expanded image tokens in input_ids
        img_id = model.config.image_token_id
        vid_id = getattr(model.config, "video_token_id", -1)
        is_vision = (ids == img_id) | (ids == vid_id)

        vision_pos = is_vision.nonzero(as_tuple=True)[0]
        if len(vision_pos) == 0:
            return {"pre_image": list(range(len(ids))),
                    "image": [], "post_image": []}
        img_start = vision_pos[0].item()
        img_end = vision_pos[-1].item() + 1
        return {
            "pre_image":  list(range(0, img_start)),
            "image":      list(range(img_start, img_end)),
            "post_image": list(range(img_end, len(ids))),
        }

    elif model_type == "llava":
        # LLaVA: find the single <image> placeholder in tokenized input_ids
        img_id = getattr(model.config, "image_token_index",
                         processor.tokenizer.convert_tokens_to_ids("<image>"))
        positions = (ids == img_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            total = expanded_seq_len if expanded_seq_len else len(ids)
            return {"pre_image": list(range(total)),
                    "image": [], "post_image": []}

        placeholder_pos = positions[0].item()
        llava_img_tokens = 576  # LLaVA-1.5 standard

        if expanded_seq_len:
            # Work in expanded space (for attention analysis)
            # Before <image>: positions 0..placeholder_pos-1
            # Image tokens:   positions placeholder_pos..placeholder_pos+576-1
            # After <image>:  the remaining text tokens
            img_start = placeholder_pos
            img_end = placeholder_pos + llava_img_tokens
            return {
                "pre_image":  list(range(0, img_start)),
                "image":      list(range(img_start, img_end)),
                "post_image": list(range(img_end, expanded_seq_len)),
            }
        else:
            # Pre-expansion space (for token counting etc.)
            return {
                "pre_image":  list(range(0, placeholder_pos)),
                "image":      list(range(placeholder_pos, placeholder_pos + 1)),
                "post_image": list(range(placeholder_pos + 1, len(ids))),
            }
    else:
        raise ValueError(model_type)


@torch.no_grad()
def run_attention_analysis(args):
    model, processor = load_model_and_processor(
        args.model_type, args.model_path, args.cache_dir,
        getattr(args, "revision", None), getattr(args, "max_pixels", 401408),
    )
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Disable FastV for clean attention collection
    set_fastv_params(model, args.model_type, K=0, R_prune=0.0)

    samples = load_aokvqa(args.num_samples, args.data_dir)
    if not samples:
        print("No samples!"); return

    cfg = getattr(model.config, "text_config", model.config)
    num_layers = cfg.num_hidden_layers
    # Accumulators: sum of attention allocation per layer per category
    alloc_sums = {cat: np.zeros(num_layers)
                  for cat in ["pre_image", "image", "post_image"]}
    efficiency_sums = {cat: np.zeros(num_layers)
                       for cat in ["pre_image", "image", "post_image"]}
    count_sums = {cat: np.zeros(num_layers)
                  for cat in ["pre_image", "image", "post_image"]}
    n_processed = 0

    print(f"\nAttention analysis: {len(samples)} samples, {num_layers} layers")

    for s in tqdm(samples, desc="Attention analysis"):
        try:
            inputs, _ = prepare_inputs(
                model, processor, args.model_type,
                s["image"], s["question"], s["options_str"],
            )

            # For Qwen2-VL, compute proper 3D position_ids
            extra_kwargs = {}
            if args.model_type == "qwen2vl":
                position_ids, rope_deltas = model.get_rope_index(
                    inputs["input_ids"],
                    inputs.get("image_grid_thw"),
                    inputs.get("video_grid_thw"),
                    inputs.get("attention_mask"),
                )
                extra_kwargs["position_ids"] = position_ids
                extra_kwargs["rope_deltas"] = rope_deltas

            outputs = model(
                **inputs,
                **extra_kwargs,
                output_attentions=True,
                return_dict=True,
            )

            # Get the actual (possibly expanded) sequence length from attention
            # For LLaVA: tokenizer has ~69 tokens, but model expands to ~644
            expanded_seq_len = outputs.attentions[0].shape[-1]

            categories = _get_token_categories(
                inputs["input_ids"], args.model_type, model, processor,
                expanded_seq_len=expanded_seq_len,
            )
            if len(categories["image"]) == 0:
                continue

            # outputs.attentions: tuple of [bsz, num_heads, seq_len, seq_len]
            for layer_idx, attn in enumerate(outputs.attentions):
                # Average over heads → [seq_len, seq_len]
                avg_attn = attn.float().mean(dim=1)[0]  # [seq_len, seq_len]
                # Attention from the LAST token to all others
                last_tok_attn = avg_attn[-1]  # [seq_len]

                for cat, indices in categories.items():
                    if len(indices) == 0:
                        continue
                    idx_t = torch.tensor(indices, device=last_tok_attn.device)
                    allocation = last_tok_attn[idx_t].sum().item()
                    efficiency = allocation / len(indices)
                    alloc_sums[cat][layer_idx] += allocation
                    efficiency_sums[cat][layer_idx] += efficiency
                    count_sums[cat][layer_idx] += len(indices)

            n_processed += 1

            # Free memory
            del outputs, attn, avg_attn, last_tok_attn
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Warning: sample failed: {e}")
            continue

    if n_processed == 0:
        print("No samples processed!"); return

    # Average
    result = {
        "model": args.model_path,
        "model_type": args.model_type,
        "num_samples": n_processed,
        "num_layers": num_layers,
        "attention_allocation": {},
        "attention_efficiency": {},
        "avg_token_counts": {},
    }
    for cat in ["pre_image", "image", "post_image"]:
        result["attention_allocation"][cat] = (
            alloc_sums[cat] / n_processed
        ).tolist()
        result["attention_efficiency"][cat] = (
            efficiency_sums[cat] / n_processed
        ).tolist()
        result["avg_token_counts"][cat] = round(
            count_sums[cat].mean() / n_processed, 1
        )

    out_path = out_dir / f"attention_{args.model_type}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # Quick summary
    print(f"\n  Avg attention allocation (averaged over {n_processed} samples):")
    print(f"  {'Layer':<8}", end="")
    for cat in ["pre_image", "image", "post_image"]:
        print(f" {cat:>12}", end="")
    print()
    # Show layers 0, 1, 2, mid, last
    show_layers = sorted(set([0, 1, 2, num_layers//2, num_layers-1]))
    for li in show_layers:
        print(f"  {li:<8}", end="")
        for cat in ["pre_image", "image", "post_image"]:
            val = result["attention_allocation"][cat][li]
            print(f" {val:>11.4f}", end="")
        print()


# ════════════════════════════════════════════════════════════════
# 10. EXPERIMENT: ABLATION – Random vs Attention Pruning (Paper Table 7)
# ════════════════════════════════════════════════════════════════

# Configs for ablation: compare attention-ranked vs random pruning.
ABLATION_CONFIGS = [
    {"K": 0,  "R_prune": 0.0,  "random": False, "label": "Baseline"},
    {"K": 2,  "R_prune": 0.5,  "random": False, "label": "K=2,R=50% (attn)"},
    {"K": 2,  "R_prune": 0.5,  "random": True,  "label": "K=2,R=50% (random)"},
    {"K": 2,  "R_prune": 0.75, "random": False, "label": "K=2,R=75% (attn)"},
    {"K": 2,  "R_prune": 0.75, "random": True,  "label": "K=2,R=75% (random)"},
    {"K": 2,  "R_prune": 0.9,  "random": False, "label": "K=2,R=90% (attn)"},
    {"K": 2,  "R_prune": 0.9,  "random": True,  "label": "K=2,R=90% (random)"},
]


def run_ablation(args):
    """Ablation study: attention-ranked pruning vs random pruning.
    Reproduces Paper Table 7, rows (b) vs (c)."""
    model, processor = load_model_and_processor(
        args.model_type, args.model_path, args.cache_dir,
        getattr(args, "revision", None), getattr(args, "max_pixels", 401408),
    )
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_name = args.benchmark
    print(f"\n{'='*60}")
    print(f"  ABLATION: random vs attention pruning on {bench_name.upper()}")
    print(f"  Model: {args.model_path}")
    print(f"{'='*60}")

    if bench_name == "aokvqa":
        samples = load_aokvqa(args.num_samples, args.data_dir)
    elif bench_name == "mmmu":
        samples = load_mmmu(args.num_samples, args.data_dir)
    elif bench_name == "sciqa":
        samples = load_sciqa(args.num_samples, args.data_dir)
    else:
        print(f"Unknown benchmark: {bench_name}"); return

    if not samples:
        print("No samples loaded!"); return

    # Estimate token counts for FLOPs
    n_total, n_img = estimate_token_counts(
        model, processor, args.model_type, samples
    )
    print(f"  Avg tokens: total={n_total}, image={n_img}")

    results = []
    for cfg in ABLATION_CONFIGS:
        K, R_prune = cfg["K"], cfg["R_prune"]
        random_pruning = cfg["random"]
        label = cfg["label"]

        set_fastv_params(model, args.model_type, K, R_prune,
                         random_pruning=random_pruning)

        print(f"\n  Config: {label}")
        acc = evaluate(model, processor, args.model_type, samples, desc=label)

        _, _, flops_ratio = compute_flops(
            model.config, n_total, n_img, K, R_prune
        )

        results.append({
            "label": label, "K": K, "R_prune": R_prune,
            "random_pruning": random_pruning,
            "accuracy": acc, "flops_ratio": flops_ratio,
            "flops_reduction": round(1.0 - flops_ratio, 4),
            "num_samples": len(samples),
        })
        print(f"    Accuracy: {acc}%  |  FLOPs ratio: {flops_ratio:.2%}")

        gc.collect(); torch.cuda.empty_cache()

    # Save
    out_path = out_dir / f"ablation_{args.model_type}_{bench_name}.json"
    payload = {
        "model": args.model_path,
        "model_type": args.model_type,
        "benchmark": bench_name,
        "n_total": n_total,
        "n_img": n_img,
        "configs": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # Summary table
    print(f"\n  {'Label':<28} {'Acc':>7} {'FLOPs%':>8}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['label']:<28} {r['accuracy']:>6.1f}% "
              f"{r['flops_ratio']:>7.1%}")


# ════════════════════════════════════════════════════════════════
# 11. MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FastV Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache_dir", default="./checkpoints")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--results_dir", default="./fastv_results")
    parser.add_argument("--max_pixels", type=int, default=401408,
                        help="Max image pixels for Qwen2-VL (lower to avoid OOM with eager attn)")

    sub = parser.add_subparsers(dest="experiment", required=True)

    # ── sweep ──
    p = sub.add_parser("sweep", help="K/R sweep (Table 1 + Figure 1)")
    p.add_argument("--model_type", required=True, choices=["qwen2vl", "llava"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--revision", default=None, help="HF model revision (LLaVA)")
    p.add_argument("--benchmarks", nargs="+", default=["aokvqa"],
                   choices=["aokvqa", "mmmu", "sciqa"])
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit samples (None=full dataset)")

    # ── latency ──
    p = sub.add_parser("latency", help="Latency measurement (Table 4)")
    p.add_argument("--model_type", required=True, choices=["qwen2vl", "llava"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--num_samples", type=int, default=200)

    # ── attention ──
    p = sub.add_parser("attention", help="Attention pattern analysis (§3)")
    p.add_argument("--model_type", required=True, choices=["qwen2vl", "llava"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--num_samples", type=int, default=100)

    # ── ablation ──
    p = sub.add_parser("ablation", help="Random vs attention pruning (Table 7)")
    p.add_argument("--model_type", required=True, choices=["qwen2vl", "llava"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--benchmark", default="aokvqa",
                   choices=["aokvqa", "mmmu", "sciqa"])
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit samples (None=full dataset)")

    args = parser.parse_args()

    if args.experiment == "sweep":
        run_sweep(args)
    elif args.experiment == "latency":
        run_latency(args)
    elif args.experiment == "attention":
        run_attention_analysis(args)
    elif args.experiment == "ablation":
        run_ablation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()