#!/usr/bin/env python3
"""
Quick diagnostic: is FastV actually pruning tokens on LLaVA-1.5-7b?

Usage:
  python test_fastv_llava.py --model_path llava-hf/llava-1.5-7b-hf --image_path /path/to/any/image.jpg

What it does:
  1. Loads LLaVA-1.5-7b with your modified modeling files
  2. Monkey-patches fastv_forward to log sequence lengths per layer
  3. Runs the SAME prompt+image three ways:
       (a) Baseline  – no FastV
       (b) Buggy     – FastV K=2, R=50%, image range from raw input_ids  (current code)
       (c) Fixed     – FastV K=2, R=50%, image range from expanded embeddings
  4. Prints a clear comparison so you can see whether pruning happened
"""

import argparse, torch, gc
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

# ── helpers ──────────────────────────────────────────────────────
def load(model_path, cache_dir="./checkpoints"):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    try:
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    except TypeError:
        from transformers import AutoTokenizer, AutoImageProcessor, LlavaProcessor
        tok = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        img_proc = AutoImageProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        processor = LlavaProcessor(tokenizer=tok, image_processor=img_proc)
    model.eval()
    return model, processor


def prepare(model, processor, image, question):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    return inputs


def set_fastv(model, K, R_prune, img_start, img_len):
    """Set fastv_config on model. Pass None to disable."""
    if R_prune == 0.0:
        model.config.fastv_config = None
    else:
        model.config.fastv_config = {
            "use_fastv": True,
            "fastv_k": K,
            "fastv_r": R_prune,
            "image_token_start_index": img_start,
            "image_token_length": img_len,
        }


def get_buggy_image_range(input_ids, model, processor):
    """Current code: scans raw input_ids (pre-expansion)."""
    img_tok_id = getattr(model.config, "image_token_index",
                         processor.tokenizer.convert_tokens_to_ids("<image>"))
    positions = (input_ids[0] == img_tok_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        return positions[0].item(), len(positions)
    return 0, 0


def get_fixed_image_range(input_ids, model, processor):
    """Fixed: account for token expansion (1 placeholder → 576 patches)."""
    img_tok_id = getattr(model.config, "image_token_index",
                         processor.tokenizer.convert_tokens_to_ids("<image>"))
    positions = (input_ids[0] == img_tok_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        first_pos = positions[0].item()
        num_patches = 576  # LLaVA-1.5 with 336×336 → 576 image tokens
        return first_pos, len(positions) * num_patches
    return 0, 0


# ── monkey-patch to log seq lengths ─────────────────────────────
_seq_log = []

def make_logging_hook(model):
    """Wrap the LlamaModel.fastv_forward to print seq-length changes."""
    llama_model = model.language_model.model  # the LlamaModel
    original_fastv = llama_model.fastv_forward

    def logging_fastv_forward(*args, **kwargs):
        fastv_config = kwargs.get("fastv_config") or (args[10] if len(args) > 10 else None)
        K = fastv_config["fastv_k"]
        R = fastv_config["fastv_r"]
        img_start = fastv_config["image_token_start_index"]
        img_len = fastv_config["image_token_length"]
        tokens_to_keep = round(img_len * (1 - R))

        print(f"  [FastV config] K={K}, R_prune={R}, "
              f"img_start={img_start}, img_len={img_len}, "
              f"tokens_to_keep={tokens_to_keep}")
        _seq_log.clear()

        # Hook into each decoder layer to capture hidden_states shape
        hooks = []
        for layer_idx, layer in enumerate(llama_model.layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    hs = output[0]
                    _seq_log.append((idx, hs.shape[1]))
                return hook_fn
            h = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            result = original_fastv(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        # Print summary
        if _seq_log:
            pre_prune = _seq_log[0][1]
            post_prune = _seq_log[K][1] if K < len(_seq_log) else _seq_log[-1][1]
            print(f"  [Seq lengths] layer 0: {pre_prune} → layer {K}: {post_prune}  "
                  f"(dropped {pre_prune - post_prune} tokens)")
        return result

    llama_model.fastv_forward = logging_fastv_forward
    return original_fastv  # return so we can restore if needed

import requests
from PIL import Image
# ── main ─────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image_path", default=None,
                        help="Path to test image. If omitted, creates a synthetic one.")
    parser.add_argument("--cache_dir", default="./checkpoints")
    args = parser.parse_args()

    # Load
    model, processor = load(args.model_path, args.cache_dir)

    # Image
    if args.image_path:
        image = Image.open(requests.get(args.image_path, stream=True).raw)
    else:
        print("No --image_path given; using a synthetic 336×336 image.")
        import numpy as np
        image = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))

    question = "What do you see in this image? Answer briefly."
    inputs = prepare(model, processor, image, question)
    input_ids = inputs["input_ids"]

    # Show token info
    print(f"\n{'='*60}")
    print(f"  Raw input_ids length: {input_ids.shape[1]}")
    buggy_start, buggy_len = get_buggy_image_range(input_ids, model, processor)
    fixed_start, fixed_len = get_fixed_image_range(input_ids, model, processor)
    print(f"  Buggy image range: start={buggy_start}, len={buggy_len}")
    print(f"  Fixed image range: start={fixed_start}, len={fixed_len}")
    print(f"{'='*60}\n")

    # Install logging hook
    orig_fn = make_logging_hook(model)

    K, R_prune = 2, 0.5
    results = {}

    # ── (a) Baseline ──
    print("━" * 60)
    print("TEST A: Baseline (no FastV)")
    print("━" * 60)
    set_fastv(model, 0, 0.0, 0, 0)
    out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    text_a = processor.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output: {text_a!r}\n")
    results["baseline"] = text_a
    gc.collect(); torch.cuda.empty_cache()

    # ── (b) Buggy FastV ──
    print("━" * 60)
    print(f"TEST B: FastV K={K}, R={R_prune} — BUGGY image range (from raw input_ids)")
    print("━" * 60)
    set_fastv(model, K, R_prune, buggy_start, buggy_len)
    out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    text_b = processor.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output: {text_b!r}\n")
    results["buggy_fastv"] = text_b
    gc.collect(); torch.cuda.empty_cache()

    # ── (c) Fixed FastV ──
    print("━" * 60)
    print(f"TEST C: FastV K={K}, R={R_prune} — FIXED image range (576 expanded tokens)")
    print("━" * 60)
    set_fastv(model, K, R_prune, fixed_start, fixed_len)
    out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    text_c = processor.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output: {text_c!r}\n")
    results["fixed_fastv"] = text_c

    # ── Verdict ──
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    if text_a == text_b:
        print("  ⚠  Buggy FastV output == Baseline  →  pruning is NOT happening!")
    else:
        print("  ✓  Buggy FastV output differs from Baseline")

    if text_a == text_c:
        print("  ⚠  Fixed FastV output == Baseline  (possible, but check seq lengths above)")
    else:
        print("  ✓  Fixed FastV output differs from Baseline  →  pruning IS active")

    if text_b == text_c:
        print("  ⚠  Buggy == Fixed  →  fix made no difference (unlikely if img_len changed)")
    else:
        print("  ✓  Buggy ≠ Fixed  →  the image range fix changes behavior")

    print("\nDone. Check the [Seq lengths] lines above to confirm token counts.\n")


if __name__ == "__main__":
    main()