"""
RKNN-LLM Perplexity Evaluator
GET_LOGITS 모드를 사용하여 wikitext-2에서 perplexity 측정.
MNN evaluate_perplexity.py와 동일한 sliding window 알고리즘 사용.

Usage:
    python ppl_rkllm.py --model_path ./model.rkllm --output_dir ./results/raw
"""
import ctypes
import os
import sys
import json
import time
import argparse
import numpy as np

# ── RKLLM C API bindings (reused from bench_rkllm.py) ────────────────
RKLLM_LIB_PATH = "/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so"

rkllm_lib = ctypes.CDLL(RKLLM_LIB_PATH)
RKLLM_Handle_t = ctypes.c_void_p

LLMCallState_NORMAL = 0
LLMCallState_FINISH = 2
LLMCallState_ERROR = 3

RKLLM_INPUT_TOKEN = 1
RKLLM_INFER_GET_LOGITS = 2


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 106),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
    ]


# ── Global state for logits callback ──────────────────────────────────
_collected_logits = None
_logits_ready = False
_logits_error = False


def _logits_callback(result, userdata, state):
    global _collected_logits, _logits_ready, _logits_error
    if state == LLMCallState_NORMAL:
        logits_result = result.contents.logits
        vocab_size = logits_result.vocab_size
        num_tokens = logits_result.num_tokens

        if vocab_size > 0 and num_tokens > 0:
            total = num_tokens * vocab_size
            arr = np.ctypeslib.as_array(logits_result.logits, shape=(total,))
            _collected_logits = arr.reshape(num_tokens, vocab_size).copy()
    elif state == LLMCallState_FINISH:
        _logits_ready = True
    elif state == LLMCallState_ERROR:
        _logits_error = True
        _logits_ready = True


_callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
_callback = _callback_type(_logits_callback)


# ── Model management ─────────────────────────────────────────────────
def init_model(model_path, max_context_len=4096):
    param = RKLLMParam()
    param.model_path = model_path.encode("utf-8")
    param.max_context_len = max_context_len
    param.max_new_tokens = 1
    param.skip_special_token = False
    param.n_keep = -1
    param.top_k = 1
    param.top_p = 1.0
    param.temperature = 1.0
    param.repeat_penalty = 1.0
    param.frequency_penalty = 0.0
    param.presence_penalty = 0.0
    param.mirostat = 0
    param.mirostat_tau = 5.0
    param.mirostat_eta = 0.1
    param.is_async = False
    param.img_start = b""
    param.img_end = b""
    param.img_content = b""
    param.extend_param.base_domain_id = 0
    param.extend_param.enabled_cpus_num = 4
    param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

    handle = RKLLM_Handle_t()
    rkllm_init = rkllm_lib.rkllm_init
    rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), _callback_type]
    rkllm_init.restype = ctypes.c_int
    ret = rkllm_init(ctypes.byref(handle), ctypes.byref(param), _callback)
    if ret != 0:
        raise RuntimeError(f"rkllm_init failed: {ret}")
    return handle


def get_logits(handle, token_ids, keep_history=0):
    """Feed token IDs and get logits via GET_LOGITS mode.

    keep_history:
        0 — default, clears KV state before running. Used for PPL sliding-window.
        1 — retains KV state from previous calls. Used for verify/speculative-decoding
            where we want to score `k` follow-up tokens against a pre-filled prefix.
    """
    global _collected_logits, _logits_ready, _logits_error

    _collected_logits = None
    _logits_ready = False
    _logits_error = False

    n_tokens = len(token_ids)
    c_ids = (ctypes.c_int32 * n_tokens)(*token_ids)

    rkllm_input = RKLLMInput()
    rkllm_input.input_mode = RKLLM_INPUT_TOKEN
    rkllm_input.input_data.token_input.input_ids = c_ids
    rkllm_input.input_data.token_input.n_tokens = n_tokens

    infer_param = RKLLMInferParam()
    ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
    infer_param.mode = RKLLM_INFER_GET_LOGITS
    infer_param.keep_history = int(keep_history)

    rkllm_run = rkllm_lib.rkllm_run
    rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    rkllm_run.restype = ctypes.c_int

    ret = rkllm_run(handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None)

    if ret != 0 or _logits_error:
        print(f"[ERROR] get_logits failed (ret={ret})")
        return None

    return _collected_logits


def clear_kv_cache(handle):
    """Reset KV cache state between independent probe runs."""
    fn = rkllm_lib.rkllm_clear_kv_cache
    fn.argtypes = [RKLLM_Handle_t]
    fn.restype = ctypes.c_int
    return fn(handle)


def destroy_model(handle):
    rkllm_destroy = rkllm_lib.rkllm_destroy
    rkllm_destroy.argtypes = [RKLLM_Handle_t]
    rkllm_destroy.restype = ctypes.c_int
    rkllm_destroy(handle)


# ── Perplexity evaluation ─────────────────────────────────────────────
def load_wikitext2_tokens(tokenizer_path=None):
    """Load wikitext-2 and tokenize."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(dataset["text"])

    # Use HuggingFace tokenizer if available
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokens = tokenizer.encode(full_text)
    else:
        # Simple whitespace tokenization fallback (not recommended)
        print("[WARN] No tokenizer provided, using rough approximation")
        tokens = list(range(1000))  # placeholder

    return tokens


def evaluate_perplexity(model_path, tokenizer_path, stride=512, context_length=768,
                        max_tokens=None):
    """Evaluate perplexity using sliding window approach (matching MNN's method)."""
    import torch

    print("Loading model...")
    handle = init_model(model_path, max_context_len=context_length + 256)

    print("Loading and tokenizing wikitext-2...")
    input_ids = load_wikitext2_tokens(tokenizer_path)
    seq_len = len(input_ids)
    if max_tokens:
        seq_len = min(seq_len, max_tokens)
    print(f"Total tokens: {seq_len}")

    nlls = []
    prev_end_loc = 0
    criterion = torch.nn.CrossEntropyLoss()

    num_chunks = (seq_len - 1) // stride + 1
    print(f"Evaluating {num_chunks} chunks (stride={stride}, context={context_length})...")

    for i, begin_loc in enumerate(range(0, seq_len, stride)):
        end_loc = min(begin_loc + context_length, seq_len)
        chunk_ids = input_ids[begin_loc:end_loc]

        # Get logits from RKLLM
        logits = get_logits(handle, chunk_ids)

        if logits is None:
            print(f"  [WARN] Chunk {i+1}: logits retrieval failed, skipping")
            continue

        logits_tensor = torch.from_numpy(logits).float()
        target_ids = torch.tensor(chunk_ids, dtype=torch.long)
        num_logits = logits_tensor.shape[0]
        num_targets = len(chunk_ids)

        # RKLLM may return N-1 logits for N input tokens (no logit for first token)
        # Align: logits[i] predicts target[i+1] if num_logits == num_targets
        #        logits[i] predicts target[i+1] if num_logits == num_targets - 1
        if num_logits == num_targets:
            # Standard: logits[:-1] predicts target[1:]
            pred_logits = logits_tensor[:-1, :]
            pred_targets = target_ids[1:]
        elif num_logits == num_targets - 1:
            # RKLLM returns N-1 logits: logits[i] predicts target[i+1]
            pred_logits = logits_tensor
            pred_targets = target_ids[1:]
        else:
            print(f"  [WARN] Chunk {i+1}: logits shape mismatch ({num_logits} vs {num_targets}), skipping")
            continue

        # Mask tokens that were seen in previous window
        trg_len = end_loc - prev_end_loc
        mask_len = len(pred_targets) - trg_len
        if mask_len > 0:
            pred_targets[:mask_len] = -100

        nll = criterion(pred_logits, pred_targets)
        nlls.append(nll)

        if (i + 1) % 10 == 0:
            current_ppl = torch.exp(torch.stack(nlls).mean()).item()
            print(f"  Chunk {i+1}/{num_chunks}: running PPL = {current_ppl:.2f}")

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    destroy_model(handle)

    if not nlls:
        print("[ERROR] No valid chunks processed")
        return None

    perplexity = torch.exp(torch.stack(nlls).mean()).item()
    print(f"\nPerplexity: {perplexity:.4f}")
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="RKNN-LLM Perplexity Evaluator")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .rkllm model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to HuggingFace tokenizer (for wikitext tokenization)")
    parser.add_argument("--output_dir", type=str, default="./results/raw")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--context_length", type=int, default=768)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max tokens to evaluate (None = full dataset)")
    args = parser.parse_args()

    ppl = evaluate_perplexity(
        args.model_path, args.tokenizer_path,
        args.stride, args.context_length, args.max_tokens
    )

    if ppl is not None:
        model_name = os.path.basename(args.model_path).replace(".rkllm", "")
        result = {
            "model": model_name,
            "framework": "rknn-llm",
            "perplexity": round(ppl, 4),
            "stride": args.stride,
            "context_length": args.context_length,
            "dataset": "wikitext-2-raw-v1",
        }

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"ppl_rkllm_{model_name}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
