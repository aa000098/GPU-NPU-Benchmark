"""
End-to-end hybrid speculative decoding runtime on RK3588.

Architecture:
- Draft: llama.cpp (CPU, 4 threads) — fast, same tokenizer as target
- Target: RKLLM W8A8 (NPU, 3 cores) — slow per-token but batchable
- Mode: sequential | async (thread-based overlap)
- Verify strategy: V1 full-prefix re-verify (safest, no KV rollback issues)

Speculative acceptance: greedy (for simplicity / determinism).
Later can add stochastic (Leviathan) if needed.

Usage:
    python benchmark/hybrid_runtime_v2.py \
        --target_model_path llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm \
        --draft_model_path llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf \
        --tokenizer_path llm_quant_bench/models/Llama-3.2-1B-Instruct \
        --prompt "The key finding of this work is that" \
        --n_gen 64 \
        --k 4 \
        --mode serial
"""

import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time

import numpy as np
from transformers import AutoTokenizer

# Set CPU affinity BEFORE importing llama_cpp so its internal threads inherit it.
# RK3588: little cores 0-3 (A55), big cores 4-7 (A76).
# RKLLM's enabled_cpus_mask=(1<<4)|(1<<5)|(1<<6)|(1<<7) → RKLLM uses big cores.
# Draft should be steered to little cores to avoid NPU-CPU contention during async overlap.
# Default: use all cores for sequential; restrict for async.
if os.environ.get("HYBRID_DRAFT_CPUS"):
    cpus = set(int(c) for c in os.environ["HYBRID_DRAFT_CPUS"].split(","))
    try:
        os.sched_setaffinity(0, cpus)
        print(f"[affinity] process pinned to CPUs {sorted(cpus)}")
    except Exception as e:
        print(f"[affinity] failed: {e}")

from llama_cpp import Llama

from llm_quant_bench.eval.ppl_rkllm import (
    init_model,
    get_logits,
    clear_kv_cache,
    destroy_model,
)


# ── Draft side ────────────────────────────────────────────────────────
class DraftEngine:
    """Wrap llama-cpp-python for draft decoding.

    Maintains a persistent KV cache across rounds using `n_tokens` rollback
    to avoid the 3-second cost of `reset()` every round. The state invariant:

        after self.start(prompt, k):
            KV contains [prompt, d_0, ..., d_{k-1}]
            self.current_prefix_len = len(prompt)

        after self.advance(a, c, k):
            KV contains [prev_prefix, d_0, ..., d_{a-1}, c, d'_0, ..., d'_{k-1}]
            self.current_prefix_len = prev_prefix_len + a + 1
    """

    def __init__(self, model_path, n_threads=4, n_ctx=4096):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            logits_all=False,
            verbose=False,
        )
        self.current_prefix_len = 0

    def _draft_k(self, k):
        drafts = []
        for _ in range(k):
            t = int(self.llm.sample(top_k=1, top_p=1.0, temp=0.0))
            drafts.append(t)
            self.llm.eval([t])
        return drafts

    def start(self, prompt_tokens, k):
        """Fresh start. Evals full prompt, drafts k."""
        self.llm.reset()
        self.llm.eval(list(prompt_tokens))
        self.current_prefix_len = len(prompt_tokens)
        return self._draft_k(k)

    def advance(self, accepted_count, correction_token, k):
        """After verify result known, rollback to keep `accepted_count` of drafts,
        eval correction, then draft new k. KV reused instead of full re-eval."""
        # Rollback KV to prev_prefix + accepted drafts
        target_n = self.current_prefix_len + accepted_count
        self.llm.n_tokens = target_n
        # Eval correction token
        self.llm.eval([int(correction_token)])
        self.current_prefix_len = target_n + 1
        return self._draft_k(k)


# ── Target side (NPU via RKLLM) ───────────────────────────────────────
class TargetVerifier:
    """V1 full-prefix verify: each round, clear KV and feed full prefix+draft."""

    def __init__(self, model_path, max_context_len=4096):
        self.handle = init_model(model_path, max_context_len=max_context_len)

    def verify(self, full_tokens):
        """Pass all tokens through target; return per-position logits (n_tokens, vocab)."""
        clear_kv_cache(self.handle)
        logits = get_logits(self.handle, full_tokens, keep_history=1)
        if logits is None:
            raise RuntimeError("Target verify failed")
        return logits

    def close(self):
        destroy_model(self.handle)


# ── Speculative accept/reject (greedy) ────────────────────────────────
def greedy_accept(target_logits_block, draft_tokens):
    """
    target_logits_block: shape (n_draft, vocab) — target's predictions for
       positions [prefix_end, prefix_end+1, ..., prefix_end+n_draft-1]
       i.e. target_logits_block[i] predicts the token that should appear at
       position (prefix_end + i + 1)
    Actually: target logits are (n, V) where the LAST row predicts token AFTER the full sequence.
    If we feed prefix + drafts (len = p+d), we get (p+d, V).
    Row i predicts token at position i+1. So:
       row p-1 predicts token at position p  (first drafted token expected)
       row p   predicts token at position p+1 (second drafted)
       ...
       row p+d-2 predicts token at position p+d-1 (last drafted)
    Last row (p+d-1) predicts what comes AFTER the drafted suffix — used for correction.

    accepted_tokens = drafted prefix where target argmax == draft_tokens[i]
    stops at first mismatch; correction token = argmax of last row.
    """
    n_draft = len(draft_tokens)
    # We assume caller has already sliced the relevant rows.
    accepted = []
    for i, drafted in enumerate(draft_tokens):
        target_top = int(np.argmax(target_logits_block[i]))
        if target_top == drafted:
            accepted.append(drafted)
        else:
            # Reject at position i. Correction = target's chosen token at this position.
            return accepted, target_top
    # All drafted accepted. Correction = argmax of the next-position logits (last row).
    correction = int(np.argmax(target_logits_block[-1]))  # this is for position after last draft
    return accepted, correction


# ── Sequential hybrid loop ────────────────────────────────────────────
def run_sequential(draft: DraftEngine, target: TargetVerifier,
                     prompt_tokens, n_gen, k, log_per_round=False):
    accepted_all = []
    trace = []
    t_total_start = time.perf_counter()

    t_draft_sum = 0.0
    t_verify_sum = 0.0
    rounds = 0

    # Round 0: fresh draft
    t0 = time.perf_counter()
    draft_tokens = draft.start(prompt_tokens, k)
    t_draft = time.perf_counter() - t0

    while len(accepted_all) < n_gen:
        current_prefix = prompt_tokens + accepted_all

        # Target verify over prefix + draft
        full = current_prefix + draft_tokens
        t0 = time.perf_counter()
        target_logits = target.verify(full)
        t_verify = time.perf_counter() - t0

        start = len(current_prefix) - 1
        n_rows = k + 1
        if start + n_rows > target_logits.shape[0]:
            raise RuntimeError(f"Target returned {target_logits.shape[0]} rows, "
                                f"needed up to {start + n_rows}")
        block = target_logits[start : start + n_rows]

        accepted, correction = greedy_accept(block, draft_tokens)
        new_tokens = accepted + [correction]
        if len(accepted_all) + len(new_tokens) > n_gen:
            new_tokens = new_tokens[: n_gen - len(accepted_all)]
        accepted_all.extend(new_tokens)

        t_draft_sum += t_draft
        t_verify_sum += t_verify
        rounds += 1

        if log_per_round:
            target_argmax_list = [int(np.argmax(block[i])) for i in range(k)]
            print(f"  round {rounds}: draft {t_draft*1000:.1f}ms, verify {t_verify*1000:.1f}ms, "
                  f"accepted {len(accepted)}/{k}")
            print(f"    draft tokens:  {draft_tokens}")
            print(f"    target argmax: {target_argmax_list}")

        trace.append({
            "round": rounds,
            "accepted": len(accepted),
            "t_draft_ms": t_draft * 1000,
            "t_verify_ms": t_verify * 1000,
            "emitted": len(new_tokens),
            "prefix_len": len(current_prefix),
        })

        # Advance draft to next round (reuses KV)
        if len(accepted_all) < n_gen:
            t0 = time.perf_counter()
            draft_tokens = draft.advance(len(accepted), correction, k)
            t_draft = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start
    return {
        "mode": "sequential",
        "n_gen_requested": n_gen,
        "n_gen_actual": len(accepted_all),
        "rounds": rounds,
        "total_ms": t_total * 1000,
        "ms_per_token": (t_total * 1000) / len(accepted_all) if accepted_all else 0,
        "tok_s": len(accepted_all) / t_total if t_total > 0 else 0,
        "mean_t_draft_ms": t_draft_sum * 1000 / rounds,
        "mean_t_verify_ms": t_verify_sum * 1000 / rounds,
        "mean_accepted_per_round": sum(r["accepted"] for r in trace) / rounds,
        "accepted_tokens": accepted_all,
        "trace": trace,
    }


# ── Async hybrid loop ─────────────────────────────────────────────────
def run_async(draft: DraftEngine, target: TargetVerifier,
                prompt_tokens, n_gen, k, log_per_round=False):
    """Pipeline: CPU advances next draft while NPU verifies current one.

    Works by running target.verify() and draft.advance() in parallel threads.
    But there's a catch: draft.advance() needs the accepted_count and correction
    from the CURRENT verify, which finishes at the end of the round.
    So true pipeline requires **speculative prefetch**: assume all drafts
    will be accepted, start computing next draft immediately. If they weren't,
    throw it away.

    For simplicity v1: run verify on NPU and draft.advance for OPTIMISTIC scenario
    (all k drafts + arbitrary correction token placeholder) in parallel.
    After verify, if prediction was wrong, redo draft.advance with real values.

    Because KV rollback is cheap (just n_tokens assignment), a wasted speculative
    advance is just the draft eval time — same as serial. But if right, we save
    draft time entirely (since it ran in parallel with verify).
    """
    accepted_all = []
    trace = []
    t_total_start = time.perf_counter()
    rounds = 0

    # Round 0: fresh draft (no overlap yet)
    draft_tokens = draft.start(prompt_tokens, k)

    while len(accepted_all) < n_gen:
        current_prefix_len = len(prompt_tokens) + len(accepted_all)

        full = prompt_tokens + accepted_all + draft_tokens

        # === Speculative prefetch: guess the correction token ===
        # If all k drafts were accepted, the correction comes from target's
        # prediction at position (prefix + k). Draft can't know that, so it
        # guesses: "draft's own argmax at that position" (reusing draft's KV).
        # We pre-compute draft_next assuming {all accepted, correction = draft_guess}.
        #
        # KV state at this point: [prefix, d_0, ..., d_{k-1}]
        # We need draft's next-token prediction at this state.
        #
        # Strategy: run target.verify (blocking NPU) and draft speculative
        # prefetch concurrently via threading.

        verify_result = {"logits": None, "t_verify": 0.0}
        spec_result = {"guess": None, "spec_drafts": None, "t_spec": 0.0,
                        "spec_start_kv": None}

        def verify_task():
            t0 = time.perf_counter()
            verify_result["logits"] = target.verify(full)
            verify_result["t_verify"] = time.perf_counter() - t0

        def spec_task():
            # Save current KV length so we can restore after spec
            initial_n = draft.llm.n_tokens
            spec_result["spec_start_kv"] = initial_n
            t0 = time.perf_counter()
            # Draft's guess for what comes AFTER [prefix, d_0..d_{k-1}]
            guess = int(draft.llm.sample(top_k=1, top_p=1.0, temp=0.0))
            draft.llm.eval([guess])
            # Now draft the next k tokens speculatively
            spec_drafts = []
            for _ in range(k):
                t = int(draft.llm.sample(top_k=1, top_p=1.0, temp=0.0))
                spec_drafts.append(t)
                draft.llm.eval([t])
            spec_result["guess"] = guess
            spec_result["spec_drafts"] = spec_drafts
            spec_result["t_spec"] = time.perf_counter() - t0

        t_round_start = time.perf_counter()
        th_v = threading.Thread(target=verify_task)
        th_s = threading.Thread(target=spec_task)
        th_v.start()
        th_s.start()
        th_v.join()
        th_s.join()
        t_round_wallclock = time.perf_counter() - t_round_start

        target_logits = verify_result["logits"]
        t_verify = verify_result["t_verify"]
        t_spec = spec_result["t_spec"]

        # Process verify result
        start = current_prefix_len - 1
        block = target_logits[start : start + k + 1]
        accepted, correction = greedy_accept(block, draft_tokens)
        new_tokens = accepted + [correction]
        if len(accepted_all) + len(new_tokens) > n_gen:
            new_tokens = new_tokens[: n_gen - len(accepted_all)]
        accepted_all.extend(new_tokens)

        # Can we reuse the speculative drafts?
        # Requires: all k drafts accepted AND draft's guess == target's correction
        spec_hit = (len(accepted) == k) and (spec_result["guess"] == correction)

        if len(accepted_all) >= n_gen:
            t_draft = 0.0
            draft_tokens_for_next = []
        elif spec_hit:
            # KV already advanced by k+1+k = 2k+1 tokens after spec_task.
            # But we need KV at: prev_prefix_len + k + 1 (= current n_tokens BEFORE spec).
            # Actually: spec_task started with n_tokens = initial_n = current_prefix_len + k
            # spec_task advanced to: initial_n + 1 + k = current_prefix_len + k + 1 + k  -- wait
            # Let me re-check. Before spec_task: KV has [prefix, drafts] of length P + k.
            # spec_task: sample guess, eval(guess) → KV = P+k+1. Then k draftings → KV = P+k+1+k.
            # After accept all k + correction (= guess), NEW prefix = prefix + drafts + correction.
            # This new prefix has length P + k + 1. draft.current_prefix_len should be P + k + 1.
            # Then draft_tokens_for_next = spec_drafts (which are predictions at P+k+1, ..., P+2k).
            # KV is already at P + k + 1 + k. ✓
            draft.current_prefix_len = current_prefix_len + k + 1
            draft_tokens_for_next = spec_result["spec_drafts"]
            t_draft = 0.0  # overlapped — no extra cost
        else:
            # Spec miss. Rollback to before spec_task, then advance properly.
            draft.llm.n_tokens = spec_result["spec_start_kv"]
            t0 = time.perf_counter()
            draft_tokens_for_next = draft.advance(len(accepted), correction, k)
            t_draft = time.perf_counter() - t0

        draft_tokens = draft_tokens_for_next

        rounds += 1
        # Actual round time = max(verify, spec) + (t_draft if miss else 0)
        effective_round_ms = t_round_wallclock * 1000 + t_draft * 1000

        if log_per_round:
            print(f"  round {rounds}: verify {t_verify*1000:.1f}ms, spec {t_spec*1000:.1f}ms "
                  f"(parallel), wall {t_round_wallclock*1000:.1f}ms, "
                  f"accepted {len(accepted)}/{k}, spec_hit={spec_hit}, "
                  f"post_draft={t_draft*1000:.1f}ms")

        trace.append({
            "round": rounds,
            "accepted": len(accepted),
            "t_verify_ms": t_verify * 1000,
            "t_spec_ms": t_spec * 1000,
            "t_wallclock_ms": t_round_wallclock * 1000,
            "t_post_draft_ms": t_draft * 1000,
            "spec_hit": spec_hit,
            "emitted": len(new_tokens),
        })

    t_total = time.perf_counter() - t_total_start
    spec_hit_rate = sum(1 for r in trace if r.get("spec_hit")) / rounds if rounds > 0 else 0

    return {
        "mode": "async",
        "n_gen_requested": n_gen,
        "n_gen_actual": len(accepted_all),
        "rounds": rounds,
        "total_ms": t_total * 1000,
        "ms_per_token": (t_total * 1000) / len(accepted_all) if accepted_all else 0,
        "tok_s": len(accepted_all) / t_total if t_total > 0 else 0,
        "mean_accepted_per_round": sum(r["accepted"] for r in trace) / rounds,
        "spec_hit_rate": spec_hit_rate,
        "accepted_tokens": accepted_all,
        "trace": trace,
    }


# ── Prompt-Lookup Decoding (PLD) — draft-free speculative ─────────────
class PromptLookupDrafter:
    """Proposes k tokens by searching an n-gram pool built from prompt + generated history.

    Algorithm (Saxena 2023):
      - Maintain a token buffer = prompt + all generated tokens so far.
      - To draft next k tokens: take the last `n_gram_max` tokens of the buffer
        as a query, scan the buffer for matches (right-to-left, largest n-gram first),
        and return up to `k` subsequent tokens from the match.
      - If no n-gram of size >= n_gram_min matches, return empty (fallback to target-only).

    Cost: O(buffer_len × n_gram_max) per propose. For ctx 4096, ~40μs on A76. Negligible vs NPU verify (~100ms).
    """
    def __init__(self, n_gram_min=2, n_gram_max=3):
        self.n_gram_min = n_gram_min
        self.n_gram_max = n_gram_max

    def propose(self, token_buffer, k):
        """Return up to k draft tokens, or empty list if no match."""
        if k <= 0:
            return []
        L = len(token_buffer)
        # Try largest n-gram first
        for n in range(self.n_gram_max, self.n_gram_min - 1, -1):
            if L <= n:
                continue
            query = tuple(token_buffer[-n:])
            # Search right-to-left, skipping the trailing query itself
            for i in range(L - n - 1, -1, -1):
                if tuple(token_buffer[i : i + n]) == query:
                    candidate = token_buffer[i + n : i + n + k]
                    if candidate:
                        return list(candidate)
        return []


def run_pld(target: TargetVerifier, prompt_tokens, n_gen, k,
             n_gram_min=2, n_gram_max=3, log_per_round=False):
    """PLD: target-only (no external draft) speculative decoding.

    Each round:
      1. Propose up to k tokens via n-gram lookup in token_buffer.
      2. Target verify (prefix + proposed).
      3. Greedy accept prefix match + correction.
      4. If no proposal possible (empty n-gram match), do single-step target decode.
    """
    drafter = PromptLookupDrafter(n_gram_min=n_gram_min, n_gram_max=n_gram_max)
    accepted_all = []
    trace = []
    total_ngram_hits = 0
    t_total_start = time.perf_counter()

    while len(accepted_all) < n_gen:
        current_prefix = prompt_tokens + accepted_all
        token_buffer = current_prefix  # single buffer

        # Propose
        proposal = drafter.propose(token_buffer, k)
        has_proposal = len(proposal) > 0

        # Feed prefix + proposal (or just prefix if no proposal)
        full = current_prefix + proposal
        t0 = time.perf_counter()
        target_logits = target.verify(full)
        t_verify = time.perf_counter() - t0

        if has_proposal:
            # Align: target's prediction at position len(prefix)+i is at row len(prefix)+i-1
            start = len(current_prefix) - 1
            n_rows = len(proposal) + 1  # proposals + 1 correction slot
            if start + n_rows > target_logits.shape[0]:
                raise RuntimeError(f"logits too short")
            block = target_logits[start : start + n_rows]
            accepted, correction = greedy_accept(block, proposal)
            new_tokens = accepted + [correction]
        else:
            # No n-gram match: just take target's next token
            next_tok = int(np.argmax(target_logits[-1]))
            new_tokens = [next_tok]
            accepted = []

        if len(accepted_all) + len(new_tokens) > n_gen:
            new_tokens = new_tokens[: n_gen - len(accepted_all)]
        accepted_all.extend(new_tokens)

        if has_proposal:
            total_ngram_hits += 1

        if log_per_round:
            print(f"  proposal={proposal}, accepted={len(accepted)}/{len(proposal)}, "
                  f"+{len(new_tokens)}, verify={t_verify*1000:.1f}ms, "
                  f"total={len(accepted_all)}")

        trace.append({
            "round": len(trace) + 1,
            "proposal_len": len(proposal),
            "accepted": len(accepted),
            "emitted": len(new_tokens),
            "t_verify_ms": t_verify * 1000,
            "has_proposal": has_proposal,
        })

    t_total = time.perf_counter() - t_total_start
    return {
        "mode": "pld",
        "n_gen_requested": n_gen,
        "n_gen_actual": len(accepted_all),
        "rounds": len(trace),
        "total_ms": t_total * 1000,
        "ms_per_token": (t_total * 1000) / len(accepted_all) if accepted_all else 0,
        "tok_s": len(accepted_all) / t_total if t_total > 0 else 0,
        "ngram_hit_rate": total_ngram_hits / len(trace) if trace else 0,
        "mean_accepted_per_round": sum(r["accepted"] for r in trace) / len(trace) if trace else 0,
        "mean_emitted_per_round": sum(r["emitted"] for r in trace) / len(trace) if trace else 0,
        "accepted_tokens": accepted_all,
        "trace": trace,
    }


# ── NPU-only baseline for comparison ──────────────────────────────────
def run_npu_only(target: TargetVerifier, prompt_tokens, n_gen):
    """Pure target-side greedy decode, no draft."""
    accepted_all = []
    t_total_start = time.perf_counter()

    while len(accepted_all) < n_gen:
        full = prompt_tokens + accepted_all
        target_logits = target.verify(full)
        next_tok = int(np.argmax(target_logits[-1]))
        accepted_all.append(next_tok)

    t_total = time.perf_counter() - t_total_start
    return {
        "mode": "npu_only",
        "n_gen_actual": len(accepted_all),
        "total_ms": t_total * 1000,
        "ms_per_token": (t_total * 1000) / len(accepted_all),
        "tok_s": len(accepted_all) / t_total,
        "accepted_tokens": accepted_all,
    }


# ── Main ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model_path", required=True)
    p.add_argument("--draft_model_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--prompt", default="The key finding of this work is that")
    p.add_argument("--n_gen", type=int, default=64)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--mode", default="sequential",
                    choices=["sequential", "async", "pld", "npu_only", "all"])
    p.add_argument("--n_gram_min", type=int, default=2)
    p.add_argument("--n_gram_max", type=int, default=3)
    p.add_argument("--draft_threads", type=int, default=4)
    p.add_argument("--max_context_len", type=int, default=4096)
    p.add_argument("--output_json", default=None)
    p.add_argument("--log_per_round", action="store_true")
    args = p.parse_args()

    print(f"Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    prompt_tokens = tok.encode(args.prompt, add_special_tokens=True)
    print(f"Prompt: {repr(args.prompt)} → {len(prompt_tokens)} tokens")

    print(f"Loading target (RKLLM)...")
    target = TargetVerifier(args.target_model_path, max_context_len=args.max_context_len)

    print(f"Loading draft (llama.cpp)...")
    draft = DraftEngine(args.draft_model_path, n_threads=args.draft_threads,
                         n_ctx=args.max_context_len)

    results = {}

    try:
        modes_to_run = ["npu_only", "sequential", "async"] if args.mode == "all" else [args.mode]

        for mode in modes_to_run:
            print(f"\n{'='*60}")
            print(f"Running mode={mode}, k={args.k}, n_gen={args.n_gen}")
            print('='*60)
            if mode == "sequential":
                res = run_sequential(draft, target, prompt_tokens, args.n_gen, args.k,
                                      log_per_round=args.log_per_round)
            elif mode == "async":
                res = run_async(draft, target, prompt_tokens, args.n_gen, args.k,
                                 log_per_round=args.log_per_round)
            elif mode == "npu_only":
                res = run_npu_only(target, prompt_tokens, args.n_gen)
            elif mode == "pld":
                res = run_pld(target, prompt_tokens, args.n_gen, args.k,
                               n_gram_min=args.n_gram_min, n_gram_max=args.n_gram_max,
                               log_per_round=args.log_per_round)
            else:
                continue

            print(f"\n[{mode}] tok/s = {res['tok_s']:.2f}, ms/tok = {res['ms_per_token']:.2f}, "
                  f"total = {res['total_ms']:.0f}ms ({res['n_gen_actual']} tokens)")
            if "rounds" in res:
                print(f"  rounds = {res['rounds']}, avg accepted/round = {res['mean_accepted_per_round']:.2f}")
            if "spec_hit_rate" in res:
                print(f"  spec_hit_rate = {res['spec_hit_rate']:.2%}")
            results[mode] = res

        # Compare
        if "sequential" in results and "npu_only" in results:
            print(f"\n=== Sequential speedup vs NPU-only ===")
            print(f"  {results['sequential']['tok_s'] / results['npu_only']['tok_s']:.3f}×")
        if "async" in results and "npu_only" in results:
            print(f"\n=== Async speedup vs NPU-only ===")
            print(f"  {results['async']['tok_s'] / results['npu_only']['tok_s']:.3f}×")
        if "sequential" in results and "async" in results:
            print(f"\n=== Async speedup vs Sequential ===")
            print(f"  {results['async']['tok_s'] / results['sequential']['tok_s']:.3f}×")

        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            # Strip large fields for output
            clean = {}
            for m, r in results.items():
                c = {k: v for k, v in r.items() if k not in ("accepted_tokens", "trace")}
                clean[m] = c
            with open(args.output_json, "w") as f:
                json.dump({"args": vars(args), "results": clean}, f, indent=2)
            print(f"\nSaved: {args.output_json}")

    finally:
        target.close()


if __name__ == "__main__":
    main()
