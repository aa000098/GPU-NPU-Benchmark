"""KV-reuse aware hybrid speculative decoding runtime.

Key change from v2: target.verify uses incremental decode (KV reuse).
- Prefill prefix once (V1-cost, but only once per prompt).
- Each spec round: feed k drafts via incremental verify (cheap, GEMM-amortized).
- On reject: rollback via clear + re-prefill of [prefix + accepted] (V1 cost).

If RKLLM's get_logits returns (n_fed, V) shape, then a single
verify_incremental(drafts) call gives us logits[i] = prediction AFTER fed[i].
Combined with the previous step's last logit (prediction at the boundary),
we have k+1 logits to drive accept/reject for k drafts.
"""
import time
import numpy as np

from llm_quant_bench.eval.ppl_rkllm import (
    init_model, get_logits, clear_kv_cache, destroy_model,
)


class TargetVerifierV3:
    """KV-reuse aware verifier."""

    def __init__(self, model_path, max_context_len=4096):
        self.handle = init_model(model_path, max_context_len=max_context_len)
        self.kv_tokens = []

    def prefill(self, prefix_tokens):
        clear_kv_cache(self.handle)
        logits = get_logits(self.handle, list(prefix_tokens), keep_history=1)
        if logits is None:
            raise RuntimeError("prefill failed")
        self.kv_tokens = list(prefix_tokens)
        return logits   # shape (len(prefix), V)

    def verify_incremental(self, new_tokens):
        if not new_tokens:
            return None
        logits = get_logits(self.handle, list(new_tokens), keep_history=1)
        if logits is None:
            raise RuntimeError("incremental verify failed")
        self.kv_tokens.extend(new_tokens)
        return logits   # shape (len(new_tokens), V)

    def rollback_to_length(self, target_len):
        """Truncate KV to first target_len tokens (V1 fallback)."""
        if target_len < 0 or target_len > len(self.kv_tokens):
            raise ValueError(f"invalid rollback {target_len}/{len(self.kv_tokens)}")
        if target_len == len(self.kv_tokens):
            return
        kept = self.kv_tokens[:target_len]
        clear_kv_cache(self.handle)
        if kept:
            get_logits(self.handle, kept, keep_history=1)
        self.kv_tokens = list(kept)

    @property
    def kv_len(self):
        return len(self.kv_tokens)

    def close(self):
        destroy_model(self.handle)


def run_npu_only_kvreuse(target: TargetVerifierV3, prompt_tokens, n_gen):
    t0 = time.perf_counter()
    logits = target.prefill(prompt_tokens)   # (P, V)
    accepted = []
    last_logit = logits[-1]   # predicts position P+1
    while len(accepted) < n_gen:
        nxt = int(np.argmax(last_logit))
        accepted.append(nxt)
        if len(accepted) >= n_gen:
            break
        new_logits = target.verify_incremental([nxt])  # (1, V)
        last_logit = new_logits[-1]
    t = time.perf_counter() - t0
    return {
        "mode": "npu_only_kvreuse",
        "n_gen_actual": len(accepted),
        "total_ms": t * 1000,
        "ms_per_token": t * 1000 / len(accepted),
        "tok_s": len(accepted) / t,
        "accepted_tokens": accepted,
    }


def run_sequential_kvreuse(draft, target: TargetVerifierV3, prompt_tokens, n_gen, k=4):
    """Spec decoding with KV-reuse + V1 fallback on reject.

    Round structure:
        - Have last_logit (predicts next position).
        - Make k drafts.
        - verify_incremental(drafts) → returns logits shape (k, V).
        - Accept logic uses [last_logit] + logits[0..k-1] as k+1 prediction logits;
          but row[i] predicts AFTER drafts[i] (= position+1 after draft i).
          So pre_logit[0] = last_logit (predicts draft[0]'s position)
              pre_logit[i] = logits[i-1]  for i in 1..k  (predicts draft[i]'s position; i=k is the correction-position)
        - Iterate i=0..k-1, accept if argmax(pre_logit[i]) == drafts[i].
        - On first mismatch, correction = argmax(pre_logit[mismatch_index]).
        - If all match, correction = argmax(pre_logit[k]) = argmax(logits[k-1]).
        - KV after verify_incremental: [prefix + accepted_all + drafts(k)].
        - Desired KV after round: [prefix + accepted_all + accepted_in_round + correction].
        - If full accept: rollback not needed; KV needs +1 (correction). verify_incremental([correction]).
        - If partial: rollback to len(prefix) + len(accepted_all) + len(accepted_in_round), then verify_incremental([correction]).
    """
    t_total_start = time.perf_counter()
    P = len(prompt_tokens)
    initial_logits = target.prefill(prompt_tokens)
    last_logit = initial_logits[-1]

    accepted_all = []
    drafts = draft.start(prompt_tokens, k)
    rounds = 0
    rollbacks = 0
    full_accepts = 0
    accept_counts = []

    while len(accepted_all) < n_gen:
        # Send drafts in one shot (KV: prefix + accepted_all → + drafts)
        verify_logits = target.verify_incremental(list(drafts))  # (k, V)

        # Build (k+1) prediction logits
        pre_logits = [last_logit] + [verify_logits[i] for i in range(k)]
        # pre_logits[i] predicts position (P + len(accepted_all) + i + 1)

        accepted_round = []
        correction = None
        all_match = True
        for i in range(k):
            if int(np.argmax(pre_logits[i])) == drafts[i]:
                accepted_round.append(drafts[i])
            else:
                correction = int(np.argmax(pre_logits[i]))
                all_match = False
                break
        if all_match:
            correction = int(np.argmax(pre_logits[k]))
            full_accepts += 1
        accept_counts.append(len(accepted_round))

        new_tokens = accepted_round + [correction]
        # Truncate to n_gen
        if len(accepted_all) + len(new_tokens) > n_gen:
            new_tokens = new_tokens[: n_gen - len(accepted_all)]
            accepted_all.extend(new_tokens)
            rounds += 1
            break

        # KV management
        if all_match:
            # KV is at [prefix + accepted_all + drafts(k)] = correct prefix for next round
            # We need to also feed correction so KV ends at [prefix + new_accepted_all]
            corr_logits = target.verify_incremental([correction])
            last_logit = corr_logits[-1]
        else:
            # KV has prefix+accepted_all+drafts. Roll back to prefix+accepted_all+accepted_round, then add correction.
            target_kv_len = P + len(accepted_all) + len(accepted_round)
            target.rollback_to_length(target_kv_len)
            rollbacks += 1
            # After rollback, get last logit by feeding correction
            corr_logits = target.verify_incremental([correction])
            last_logit = corr_logits[-1]

        accepted_all.extend(new_tokens)
        rounds += 1

        if len(accepted_all) >= n_gen:
            break

        # Generate next k drafts (CPU side advances its own KV)
        drafts = draft.advance(len(accepted_round), correction, k)

    t_total = time.perf_counter() - t_total_start
    return {
        "mode": "sequential_kvreuse",
        "n_gen_actual": len(accepted_all),
        "rounds": rounds,
        "rollbacks": rollbacks,
        "full_accepts": full_accepts,
        "mean_accepted_per_round": float(np.mean(accept_counts)) if accept_counts else 0,
        "total_ms": t_total * 1000,
        "ms_per_token": t_total * 1000 / len(accepted_all) if accepted_all else 0,
        "tok_s": len(accepted_all) / t_total if t_total > 0 else 0,
        "accepted_tokens": accepted_all,
    }
