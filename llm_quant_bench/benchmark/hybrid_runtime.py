"""
Greedy hybrid speculative decoding runtime prototype.

This is a systems scaffold that connects:
- CPU draft generation
- RKLLM NPU verification
- context-aware scheduling policy

Current status:
- Designed for architecture validation and implementation bring-up
- Uses greedy acceptance for now
- Exact stochastic speculative decoding can be added on top of the same interfaces

Example:
    python benchmark/hybrid_runtime.py \
        --target_model_path ./rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm \
        --draft_model_path ./models/Llama-3.2-1B-Instruct \
        --tokenizer_path ./models/Llama-3.2-1B-Instruct \
        --prompt "Hello" \
        --max_new_tokens 32
"""
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from llm_quant_bench.analysis.hybrid_scheduler_policy import (
    PolicyDecision,
    choose_policy,
    load_verify_profile,
)
from llm_quant_bench.eval.ppl_rkllm import destroy_model, get_logits, init_model


@dataclass
class RoundTrace:
    round_idx: int
    context_len: int
    requested_mode: str
    executed_mode: str
    k: Optional[int]
    accepted_tokens: int
    emitted_tokens: int
    latency_ms: float
    text_fragment: str


class BaseDraftModel:
    def generate_draft(self, prefix_ids: Sequence[int], k: int) -> List[int]:
        raise NotImplementedError

    def greedy_next(self, prefix_ids: Sequence[int]) -> int:
        raise NotImplementedError


class HuggingFaceDraftModel(BaseDraftModel):
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

    def _next_token(self, prefix_ids: Sequence[int]) -> int:
        import torch

        input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        return int(torch.argmax(logits[0, -1, :]).item())

    def generate_draft(self, prefix_ids: Sequence[int], k: int) -> List[int]:
        tokens = list(prefix_ids)
        draft = []
        for _ in range(k):
            nxt = self._next_token(tokens)
            draft.append(nxt)
            tokens.append(nxt)
        return draft

    def greedy_next(self, prefix_ids: Sequence[int]) -> int:
        return self._next_token(prefix_ids)


class RkllmTargetVerifier:
    def __init__(self, model_path: str, max_context_len: int = 4096):
        self.model_path = model_path
        self.max_context_len = max_context_len
        self.handle = init_model(model_path, max_context_len=max_context_len)

    def close(self):
        if self.handle is not None:
            destroy_model(self.handle)
            self.handle = None

    def get_full_logits(self, token_ids: Sequence[int]) -> Optional[np.ndarray]:
        return get_logits(self.handle, list(token_ids))

    def greedy_next(self, prefix_ids: Sequence[int]) -> int:
        logits = self.get_full_logits(prefix_ids)
        if logits is None:
            raise RuntimeError("RKLLM get_logits failed")
        row = self._next_token_row(logits, len(prefix_ids))
        return int(np.argmax(row))

    @staticmethod
    def _next_token_row(logits: np.ndarray, n_tokens: int) -> np.ndarray:
        """
        Return the row to use for next-token prediction from a full-sequence logits tensor.

        Empirical RKLLM behavior may be:
        - num_logits == n_tokens
        - num_logits == n_tokens - 1

        We use the last available row as the next-step proxy for this greedy prototype.
        """
        _ = n_tokens
        return logits[-1]

    @staticmethod
    def predicted_rows_for_input_tokens(
        logits: np.ndarray,
        input_len: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Normalize RKLLM output so that row i predicts input token at position i+1.

        Returns:
        - normalized logits matrix
        - offset such that normalized[row_idx] predicts token input_ids[row_idx + 1]
        """
        num_logits = logits.shape[0]
        if num_logits == input_len:
            return logits[:-1], 1
        if num_logits == input_len - 1:
            return logits, 1
        raise ValueError(f"Unexpected logits shape: num_logits={num_logits}, input_len={input_len}")

    def greedy_verify(
        self,
        prefix_ids: Sequence[int],
        draft_ids: Sequence[int],
    ) -> Tuple[int, Optional[int]]:
        """
        Greedy verification:
        - Accept drafted token while target argmax matches draft token
        - On first mismatch, return the target greedy token as correction

        Returns:
        - number of accepted drafted tokens
        - correction token if mismatch happened, else None
        """
        combined = list(prefix_ids) + list(draft_ids)
        logits = self.get_full_logits(combined)
        if logits is None:
            raise RuntimeError("RKLLM get_logits failed during verify")

        normalized, offset = self.predicted_rows_for_input_tokens(logits, len(combined))
        prefix_len = len(prefix_ids)

        accepted = 0
        correction = None
        for i, draft_tok in enumerate(draft_ids):
            input_pos = prefix_len + i
            row_idx = input_pos - offset
            if row_idx < 0 or row_idx >= normalized.shape[0]:
                raise IndexError(
                    f"Verification alignment failed: row_idx={row_idx}, "
                    f"normalized_rows={normalized.shape[0]}"
                )
            target_tok = int(np.argmax(normalized[row_idx]))
            if target_tok == draft_tok:
                accepted += 1
            else:
                correction = target_tok
                break

        return accepted, correction


class HybridRuntime:
    def __init__(
        self,
        tokenizer,
        draft_model: BaseDraftModel,
        target_verifier: RkllmTargetVerifier,
        verify_profile: Dict[int, Dict[int, float]],
        cpu_target_model: Optional[BaseDraftModel] = None,
    ):
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.target_verifier = target_verifier
        self.verify_profile = verify_profile
        self.cpu_target_model = cpu_target_model

    def _cpu_only_step(self, prefix_ids: List[int]) -> List[int]:
        if self.cpu_target_model is None:
            raise RuntimeError("CPU-only mode requested but no cpu_target_model is configured")
        return [self.cpu_target_model.greedy_next(prefix_ids)]

    def _npu_only_step(self, prefix_ids: List[int]) -> List[int]:
        return [self.target_verifier.greedy_next(prefix_ids)]

    def _hybrid_step(
        self,
        prefix_ids: List[int],
        acceptance_hint: float,
    ) -> Tuple[List[int], PolicyDecision]:
        decision = choose_policy(
            context=len(prefix_ids),
            acceptance=acceptance_hint,
            verify_profile=self.verify_profile,
        )

        if decision.best_mode != "Hybrid":
            if decision.best_mode == "CPU-only":
                return self._cpu_only_step(prefix_ids), decision
            return self._npu_only_step(prefix_ids), decision

        k = decision.best_k or 1
        draft_ids = self.draft_model.generate_draft(prefix_ids, k)
        accepted, correction = self.target_verifier.greedy_verify(prefix_ids, draft_ids)

        emitted = list(draft_ids[:accepted])
        if accepted < len(draft_ids) and correction is not None:
            emitted.append(correction)

        if not emitted:
            emitted.append(self.target_verifier.greedy_next(prefix_ids))

        return emitted, decision

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        initial_acceptance: float = 0.90,
    ) -> Tuple[str, List[RoundTrace]]:
        prefix_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids: List[int] = []
        traces: List[RoundTrace] = []
        acceptance_hint = initial_acceptance
        round_idx = 0

        while len(generated_ids) < max_new_tokens:
            full_prefix = prefix_ids + generated_ids
            t0 = time.perf_counter()
            emitted, decision = self._hybrid_step(full_prefix, acceptance_hint)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            remaining = max_new_tokens - len(generated_ids)
            emitted = emitted[:remaining]
            generated_ids.extend(emitted)

            # Simple online hint update for future rounds
            if decision.best_mode == "Hybrid" and decision.best_k:
                acceptance_hint = 0.7 * acceptance_hint + 0.3 * (len(emitted) / decision.best_k)

            fragment = self.tokenizer.decode(emitted, skip_special_tokens=False)
            traces.append(RoundTrace(
                round_idx=round_idx,
                context_len=len(full_prefix),
                requested_mode=decision.best_mode,
                executed_mode=decision.best_mode,
                k=decision.best_k,
                accepted_tokens=len(emitted),
                emitted_tokens=len(emitted),
                latency_ms=elapsed_ms,
                text_fragment=fragment,
            ))
            round_idx += 1

        final_text = self.tokenizer.decode(prefix_ids + generated_ids, skip_special_tokens=False)
        return final_text, traces


def build_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path)


def main():
    parser = argparse.ArgumentParser(description="Greedy hybrid speculative runtime prototype")
    parser.add_argument("--target_model_path", required=True, help="Path to target .rkllm model")
    parser.add_argument("--draft_model_path", required=True, help="Path to draft HF model")
    parser.add_argument("--tokenizer_path", required=True, help="Tokenizer path")
    parser.add_argument("--verify_json", default=None, help="Merged verify profile JSON")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_context_len", type=int, default=4096)
    parser.add_argument("--cpu_target_model_path", default=None,
                        help="Optional CPU target model for CPU-only policy execution")
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.tokenizer_path)
    verify_profile = load_verify_profile(args.verify_json)
    draft_model = HuggingFaceDraftModel(args.draft_model_path)
    cpu_target_model = None
    if args.cpu_target_model_path:
        cpu_target_model = HuggingFaceDraftModel(args.cpu_target_model_path)

    verifier = RkllmTargetVerifier(args.target_model_path, max_context_len=args.max_context_len)
    runtime = HybridRuntime(
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_verifier=verifier,
        verify_profile=verify_profile,
        cpu_target_model=cpu_target_model,
    )

    try:
        text, traces = runtime.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print("\n=== Round Trace ===")
        for tr in traces:
            print(
                f"round={tr.round_idx} ctx={tr.context_len} mode={tr.executed_mode} "
                f"k={tr.k} emitted={tr.emitted_tokens} latency_ms={tr.latency_ms:.2f} "
                f"fragment={tr.text_fragment!r}"
            )
        print("\n=== Final Text ===")
        print(text)
    finally:
        verifier.close()


if __name__ == "__main__":
    main()
