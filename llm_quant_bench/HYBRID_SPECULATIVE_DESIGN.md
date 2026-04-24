# Hybrid Speculative Decoding for Edge NPU: Paper-Grade System Design

## 1. Problem Statement

### 1.1 Observation from our characterization

On RK3588, autoregressive decode on NPU slows down linearly with context length:

- `decode_ms = 50.2 + 0.00926 * ctx`
- At `ctx=4096`, NPU decode falls to `11.3 tok/s`
- CPU decode remains nearly context-invariant and overtakes NPU at long context

Our microbench shows the root cause is not KV bandwidth but **M=1 attention matmul under-utilization**:

- Attention-like NPU matmul `(1, 2048) x (2048, ctx)` scales linearly with `ctx`
- Sustained throughput is only about `11 GFLOPS` vs `3 TFLOPS` FP16 peak
- Effective utilization is about `0.37%`

Therefore, the edge-NPU decode bottleneck is:

> The target model is forced into a sequence of tiny `M=1` decode steps, which wastes NPU parallelism.

### 1.2 Design opportunity

Speculative decoding naturally creates a **multi-token verification phase**.
If the target model can verify `k` drafted tokens in one call, then the target-side attention changes from:

- baseline decode: one `M=1` step repeated `k` times

to:

- speculative verify: one `M=k` scoring call over the drafted suffix

This is exactly the missing parallelism the NPU needs.

### 1.3 System hypothesis

We propose:

> Use a small CPU-resident draft model for cheap token proposals, and use the NPU-resident target model only for batched verification, where the target sees `M=k` instead of `M=1`.

If the NPU is fundamentally bad at single-token decode but better at multi-token scoring, this hybridization should outperform both:

- CPU-only autoregressive generation for long context
- NPU-only autoregressive generation for long context


## 2. Research Questions

The system is designed to answer four paper-level questions.

### RQ1. Can RKLLM NPU perform true multi-token verification?

Feasibility criterion:

- `GET_LOGITS + TOKEN_INPUT` accepts `n_tokens > 1`
- Returned logits shape is compatible with scoring a suffix of length `k`
- Verify latency grows sublinearly relative to `k` serial target decode steps

### RQ2. When does CPU-draft / NPU-verify beat the best monolithic backend?

We expect:

- Short context: plain NPU decode or plain CPU decode may still win
- Mid/long context: hybrid wins as target-side verification amortizes NPU fixed cost and improves compute utilization

### RQ3. What draft length is optimal as a function of context?

The optimal `k` should depend on:

- context length
- draft acceptance rate
- CPU draft throughput
- NPU verify batching efficiency

### RQ4. Is one static backend ever optimal across the full context range?

Likely no. This motivates a **dynamic dispatch policy**:

- pure NPU for short context or low verify profitability
- hybrid for long context and sufficiently high acceptance
- optionally CPU-only fallback when NPU verify batching is ineffective


## 3. Design Goals

### Functional goals

- Preserve exact target-model distribution up to standard speculative-decoding equivalence
- Exploit NPU only where it benefits from batch-like parallelism
- Support context lengths up to RK3588 target limit

### Systems goals

- Avoid modifying RKLLM internals
- Treat RKLLM as a black-box target scorer with logits access
- Keep draft-side implementation fully open and controllable
- Minimize cross-device synchronization frequency

### Evaluation goals

- Demonstrate end-to-end throughput gain
- Explain gain using first-principles utilization and latency decomposition
- Characterize failure modes, not only successes


## 4. System Architecture

## 4.1 Components

### Draft engine

- Backend: CPU
- Runtime: MNN or lightweight CPU inference runtime
- Model: small draft LM, e.g. Llama/Qwen class at 0.1B-0.5B, or a quantized sub-1B variant
- Role: generate `k` candidate tokens cheaply under the current context

### Target engine

- Backend: RK3588 NPU
- Runtime: RKLLM
- Model: full target LM, e.g. Llama-3.2-1B W8A8
- Role: verify drafted suffix by batched scoring through `GET_LOGITS`

### Scheduler

- Maintains generation state
- Chooses mode: CPU-only, NPU-only, or hybrid speculative
- Chooses draft length `k`
- Updates policy online based on observed acceptance and latency

### Token/state manager

- Holds canonical accepted prefix
- Builds target verification sequence from accepted prefix plus drafted suffix
- Tracks context window and truncation if needed


## 4.2 Execution model

At each outer iteration:

1. CPU draft model proposes `k` tokens
2. NPU target scores the drafted segment in one batched verify call
3. Scheduler computes longest accepted prefix
4. Accepted drafted tokens are committed
5. If rejection occurs, one correction token is sampled from target distribution
6. Repeat

This converts many target `M=1` decode calls into fewer target `M=k` verify calls.


## 5. Algorithm

## 5.1 Baseline autoregressive decode

For reference, vanilla target decoding repeats:

1. run target on current prefix
2. obtain next-token distribution
3. sample one token
4. append token

This costs one target pass per output token.


## 5.2 Hybrid speculative decode

Let:

- `p` be the target model distribution
- `q` be the draft model distribution
- `x` be the accepted prefix
- `k` be draft length

### Draft phase

The CPU draft model generates:

- `d_1, d_2, ..., d_k ~ q(. | x, d_<i)`

### Verify phase

The NPU target model scores the sequence:

- `x + d_1 + d_2 + ... + d_k`

via a single multi-token logits request.

This yields target logits for each drafted position:

- `p(. | x)`
- `p(. | x, d_1)`
- ...
- `p(. | x, d_<k)`

Depending on exact RKLLM alignment, the logits tensor may have length `k` or `k-1`; the implementation must explicitly align positions.

### Accept/reject rule

Use standard speculative decoding acceptance:

- for each drafted token `d_i`, accept with probability `min(1, p_i(d_i) / q_i(d_i))`
- stop at first rejection
- if rejection occurs at position `j`, sample correction token from residual target distribution

This preserves the target distribution exactly under standard assumptions.

### Edge-oriented simplification

If exact acceptance bookkeeping is too expensive initially, we can implement a staged system:

- Stage A: greedy draft + greedy verify as a feasibility prototype
- Stage B: full stochastic speculative decoding for paper-quality correctness

For the final paper, the exact algorithm should be the default evaluation path.


## 6. RK3588-Specific Verify Design

## 6.0 Runtime reality on this platform

One practical lesson from bring-up is important:

- `/dev/rknpu` is **not** the canonical health check on this machine
- `lsmod | grep rknpu` may also be empty even when NPU execution works
- the kernel-side NPU support may be built-in rather than exposed as a loadable module
- user-space execution can still succeed through the RKNN/RKLLM runtime stack

In other words, the following reasoning is unreliable:

> "I do not see `/dev/rknpu`, therefore RKLLM cannot use the NPU."

For this system, the more reliable checks are:

- whether RKNN matmul microbench runs successfully
- whether RKLLM succeeds with the correct runtime library path
- whether the probe/eval scripts produce output JSON successfully

### 6.0.1 Correct debugging priority

When RKLLM emits:

- `failed to open rknpu module`
- `failed to open rknn device`

this should be treated as a **generic runtime failure message**, not immediate proof that a `/dev/rknpu` node must exist.

The first thing to check is the runtime library path:

```bash
export LD_LIBRARY_PATH=/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:$LD_LIBRARY_PATH
```

This matters because loading the wrong RKNN runtime from the default system path can create ABI mismatches that surface as misleading device-open failures.

### 6.0.2 Practical execution rule

All RKLLM verify experiments in this project should be launched with:

- the project root as current working directory
- the project virtualenv activated
- `PYTHONPATH=.`
- the RKLLM runtime path prepended to `LD_LIBRARY_PATH`

To reduce bring-up errors, use the wrapper script:

- `scripts/run_npu_verify_probe.sh`

## 6.1 Why this should help on this hardware

Our characterization says the NPU loses because decode attention uses `M=1`.

In verify, the target processes `k` drafted positions together, so the dominant attention-like work becomes closer to:

- `Q [k, d] x K^T [d, ctx+k]`

instead of repeating:

- `Q [1, d] x K^T [d, ctx+i]` for `i=1...k`

This improves:

- NPU MAC occupancy
- amortization of dispatch and framework overhead
- effective throughput per accepted output token

### Key insight

The purpose of speculative decoding here is not only fewer target calls.
It is specifically to **reshape target computation into a form the NPU is better at**.


## 6.2 Black-box target interface

We assume only the following RKLLM capabilities:

- token input mode
- logits retrieval mode
- ability to pass `n_tokens > 1`

We do **not** assume:

- direct KV-cache manipulation
- fused draft/verify kernels
- hidden internal scheduler controls
- exposed low-level attention kernels

This is important for the paper contribution:

> The design is deployable even with a closed NPU runtime, as long as batched logits scoring is available.


## 6.3 Verify call shapes

We consider two verify modes.

### Mode V1: Full-prefix scoring

For each verify, pass the full sequence:

- `[accepted_prefix + drafted_suffix]`

Pros:

- simplest implementation
- directly compatible with black-box `GET_LOGITS`

Cons:

- repeats scoring over the accepted prefix every verify iteration
- may increase fixed cost for long contexts

### Mode V2: Prefix-cached scoring

If RKLLM supports prompt caching or history reuse for token inputs:

- prefill accepted prefix once
- verify only drafted suffix against the cached prefix

Pros:

- closer to true speculative verification
- avoids redundant target work on stable prefix

Cons:

- may be unsupported or poorly documented
- correctness/latency behavior must be validated carefully

### Paper strategy

Implement V1 first for robustness.
Treat V2 as an optimization or ablation if runtime support proves usable.


## 7. Latency Model

We need a predictive model to explain when hybrid wins.

## 7.1 Variables

Let:

- `T_draft(k, c)` = CPU time to draft `k` tokens at context `c`
- `T_verify(k, c)` = NPU time to verify drafted suffix of length `k` at context `c`
- `a(c, k)` = expected accepted tokens per verify round
- `T_corr(c)` = expected correction overhead after rejection

Then expected time per accepted output token is:

`E[T/token] = (T_draft + T_verify + T_corr) / a`

Hybrid beats baseline CPU-only if:

`(T_draft + T_verify + T_corr) / a < T_cpu_decode`

Hybrid beats baseline NPU-only if:

`(T_draft + T_verify + T_corr) / a < T_npu_decode`


## 7.2 Expected structure

Based on current findings:

- `T_npu_decode(c)` increases roughly linearly with `c`
- `T_verify(k, c)` should increase with both `c` and `k`, but ideally sublinearly in `k` relative to `k * T_npu_decode(c)`
- `T_draft(k, c)` scales roughly linearly with `k`, but CPU draft is cheap if the model is small

Thus, hybrid improves as long as:

- acceptance is high enough
- verify batching provides enough NPU efficiency gain


## 7.3 Derived metrics for analysis

We recommend reporting:

- `verify_efficiency(k, c) = k * T_npu_decode(c) / T_verify(k, c)`
- `accepted_tokens_per_verify = a(c, k)`
- `effective_target_M_gain = T_npu_decode(c) / (T_verify(k, c) / k)`
- `hybrid_speedup_vs_cpu`
- `hybrid_speedup_vs_npu`

These metrics directly connect the system results back to the `M=1` characterization.


## 8. Scheduler Design

## 8.1 Static scheduler

The simplest policy uses fixed thresholds:

- if `context < c_short`, use NPU-only
- else use hybrid with fixed draft length `k`

This is useful as a baseline but not sufficient for the full paper.


## 8.2 Dynamic scheduler

The paper system should use online adaptation.

State variables:

- current context length `c`
- recent acceptance rate `r_acc`
- recent median draft latency
- recent median verify latency
- estimated effective time per accepted token

Decision outputs:

- backend mode: `CPU-only`, `NPU-only`, `Hybrid`
- draft length `k`

### Candidate policy

At each scheduling epoch:

1. estimate expected token cost for each mode
2. select the mode with minimum estimated cost
3. if `Hybrid` is chosen, select `k` from a discrete set such as `{2, 4, 6, 8}`

### Policy objective

Minimize:

- wall-clock time per emitted target token

subject to:

- exact target semantics
- bounded memory usage


## 8.3 Adaptive draft length selection

The optimal `k` trades off:

- larger `k`: better NPU batch efficiency, fewer target invocations
- smaller `k`: less wasted draft work on rejection, lower verify cost

We recommend a contextual bandit or low-overhead hill-climbing policy:

- probe neighboring `k` values occasionally
- update reward using achieved accepted tokens per second
- converge to per-context optimal `k`

For the paper, even a robust heuristic is enough:

- start at `k=4`
- increase `k` if acceptance remains high and verify efficiency improves
- decrease `k` if rejection rises or verify latency grows too sharply


## 9. Implementation Plan

## 9.1 Phase I: Verify feasibility

Objective:

- prove that RKLLM accepts multi-token `GET_LOGITS` and quantify latency scaling

Artifact:

- `benchmark/npu_verify_probe.py`

Measurements:

- verify length `k in {1, 2, 4, 8, 16}`
- contexts `c in {32, 256, 1024, 4096}`
- logits alignment and shape
- latency and throughput


## 9.2 Phase II: Oracle hybrid simulator

Before full online generation, build an oracle simulator:

- generate target text offline
- emulate draft/verify using logged target probabilities
- sweep `k`, acceptance, and latency parameters

Purpose:

- validate the scheduler design
- identify profitable regions before implementing full runtime integration


## 9.3 Phase III: End-to-end hybrid runtime

Implement:

- CPU draft model loop
- RKLLM verify wrapper
- exact speculative accept/reject logic
- dynamic scheduler

Outputs:

- generated text
- per-round acceptance trace
- per-stage latency breakdown


## 9.4 Phase IV: Optimized runtime variants

Optional enhancements:

- prefix caching / history reuse
- overlap CPU draft of next round with NPU verification of current round
- pin CPU threads for draft and runtime helpers
- preallocated token buffers to reduce Python overhead


## 10. Experimental Plan

## 10.1 Baselines

Required baselines:

- CPU-only target decode
- NPU-only target decode
- CPU-only draft model decode
- Hybrid speculative with fixed `k`
- Hybrid speculative with dynamic `k`

If possible:

- GPU baseline for completeness


## 10.2 Independent variables

- context length: `32` to `4096`
- draft length `k`: `1, 2, 4, 6, 8, 12, 16`
- draft model size
- target quantization variant
- acceptance regime


## 10.3 Dependent variables

- end-to-end output throughput
- TTFT
- time per accepted token
- accepted tokens per verify round
- correction frequency
- energy or average power if available


## 10.4 Ablations

Essential ablations:

- fixed `k` vs adaptive `k`
- greedy speculative vs exact stochastic speculative
- full-prefix verify vs cached-prefix verify
- different draft model families/sizes
- short vs long context


## 10.5 Generalization matrix

To strengthen the paper:

- target models: at least 2 families if possible
- quantization variants: at least 2 target variants
- context buckets: short / medium / long

Even if hardware is fixed to RK3588, model diversity helps.


## 11. Threats and Risks

## 11.1 Risk A: Verify batching exists but gives little speedup

Interpretation:

- RKLLM may internally still execute near-serial logic
- framework overhead may dominate

Paper value remains:

- negative result showing speculative decoding is less effective on closed edge NPUs than expected


## 11.2 Risk B: Acceptance rate is too low

Interpretation:

- draft model too weak
- draft length too aggressive

Mitigation:

- use slightly larger draft model
- adaptive `k`
- context-aware mode switching


## 11.3 Risk C: Full-prefix verify is too expensive

Interpretation:

- without prefix caching, repeated prefix scoring may erase gains

Mitigation:

- investigate RKLLM prompt cache / keep_history support
- if unsupported, frame this as a key interface limitation of closed edge runtimes


## 11.4 Risk D: Python control overhead distorts measurements

Mitigation:

- isolate per-stage overhead
- preallocate buffers
- move hot loop to C++ only if needed after proof of concept


## 12. Expected Contributions

If successful, the paper contributions become:

### C1. Root-cause characterization

- NPU decode slowdown comes from `M=1` under-utilization, not KV bandwidth

### C2. Hardware-aligned system design

- A CPU-draft / NPU-verify hybrid speculative decoder that specifically targets edge-NPU parallelism waste

### C3. Context-aware scheduling insight

- No single backend is optimal across all contexts; dynamic dispatch is necessary

### C4. Empirical feasibility limits of closed edge NPUs

- Quantifies when speculative verification works, and where runtime interfaces block further gains


## 13. Paper Narrative

The story should be:

1. We characterize a surprising edge-NPU pathology
2. We explain it from first principles as `M=1` utilization collapse
3. We derive a systems remedy, not by generic optimization, but by matching the algorithm to the hardware bottleneck
4. We show that speculative decoding is especially valuable on edge NPUs because it creates the missing target-side parallelism
5. We further show that profitability depends on context and acceptance, motivating dynamic dispatch

This narrative is stronger than a pure negative-result paper because it closes the loop from diagnosis to remedy.


## 14. Concrete Next Experiments

Immediate next steps:

1. Run `npu_verify_probe.py` on the real RK3588 device and measure `T_verify(k, c)`
2. Determine whether RKLLM supports usable prefix reuse for token-input verification
3. Build the oracle latency model and identify profitable `(context, k)` regions
4. Select 1-2 draft models that maximize acceptance per CPU cost
5. Implement exact speculative acceptance and end-to-end scheduler


## 15. Success Criteria

We should consider the project successful if all of the following hold:

- multi-token NPU verify is functionally supported
- `T_verify(k, c) < k * T_npu_decode(c)` by a meaningful margin for `k >= 4`
- hybrid beats CPU-only and NPU-only at long context
- dynamic scheduler outperforms any single static mode across the full context range

Even if the end-to-end gain is modest, the work is publishable if it clearly explains:

- why edge-NPU speculative decoding helps less or more than expected
- how closed runtimes shape the achievable design space
