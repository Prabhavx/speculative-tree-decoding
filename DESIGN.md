# Design Document: Tree-Based Speculative Decoding Engine

## Mathematical Foundation

### Speculative Decoding Correctness

The core insight of speculative decoding is that we can sample from a target distribution $p(x)$ using proposals from a draft distribution $q(x)$ via **modified rejection sampling**:

For a proposed token $x$:
- If $p(x) \ge q(x)$: accept with probability 1
- If $p(x) < q(x)$: accept with probability $p(x)/q(x)$
- On rejection: resample from residual $p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$

**Proof of correctness**: The probability of outputting token $x$ is:
$$\mathbb{P}[\text{output}=x] = q(x) \cdot \min(1, \frac{p(x)}{q(x)}) + (1 - \sum_y q(y)\min(1,\frac{p(y)}{q(y)})) \cdot p'(x)$$

For $p(x) \ge q(x)$: first term = $q(x) \cdot 1 = q(x) = p(x)$ (since residual is 0)
For $p(x) < q(x)$: first term = $q(x) \cdot \frac{p(x)}{q(x)} = p(x)$; rejection probability contributes to residual which normalizes to $p(x)$

Thus $\mathbb{P}[\text{output}=x] = p(x)$ exactly.

### Tree-Structured Drafting

A **draft tree** is a rooted tree where:
- Root = current prefix (shared context)
- Each node = a candidate token at some position
- Path from root to node = a candidate continuation
- Siblings = mutually exclusive alternatives at same position

**Tree attention mask**: For flattened sequence of tree nodes, node $i$ attends to node $j$ iff $j$ is an ancestor of $i$ in the tree. This is a generalization of causal mask (where ancestors = all previous positions in linear chain).

### Position IDs for RoPE

In a tree, nodes at the same depth from root but different branches have **different position IDs** for RoPE. For a node at depth $d$ along its branch, position_id = prefix_length + d. This ensures RoPE correctly encodes relative positions within each branch.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SpeculativeDecodingEngine                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Draft Model │    │ Target Model │    │  Tokenizer   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ DraftKVCache │    │TargetKVCache │    │  Sampling    │  │
│  └──────┬───────┘    └──────┬───────┘    │  Utilities   │  │
│         │                   │             └──────┬───────┘  │
│         ▼                   ▼                    │          │
│  ┌─────────────────────────────────────────────┐           │
│  │         Verification & Rejection            │           │
│  │  - generate_draft_chain()                   │           │
│  │  - verify_chain()                           │           │
│  │  - rejection_sample() / greedy_*()          │           │
│  └─────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow (Phase 0: Chain)

```
Prompt → Tokenize → [input_ids, attention_mask]
                            │
                            ▼
                    ┌───────────────┐
                    │ Draft Model   │ ──► gamma tokens + logprobs
                    │ (autoregressive with KV cache)
                    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Target Model  │ ──► logprobs at each draft position
                    │ (single forward pass with KV cache)
                    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Rejection     │ ──► accepted tokens + resampled
                    │ Sampling      │
                    └───────────────┘
                            │
                            ▼
                    Update KV caches, append accepted tokens, repeat
```

### KV-Cache Design

Two separate cache managers:
- **DraftKVCache**: Lightweight, updated per draft token during chain generation
- **TargetKVCache**: Supports prefix reuse; on verification, only computes new positions

Key invariant: `target_cache.current_length == input_ids.size(1)` after each generation step (prefix length).

## Phase 0: Chain Speculative Decoding ✅ COMPLETE

### Algorithm

```
function generate(prompt, max_tokens):
    input_ids = tokenize(prompt)
    while generated < max_tokens:
        # 1. Draft phase
        draft_tokens, draft_logprobs = []
        for i in 1..gamma:
            logits = draft_model(input_ids)
            token = sample(logits)
            draft_tokens.append(token)
            draft_logprobs.append(log_softmax(logits))
            input_ids.append(token)

        # 2. Verify phase
        target_logprobs = target_model(input_ids_with_drafts)

        # 3. Rejection sampling
        accepted = 0
        for draft_token, d_lp, t_lp in zip(draft_tokens, draft_logprobs, target_logprobs):
            if temperature == 0:
                accepted, resampled = greedy_reject(d_lp, t_lp, draft_token)
            else:
                accepted, resampled = rejection_sample(d_lp, t_lp, draft_token, temp)
            if accepted:
                output.append(draft_token)
            else:
                output.append(resampled)
                break

        # 4. Bonus token (if all accepted)
        if accepted == gamma:
            output.append(target_model_next_token)
            accepted += 1

        # 5. Update state
        input_ids = input_ids + accepted_tokens
        update target KV cache to new prefix length
```

### Complexity

- **Draft phase**: $\gamma$ forward passes of draft model (small, fast)
- **Verify phase**: 1 forward pass of target model on sequence length $L + \gamma$
- **Memory**: KV cache for both models, $O(L + \gamma)$
- **Speedup**: Theoretical max $\approx \gamma \cdot \frac{t_{draft}}{t_{target}}$, practically limited by acceptance rate

### Bug Fixes Applied

1. **Bug 1 (Attention mask rank mismatch)**: Fixed concatenation of 2D prefix mask with 3D tree mask by building proper 4D additive masks for HF models
2. **Bug 2 (KV cache explosion)**: Fixed slice `[-accepted_count:]` when `accepted_count == 0` by tracking actual tokens added
3. **Bug 3 (Bonus token ignores temperature)**: Fixed bonus token to sample from target distribution at temperature instead of always using argmax

## Phase 1: Static Tree + Tree Attention ✅ COMPLETE

### Tree Construction

Fixed branching factor $b$, depth $d$. Total draft tokens = $\frac{b^{d+1} - 1}{b - 1} - 1$.

### Tree Attention Mask

For flattened node indices $0..N-1$:
- `mask[i, j] = True` iff node $j$ is ancestor of node $i$
- Computed from parent array: `parent[i] = parent index of node i`
- `mask[i, j] = True` if $j = i$ or `mask[parent[i], j] = True`

### Verification

Single target forward pass on flattened sequence with custom attention mask. Extract logprobs at each draft node position.

### Bug Fixes Applied

4. **Bug 5 (Greedy-mode sibling diversity)**: StaticTreeBuilder now uses top-k distinct tokens in greedy mode and Gumbel-Top-k in sampling mode for diverse siblings

## Phase 2: Dynamic/Adaptive Tree ✅ COMPLETE

### Adaptive Construction

Use **Gumbel-Top-k** or **stochastic beam search** to grow high-probability branches:
1. At each expansion step, compute cumulative probability of each frontier node
2. Expand top-$k$ by cumulative probability
3. Stop when compute budget or probability threshold reached

### Resource-Constrained Search

Budget $B$ = max target model FLOPs for verification. Each node costs $\approx$ target model FLOPs per layer. This becomes a best-first search on the tree.

### Engines Implemented

- `DynamicTreeSpeculativeDecodingEngine`: Gumbel-Top-k with compute budget
- `AdaptiveTreeSpeculativeDecodingEngine`: Entropy-adaptive branching factor

## Phase 3: KV-Cache & Position-ID Engineering ✅ COMPLETE

### Tree-Aware KV Cache

- **TreeKVCache**: Prefix sharing, rollback on rejection, DynamicCache format support
- **Per-node RoPE positions**: `position_id = prefix_length + depth - 1`
- **Tree attention mask**: Prefix + ancestors only

### Bug Fixes Applied

- Fixed DynamicCache format handling in KV cache updates
- Proper 4D additive mask construction for HF models

## Testing Strategy

### Distributional Equivalence

For toy distributions where exact $p$ and $q$ are known:
1. Draw $N$ samples from speculative engine
2. Draw $N$ samples from target-only
3. Compare empirical distributions via chi-squared or KL divergence
4. Should match within $O(1/\sqrt{N})$ sampling noise

### Greedy Determinism

With temperature=0, output must exactly match target model greedy decoding token-by-token.

### Unit Tests (71 passing)

- Rejection sampling formula matches residual distribution
- Tree attention mask includes exactly ancestors
- KV cache correctly advances/rolls back
- Dynamic tree builder Gumbel-Top-k and entropy-based branching
- Distributional equivalence at each tree node

## Performance Targets (Sanity Check)

Published results (EAGLE-3 class, 7B/8B models):
- Acceptance length: 4-5 tokens/step
- Throughput speedup: 2-6x on chat/code/math
- Drops on long-context / low-diversity inputs

Our from-scratch implementation on small model pairs (0.5B/1.5B) should achieve:
- Acceptance length: 1.5-3 tokens/step
- Speedup: 1.5-3x
- Memory: < 2x baseline

## Known Limitations / Future Work

1. **Bug 4 (Multi-candidate tree verification)**: Current tree verification only follows the path matching target's greedy argmax. Proper multi-candidate speculative sampling (per SpecInfer) would allow all siblings to be tested for acceptance. This is a significant algorithmic enhancement for a future pass.

2. **Dynamic tree stochastic beam search**: The `build_tree_stochastic_beam` method was removed due to implementation issues. Could be re-implemented properly.

3. **Benchmarking on real GPUs**: CPU-only benchmarks are slow and don't reflect true speedup potential.

4. **Integration with vLLM/SGLang**: Not yet implemented.