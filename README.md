# Speculative Tree Decoding

A from-scratch, research-oriented implementation of speculative decoding for
autoregressive language models.

Speculative decoding speeds up text generation by using a smaller draft model to
guess several possible next tokens, then asking a larger target model to verify
those guesses in fewer expensive forward passes. This repository implements the
classic chain version and experiments with tree-shaped draft proposals, where the
draft model explores multiple possible continuations at once.

## What is inside

| Component | Description |
| --- | --- |
| Chain speculative decoding | Standard draft-then-verify decoding with modified rejection sampling. |
| Static draft trees | Fixed-width, fixed-depth tree construction for candidate continuations. |
| Dynamic draft trees | Best-first tree expansion with Gumbel-Top-k candidate selection. |
| Adaptive draft trees | Entropy-based branching that changes width based on draft confidence. |
| Tree attention masks | Masks that let each tree node attend only to the prompt and its ancestors. |
| Tree-aware KV cache | Experimental cache manager with prefix sharing and rollback hooks. |
| Correctness tests | 78 tests covering rejection sampling, tree masks, caches, engine regressions, and toy-distribution equivalence. |
| Benchmarks and demo | A benchmark runner plus a Streamlit UI for interactive experimentation. |

## Project status

This is best understood as an educational and research prototype, not a
production inference engine.

The chain speculative decoder is the strongest part of the repository: it
implements the modified rejection sampling logic used by speculative decoding
papers and is covered by distributional tests.

The tree decoder code is useful and non-trivial, but still experimental. It has
the important building blocks: tree construction, tree attention masks, per-node
position IDs, verification logic, and a cache abstraction. However, it does not
yet implement every optimization or multi-branch acceptance detail from systems
such as SpecInfer, Medusa, or EAGLE. Treat benchmark results as exploratory until
you validate them on real hardware and model pairs.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/YOUR_USERNAME/speculative-tree-decoding.git
cd speculative-tree-decoding
python -m pip install -e ".[dev,demo]"
```

For only the core library:

```bash
python -m pip install -e .
```

CUDA is strongly recommended for meaningful model benchmarks. CPU execution is
supported by PyTorch but can be very slow for real language models.

## Quick start

```python
import torch

from speculative_tree_decoding import (
    load_models,
    SpeculativeDecodingEngine,
    TreeSpeculativeDecodingEngine,
    DynamicTreeSpeculativeDecodingEngine,
    AdaptiveTreeSpeculativeDecodingEngine,
    autoregressive_generate,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

draft_model, target_model, tokenizer = load_models(
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    device=device,
    dtype=dtype,
)

engine = SpeculativeDecodingEngine(
    draft_model=draft_model,
    target_model=target_model,
    tokenizer=tokenizer,
    gamma=4,
    temperature=0.7,
    top_p=0.9,
)

text = engine.generate("Explain speculative decoding simply:", max_new_tokens=80)
print(text)
```

Tree-based variants are available through the same `generate(...)` interface:

```python
tree_engine = DynamicTreeSpeculativeDecodingEngine(
    draft_model=draft_model,
    target_model=target_model,
    tokenizer=tokenizer,
    max_nodes=15,
    max_depth=4,
    branching_factor=4,
    temperature=0.7,
    top_p=0.9,
)

print(tree_engine.generate("Write a short poem about GPUs:", max_new_tokens=80))
```

## Command line usage

After installing the package, you can generate from the terminal:

```bash
speculative-tree-decode \
  --engine chain \
  --draft Qwen/Qwen2.5-0.5B-Instruct \
  --target Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Explain speculative decoding simply:" \
  --max-new-tokens 80 \
  --device auto \
  --show-timing
```

Available engines are `autoregressive`, `chain`, `static-tree`, `dynamic-tree`,
and `adaptive-tree`.

## Run the demo

```bash
streamlit run app.py
```

The demo lets you compare speculative decoding against normal autoregressive
generation, tune decoding parameters, and inspect tree-shaped candidate
proposals.

## Run tests

```bash
python -m pytest
```

Useful targeted test runs:

```bash
python -m pytest tests/test_distribution_equivalence.py -q
python -m pytest tests/test_tree_attention_mask.py -q
python -m pytest tests/test_tree_kv_cache.py -q
python -m pytest tests/test_dynamic_tree.py -q
```

## Run benchmarks

```bash
python benchmarks/run_benchmark.py \
  --draft Qwen/Qwen2.5-0.5B-Instruct \
  --target Qwen/Qwen2.5-1.5B-Instruct \
  --engine all \
  --max-tokens 100 \
  --device cuda \
  --output benchmarks/results/benchmark.json
```

The benchmark reports wall-clock latency, tokens per second, acceptance length,
and relative speedup against autoregressive decoding.

## How speculative decoding works

Normal language model generation produces one token at a time. Each new token
requires another pass through the large model.

Speculative decoding adds a smaller draft model:

1. The draft model quickly proposes several next tokens.
2. The target model checks those tokens in one larger verification pass.
3. Tokens that match the target model's distribution are accepted.
4. If a token is rejected, the algorithm samples a replacement from the residual
   target distribution so the final output still matches target-model sampling.

Tree speculative decoding generalizes this idea. Instead of drafting one chain
of future tokens, the draft model builds a tree of possible futures. A tree can
spend compute on multiple plausible continuations, which can improve acceptance
when there are several likely next-token paths.

## Repository layout

```text
speculative-tree-decoding/
├── src/speculative_tree_decoding/
│   ├── models.py
│   ├── sampling.py
│   ├── engine.py
│   ├── kv_cache.py
│   ├── tree_builder.py
│   ├── dynamic_tree_builder.py
│   ├── tree_attention.py
│   ├── tree_kv_cache.py
│   └── verify.py
├── tests/
├── benchmarks/
├── notebooks/
├── app.py
├── demo.py
├── DESIGN.md
├── REFERENCES.md
├── pyproject.toml
└── README.md
```

## References

The implementation is based on ideas from speculative decoding and tree-based
LLM inference work. See [REFERENCES.md](REFERENCES.md) for the bibliography.

Key papers include:

- Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding"
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling"
- Miao et al., "SpecInfer: Accelerating LLM Serving with Tree-Based Speculative Inference"
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework"
- Li et al., "EAGLE", "EAGLE-2", and "EAGLE-3"

## License

MIT. See [LICENSE](LICENSE).
