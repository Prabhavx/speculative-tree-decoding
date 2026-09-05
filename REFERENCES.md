# References

## Core Papers

### Speculative Decoding Foundations
1. **Leviathan, Kalman, Matias (2023)** — *Fast Inference from Transformers via Speculative Decoding*
   - Original speculative decoding paper (Google)
   - Introduces chain-based drafting with rejection sampling
   - arXiv:2302.01318

2. **Chen et al. (2023, DeepMind)** — *Accelerating Large Language Model Decoding with Speculative Sampling*
   - Independent concurrent work
   - Formalizes modified rejection sampling proof
   - arXiv:2302.01318 (same number, different paper)

### Tree-Based Speculative Decoding
3. **Miao et al. (2024, ASPLOS)** — *SpecInfer: Accelerating LLM Serving with Tree-Based Speculative Inference and Verification*
   - First tree-based approach
   - Static tree with custom attention mask
   - Introduces tree attention for parallel verification

4. **Cai et al. (2024)** — *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*
   - Attached draft heads (not separate model)
   - Multiple heads predict different positions
   - Tree-structured candidates from multiple heads

5. **Li et al. (2024)** — *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*
   - Feature-level autoregressive drafting (not token-level)
   - Draft head on target model's hidden states
   - Significantly higher acceptance rates

6. **Li et al. (2024)** — *EAGLE-2: Faster Inference with Dynamic Draft Trees*
   - Dynamic/adaptive tree construction
   - Gumbel-Top-k for diverse branch selection
   - Compute-budget-aware expansion

7. **Li et al. (2025)** — *EAGLE-3: Training-Free Speculative Decoding*
   - Training-free dynamic drafting
   - Improved tree construction heuristics

### Adaptive/Advanced Methods
8. **Chen et al. (2024, NeurIPS)** — *Sequoia: Scalable, Robust, and Hardware-Aware Speculative Decoding*
   - Hardware-aware tree construction
   - Robustness to model mismatch
   - Production serving optimizations

9. **Wang et al. (2024/2025)** — *OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure*
   - Adaptive tree structure based on uncertainty
   - Dynamic branching factor

10. **Jeon et al. (2024)** — *Recursive Speculative Decoding*
    - Gumbel-Top-k / stochastic beam search for draft tree
    - Recursive application of speculative decoding

11. **Xia et al. (2024)** — *Spec-Bench: A Standard Cross-Method Evaluation Suite*
    - Standardized benchmarks for speculative decoding
    - Multiple model pairs, tasks, metrics

## Theoretical Background

### Rejection Sampling
- **Von Neumann (1951)** — *Various techniques used in connection with random digits*
  - Original rejection sampling
- **Robert & Casella (2004)** — *Monte Carlo Statistical Methods*
  - Comprehensive treatment of MCMC including rejection sampling

### Tree Algorithms
- **Knuth (1997)** — *The Art of Computer Programming, Vol 1: Fundamental Algorithms*
  - Tree traversal, representation
- **Cormen et al. (2009)** — *Introduction to Algorithms*
  - Graph algorithms, best-first search

## Implementation References

### PyTorch/Transformers
- **Paszke et al. (2019)** — *PyTorch: An Imperative Style, High-Performance Deep Learning Library*
- **Wolf et al. (2020)** — *Transformers: State-of-the-Art Natural Language Processing*
  - HuggingFace transformers library

### KV-Cache Management
- **Pope et al. (2023)** — *Efficiently Scaling Transformer Inference*
  - KV-cache optimization techniques
- **Kwon et al. (2023)** — *vLLM: Easy, Fast, and Cheap LLM Serving*
  - PagedAttention, KV-cache management in production

### Attention Masks
- **Child et al. (2019)** — *Generating Long Sequences with Sparse Transformers*
  - Custom attention patterns
- **Beltagy et al. (2020)** — *Longformer: The Long-Document Transformer*
  - Dilated/sliding window attention masks

## Model Families for Testing

### Qwen Family (Recommended)
- `Qwen/Qwen2.5-0.5B-Instruct` + `Qwen/Qwen2.5-1.5B-Instruct` (consumer GPU)
- `Qwen/Qwen2.5-1.5B-Instruct` + `Qwen/Qwen2.5-7B-Instruct` (larger)
- Same tokenizer, same architecture

### Llama Family
- `meta-llama/Llama-3.2-1B-Instruct` + `meta-llama/Llama-3.1-8B-Instruct`
- Requires HF token

### SmolLM Family (Low Resource)
- `HuggingFaceTB/SmolLM2-135M-Instruct` + `HuggingFaceTB/SmolLM2-1.7B-Instruct`
  - Runs on 8GB GPU or CPU
  - Good for correctness validation

## Evaluation Benchmarks

### Standard Datasets
- **MT-Bench**: Multi-turn conversation benchmark
- **HumanEval**: Code generation (164 problems)
- **GSM8K**: Grade school math reasoning
- **MMLU**: Massive multitask language understanding
- **Spec-Bench**: Standardized speculative decoding evaluation

### Metrics
- **Acceptance Length**: Average tokens accepted per verification step
- **Tokens/second**: Wall-clock throughput
- **Speedup Ratio**: vs. autoregressive baseline
- **Peak Memory**: GPU memory usage
- **Distributional Fidelity**: KL divergence from target-only

## Key Implementation Details from Papers

### SpecInfer (Miao et al.)
- Tree construction: fixed branching factor, depth
- Attention mask: ancestor-only in flattened sequence
- Position IDs: per-node depth from root
- Verification: single forward pass with custom mask

### EAGLE (Li et al.)
- Draft head: linear layer on target hidden states
- Feature autoregression: predicts next hidden state, then LM head
- Loss: MSE on hidden states + cross-entropy on tokens
- Dynamic tree: expand by cumulative probability

### EAGLE-2 (Li et al.)
- Gumbel-Top-k sampling for diverse branches
- Stochastic beam search without replacement
- Compute budget constraint

### Sequoia (Chen et al.)
- Hardware-aware: considers GPU SM occupancy, memory bandwidth
- Robust tree: handles draft-target mismatch
- Batched verification for multi-request

## Useful Code References

### HuggingFace Generation
- `transformers.generation.utils.GenerationMixin`
- `transformers.generation.logits_process.LogitsProcessorList`

### PyTorch Attention
- `torch.nn.functional.scaled_dot_product_attention`
- Custom `attn_mask` parameter for tree attention

### KV-Cache Patterns
- `past_key_values` format: tuple of (key, value) per layer
- Shape: [batch, num_heads, seq_len, head_dim]

## Further Reading

### Surveys
- **He et al. (2024)** — *A Survey on Speculative Decoding for Large Language Models*
- **Zhang et al. (2024)** — *Efficient LLM Inference: A Comprehensive Survey*

### Blogs/Tutorials
- **HuggingFace Blog**: "Assisted Generation: A New Direction for Low-Latency Text Generation"
- **vLLM Blog**: "Speculative Decoding in vLLM"
- **NVIDIA Technical Blog**: "Accelerating LLM Inference with Speculative Decoding"