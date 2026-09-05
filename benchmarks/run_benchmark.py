"""Benchmark harness for speculative decoding evaluation."""
import torch
import time
import json
import logging
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from speculative_tree_decoding.models import load_models
from speculative_tree_decoding.engine import (
    SpeculativeDecodingEngine,
    TreeSpeculativeDecodingEngine,
    DynamicTreeSpeculativeDecodingEngine,
    AdaptiveTreeSpeculativeDecodingEngine,
    autoregressive_generate,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    engine_type: str
    model_pair: str
    gamma: int = 0
    branching_factor: int = 0
    max_depth: int = 0
    max_nodes: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
    num_prompts: int = 0
    category: str = ""
    total_tokens_spec: int = 0
    total_tokens_ar: int = 0
    time_spec: float = 0.0
    time_ar: float = 0.0
    speedup: float = 0.0
    avg_acceptance_length: float = 0.0
    peak_memory_mb: float = 0.0
    tokens_per_sec_spec: float = 0.0
    tokens_per_sec_ar: float = 0.0


def load_test_prompts(category: str) -> List[str]:
    """Load test prompts for benchmarking."""
    if category == "chat":
        return [
            "Write a short story about a robot learning to paint.",
            "Explain the concept of recursion to a 10-year-old.",
            "What are the benefits of meditation?",
            "Write a Python function to compute fibonacci numbers.",
            "Describe the lifecycle of a butterfly.",
            "How does a hash table work?",
            "What is the difference between TCP and UDP?",
            "Write a haiku about debugging.",
            "Explain how a transformer model works.",
            "What are the phases of the moon?",
        ]
    elif category == "code":
        return [
            "Write a function to reverse a linked list.",
            "Implement binary search in Python.",
            "Write a class for a stack with push, pop, and peek.",
            "Implement quicksort algorithm.",
            "Write a function to check if a string is a palindrome.",
            "Create a decorator that measures execution time.",
            "Write a context manager for file handling.",
            "Implement a LRU cache.",
            "Write a function to merge two sorted arrays.",
            "Create a simple neural network from scratch.",
        ]
    elif category == "math":
        return [
            "Solve: 2x + 5 = 15",
            "What is the derivative of x^3 + 2x^2 - 5x + 1?",
            "Calculate the integral of sin(x) from 0 to pi.",
            "Find the roots of x^2 - 5x + 6 = 0.",
            "What is the sum of the first 100 natural numbers?",
            "Prove that sqrt(2) is irrational.",
            "What is the limit of (1 + 1/n)^n as n approaches infinity?",
            "Calculate the area of a circle with radius 5.",
            "Solve the system: x + y = 10, 2x - y = 5",
            "What is the probability of rolling a sum of 7 with two dice?",
        ]
    else:
        return [
            "Hello, how are you?",
            "Tell me a joke.",
            "What is the meaning of life?",
        ]


def create_engine(engine_type: str, draft_model, target_model, tokenizer, **kwargs):
    """Create the appropriate engine instance."""
    if engine_type == "chain":
        return SpeculativeDecodingEngine(
            draft_model, target_model, tokenizer,
            gamma=kwargs.get("gamma", 4),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 0),
        )
    elif engine_type == "static_tree":
        return TreeSpeculativeDecodingEngine(
            draft_model, target_model, tokenizer,
            branching_factor=kwargs.get("branching_factor", 3),
            max_depth=kwargs.get("max_depth", 2),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 0),
        )
    elif engine_type == "dynamic_tree":
        return DynamicTreeSpeculativeDecodingEngine(
            draft_model, target_model, tokenizer,
            max_nodes=kwargs.get("max_nodes", 15),
            max_depth=kwargs.get("max_depth", 4),
            branching_factor=kwargs.get("branching_factor", 4),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 0),
            cum_prob_threshold=kwargs.get("cum_prob_threshold", 0.0),
            use_gumbel_topk=kwargs.get("use_gumbel_topk", True),
            gumbel_temperature=kwargs.get("gumbel_temperature", 1.0),
            min_branch_prob=kwargs.get("min_branch_prob", 0.01),
        )
    elif engine_type == "adaptive_tree":
        return AdaptiveTreeSpeculativeDecodingEngine(
            draft_model, target_model, tokenizer,
            max_nodes=kwargs.get("max_nodes", 20),
            max_depth=kwargs.get("max_depth", 4),
            base_branching=kwargs.get("base_branching", 3),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 0),
            entropy_threshold=kwargs.get("entropy_threshold", 2.0),
        )
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


def run_single_benchmark(
    engine,
    engine_type: str,
    model_pair: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    category: str,
    **engine_kwargs
) -> BenchmarkResult:
    """Run benchmark for a single engine on a category of prompts."""
    total_tokens_spec = 0
    total_tokens_ar = 0
    time_spec = 0.0
    time_ar = 0.0

    device = next(engine.draft_model.parameters()).device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for prompt in prompts:
        # Speculative
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        spec_output = engine.generate(prompt, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_spec += time.time() - start

        # Autoregressive
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        ar_output = autoregressive_generate(
            engine.target_model, engine.tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, device=device
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_ar += time.time() - start

        spec_tokens = len(engine.tokenizer.encode(spec_output))
        ar_tokens = len(engine.tokenizer.encode(ar_output))
        total_tokens_spec += spec_tokens
        total_tokens_ar += ar_tokens

    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    speedup = time_ar / time_spec if time_spec > 0 else 0
    tokens_per_sec_spec = total_tokens_spec / time_spec if time_spec > 0 else 0
    tokens_per_sec_ar = total_tokens_ar / time_ar if time_ar > 0 else 0

    # Approximate acceptance length: total tokens / number of steps
    # For chain: steps = total_tokens / gamma
    # For tree: more complex, use average
    if engine_type == "chain":
        gamma = engine_kwargs.get("gamma", 4)
        avg_acceptance = total_tokens_spec / max(1, total_tokens_spec / gamma)
    else:
        # For tree, we'd need to track internally; approximate
        avg_acceptance = total_tokens_spec / max(1, len(prompts) * 2)  # rough estimate

    result = BenchmarkResult(
        engine_type=engine_type,
        model_pair=model_pair,
        gamma=engine_kwargs.get("gamma", 0),
        branching_factor=engine_kwargs.get("branching_factor", 0),
        max_depth=engine_kwargs.get("max_depth", 0),
        max_nodes=engine_kwargs.get("max_nodes", 0),
        temperature=temperature,
        top_p=top_p,
        num_prompts=len(prompts),
        category=category,
        total_tokens_spec=total_tokens_spec,
        total_tokens_ar=total_tokens_ar,
        time_spec=time_spec,
        time_ar=time_ar,
        speedup=speedup,
        avg_acceptance_length=avg_acceptance,
        peak_memory_mb=peak_memory,
        tokens_per_sec_spec=tokens_per_sec_spec,
        tokens_per_sec_ar=tokens_per_sec_ar,
    )

    logger.info(f"  [{engine_type}] {category}: {time_spec:.2f}s vs {time_ar:.2f}s, "
                f"speedup={speedup:.2f}x, tok/s={tokens_per_sec_spec:.1f}, "
                f"mem={peak_memory:.0f}MB")

    return result


def run_all_benchmarks(
    draft_model_name: str,
    target_model_name: str,
    engine_types: List[str] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    categories: List[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    **engine_kwargs
) -> List[BenchmarkResult]:
    """Run benchmarks for all engine types across all categories."""
    if engine_types is None:
        engine_types = ["chain", "static_tree", "dynamic_tree", "adaptive_tree"]
    if categories is None:
        categories = ["chat", "code", "math"]

    logger.info(f"Loading models: {draft_model_name} + {target_model_name}")
    draft_model, target_model, tokenizer = load_models(
        draft_model_name, target_model_name, device=device, dtype=dtype
    )

    model_pair = f"{draft_model_name.split('/')[-1]}+{target_model_name.split('/')[-1]}"
    results = []

    for engine_type in engine_types:
        logger.info(f"\n=== Testing {engine_type} ===")
        engine = create_engine(engine_type, draft_model, target_model, tokenizer, **engine_kwargs)

        for category in categories:
            prompts = load_test_prompts(category)
            logger.info(f"  Category: {category} ({len(prompts)} prompts)")
            result = run_single_benchmark(
                engine, engine_type, model_pair, prompts,
                max_new_tokens, temperature, top_p, category, **engine_kwargs
            )
            results.append(result)

    return results


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Results saved to {output}")


def print_summary(results: List[BenchmarkResult]):
    """Print a formatted summary of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by engine type
    by_engine = {}
    for r in results:
        if r.engine_type not in by_engine:
            by_engine[r.engine_type] = []
        by_engine[r.engine_type].append(r)

    for engine_type, engine_results in by_engine.items():
        print(f"\n--- {engine_type.upper()} ---")
        for r in engine_results:
            print(f"  {r.category:10s} | Speedup: {r.speedup:5.2f}x | "
                  f"Tok/s: {r.tokens_per_sec_spec:6.1f} | "
                  f"Accept: {r.avg_acceptance_length:4.2f} | "
                  f"Mem: {r.peak_memory_mb:6.0f}MB")

    # Overall best
    best = max(results, key=lambda r: r.speedup)
    print(f"\nBEST: {best.engine_type} on {best.category} - {best.speedup:.2f}x speedup")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--engine", default="all", choices=["all", "chain", "static_tree", "dynamic_tree", "adaptive_tree"])
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="benchmarks/results/benchmark.json")
    parser.add_argument("--categories", nargs="+", default=["chat", "code", "math"])
    args = parser.parse_args()

    dtype = torch.float16 if args.device == "cuda" else torch.float32

    if args.engine == "all":
        engine_types = ["chain", "static_tree", "dynamic_tree", "adaptive_tree"]
    else:
        engine_types = [args.engine]

    engine_kwargs = {
        "gamma": args.gamma,
        "branching_factor": args.branching_factor,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
    }

    results = run_all_benchmarks(
        args.draft, args.target,
        engine_types=engine_types,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        categories=args.categories,
        device=args.device,
        dtype=dtype,
        **engine_kwargs
    )

    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()