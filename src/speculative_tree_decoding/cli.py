"""Command-line interface for speculative tree decoding."""
import argparse
import time

import torch

from .engine import (
    AdaptiveTreeSpeculativeDecodingEngine,
    DynamicTreeSpeculativeDecodingEngine,
    SpeculativeDecodingEngine,
    TreeSpeculativeDecodingEngine,
    autoregressive_generate,
)
from .models import load_models


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _create_engine(args, draft_model, target_model, tokenizer, device: torch.device):
    if args.engine == "chain":
        return SpeculativeDecodingEngine(
            draft_model,
            target_model,
            tokenizer,
            gamma=args.gamma,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
    if args.engine == "static-tree":
        return TreeSpeculativeDecodingEngine(
            draft_model,
            target_model,
            tokenizer,
            branching_factor=args.branching_factor,
            max_depth=args.max_depth,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
    if args.engine == "dynamic-tree":
        return DynamicTreeSpeculativeDecodingEngine(
            draft_model,
            target_model,
            tokenizer,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            branching_factor=args.branching_factor,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
    if args.engine == "adaptive-tree":
        return AdaptiveTreeSpeculativeDecodingEngine(
            draft_model,
            target_model,
            tokenizer,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            base_branching=args.branching_factor,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
    raise ValueError(f"Unknown engine: {args.engine}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speculative-tree-decode",
        description="Generate text with chain or tree speculative decoding.",
    )
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--engine",
        default="chain",
        choices=["autoregressive", "chain", "static-tree", "dynamic-tree", "adaptive-tree"],
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-timing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device_name = _resolve_device(args.device)
    dtype = torch.float16 if device_name == "cuda" else torch.float32
    draft_model, target_model, tokenizer = load_models(
        args.draft,
        args.target,
        device=device_name,
        dtype=dtype,
    )
    device = torch.device(device_name)

    start = time.time()
    if args.engine == "autoregressive":
        output = autoregressive_generate(
            target_model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
    else:
        engine = _create_engine(args, draft_model, target_model, tokenizer, device)
        output = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    elapsed = time.time() - start

    print(output)
    if args.show_timing:
        print(f"\nGenerated in {elapsed:.2f}s on {device_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
