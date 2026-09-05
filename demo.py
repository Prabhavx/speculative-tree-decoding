"""Demo script to test speculative decoding engine."""
import torch
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_tree_decoding.models import load_models
from speculative_tree_decoding.engine import SpeculativeDecodingEngine, autoregressive_generate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Use smaller models for testing - adjust based on available GPU memory
    draft_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    target_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Loading models: {draft_model_name} -> {target_model_name}")

    draft_model, target_model, tokenizer = load_models(
        draft_model_name,
        target_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = next(draft_model.parameters()).device
    print(f"Models loaded on {device}")

    engine = SpeculativeDecodingEngine(
        draft_model,
        target_model,
        tokenizer,
        gamma=4,
        temperature=0.7,
        top_p=0.9,
    )

    prompts = [
        "Write a short poem about coding:",
        "Explain quantum computing in simple terms:",
        "What is the capital of France?",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        # Speculative decoding
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        spec_output = engine.generate(prompt, max_new_tokens=50)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        spec_time = time.time() - start

        # Autoregressive baseline
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        ar_output = autoregressive_generate(
            target_model, tokenizer, prompt, max_new_tokens=50,
            temperature=0.7, top_p=0.9, device=device
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ar_time = time.time() - start

        print(f"Speculative: {spec_output}")
        print(f"Autoregressive: {ar_output}")
        print(f"Speculative time: {spec_time:.3f}s")
        print(f"Autoregressive time: {ar_time:.3f}s")
        print(f"Speedup: {ar_time/spec_time:.2f}x")


if __name__ == "__main__":
    main()