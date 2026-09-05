"""Model loading utilities for draft and target models with shared tokenizer validation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_models(
    draft_model_name: str,
    target_model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    draft_device: Optional[str] = None,
    target_device: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """
    Load draft and target models with shared tokenizer validation.

    Args:
        draft_model_name: HuggingFace model name for draft model
        target_model_name: HuggingFace model name for target model
        device: Default device for both models
        dtype: Torch dtype for model weights
        draft_device: Optional separate device for draft model
        target_device: Optional separate device for target model

    Returns:
        Tuple of (draft_model, target_model, shared_tokenizer)

    Raises:
        ValueError: If tokenizers don't match
    """
    draft_dev = draft_device or device
    target_dev = target_device or device

    logger.info(f"Loading draft model: {draft_model_name} on {draft_dev}")
    logger.info(f"Loading target model: {target_model_name} on {target_dev}")

    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    if draft_tokenizer.vocab_size != target_tokenizer.vocab_size:
        raise ValueError(
            f"Vocab size mismatch: draft={draft_tokenizer.vocab_size}, "
            f"target={target_tokenizer.vocab_size}"
        )

    if draft_tokenizer.get_vocab() != target_tokenizer.get_vocab():
        raise ValueError("Tokenizer vocabularies differ between draft and target models")

    if draft_tokenizer.chat_template != target_tokenizer.chat_template:
        logger.warning("Chat templates differ between draft and target tokenizers")

    tokenizer = target_tokenizer

    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=dtype,
        device_map=draft_dev if draft_dev != "auto" else None,
        low_cpu_mem_usage=True,
    ).eval()

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=dtype,
        device_map=target_dev if target_dev != "auto" else None,
        low_cpu_mem_usage=True,
    ).eval()

    if draft_dev != "auto":
        draft_model = draft_model.to(draft_dev)
    if target_dev != "auto":
        target_model = target_model.to(target_dev)

    logger.info(f"Draft model loaded: {draft_model.config.num_hidden_layers} layers, "
                f"{draft_model.config.hidden_size} hidden dim")
    logger.info(f"Target model loaded: {target_model.config.num_hidden_layers} layers, "
                f"{target_model.config.hidden_size} hidden dim")

    return draft_model, target_model, tokenizer


def get_model_device(model: AutoModelForCausalLM) -> torch.device:
    """Get the device of a model's parameters."""
    return next(model.parameters()).device


def prepare_inputs(
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    max_length: Optional[int] = None,
) -> dict:
    """Tokenize and prepare inputs for model."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in inputs.items()}