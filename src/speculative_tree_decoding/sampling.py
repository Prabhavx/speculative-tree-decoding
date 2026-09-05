"""Sampling utilities for speculative decoding."""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Sample token from logits with temperature, top-p, and top-k filtering.

    Args:
        logits: [batch_size, vocab_size] or [vocab_size]
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling limit (0 = disabled)

    Returns:
        Sampled token ids [batch_size] or scalar
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if temperature == 0.0:
        token = logits.argmax(dim=-1)
    else:
        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return token.squeeze(0) if squeeze else token


def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities from logits."""
    return F.log_softmax(logits, dim=-1)


def rejection_sample(
    draft_logprobs: torch.Tensor,
    target_logprobs: torch.Tensor,
    draft_token: int,
    temperature: float = 1.0,
) -> Tuple[bool, Optional[int]]:
    """
    Modified rejection sampling for speculative decoding.

    Args:
        draft_logprobs: Log probabilities from draft model [vocab_size]
        target_logprobs: Log probabilities from target model [vocab_size]
        draft_token: The token proposed by draft model
        temperature: Sampling temperature

    Returns:
        (accepted: bool, resampled_token: Optional[int])
        If accepted=True, resampled_token is None (use draft_token).
        If accepted=False, resampled_token is drawn from residual distribution.
    """
    draft_prob = draft_logprobs[draft_token].exp()
    target_prob = target_logprobs[draft_token].exp()

    if target_prob >= draft_prob:
        return True, None

    accept_prob = target_prob / draft_prob
    if torch.rand(1).item() < accept_prob:
        return True, None

    residual = torch.clamp(target_logprobs.exp() - draft_logprobs.exp(), min=0.0)
    residual_sum = residual.sum()

    if residual_sum > 0:
        residual_probs = residual / residual_sum
        resampled = torch.multinomial(residual_probs, num_samples=1).item()
        return False, resampled
    else:
        return False, target_logprobs.argmax().item()


def greedy_rejection_sample(
    draft_logprobs: torch.Tensor,
    target_logprobs: torch.Tensor,
    draft_token: int,
) -> Tuple[bool, Optional[int]]:
    """
    Greedy rejection sampling (temperature=0).

    Accepts if target's top token matches draft token, otherwise rejects.
    """
    target_token = target_logprobs.argmax().item()
    if target_token == draft_token:
        return True, None
    return False, target_token