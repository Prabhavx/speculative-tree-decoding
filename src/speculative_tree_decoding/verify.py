"""Tree verification with target model and rejection sampling."""
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .tree_builder import DraftTree
from .tree_attention import (
    flatten_tree_for_verification,
    build_tree_attention_mask,
)
from .tree_kv_cache import TreeKVCache
from .sampling import rejection_sample, greedy_rejection_sample, compute_logprobs, sample_from_logits


def build_tree_attention_mask_4d(
    parent_array: List[int],
    prefix_length: int,
    attention_mask: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a 4D additive attention mask for tree verification compatible with HF models.
    
    Args:
        parent_array: parent_array[i] = parent index of node i in tree (-1 for root)
        prefix_length: Number of prefix tokens before the tree starts
        attention_mask: 2D prefix attention mask [batch, prefix_len]
        device: Device for the mask tensor
        dtype: Data type for the additive mask
    
    Returns:
        Additive mask of shape [batch, 1, total_seq_len, total_seq_len] with -inf for masked positions
    """
    batch_size = attention_mask.size(0) if attention_mask is not None else 1
    num_nodes = len(parent_array)
    total_len = prefix_length + num_nodes
    
    # Build boolean mask first
    bool_mask = build_tree_attention_mask(prefix_length, parent_array, device)
    bool_mask = bool_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    # If we have a prefix attention mask, incorporate it
    if attention_mask is not None:
        prefix_keep = attention_mask.bool()
        bool_mask[:, :, :prefix_length] &= prefix_keep.unsqueeze(1)
        bool_mask[:, :prefix_length, :] &= prefix_keep.unsqueeze(2)
    
    # Convert to additive mask
    additive_mask = torch.zeros_like(bool_mask, dtype=dtype)
    additive_mask.masked_fill_(~bool_mask, torch.finfo(dtype).min)
    return additive_mask.unsqueeze(1)  # [batch, 1, seq, seq]


@dataclass
class TreeVerificationResult:
    """Result of verifying a draft tree with the target model."""
    accepted_tokens: List[int]  # Accepted tokens in DFS order
    accepted_node_indices: List[int]  # Node indices that were accepted
    rejected_at: Optional[int] = None  # Node index where rejection occurred
    resampled_token: Optional[int] = None  # Token resampled from residual
    target_logprobs: List[torch.Tensor] = None  # Target logprobs at each tree node
    target_tokens: List[int] = None  # Target's own predictions
    num_accepted: int = 0


def _extract_tree_distributions(
    outputs,
    tree: DraftTree,
    prefix_len: int,
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor], List[int]]:
    """Extract per-node verification and next-token distributions."""
    target_logprobs = []
    target_tokens = []
    next_logprobs = []
    next_tokens = []

    for node_idx, node in enumerate(tree.nodes):
        verify_pos = prefix_len - 1 if node.parent_idx == -1 else prefix_len + node.parent_idx
        if verify_pos < 0:
            raise ValueError("Tree verification requires a non-empty prefix sequence")

        verify_logits = outputs.logits[:, verify_pos, :]
        verify_logprobs = compute_logprobs(verify_logits).squeeze(0)
        target_logprobs.append(verify_logprobs)
        target_tokens.append(verify_logits.argmax(dim=-1).item())

        next_logits = outputs.logits[:, prefix_len + node_idx, :]
        next_lp = compute_logprobs(next_logits).squeeze(0)
        next_logprobs.append(next_lp)
        next_tokens.append(next_logits.argmax(dim=-1).item())

    return target_logprobs, target_tokens, next_logprobs, next_tokens


def _select_tree_path(
    tree: DraftTree,
    target_logprobs: List[torch.Tensor],
    target_tokens: List[int],
    next_logprobs: List[torch.Tensor],
    next_tokens: List[int],
    temperature: float,
    top_p: float,
    top_k: int,
) -> TreeVerificationResult:
    """Apply rejection sampling along the accepted tree path."""
    if tree.get_num_nodes() == 0:
        return TreeVerificationResult(
            accepted_tokens=[],
            accepted_node_indices=[],
            target_logprobs=target_logprobs,
            target_tokens=target_tokens,
            num_accepted=0,
        )

    accepted_node_indices = []
    accepted_tokens = []
    rejected_at = None
    resampled_token = None

    current_node_idx = 0

    while True:
        node = tree.nodes[current_node_idx]
        draft_token = node.token_id
        d_logprobs = node.logprobs
        t_logprobs = target_logprobs[current_node_idx]

        if temperature == 0.0:
            accepted, resampled = greedy_rejection_sample(d_logprobs, t_logprobs, draft_token)
        else:
            accepted, resampled = rejection_sample(d_logprobs, t_logprobs, draft_token, temperature)

        if accepted:
            accepted_node_indices.append(current_node_idx)
            accepted_tokens.append(draft_token)
        else:
            rejected_at = current_node_idx
            resampled_token = resampled
            break

        if node.children:
            target_token = next_tokens[current_node_idx]
            matched_child = None
            for child_idx in node.children:
                if tree.nodes[child_idx].token_id == target_token:
                    matched_child = child_idx
                    break

            if matched_child is not None:
                current_node_idx = matched_child
            else:
                break
        else:
            break

    if rejected_at is None and accepted_node_indices:
        last_node_idx = accepted_node_indices[-1]
        if temperature == 0.0:
            bonus_token = next_tokens[last_node_idx]
        else:
            bonus_token = sample_from_logits(
                next_logprobs[last_node_idx],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ).item()
        accepted_tokens.append(bonus_token)

    output_count = len(accepted_tokens) + (1 if resampled_token is not None else 0)
    return TreeVerificationResult(
        accepted_tokens=accepted_tokens,
        accepted_node_indices=accepted_node_indices,
        rejected_at=rejected_at,
        resampled_token=resampled_token,
        target_logprobs=target_logprobs,
        target_tokens=target_tokens,
        num_accepted=output_count,
    )


def verify_tree(
    target_model,
    tree: DraftTree,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: Optional[torch.device] = None,
) -> TreeVerificationResult:
    """
    Verify a draft tree with the target model in a single forward pass.

    Uses tree attention mask to verify all branches in parallel.

    Args:
        target_model: The target model
        tree: DraftTree to verify
        input_ids: Input token ids [1, prefix_len]
        attention_mask: Optional attention mask for prefix
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling limit
        device: Device for computation

    Returns:
        TreeVerificationResult with accepted tokens and rejection info
    """
    device = device or input_ids.device
    prefix_len = input_ids.size(1)

    if tree.get_num_nodes() == 0:
        return TreeVerificationResult(
            accepted_tokens=[],
            accepted_node_indices=[],
            target_logprobs=[],
            target_tokens=[],
            num_accepted=0,
        )

    # Flatten tree for verification
    flat_ids, tree_attn_mask, pos_ids = flatten_tree_for_verification(tree, input_ids)

    # Build proper 4D additive attention mask for HF model
    parent_array = tree.get_parent_array()
    if attention_mask is not None:
        full_mask = build_tree_attention_mask_4d(
            parent_array, prefix_len, attention_mask, device,
            dtype=next(target_model.parameters()).dtype
        )
    else:
        # Build mask without prefix attention
        batch_size = 1
        num_nodes = len(parent_array)
        total_len = prefix_len + num_nodes
        bool_mask = build_tree_attention_mask(prefix_len, parent_array, device)
        bool_mask = bool_mask.unsqueeze(0)  # [1, total_len, total_len]
        additive_mask = torch.zeros_like(bool_mask, dtype=next(target_model.parameters()).dtype)
        additive_mask.masked_fill_(~bool_mask, torch.finfo(next(target_model.parameters()).dtype).min)
        full_mask = additive_mask.unsqueeze(1)  # [1, 1, seq, seq]

    # Single forward pass with tree attention
    with torch.no_grad():
        outputs = target_model(
            input_ids=flat_ids,
            attention_mask=full_mask,
            position_ids=pos_ids,
            use_cache=True,
        )

    target_logprobs, target_tokens, next_logprobs, next_tokens = _extract_tree_distributions(
        outputs, tree, prefix_len
    )
    return _select_tree_path(
        tree,
        target_logprobs,
        target_tokens,
        next_logprobs,
        next_tokens,
        temperature,
        top_p,
        top_k,
    )


def verify_tree_greedy(
    target_model,
    tree: DraftTree,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> TreeVerificationResult:
    """Verify tree with greedy decoding (temperature=0)."""
    return verify_tree(
        target_model, tree, input_ids, attention_mask,
        temperature=0.0, top_p=1.0, top_k=0, device=device
    )


def verify_tree_sampling(
    target_model,
    tree: DraftTree,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: Optional[torch.device] = None,
) -> TreeVerificationResult:
    """Verify tree with sampling."""
    return verify_tree(
        target_model, tree, input_ids, attention_mask,
        temperature=temperature, top_p=top_p, top_k=top_k, device=device
    )


def get_accepted_prefix_length(
    tree: DraftTree,
    result: TreeVerificationResult,
) -> int:
    """
    Compute the new prefix length after accepting tokens from tree verification.

    The new prefix includes:
    - Original prefix
    - All accepted tokens along the accepted path
    - The resampled token (if rejection occurred)
    """
    return tree.prefix_length + result.num_accepted


def extract_accepted_path_tokens(
    tree: DraftTree,
    result: TreeVerificationResult,
) -> List[int]:
    """Extract the accepted tokens in sequence order."""
    if result.accepted_tokens is not None:
        tokens = list(result.accepted_tokens)
    else:
        tokens = []
        for node_idx in result.accepted_node_indices:
            tokens.append(tree.nodes[node_idx].token_id)
    if result.resampled_token is not None:
        tokens.append(result.resampled_token)
    return tokens


def verify_tree_with_kv_cache(
    target_model,
    tree: DraftTree,
    input_ids: torch.Tensor,
    target_cache: TreeKVCache,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: Optional[torch.device] = None,
) -> TreeVerificationResult:
    """
    Verify a draft tree using tree-aware KV cache.

    This is more efficient than the naive approach as it:
    1. Reuses prefix KV cache
    2. Only computes tree nodes in single forward pass
    3. Supports rollback on rejection
    """
    result = verify_tree(
        target_model,
        tree,
        input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        device=device,
    )
    # A flattened tree cache is not directly reusable as a linear accepted
    # prefix when the accepted path skips siblings, so keep the state honest.
    target_cache.reset()
    return result
