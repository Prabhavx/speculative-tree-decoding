"""Tree verification with target model and rejection sampling."""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from .tree_builder import DraftTree, TreeNode
from .tree_attention import (
    flatten_tree_for_verification,
    build_tree_attention_mask,
    build_tree_position_ids,
    TreeAttentionMask,
)
from .tree_kv_cache import TreeKVCache, build_tree_position_ids as build_tree_pos_ids, build_tree_attention_mask as build_tree_attn_mask
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
    bool_mask = bool_mask.unsqueeze(0)  # [1, total_len, total_len]
    
    # If we have a prefix attention mask, incorporate it
    if attention_mask is not None:
        # attention_mask is [batch, prefix_len], expand to [batch, total_len, total_len]
        prefix_mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=device)
        prefix_mask[:, :prefix_length, :prefix_length] = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
        # Tree nodes attend to prefix based on attention_mask
        for node_idx in range(num_nodes):
            seq_idx = prefix_length + node_idx
            prefix_mask[:, seq_idx, :prefix_length] = attention_mask.bool()
        bool_mask = bool_mask & prefix_mask
    
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

    # Extract target logprobs at each tree node position
    target_logprobs = []
    target_tokens = []

    for node_idx in range(tree.get_num_nodes()):
        seq_pos = prefix_len + node_idx
        logits = outputs.logits[:, seq_pos - 1, :]
        logprobs = compute_logprobs(logits).squeeze(0)
        target_logprobs.append(logprobs)
        target_tokens.append(logits.argmax(dim=-1).item())

    # Perform rejection sampling along each branch
    # We need to find the longest accepted path
    accepted_node_indices = []
    accepted_tokens = []
    rejected_at = None
    resampled_token = None

    # Start from root, follow the path with highest acceptance
    # For tree, we accept nodes along a single root-to-leaf path
    # The path is determined by which child gets accepted at each branch point

    current_node_idx = 0  # Root
    path = [0]

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

        # If this node has children, pick the one that matches target's prediction
        # (or sample from accepted children)
        if node.children:
            # Find which child the target model would predict
            target_token = target_tokens[current_node_idx]
            # Look for a child with matching token
            matched_child = None
            for child_idx in node.children:
                if tree.nodes[child_idx].token_id == target_token:
                    matched_child = child_idx
                    break

            if matched_child is not None:
                current_node_idx = matched_child
                path.append(current_node_idx)
            else:
                # No matching child - we can't continue this branch
                # In standard tree spec decoding, we stop here
                break
        else:
            # Leaf node, no more children
            break

    # Bonus token: if we accepted all nodes on the path, sample one extra target token.
    if rejected_at is None and accepted_node_indices:
        last_node_idx = accepted_node_indices[-1]
        if temperature == 0.0:
            bonus_token = target_tokens[last_node_idx]
            accepted_tokens.append(bonus_token)
        else:
            bonus_token = sample_from_logits(
                target_logprobs[last_node_idx],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ).item()
            accepted_tokens.append(bonus_token)

    return TreeVerificationResult(
        accepted_tokens=accepted_tokens,
        accepted_node_indices=accepted_node_indices,
        rejected_at=rejected_at,
        resampled_token=resampled_token,
        target_logprobs=target_logprobs,
        target_tokens=target_tokens,
        num_accepted=len(accepted_tokens),
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
    device = device or input_ids.device
    prefix_len = input_ids.size(1)

    # If cache doesn't match prefix, reset and recompute prefix
    if target_cache.get_prefix_length() != prefix_len:
        target_cache.reset()
        target_cache.set_prefix(prefix_len)

    num_nodes = tree.get_num_nodes()

    # Prepare cache for tree verification
    target_cache.prepare_for_tree_verification(num_nodes)

    # Flatten tree for verification
    flat_ids, tree_attn_mask, pos_ids = flatten_tree_for_verification(tree, input_ids)

    # Build proper 4D additive attention mask for tree nodes only
    parent_array = tree.get_parent_array()
    tree_prefix_len = prefix_len
    total_tree_len = num_nodes
    
    if attention_mask is not None:
        # Build full mask for the whole sequence, then slice for tree portion
        full_bool_mask = build_tree_attention_mask(tree_prefix_len, parent_array, device)
        # We only need the tree portion of the mask (rows for tree nodes, cols for prefix + tree)
        tree_bool_mask = full_bool_mask[tree_prefix_len:, :].unsqueeze(0)  # [1, num_nodes, total_len]
        # Apply prefix attention mask
        prefix_mask = attention_mask.bool().unsqueeze(1).expand(-1, num_nodes, -1)  # [batch, num_nodes, prefix_len]
        tree_bool_mask[:, :, :tree_prefix_len] = tree_bool_mask[:, :, :tree_prefix_len] & prefix_mask
        # Convert to additive mask
        dtype = next(target_model.parameters()).dtype
        additive_mask = torch.zeros_like(tree_bool_mask, dtype=dtype)
        additive_mask.masked_fill_(~tree_bool_mask, torch.finfo(dtype).min)
        tree_attn_mask_4d = additive_mask.unsqueeze(1)  # [batch, 1, num_nodes, total_len]
    else:
        # Build mask without prefix attention
        bool_mask = build_tree_attention_mask(tree_prefix_len, parent_array, device)
        tree_bool_mask = bool_mask[tree_prefix_len:, :].unsqueeze(0)  # [1, num_nodes, total_len]
        dtype = next(target_model.parameters()).dtype
        additive_mask = torch.zeros_like(tree_bool_mask, dtype=dtype)
        additive_mask.masked_fill_(~tree_bool_mask, torch.finfo(dtype).min)
        tree_attn_mask_4d = additive_mask.unsqueeze(1)  # [1, 1, num_nodes, total_len]

    # Single forward pass with tree attention and KV cache
    with torch.no_grad():
        outputs = target_model(
            input_ids=flat_ids[:, prefix_len:],  # Only tree nodes
            attention_mask=tree_attn_mask_4d,
            position_ids=pos_ids[:, prefix_len:],
            past_key_values=target_cache.get_past_key_values(prefix_len),
            use_cache=True,
        )

    # Update cache with new tree nodes
    # Handle both DynamicCache and legacy tuple formats
    past_kv = outputs.past_key_values
    if hasattr(past_kv, 'layers'):
        # DynamicCache format
        layer_0 = past_kv.layers[0]
        key_0, value_0 = layer_0.keys, layer_0.values
    else:
        # Legacy tuple format
        key_0, value_0 = past_kv[0][0], past_kv[0][1]
    target_cache.update_from_model_output(0, key_0, value_0, prefix_len)

    # Extract target logprobs at each tree node position
    target_logprobs = []
    target_tokens = []

    for node_idx in range(num_nodes):
        logits = outputs.logits[:, node_idx, :]
        logprobs = compute_logprobs(logits).squeeze(0)
        target_logprobs.append(logprobs)
        target_tokens.append(logits.argmax(dim=-1).item())

    # Perform rejection sampling along the accepted path
    accepted_node_indices = []
    accepted_tokens = []
    rejected_at = None
    resampled_token = None

    current_node_idx = 0  # Root

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

        # If this node has children, pick the one that matches target's prediction
        if node.children:
            target_token = target_tokens[current_node_idx]
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

    # Handle rejection/acceptance for cache
    if rejected_at is not None:
        target_cache.reject_tree()
    else:
        target_cache.accept_tree_path(len(accepted_node_indices))

    # Bonus token
    if rejected_at is None and accepted_node_indices:
        last_node_idx = accepted_node_indices[-1]
        if temperature == 0.0:
            bonus_token = target_tokens[last_node_idx]
            accepted_tokens.append(bonus_token)
        else:
            # Sample bonus token from target distribution at temperature
            bonus_logprobs = target_logprobs[last_node_idx]
            bonus_token = sample_from_logits(
                bonus_logprobs,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ).item()
            accepted_tokens.append(bonus_token)

    return TreeVerificationResult(
        accepted_tokens=accepted_tokens,
        accepted_node_indices=accepted_node_indices,
        rejected_at=rejected_at,
        resampled_token=resampled_token,
        target_logprobs=target_logprobs,
        target_tokens=target_tokens,
        num_accepted=len(accepted_tokens),
    )
