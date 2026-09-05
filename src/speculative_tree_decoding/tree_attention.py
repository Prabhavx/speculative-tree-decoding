"""Tree attention mask construction for speculative decoding."""
import torch
from typing import List, Tuple, Optional
from .tree_builder import DraftTree


def build_tree_attention_mask(
    prefix_length: int,
    parent_array: List[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a causal attention mask for a flattened tree.

    Each node can attend to:
    - All prefix tokens (indices 0 to prefix_length-1)
    - Its own ancestors in the tree (including itself)

    Nodes CANNOT attend to:
    - Siblings or other branches (mutually exclusive futures)
    - Descendants (future tokens)

    Args:
        prefix_length: Number of prefix tokens before the tree starts
        parent_array: parent_array[i] = parent index of node i in tree (-1 for root)
        device: Device for the mask tensor

    Returns:
        Boolean mask of shape [total_seq_len, total_seq_len] where total_seq_len = prefix_length + num_nodes
        mask[i, j] = True means position i can attend to position j
    """
    device = device or torch.device("cpu")
    num_nodes = len(parent_array)
    total_len = prefix_length + num_nodes

    # Initialize mask: prefix attends to prefix (causal), tree nodes attend to prefix
    mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)

    # Prefix: standard causal mask
    mask[:prefix_length, :prefix_length] = torch.tril(
        torch.ones((prefix_length, prefix_length), dtype=torch.bool, device=device)
    )

    # Tree nodes: each node attends to prefix + its ancestors
    for node_idx in range(num_nodes):
        seq_idx = prefix_length + node_idx

        # Attend to all prefix tokens
        mask[seq_idx, :prefix_length] = True

        # Attend to self and ancestors in tree
        curr = node_idx
        while curr != -1:
            ancestor_seq_idx = prefix_length + curr
            mask[seq_idx, ancestor_seq_idx] = True
            curr = parent_array[curr]

    return mask


def build_tree_position_ids(
    prefix_length: int,
    parent_array: List[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build position IDs for RoPE in a tree-structured sequence.

    Each node's position_id = prefix_length + depth_from_root.
    This ensures RoPE correctly encodes relative positions within each branch.

    Args:
        prefix_length: Number of prefix tokens before the tree starts
        parent_array: parent_array[i] = parent index of node i in tree (-1 for root)
        device: Device for the tensor

    Returns:
        Position IDs tensor of shape [total_seq_len]
    """
    device = device or torch.device("cpu")
    num_nodes = len(parent_array)
    total_len = prefix_length + num_nodes

    position_ids = torch.arange(total_len, device=device, dtype=torch.long)

    # Prefix positions are just 0, 1, ..., prefix_length-1
    # Tree nodes: position = prefix_length + depth
    for node_idx in range(num_nodes):
        # Compute depth by walking up to root
        depth = 0
        curr = node_idx
        while curr != -1:
            depth += 1
            curr = parent_array[curr]
        # Depth from root (root has depth 1)
        position_ids[prefix_length + node_idx] = prefix_length + depth - 1

    return position_ids


def compute_node_depths(parent_array: List[int]) -> List[int]:
    """Compute depth of each node from root."""
    depths = []
    for i in range(len(parent_array)):
        depth = 0
        curr = i
        while curr != -1:
            depth += 1
            curr = parent_array[curr]
        depths.append(depth - 1)  # root has depth 0
    return depths


def build_tree_attention_mask_batched(
    parent_arrays: List[List[int]],
    prefix_lengths: List[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build batched tree attention masks for multiple sequences.

    Args:
        parent_arrays: List of parent arrays for each sequence
        prefix_lengths: List of prefix lengths for each sequence
        device: Device for the mask tensor

    Returns:
        Boolean mask of shape [batch, max_seq_len, max_seq_len]
    """
    device = device or torch.device("cpu")
    batch_size = len(parent_arrays)
    max_nodes = max(len(p) for p in parent_arrays)
    max_prefix = max(prefix_lengths)
    max_len = max_prefix + max_nodes

    mask = torch.zeros((batch_size, max_len, max_len), dtype=torch.bool, device=device)

    for b in range(batch_size):
        parent_array = parent_arrays[b]
        prefix_len = prefix_lengths[b]
        num_nodes = len(parent_array)

        # Prefix causal
        mask[b, :prefix_len, :prefix_len] = torch.tril(
            torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=device)
        )

        # Tree nodes
        for node_idx in range(num_nodes):
            seq_idx = prefix_len + node_idx
            mask[b, seq_idx, :prefix_len] = True

            curr = node_idx
            while curr != -1:
                ancestor_seq_idx = prefix_len + curr
                mask[b, seq_idx, ancestor_seq_idx] = True
                curr = parent_array[curr]

    return mask


def verify_tree_mask_correctness(
    parent_array: List[int],
    prefix_length: int = 0,
) -> Tuple[bool, List[str]]:
    """
    Verify the tree attention mask is correct.

    Checks:
    1. Each node attends to all its ancestors (including self)
    2. No node attends to non-ancestors (siblings, cousins, descendants)
    3. Prefix has standard causal mask
    4. All tree nodes attend to full prefix

    Returns:
        (is_correct, list_of_errors)
    """
    errors = []
    num_nodes = len(parent_array)
    total_len = prefix_length + num_nodes

    mask = build_tree_attention_mask(prefix_length, parent_array)

    # Check prefix causal
    for i in range(prefix_length):
        for j in range(prefix_length):
            expected = i >= j
            if mask[i, j] != expected:
                errors.append(f"Prefix[{i},{j}]: expected {expected}, got {mask[i, j]}")

    # Check tree nodes
    for node_idx in range(num_nodes):
        seq_idx = prefix_length + node_idx

        # Should attend to all prefix
        for j in range(prefix_length):
            if not mask[seq_idx, j]:
                errors.append(f"Node {node_idx} (seq {seq_idx}) should attend to prefix {j}")

        # Should attend to self and ancestors
        ancestors = set()
        curr = node_idx
        while curr != -1:
            ancestors.add(curr)
            curr = parent_array[curr]

        for anc in ancestors:
            anc_seq = prefix_length + anc
            if not mask[seq_idx, anc_seq]:
                errors.append(f"Node {node_idx} should attend to ancestor {anc} (seq {anc_seq})")

        # Should NOT attend to non-ancestors
        for other_idx in range(num_nodes):
            if other_idx not in ancestors:
                other_seq = prefix_length + other_idx
                if mask[seq_idx, other_seq]:
                    errors.append(f"Node {node_idx} should NOT attend to non-ancestor {other_idx} (seq {other_seq})")

    return len(errors) == 0, errors


def flatten_tree_for_verification(
    tree: DraftTree,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare flattened tree for target model verification.

    Returns:
        flat_input_ids: [1, prefix_len + num_nodes]
        attention_mask: [1, prefix_len + num_nodes] (tree mask)
        position_ids: [1, prefix_len + num_nodes]
    """
    prefix_len = input_ids.size(1)
    device = input_ids.device

    # Flatten tokens: prefix + tree nodes in DFS order
    tree_tokens = tree.get_flattened_tokens()
    flat_ids = torch.cat([
        input_ids,
        torch.tensor(tree_tokens, device=device).unsqueeze(0)
    ], dim=1)

    # Build attention mask
    parent_array = tree.get_parent_array()
    attn_mask = build_tree_attention_mask(prefix_len, parent_array, device)
    attn_mask = attn_mask.unsqueeze(0)  # Add batch dim

    # Build position IDs
    pos_ids = build_tree_position_ids(prefix_len, parent_array, device)
    pos_ids = pos_ids.unsqueeze(0)

    return flat_ids, attn_mask, pos_ids


class TreeAttentionMask:
    """Callable class for building tree attention masks (for use with HF models)."""

    def __init__(self, parent_array: List[int], prefix_length: int = 0):
        self.parent_array = parent_array
        self.prefix_length = prefix_length
        self._mask = None

    def __call__(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return attention mask in format expected by HF models."""
        if self._mask is None or self._mask.device != device:
            self._mask = build_tree_attention_mask(
                self.prefix_length, self.parent_array, device
            )
        # Convert to additive mask for attention scores
        additive_mask = torch.zeros_like(self._mask, dtype=dtype)
        additive_mask.masked_fill_(~self._mask, torch.finfo(dtype).min)
        return additive_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    def get_boolean_mask(self, device: torch.device) -> torch.Tensor:
        """Get boolean mask for custom attention implementation."""
        return build_tree_attention_mask(self.prefix_length, self.parent_array, device)