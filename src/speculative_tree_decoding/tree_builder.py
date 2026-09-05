"""Static tree construction for speculative decoding."""
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .sampling import sample_from_logits, compute_logprobs


@dataclass
class TreeNode:
    """A node in the draft tree."""
    token_id: int
    parent_idx: int
    depth: int
    logprobs: torch.Tensor  # [vocab_size]
    children: List[int] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class DraftTree:
    """A tree of draft tokens for speculative decoding."""
    nodes: List[TreeNode]
    root_idx: int = 0
    prefix_length: int = 0

    def get_flattened_tokens(self) -> List[int]:
        """Get tokens in DFS order (root first, then children recursively)."""
        return [node.token_id for node in self.nodes]

    def get_flattened_logprobs(self) -> List[torch.Tensor]:
        """Get logprobs in DFS order."""
        return [node.logprobs for node in self.nodes]

    def get_parent_array(self) -> List[int]:
        """Get parent index for each node (for attention mask)."""
        return [node.parent_idx for node in self.nodes]

    def get_depth_array(self) -> List[int]:
        """Get depth of each node from root."""
        return [node.depth for node in self.nodes]

    def get_num_nodes(self) -> int:
        return len(self.nodes)

    def get_leaf_indices(self) -> List[int]:
        """Get indices of leaf nodes (nodes with no children)."""
        return [i for i, node in enumerate(self.nodes) if len(node.children) == 0]

    def get_branch_paths(self) -> List[List[int]]:
        """Get all root-to-leaf paths as lists of node indices."""
        paths = []
        leaves = self.get_leaf_indices()
        for leaf_idx in leaves:
            path = []
            curr = leaf_idx
            while curr != -1:
                path.append(curr)
                curr = self.nodes[curr].parent_idx
            paths.append(list(reversed(path)))
        return paths


class StaticTreeBuilder:
    """
    Builds a fixed-shape tree for speculative decoding.

    Tree structure:
    - Root at depth 0 (the first draft token after prefix)
    - Each node at depth < max_depth has exactly branching_factor children
    - Total nodes = (branching_factor^(max_depth+1) - 1) / (branching_factor - 1)
    """

    def __init__(
        self,
        branching_factor: int = 3,
        max_depth: int = 2,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def build_tree(
        self,
        draft_model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> DraftTree:
        """
        Build a static tree by expanding nodes breadth-first.

        Args:
            draft_model: The draft model
            input_ids: Input token ids [1, prefix_len]
            attention_mask: Attention mask [1, prefix_len]
            device: Device for computation

        Returns:
            DraftTree with nodes in DFS order
        """
        device = device or input_ids.device
        prefix_len = input_ids.size(1)

        # Root node: first draft token after prefix
        nodes = []
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone() if attention_mask is not None else None

        # Generate root token
        with torch.no_grad():
            outputs = draft_model(
                input_ids=current_ids,
                attention_mask=current_mask,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            next_logprobs = compute_logprobs(next_logits)

            root_token = sample_from_logits(
                next_logits.squeeze(0),
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            ).item()

        root_node = TreeNode(
            token_id=root_token,
            parent_idx=-1,
            depth=0,
            logprobs=next_logprobs.squeeze(0),
        )
        nodes.append(root_node)

        # Update current_ids for children generation
        current_ids = torch.cat([current_ids, torch.tensor([[root_token]], device=device)], dim=1)
        if current_mask is not None:
            current_mask = torch.cat([current_mask, torch.ones_like(current_mask[:, :1])], dim=1)

        # BFS queue: (node_idx, current_ids, current_mask, past_key_values)
        from collections import deque
        queue = deque()
        queue.append((0, current_ids, current_mask, outputs.past_key_values))

        while queue:
            parent_idx, parent_ids, parent_mask, parent_kv = queue.popleft()
            parent_node = nodes[parent_idx]

            if parent_node.depth >= self.max_depth:
                continue

            # Generate children for this parent
            # In greedy mode, use top-k distinct tokens; in sampling mode, use Gumbel-Top-k for diversity
            if self.temperature == 0.0:
                # Greedy mode: get top-k distinct tokens by logit
                with torch.no_grad():
                    outputs = draft_model(
                        input_ids=parent_ids[:, -1:],
                        attention_mask=parent_mask[:, -1:] if parent_mask is not None else None,
                        past_key_values=parent_kv,
                        use_cache=True,
                    )
                next_logits = outputs.logits[:, -1, :]
                next_logprobs = compute_logprobs(next_logits).squeeze(0)
                
                # Get top-k distinct tokens
                topk = min(self.branching_factor, next_logits.size(-1))
                topk_values, topk_indices = torch.topk(next_logits.squeeze(0), topk)
                child_tokens = topk_indices.tolist()
                
                for child_i, child_token in enumerate(child_tokens):
                    child_idx = len(nodes)
                    child_node = TreeNode(
                        token_id=child_token,
                        parent_idx=parent_idx,
                        depth=parent_node.depth + 1,
                        logprobs=next_logprobs,
                    )
                    nodes.append(child_node)
                    parent_node.children.append(child_idx)
                    
                    # Prepare for this child's children
                    new_ids = torch.cat([parent_ids, torch.tensor([[child_token]], device=device)], dim=1)
                    new_mask = None
                    if parent_mask is not None:
                        new_mask = torch.cat([parent_mask, torch.ones_like(parent_mask[:, :1])], dim=1)
                    queue.append((child_idx, new_ids, new_mask, outputs.past_key_values))
            else:
                # Sampling mode: use Gumbel-Top-k for diverse children
                with torch.no_grad():
                    outputs = draft_model(
                        input_ids=parent_ids[:, -1:],
                        attention_mask=parent_mask[:, -1:] if parent_mask is not None else None,
                        past_key_values=parent_kv,
                        use_cache=True,
                    )
                next_logits = outputs.logits[:, -1, :]
                next_logprobs = compute_logprobs(next_logits).squeeze(0)
                
                # Gumbel-Top-k for diverse sampling without replacement
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(next_logprobs) + 1e-10) + 1e-10)
                perturbed = next_logprobs + gumbel_noise
                topk = min(self.branching_factor, perturbed.size(-1))
                topk_values, topk_indices = torch.topk(perturbed, topk)
                child_tokens = topk_indices.tolist()
                
                for child_i, child_token in enumerate(child_tokens):
                    child_idx = len(nodes)
                    child_node = TreeNode(
                        token_id=child_token,
                        parent_idx=parent_idx,
                        depth=parent_node.depth + 1,
                        logprobs=next_logprobs,
                    )
                    nodes.append(child_node)
                    parent_node.children.append(child_idx)
                    
                    # Prepare for this child's children
                    new_ids = torch.cat([parent_ids, torch.tensor([[child_token]], device=device)], dim=1)
                    new_mask = None
                    if parent_mask is not None:
                        new_mask = torch.cat([parent_mask, torch.ones_like(parent_mask[:, :1])], dim=1)
                    queue.append((child_idx, new_ids, new_mask, outputs.past_key_values))

        return DraftTree(nodes=nodes, prefix_length=prefix_len)

    def get_tree_stats(self, tree: DraftTree) -> dict:
        """Get statistics about the tree."""
        return {
            "num_nodes": tree.get_num_nodes(),
            "max_depth": max(n.depth for n in tree.nodes) if tree.nodes else 0,
            "num_leaves": len(tree.get_leaf_indices()),
            "branching_factor": self.branching_factor,
            "max_depth_config": self.max_depth,
        }


def build_static_tree_from_logits(
    draft_logits_per_step: List[torch.Tensor],
    branching_factor: int,
    max_depth: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> DraftTree:
    """
    Build a static tree from pre-computed logits (for testing).

    Args:
        draft_logits_per_step: List of logits [vocab_size] for each expansion step
        branching_factor: Number of children per node
        max_depth: Maximum depth of tree
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling limit

    Returns:
        DraftTree built from the provided logits
    """
    nodes = []

    # Root from first logits
    root_logprobs = compute_logprobs(draft_logits_per_step[0].unsqueeze(0)).squeeze(0)
    root_token = sample_from_logits(
        draft_logits_per_step[0],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    ).item()

    root_node = TreeNode(
        token_id=root_token,
        parent_idx=-1,
        depth=0,
        logprobs=root_logprobs,
    )
    nodes.append(root_node)

    logit_idx = 1
    queue = [(0, 0)]  # (node_idx, depth)

    while queue and logit_idx < len(draft_logits_per_step):
        parent_idx, depth = queue.pop(0)
        parent_node = nodes[parent_idx]

        if depth >= max_depth:
            continue

        for _ in range(branching_factor):
            if logit_idx >= len(draft_logits_per_step):
                break

            logprobs = compute_logprobs(draft_logits_per_step[logit_idx].unsqueeze(0)).squeeze(0)
            token = sample_from_logits(
                draft_logits_per_step[logit_idx],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ).item()

            child_idx = len(nodes)
            child_node = TreeNode(
                token_id=token,
                parent_idx=parent_idx,
                depth=depth + 1,
                logprobs=logprobs,
            )
            nodes.append(child_node)
            parent_node.children.append(child_idx)

            queue.append((child_idx, depth + 1))
            logit_idx += 1

    return DraftTree(nodes=nodes)