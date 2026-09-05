"""Dynamic/adaptive tree construction for speculative decoding."""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import heapq
import math

from .tree_builder import DraftTree, TreeNode
from .sampling import sample_from_logits, compute_logprobs


@dataclass
class FrontierNode:
    """A node on the expansion frontier with its cumulative probability."""
    node_idx: int
    cumulative_logprob: float
    depth: int

    def __lt__(self, other):
        # For max-heap (highest probability first), negate
        return self.cumulative_logprob > other.cumulative_logprob


class DynamicTreeBuilder:
    """
    Builds an adaptive draft tree using Gumbel-Top-k or stochastic beam search.

    Key features:
    - Expands nodes with highest cumulative probability first
    - Uses Gumbel-Top-k for diverse branch selection without replacement
    - Stops when compute budget or cumulative probability threshold is reached
    - Supports both fixed budget (max nodes) and adaptive thresholds
    """

    def __init__(
        self,
        max_nodes: int = 15,
        max_depth: int = 4,
        branching_factor: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        cum_prob_threshold: float = 0.0,  # Stop if total prob of frontier < threshold
        use_gumbel_topk: bool = True,
        gumbel_temperature: float = 1.0,
        min_branch_prob: float = 0.01,  # Minimum prob for a branch to be expanded
    ):
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.cum_prob_threshold = cum_prob_threshold
        self.use_gumbel_topk = use_gumbel_topk
        self.gumbel_temperature = gumbel_temperature
        self.min_branch_prob = min_branch_prob

    def build_tree(
        self,
        draft_model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> DraftTree:
        """
        Build an adaptive tree by expanding high-probability branches first.

        Uses best-first search on the draft tree, guided by draft model probabilities.
        """
        device = device or input_ids.device
        prefix_len = input_ids.size(1)

        nodes = []

        # Generate root token
        with torch.no_grad():
            outputs = draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            next_logprobs = compute_logprobs(next_logits).squeeze(0)

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
            logprobs=next_logprobs,
        )
        nodes.append(root_node)

        # Priority queue for frontier nodes (max-heap by cumulative logprob)
        # Each entry: (negative_cum_logprob, node_idx, depth, input_ids, attention_mask, past_key_values)
        frontier = []
        root_cum_logprob = next_logprobs[root_token].item()
        heapq.heappush(frontier, (-root_cum_logprob, 0, 0, 
                                   outputs.past_key_values,
                                   input_ids,
                                   attention_mask))

        nodes_created = 1

        while frontier and nodes_created < self.max_nodes:
            # Pop highest probability node to expand
            neg_cum_logprob, node_idx, depth, parent_kv, parent_ids, parent_mask = heapq.heappop(frontier)
            cum_logprob = -neg_cum_logprob

            parent_node = nodes[node_idx]

            if depth >= self.max_depth:
                continue

            # Check if we should stop expanding this branch
            if self.cum_prob_threshold > 0 and math.exp(cum_logprob) < self.cum_prob_threshold:
                continue

            # Generate children for this parent
            # We generate multiple candidates and select top-k using Gumbel-Top-k
            child_candidates = self._generate_child_candidates(
                draft_model, parent_ids, parent_mask, parent_kv, device
            )

            for child_token, child_logprobs, child_kv, child_ids, child_mask, child_logprob in child_candidates:
                if nodes_created >= self.max_nodes:
                    break

                # Check minimum branch probability
                new_cum_logprob = cum_logprob + child_logprob
                if self.min_branch_prob > 0 and math.exp(new_cum_logprob) < self.min_branch_prob:
                    continue

                child_idx = len(nodes)
                child_node = TreeNode(
                    token_id=child_token,
                    parent_idx=node_idx,
                    depth=depth + 1,
                    logprobs=child_logprobs,
                )
                nodes.append(child_node)
                parent_node.children.append(child_idx)
                nodes_created += 1

                # Add to frontier for further expansion
                heapq.heappush(frontier, (-new_cum_logprob, child_idx, depth + 1,
                                          child_kv, child_ids, child_mask))

        return DraftTree(nodes=nodes, prefix_length=prefix_len)

    def _generate_child_candidates(
        self,
        draft_model,
        parent_ids: torch.Tensor,
        parent_mask: Optional[torch.Tensor],
        parent_kv,
        device: torch.device,
    ) -> List[Tuple[int, torch.Tensor, Any, torch.Tensor, Optional[torch.Tensor], float]]:
        """
        Generate multiple child candidates for a parent node.

        Returns list of (token_id, logprobs, kv_cache, input_ids, attention_mask, logprob)
        """
        candidates = []

        if self.use_gumbel_topk:
            # Use Gumbel-Top-k for diverse sampling without replacement
            with torch.no_grad():
                outputs = draft_model(
                    input_ids=parent_ids[:, -1:],
                    attention_mask=parent_mask[:, -1:] if parent_mask is not None else None,
                    past_key_values=parent_kv,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :] / self.gumbel_temperature
            logprobs = compute_logprobs(logits).squeeze(0)

            # Gumbel-Top-k: add Gumbel noise and take top-k
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logprobs) + 1e-10) + 1e-10)
            perturbed = logprobs + gumbel_noise
            topk_values, topk_indices = torch.topk(perturbed, min(self.branching_factor, logprobs.size(-1)))

            for i in range(topk_indices.size(0)):
                token = topk_indices[i].item()
                logprob = logprobs[token].item()
                candidates.append((
                    token,
                    logprobs,  # Full logprobs for this position
                    outputs.past_key_values,
                    torch.cat([parent_ids, torch.tensor([[token]], device=device)], dim=1),
                    torch.cat([parent_mask, torch.ones_like(parent_mask[:, :1])], dim=1) if parent_mask is not None else None,
                    logprob,
                ))
        else:
            # Standard top-k sampling with temperature
            for _ in range(self.branching_factor):
                with torch.no_grad():
                    outputs = draft_model(
                        input_ids=parent_ids[:, -1:],
                        attention_mask=parent_mask[:, -1:] if parent_mask is not None else None,
                        past_key_values=parent_kv,
                        use_cache=True,
                    )
                logits = outputs.logits[:, -1, :]
                logprobs = compute_logprobs(logits).squeeze(0)

                token = sample_from_logits(
                    logits.squeeze(0),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                ).item()
                logprob = logprobs[token].item()

                candidates.append((
                    token,
                    logprobs,
                    outputs.past_key_values,
                    torch.cat([parent_ids, torch.tensor([[token]], device=device)], dim=1),
                    torch.cat([parent_mask, torch.ones_like(parent_mask[:, :1])], dim=1) if parent_mask is not None else None,
                    logprob,
                ))

        # Sort by logprob descending
        candidates.sort(key=lambda x: x[5], reverse=True)
        return candidates[:self.branching_factor]


class AdaptiveTreeBuilder:
    """
    Adaptive tree builder that adjusts structure based on draft model confidence.

    - High entropy -> wider branching (more exploration)
    - Low entropy -> deeper branching (more exploitation)
    - Uses cumulative probability to decide where to allocate budget
    """

    def __init__(
        self,
        max_nodes: int = 20,
        max_depth: int = 4,
        base_branching: int = 3,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        entropy_threshold: float = 2.0,  # High entropy -> more branches
    ):
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.base_branching = base_branching
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.entropy_threshold = entropy_threshold

    def _compute_entropy(self, logprobs: torch.Tensor) -> float:
        """Compute entropy of a probability distribution."""
        probs = logprobs.exp()
        return -(probs * logprobs).sum().item()

    def _get_branching_for_node(self, logprobs: torch.Tensor) -> int:
        """Determine branching factor based on entropy."""
        entropy = self._compute_entropy(logprobs)
        if entropy > self.entropy_threshold:
            return min(self.base_branching + 1, 6)  # More exploration
        else:
            return max(self.base_branching - 1, 1)  # More exploitation

    def build_tree(
        self,
        draft_model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> DraftTree:
        """Build adaptive tree with entropy-based branching."""
        device = device or input_ids.device
        prefix_len = input_ids.size(1)

        nodes = []

        # Generate root
        with torch.no_grad():
            outputs = draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            next_logprobs = compute_logprobs(next_logits).squeeze(0)

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
            logprobs=next_logprobs,
        )
        nodes.append(root_node)

        # BFS with adaptive branching
        queue = deque()
        branching = self._get_branching_for_node(next_logprobs)
        queue.append((0, outputs.past_key_values,
                     torch.cat([input_ids, torch.tensor([[root_token]], device=device)], dim=1),
                     torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1) if attention_mask is not None else None,
                     0))  # (node_idx, kv, ids, mask, depth)

        nodes_created = 1

        while queue and nodes_created < self.max_nodes:
            node_idx, parent_kv, parent_ids, parent_mask, depth = queue.popleft()

            if depth >= self.max_depth:
                continue

            parent_node = nodes[node_idx]
            branching = self._get_branching_for_node(parent_node.logprobs)

            for _ in range(branching):
                if nodes_created >= self.max_nodes:
                    break

                with torch.no_grad():
                    outputs = draft_model(
                        input_ids=parent_ids[:, -1:],
                        attention_mask=parent_mask[:, -1:] if parent_mask is not None else None,
                        past_key_values=parent_kv,
                        use_cache=True,
                    )

                next_logits = outputs.logits[:, -1, :]
                next_logprobs = compute_logprobs(next_logits).squeeze(0)

                child_token = sample_from_logits(
                    next_logits.squeeze(0),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                ).item()

                child_idx = len(nodes)
                child_node = TreeNode(
                    token_id=child_token,
                    parent_idx=node_idx,
                    depth=depth + 1,
                    logprobs=next_logprobs,
                )
                nodes.append(child_node)
                parent_node.children.append(child_idx)
                nodes_created += 1

                queue.append((child_idx, outputs.past_key_values,
                             torch.cat([parent_ids, torch.tensor([[child_token]], device=device)], dim=1),
                             torch.cat([parent_mask, torch.ones_like(parent_mask[:, :1])], dim=1) if parent_mask is not None else None,
                             depth + 1))

        return DraftTree(nodes=nodes, prefix_length=prefix_len)
