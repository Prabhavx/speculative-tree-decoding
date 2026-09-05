"""Tree-aware KV-cache management for speculative decoding."""
import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from transformers import PretrainedConfig
from .tree_attention import build_tree_position_ids, build_tree_attention_mask


@dataclass
class TreeKVCacheState:
    """State snapshot for tree KV-cache rollback."""
    prefix_length: int
    tree_start_length: int
    num_tree_nodes: int


class TreeKVCache:
    """
    Tree-aware KV-cache for target model during tree verification.

    Key features:
    - Stores prefix KV-cache separately from tree nodes
    - Supports efficient tree verification by appending tree nodes to prefix
    - Enables rollback to prefix on rejection
    - Manages per-node position IDs for RoPE
    """

    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = 1,
        max_length: int = 4096,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        self.num_layers = config.num_hidden_layers
        # Use num_key_value_heads for GQA models, fallback to num_attention_heads
        self.num_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Main KV cache storage
        self.k_cache = torch.zeros(
            (self.num_layers, batch_size, self.num_heads, max_length, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (self.num_layers, batch_size, self.num_heads, max_length, self.head_dim),
            dtype=dtype,
            device=device,
        )

        # Track current state
        self.prefix_length = 0  # Length of verified prefix
        self.tree_start_length = 0  # Where tree nodes start in cache
        self.num_tree_nodes = 0  # Number of tree nodes currently in cache

        # History for rollback
        self._state_history: List[TreeKVCacheState] = []

    def get_seq_length(self) -> int:
        """Total sequence length in cache (prefix + tree nodes)."""
        return self.prefix_length + self.num_tree_nodes

    def get_prefix_length(self) -> int:
        return self.prefix_length

    def set_prefix(self, prefix_length: int):
        """Set the verified prefix length (after accepting tokens)."""
        self.prefix_length = prefix_length
        self.tree_start_length = prefix_length
        self.num_tree_nodes = 0

    def save_state(self) -> TreeKVCacheState:
        """Save current state for potential rollback."""
        state = TreeKVCacheState(
            prefix_length=self.prefix_length,
            tree_start_length=self.tree_start_length,
            num_tree_nodes=self.num_tree_nodes,
        )
        self._state_history.append(state)
        return state

    def rollback(self):
        """Rollback to the last saved state (on rejection)."""
        if self._state_history:
            state = self._state_history.pop()
            self.prefix_length = state.prefix_length
            self.tree_start_length = state.tree_start_length
            self.num_tree_nodes = state.num_tree_nodes

    def clear_history(self):
        """Clear rollback history."""
        self._state_history.clear()

    def prepare_for_tree_verification(
        self,
        tree_num_nodes: int,
    ) -> Tuple[int, int]:
        """
        Prepare cache for tree verification.

        Returns:
            (start_pos, end_pos) - positions in cache where tree nodes will be written
        """
        self.save_state()
        self.tree_start_length = self.prefix_length
        self.num_tree_nodes = tree_num_nodes
        return self.prefix_length, self.prefix_length + tree_num_nodes

    def accept_tree_path(
        self,
        accepted_node_count: int,
    ):
        """
        Accept a path from the tree and update prefix.

        Args:
            accepted_node_count: Number of tree nodes accepted (along the path)
        """
        self.prefix_length += accepted_node_count
        self.tree_start_length = self.prefix_length
        self.num_tree_nodes = 0
        self.clear_history()

    def reject_tree(self):
        """Reject the entire tree, rollback to prefix."""
        self.rollback()

    def update_from_model_output(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        write_pos: int,
    ):
        """
        Update cache from model output at specific position.

        Args:
            layer_idx: Layer index
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]
            write_pos: Position in cache to write
        """
        seq_len = key.size(2)
        end_pos = write_pos + seq_len
        if end_pos > self.max_length:
            raise ValueError(f"Cache overflow: {end_pos} > {self.max_length}")

        self.k_cache[layer_idx, :, :, write_pos:end_pos, :] = key
        self.v_cache[layer_idx, :, :, write_pos:end_pos, :] = value

    def get_past_key_values(self, seq_len: Optional[int] = None):
        """
        Get past_key_values as DynamicCache for modern HF models.

        Args:
            seq_len: Sequence length to return (default: prefix + tree nodes)

        Returns:
            DynamicCache for HF models
        """
        from transformers import DynamicCache
        if seq_len is None:
            seq_len = self.get_seq_length()

        dynamic_cache = DynamicCache()
        for layer_idx in range(self.num_layers):
            k = self.k_cache[layer_idx, :, :, :seq_len, :]
            v = self.v_cache[layer_idx, :, :, :seq_len, :]
            dynamic_cache.update(k, v, layer_idx)
        return dynamic_cache

    def get_tree_past_key_values(self):
        """Get past_key_values for just the tree portion (for incremental tree building)."""
        from transformers import DynamicCache
        if self.num_tree_nodes == 0:
            return self.get_past_key_values(self.prefix_length)

        dynamic_cache = DynamicCache()
        for layer_idx in range(self.num_layers):
            k = self.k_cache[layer_idx, :, :, :self.tree_start_length + self.num_tree_nodes, :]
            v = self.v_cache[layer_idx, :, :, :self.tree_start_length + self.num_tree_nodes, :]
            dynamic_cache.update(k, v, layer_idx)
        return dynamic_cache

    def reset(self):
        """Reset cache to empty state."""
        self.prefix_length = 0
        self.tree_start_length = 0
        self.num_tree_nodes = 0
        self._state_history.clear()
        self.k_cache.zero_()
        self.v_cache.zero_()

    def get_memory_usage_mb(self) -> float:
        """Get approximate memory usage in MB."""
        total_elements = self.k_cache.numel() + self.v_cache.numel()
        bytes_per_element = self.k_cache.element_size()
        return (total_elements * bytes_per_element) / (1024 ** 2)


class DraftTreeKVCache:
    """
    KV-cache for draft model during tree construction.

    Supports:
    - Building tree by expanding different branches from shared prefix
    - Maintaining separate KV states for each frontier node
    """

    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = 1,
        max_length: int = 4096,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        self.num_layers = config.num_hidden_layers
        # Use num_key_value_heads for GQA models, fallback to num_attention_heads
        self.num_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Base cache for prefix
        self.k_cache = torch.zeros(
            (self.num_layers, batch_size, self.num_heads, max_length, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (self.num_layers, batch_size, self.num_heads, max_length, self.head_dim),
            dtype=dtype,
            device=device,
        )

        self.prefix_length = 0
        self._branch_states: Dict[int, Tuple] = {}  # node_idx -> past_key_values

    def set_prefix(self, prefix_ids: torch.Tensor, draft_model, attention_mask: Optional[torch.Tensor] = None):
        """Initialize cache with prefix."""
        with torch.no_grad():
            outputs = draft_model(
                input_ids=prefix_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        self._store_past_key_values(outputs.past_key_values)
        self.prefix_length = prefix_ids.size(1)

    def _store_past_key_values(self, past_key_values):
        if hasattr(past_key_values, 'layers'):
            layers = past_key_values.layers
        else:
            layers = past_key_values
        
        for layer_idx, layer_kv in enumerate(layers):
            if hasattr(layer_kv, 'keys') and hasattr(layer_kv, 'values'):
                k, v = layer_kv.keys, layer_kv.values
            elif isinstance(layer_kv, (tuple, list)):
                k, v = layer_kv[0], layer_kv[1]
            else:
                raise ValueError(f"Unknown past_key_values format at layer {layer_idx}: {type(layer_kv)}")
            seq_len = k.size(2)
            self.k_cache[layer_idx, :, :, :seq_len, :] = k
            self.v_cache[layer_idx, :, :, :seq_len, :] = v

    def get_branch_state(self, node_idx: int) -> Optional[Tuple]:
        """Get saved KV state for a branch node."""
        return self._branch_states.get(node_idx)

    def save_branch_state(self, node_idx: int, past_key_values):
        """Save KV state for a branch node."""
        # Clone to avoid mutation
        cloned = []
        for k, v in past_key_values:
            cloned.append((k.clone(), v.clone()))
        self._branch_states[node_idx] = tuple(cloned)

    def load_branch_state(self, node_idx: int) -> Optional[Tuple]:
        """Load KV state for a branch node."""
        state = self._branch_states.get(node_idx)
        if state is None:
            return None
        # Return cloned tensors
        return tuple((k.clone(), v.clone()) for k, v in state)

    def clear_branch_states(self):
        """Clear all saved branch states."""
        self._branch_states.clear()

    def get_past_key_values(self, seq_len: int) -> Tuple:
        """Get past_key_values up to seq_len."""
        past_key_values = []
        for layer_idx in range(self.num_layers):
            k = self.k_cache[layer_idx, :, :, :seq_len, :]
            v = self.v_cache[layer_idx, :, :, :seq_len, :]
            past_key_values.append((k, v))
        return tuple(past_key_values)

    def reset(self):
        self.prefix_length = 0
        self._branch_states.clear()
        self.k_cache.zero_()
        self.v_cache.zero_()