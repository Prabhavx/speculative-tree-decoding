"""Tests for tree KV cache."""
import torch
import pytest

from speculative_tree_decoding.tree_kv_cache import TreeKVCache, DraftTreeKVCache, build_tree_position_ids, build_tree_attention_mask
from speculative_tree_decoding.tree_builder import DraftTree, TreeNode


class TestTreeKVCache:
    """Tests for tree-aware KV cache."""

    def test_tree_kv_cache_init(self):
        """Test TreeKVCache initialization."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        assert cache.num_layers == 2
        assert cache.num_heads == 4
        assert cache.head_dim == 64
        assert cache.prefix_length == 0
        assert cache.get_seq_length() == 0

    def test_set_prefix(self):
        """Test setting prefix length."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        cache.set_prefix(10)
        assert cache.prefix_length == 10
        assert cache.tree_start_length == 10
        assert cache.num_tree_nodes == 0
        assert cache.get_seq_length() == 10

    def test_prepare_for_tree_verification(self):
        """Test preparing cache for tree verification."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        cache.set_prefix(5)
        start, end = cache.prepare_for_tree_verification(7)
        
        assert start == 5
        assert end == 12
        assert cache.tree_start_length == 5
        assert cache.num_tree_nodes == 7
        assert cache.get_seq_length() == 12

    def test_accept_tree_path(self):
        """Test accepting a tree path and updating prefix."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        cache.set_prefix(5)
        cache.prepare_for_tree_verification(7)
        cache.accept_tree_path(3)
        
        assert cache.prefix_length == 8  # 5 + 3
        assert cache.tree_start_length == 8
        assert cache.num_tree_nodes == 0
        assert cache.get_seq_length() == 8

    def test_reject_tree_rollback(self):
        """Test rejecting tree rolls back to prefix."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        cache.set_prefix(5)
        cache.prepare_for_tree_verification(7)
        cache.reject_tree()
        
        assert cache.prefix_length == 5
        assert cache.tree_start_length == 5
        assert cache.num_tree_nodes == 0
        assert cache.get_seq_length() == 5

    def test_save_rollback_state(self):
        """Test saving and rolling back state."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        cache.set_prefix(5)
        cache.prepare_for_tree_verification(7)
        
        # Save state before potential rejection
        state = cache.save_state()
        assert state.prefix_length == 5
        assert state.tree_start_length == 5
        assert state.num_tree_nodes == 7
        
        # Rollback before accepting (this is the correct usage)
        cache.rollback()
        assert cache.prefix_length == 5
        assert cache.num_tree_nodes == 7
        
        # Now accept
        cache.accept_tree_path(2)
        assert cache.prefix_length == 7
        assert cache.num_tree_nodes == 0
        
        # After accept, history is cleared, so rollback does nothing
        cache.rollback()
        assert cache.prefix_length == 7

    def test_get_past_key_values(self):
        """Test getting past_key_values in DynamicCache format."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        cache.set_prefix(5)
        cache.prepare_for_tree_verification(3)
        
        past_kv = cache.get_past_key_values()
        # Now returns DynamicCache
        assert len(past_kv.layers) == 2  # num_layers
        for layer in past_kv.layers:
            assert layer.keys.shape == (1, 4, 8, 64)  # batch, heads, seq_len, head_dim
            assert layer.values.shape == (1, 4, 8, 64)

    def test_get_tree_past_key_values(self):
        """Test getting past_key_values for just tree portion."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        cache.set_prefix(5)
        cache.prepare_for_tree_verification(3)
        
        past_kv = cache.get_tree_past_key_values()
        for layer in past_kv.layers:
            assert layer.keys.shape == (1, 4, 8, 64)  # prefix + tree

    def test_reset(self):
        """Test cache reset."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = TreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        cache.set_prefix(10)
        cache.prepare_for_tree_verification(5)
        cache.reset()
        
        assert cache.prefix_length == 0
        assert cache.tree_start_length == 0
        assert cache.num_tree_nodes == 0
        assert cache.get_seq_length() == 0


class TestTreePositionIds:
    """Tests for tree position IDs."""

    def test_build_tree_position_ids_simple(self):
        """Test position IDs for simple tree."""
        # root(0) -> child1(1), child2(2)
        parent_array = [-1, 0, 0]
        prefix_length = 3
        
        pos_ids = build_tree_position_ids(prefix_length, parent_array, torch.device("cpu"))
        
        # Prefix: 0, 1, 2
        # root (depth 1): 3 + 1 - 1 = 3
        # child1 (depth 2): 3 + 2 - 1 = 4
        # child2 (depth 2): 3 + 2 - 1 = 4
        assert pos_ids[0] == 0
        assert pos_ids[1] == 1
        assert pos_ids[2] == 2
        assert pos_ids[3] == 3  # root
        assert pos_ids[4] == 4  # child1
        assert pos_ids[5] == 4  # child2

    def test_build_tree_position_ids_deep(self):
        """Test position IDs for deeper tree."""
        # root(0) -> child(1) -> grandchild(2)
        parent_array = [-1, 0, 1]
        prefix_length = 2
        
        pos_ids = build_tree_position_ids(prefix_length, parent_array, torch.device("cpu"))
        
        # Prefix: 0, 1
        # root (depth 1): 2 + 1 - 1 = 2
        # child (depth 2): 2 + 2 - 1 = 3
        # grandchild (depth 3): 2 + 3 - 1 = 4
        assert pos_ids[0] == 0
        assert pos_ids[1] == 1
        assert pos_ids[2] == 2
        assert pos_ids[3] == 3
        assert pos_ids[4] == 4


class TestTreeAttentionMask:
    """Tests for tree attention mask (from tree_kv_cache)."""

    def test_build_tree_attention_mask_simple(self):
        """Test attention mask for simple tree."""
        parent_array = [-1, 0, 0]
        prefix_length = 2
        
        mask = build_tree_attention_mask(prefix_length, parent_array, torch.device("cpu"))
        
        # Total length = 2 + 3 = 5
        assert mask.shape == (5, 5)
        
        # Prefix causal
        assert mask[0, 0] == True
        assert mask[1, 0] == True
        assert mask[1, 1] == True
        assert mask[0, 1] == False
        
        # Tree nodes attend to prefix
        for i in range(2, 5):
            assert mask[i, 0] == True
            assert mask[i, 1] == True
        
        # Root attends to self
        assert mask[2, 2] == True
        # Children attend to root and self
        assert mask[3, 2] == True  # child1 -> root
        assert mask[3, 3] == True  # child1 -> self
        assert mask[4, 2] == True  # child2 -> root
        assert mask[4, 4] == True  # child2 -> self
        
        # Siblings don't attend to each other
        assert mask[3, 4] == False
        assert mask[4, 3] == False


class TestDraftTreeKVCache:
    """Tests for draft tree KV cache."""

    def test_draft_tree_kv_cache_init(self):
        """Test DraftTreeKVCache initialization."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=256,
        )
        cache = DraftTreeKVCache(config, batch_size=1, max_length=128, device=torch.device("cpu"))
        
        assert cache.num_layers == 2
        assert cache.prefix_length == 0
        assert len(cache._branch_states) == 0

    def test_draft_tree_branch_states(self):
        """Test saving and loading branch states."""
        from transformers import PretrainedConfig
        config = PretrainedConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
        )
        cache = DraftTreeKVCache(config, batch_size=1, max_length=64, device=torch.device("cpu"))
        
        # Create dummy past_key_values
        past_kv = []
        for _ in range(1):
            k = torch.randn(1, 2, 5, 32)
            v = torch.randn(1, 2, 5, 32)
            past_kv.append((k, v))
        past_kv = tuple(past_kv)
        
        # Save branch state
        cache.save_branch_state(0, past_kv)
        
        # Load branch state
        loaded = cache.load_branch_state(0)
        assert loaded is not None
        assert len(loaded) == 1
        assert torch.allclose(loaded[0][0], past_kv[0][0])
        assert torch.allclose(loaded[0][1], past_kv[0][1])
        
        # Clear
        cache.clear_branch_states()
        assert cache.load_branch_state(0) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])