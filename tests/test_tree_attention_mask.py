"""Tests for tree attention mask."""
import torch
import pytest

from speculative_tree_decoding.tree_attention import (
    build_tree_attention_mask,
    build_tree_position_ids,
    verify_tree_mask_correctness,
    compute_node_depths,
    flatten_tree_for_verification,
    TreeAttentionMask,
)
from speculative_tree_decoding.tree_builder import DraftTree, TreeNode


class TestTreeAttentionMask:
    """Tests for tree-structured attention mask."""

    def test_chain_mask_baseline(self):
        """Baseline: linear chain mask should be lower triangular."""
        seq_len = 5
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        assert mask.sum() == seq_len * (seq_len + 1) // 2
        for i in range(seq_len):
            for j in range(seq_len):
                assert mask[i, j] == (i >= j)

    def test_simple_tree_two_children(self):
        """Tree: root(0) with two children(1,2). Prefix length = 0."""
        # Parent array: root=-1, child1=0, child2=0
        parent_array = [-1, 0, 0]
        prefix_length = 0

        mask = build_tree_attention_mask(prefix_length, parent_array)

        # Total 3 nodes
        assert mask.shape == (3, 3)

        # Root (0) attends to itself
        assert mask[0, 0] == True
        assert mask[0, 1] == False
        assert mask[0, 2] == False

        # Child 1 (1) attends to root and itself
        assert mask[1, 0] == True
        assert mask[1, 1] == True
        assert mask[1, 2] == False

        # Child 2 (2) attends to root and itself
        assert mask[2, 0] == True
        assert mask[2, 1] == False
        assert mask[2, 2] == True

        # Verify correctness
        is_correct, errors = verify_tree_mask_correctness(parent_array, prefix_length)
        assert is_correct, f"Mask errors: {errors}"

    def test_tree_with_prefix(self):
        """Tree with prefix tokens."""
        parent_array = [-1, 0, 0, 1, 1]  # root, child1, child2, grandchild1, grandchild2
        prefix_length = 3

        mask = build_tree_attention_mask(prefix_length, parent_array)
        total_len = prefix_length + len(parent_array)
        assert mask.shape == (total_len, total_len)

        # Prefix causal
        for i in range(prefix_length):
            for j in range(prefix_length):
                assert mask[i, j] == (i >= j)

        # All tree nodes attend to all prefix
        for node_idx in range(len(parent_array)):
            seq_idx = prefix_length + node_idx
            for j in range(prefix_length):
                assert mask[seq_idx, j] == True

        is_correct, errors = verify_tree_mask_correctness(parent_array, prefix_length)
        assert is_correct, f"Mask errors: {errors}"

    def test_deeper_tree(self):
        """Tree: root -> child -> grandchild (depth 2)."""
        parent_array = [-1, 0, 1]  # root, child, grandchild
        prefix_length = 0

        mask = build_tree_attention_mask(prefix_length, parent_array)

        # Root attends to self
        assert mask[0, 0] == True
        assert mask[0, 1] == False
        assert mask[0, 2] == False

        # Child attends to root and self
        assert mask[1, 0] == True
        assert mask[1, 1] == True
        assert mask[1, 2] == False

        # Grandchild attends to root, child, and self
        assert mask[2, 0] == True
        assert mask[2, 1] == True
        assert mask[2, 2] == True

        is_correct, errors = verify_tree_mask_correctness(parent_array, prefix_length)
        assert is_correct, f"Mask errors: {errors}"

    def test_tree_with_siblings_no_cross_attention(self):
        """Siblings should not attend to each other."""
        # root(0) -> child1(1), child2(2)
        # child1 -> grandchild1(3), grandchild2(4)
        # child2 -> grandchild3(5)
        parent_array = [-1, 0, 0, 1, 1, 2]
        prefix_length = 0

        mask = build_tree_attention_mask(prefix_length, parent_array)

        # Child1 (1) should NOT attend to child2 (2)
        assert mask[1, 2] == False
        assert mask[2, 1] == False

        # Grandchild1 (3) should NOT attend to grandchild3 (5)
        assert mask[3, 5] == False
        assert mask[5, 3] == False

        # Grandchild1 (3) should NOT attend to child2 (2)
        assert mask[3, 2] == False

        # Grandchild3 (5) should NOT attend to child1 (1)
        assert mask[5, 1] == False

        is_correct, errors = verify_tree_mask_correctness(parent_array, prefix_length)
        assert is_correct, f"Mask errors: {errors}"

    def test_position_ids(self):
        """Test position IDs for RoPE."""
        parent_array = [-1, 0, 0, 1, 1, 2]
        prefix_length = 2

        pos_ids = build_tree_position_ids(prefix_length, parent_array)
        total_len = prefix_length + len(parent_array)
        assert pos_ids.shape == (total_len,)

        # Prefix positions: 0, 1
        assert pos_ids[0] == 0
        assert pos_ids[1] == 1

        # Tree nodes: prefix_length + depth - 1
        # Node 0 (root): depth 1 -> pos = 2 + 1 - 1 = 2
        # Node 1 (child1): depth 2 -> pos = 2 + 2 - 1 = 3
        # Node 2 (child2): depth 2 -> pos = 3
        # Node 3 (grandchild1): depth 3 -> pos = 4
        # Node 4 (grandchild2): depth 3 -> pos = 4
        # Node 5 (grandchild3): depth 3 -> pos = 4

        assert pos_ids[2] == 2  # root
        assert pos_ids[3] == 3  # child1
        assert pos_ids[4] == 3  # child2
        assert pos_ids[5] == 4  # grandchild1
        assert pos_ids[6] == 4  # grandchild2
        assert pos_ids[7] == 4  # grandchild3

    def test_compute_node_depths(self):
        """Test depth computation."""
        parent_array = [-1, 0, 0, 1, 1, 2]
        depths = compute_node_depths(parent_array)
        assert depths == [0, 1, 1, 2, 2, 2]

    def test_flatten_tree_for_verification(self):
        """Test flattening tree for target model verification."""
        # Create a simple tree
        nodes = [
            TreeNode(token_id=10, parent_idx=-1, depth=0, logprobs=torch.randn(100)),
            TreeNode(token_id=20, parent_idx=0, depth=1, logprobs=torch.randn(100)),
            TreeNode(token_id=30, parent_idx=0, depth=1, logprobs=torch.randn(100)),
        ]
        tree = DraftTree(nodes=nodes, prefix_length=5)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        flat_ids, attn_mask, pos_ids = flatten_tree_for_verification(tree, input_ids)

        # flat_ids should be [1,2,3,4,5, 10,20,30]
        assert flat_ids.shape == (1, 8)
        assert flat_ids[0, :5].tolist() == [1, 2, 3, 4, 5]
        assert flat_ids[0, 5:].tolist() == [10, 20, 30]

        # attn_mask should be correct shape
        assert attn_mask.shape == (1, 8, 8)

        # pos_ids should be correct shape
        assert pos_ids.shape == (1, 8)

    def test_tree_attention_mask_callable(self):
        """Test TreeAttentionMask callable for HF models."""
        parent_array = [-1, 0, 0]
        prefix_length = 2

        mask_fn = TreeAttentionMask(parent_array, prefix_length)
        additive_mask = mask_fn(
            batch_size=1,
            seq_len=5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Should return additive mask with -inf for masked positions
        assert additive_mask.shape == (1, 1, 5, 5)
        # Check that masked positions have -inf
        assert additive_mask[0, 0, 2, 3] == torch.finfo(torch.float32).min  # sibling


class TestTreeBuilder:
    """Tests for static tree builder."""

    def test_build_static_tree_from_logits(self):
        """Test building tree from pre-computed logits."""
        from speculative_tree_decoding.tree_builder import build_static_tree_from_logits

        vocab_size = 10
        # Create logits for 7 nodes (1 root + 2 children + 4 grandchildren)
        logits_list = [torch.randn(vocab_size) for _ in range(7)]

        tree = build_static_tree_from_logits(
            logits_list,
            branching_factor=2,
            max_depth=2,
            temperature=1.0,
        )

        assert tree.get_num_nodes() == 7
        assert len(tree.get_leaf_indices()) == 4

    def test_tree_branch_paths(self):
        """Test extracting branch paths from tree."""
        nodes = [
            TreeNode(token_id=1, parent_idx=-1, depth=0, logprobs=torch.randn(10), children=[1, 2]),
            TreeNode(token_id=2, parent_idx=0, depth=1, logprobs=torch.randn(10), children=[3]),
            TreeNode(token_id=3, parent_idx=0, depth=1, logprobs=torch.randn(10), children=[]),
            TreeNode(token_id=4, parent_idx=1, depth=2, logprobs=torch.randn(10), children=[]),
        ]
        tree = DraftTree(nodes=nodes)

        paths = tree.get_branch_paths()
        # Should have 2 paths: [0,1,3] and [0,2]
        assert len(paths) == 2
        assert [0, 1, 3] in paths
        assert [0, 2] in paths


class TestTreeVerification:
    """Tests for tree verification logic."""

    def test_extract_accepted_path_tokens(self):
        """Test extracting accepted tokens from verification result."""
        from speculative_tree_decoding.verify import TreeVerificationResult, extract_accepted_path_tokens

        nodes = [
            TreeNode(token_id=10, parent_idx=-1, depth=0, logprobs=torch.randn(10)),
            TreeNode(token_id=20, parent_idx=0, depth=1, logprobs=torch.randn(10)),
            TreeNode(token_id=30, parent_idx=1, depth=2, logprobs=torch.randn(10)),
        ]
        tree = DraftTree(nodes=nodes)

        result = TreeVerificationResult(
            accepted_tokens=[10, 20],
            accepted_node_indices=[0, 1],
            rejected_at=None,
            resampled_token=None,
            num_accepted=2,
        )

        tokens = extract_accepted_path_tokens(tree, result)
        assert tokens == [10, 20]

    def test_extract_with_resampled(self):
        """Test extraction when rejection occurred."""
        from speculative_tree_decoding.verify import TreeVerificationResult, extract_accepted_path_tokens

        nodes = [
            TreeNode(token_id=10, parent_idx=-1, depth=0, logprobs=torch.randn(10)),
            TreeNode(token_id=20, parent_idx=0, depth=1, logprobs=torch.randn(10)),
        ]
        tree = DraftTree(nodes=nodes)

        result = TreeVerificationResult(
            accepted_tokens=[10],
            accepted_node_indices=[0],
            rejected_at=1,
            resampled_token=99,
            num_accepted=2,
        )

        tokens = extract_accepted_path_tokens(tree, result)
        assert tokens == [10, 99]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])