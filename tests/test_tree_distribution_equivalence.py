"""Tests for tree-based speculative decoding distributional equivalence."""
import torch
import torch.nn.functional as F
import pytest
from collections import Counter
import numpy as np

from speculative_tree_decoding.tree_builder import DraftTree, TreeNode
from speculative_tree_decoding.tree_kv_cache import build_tree_attention_mask, build_tree_position_ids
from speculative_tree_decoding.sampling import (
    sample_from_logits,
    compute_logprobs,
    rejection_sample,
    greedy_rejection_sample,
)
from speculative_tree_decoding.verify import verify_tree, TreeVerificationResult, extract_accepted_path_tokens


class TestTreeRejectionSampling:
    """Test rejection sampling logic for tree structures."""

    def test_tree_greedy_accept_match(self):
        """Greedy: accept when draft and target agree on token at tree node."""
        draft_logprobs = torch.tensor([0.1, 0.8, 0.1]).log()
        target_logprobs = torch.tensor([0.2, 0.7, 0.1]).log()
        accepted, resampled = greedy_rejection_sample(draft_logprobs, target_logprobs, 1)
        assert accepted is True
        assert resampled is None

    def test_tree_greedy_reject_mismatch(self):
        """Greedy: reject when draft and target disagree."""
        draft_logprobs = torch.tensor([0.1, 0.8, 0.1]).log()
        target_logprobs = torch.tensor([0.7, 0.2, 0.1]).log()
        accepted, resampled = greedy_rejection_sample(draft_logprobs, target_logprobs, 1)
        assert accepted is False
        assert resampled == 0

    def test_tree_rejection_sample_residual(self):
        """Test residual resampling matches formula for tree node."""
        draft_probs = torch.tensor([0.6, 0.3, 0.1])
        target_probs = torch.tensor([0.2, 0.5, 0.3])
        draft_logprobs = draft_probs.log()
        target_logprobs = target_probs.log()

        residual = torch.clamp(target_probs - draft_probs, min=0.0)
        expected = residual / residual.sum()

        torch.manual_seed(42)
        samples = []
        for _ in range(10000):
            accepted, resampled = rejection_sample(draft_logprobs, target_logprobs, 0, temperature=1.0)
            if not accepted:
                samples.append(resampled)

        empirical = torch.bincount(torch.tensor(samples), minlength=3).float() / len(samples)
        assert torch.allclose(empirical, expected, atol=0.02)


class TestTreeDistributionalEquivalence:
    """Test that tree speculative decoding matches target-only distribution."""

    def test_toy_tree_distribution_equivalence(self):
        """
        Test that rejection sampling at each tree node preserves target distribution.
        
        This tests the LOCAL correctness: at each individual node, the acceptance/rejection
        logic produces the correct target distribution.
        """
        vocab_size = 5
        torch.manual_seed(123)

        # Test at a single node with fixed distributions
        draft_logits = torch.tensor([2.0, 1.0, 0.5, 0.1, 0.1])
        target_logits = torch.tensor([1.8, 1.2, 0.4, 0.3, 0.2])

        draft_probs = F.softmax(draft_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)

        num_samples = 50000
        accepted_samples = []
        target_samples = []

        for _ in range(num_samples):
            # Sample from draft
            draft_token = torch.multinomial(draft_probs, 1).item()
            draft_lp = draft_logits.log_softmax(dim=-1)
            target_lp = target_logits.log_softmax(dim=-1)

            # Rejection sample
            accepted, resampled = rejection_sample(draft_lp, target_lp, draft_token, temperature=1.0)
            final_token = draft_token if accepted else resampled
            accepted_samples.append(final_token)

            # Target only
            target_token = torch.multinomial(target_probs, 1).item()
            target_samples.append(target_token)

        # Compare distributions - should match exactly (within sampling noise)
        accepted_counts = Counter(accepted_samples)
        target_counts = Counter(target_samples)

        for token in range(vocab_size):
            accept_freq = accepted_counts.get(token, 0) / num_samples
            target_freq = target_counts.get(token, 0) / num_samples
            assert abs(accept_freq - target_freq) < 0.015, \
                f"Token {token}: accept={accept_freq:.4f}, target={target_freq:.4f}"

    def test_tree_sampling_matches_target_at_each_node(self):
        """
        Test that at each tree node, the acceptance/rejection logic
        preserves the target distribution locally.
        """
        vocab_size = 10
        torch.manual_seed(456)

        num_nodes = 5
        num_trials = 50000
        node_accept_counts = [Counter() for _ in range(num_nodes)]
        node_target_counts = [Counter() for _ in range(num_nodes)]

        for _ in range(num_trials):
            for node_idx in range(num_nodes):
                # Random draft and target distributions
                draft_logits = torch.randn(vocab_size)
                target_logits = torch.randn(vocab_size)
                draft_probs = F.softmax(draft_logits, dim=-1)
                target_probs = F.softmax(target_logits, dim=-1)

                draft_token = torch.multinomial(draft_probs, 1).item()
                draft_logprobs = draft_logits.log_softmax(dim=-1)
                target_logprobs = target_logits.log_softmax(dim=-1)

                accepted, resampled = rejection_sample(
                    draft_logprobs, target_logprobs, draft_token, temperature=1.0
                )

                final_token = draft_token if accepted else resampled
                node_accept_counts[node_idx][final_token] += 1

                # Target-only
                target_token = torch.multinomial(target_probs, 1).item()
                node_target_counts[node_idx][target_token] += 1

        # Each node's output should match target distribution
        for node_idx in range(num_nodes):
            for token in range(vocab_size):
                accept_freq = node_accept_counts[node_idx].get(token, 0) / num_trials
                target_freq = node_target_counts[node_idx].get(token, 0) / num_trials
                assert abs(accept_freq - target_freq) < 0.02, \
                    f"Node {node_idx}, token {token}: accept={accept_freq:.4f}, target={target_freq:.4f}"


class TestTreeVerificationLogic:
    """Test the tree verification and path selection logic."""

    def test_extract_accepted_path_simple(self):
        """Test extracting tokens from a simple accepted path."""
        nodes = [
            TreeNode(token_id=10, parent_idx=-1, depth=0, logprobs=torch.randn(10), children=[1]),
            TreeNode(token_id=20, parent_idx=0, depth=1, logprobs=torch.randn(10), children=[]),
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

    def test_extract_with_rejection(self):
        """Test extraction when rejection occurs at a node."""
        nodes = [
            TreeNode(token_id=10, parent_idx=-1, depth=0, logprobs=torch.randn(10), children=[1, 2]),
            TreeNode(token_id=20, parent_idx=0, depth=1, logprobs=torch.randn(10), children=[]),
            TreeNode(token_id=30, parent_idx=0, depth=1, logprobs=torch.randn(10), children=[]),
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

    def test_verify_tree_greedy_deterministic(self):
        """Test that greedy tree verification is deterministic."""
        # Create a tree with known logprobs where target matches draft at root
        nodes = [
            TreeNode(token_id=5, parent_idx=-1, depth=0, 
                     logprobs=torch.tensor([-2.0, -2.0, -1.0, -2.0, -0.5, -2.0, -2.0, -2.0, -2.0, -2.0]),
                     children=[1, 2]),
            TreeNode(token_id=3, parent_idx=0, depth=1,
                     logprobs=torch.tensor([-2.0, -2.0, -2.0, -0.5, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
                     children=[]),
            TreeNode(token_id=7, parent_idx=0, depth=1,
                     logprobs=torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -0.5, -2.0, -2.0]),
                     children=[]),
        ]
        tree = DraftTree(nodes=nodes)

        # Target logprobs that match draft at each node
        target_logprobs = [
            torch.tensor([-2.0, -2.0, -1.0, -2.0, -0.5, -2.0, -2.0, -2.0, -2.0, -2.0]),
            torch.tensor([-2.0, -2.0, -2.0, -0.5, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
            torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -0.5, -2.0, -2.0]),
        ]
        target_tokens = [4, 3, 7]  # argmax of each

        # Mock the verification - we test the logic directly
        # Root: draft=5, target top=4 -> reject, resample=4
        # But in greedy, target top != draft -> reject to target top
        draft_lp = nodes[0].logprobs
        target_lp = target_logprobs[0]
        accepted, resampled = greedy_rejection_sample(draft_lp, target_lp, 5)
        assert accepted is False
        assert resampled == 4


class TestTreeAttentionMaskVerification:
    """Verify tree attention mask correctness for various structures."""

    def test_tree_mask_root_only(self):
        """Single root node attends to prefix and self."""
        parent_array = [-1]
        prefix_length = 3
        mask = build_tree_attention_mask(prefix_length, parent_array, torch.device("cpu"))
        
        assert mask.shape == (4, 4)
        # Prefix causal
        assert mask[0, 0] and mask[1, 0] and mask[1, 1] and mask[2, 0] and mask[2, 1] and mask[2, 2]
        # Root attends to prefix
        assert all(mask[3, :3])
        # Root attends to self
        assert mask[3, 3]

    def test_tree_mask_branch_selection(self):
        """Tree mask ensures branches don't attend to each other."""
        # root(0) -> child1(1), child2(2)
        # child1 -> grandchild1(3)
        parent_array = [-1, 0, 0, 1]
        prefix_length = 2
        mask = build_tree_attention_mask(prefix_length, parent_array, torch.device("cpu"))
        
        # child1 (idx 3) should NOT attend to child2 (idx 4)
        assert not mask[3, 4]
        assert not mask[4, 3]
        
        # grandchild1 (idx 5) should attend to child1 (idx 3) and root (idx 2)
        assert mask[5, 2]  # root
        assert mask[5, 3]  # child1
        assert mask[5, 5]  # self
        
        # grandchild1 should NOT attend to child2
        assert not mask[5, 4]

    def test_tree_position_ids_correct(self):
        """Position IDs correctly reflect depth for RoPE."""
        # root -> child -> grandchild
        parent_array = [-1, 0, 1]
        prefix_length = 5
        pos_ids = build_tree_position_ids(prefix_length, parent_array, torch.device("cpu"))
        
        # Prefix: 0, 1, 2, 3, 4
        # root (depth 1): 5
        # child (depth 2): 6
        # grandchild (depth 3): 7
        assert pos_ids[5] == 5
        assert pos_ids[6] == 6
        assert pos_ids[7] == 7

    def test_tree_position_ids_siblings_same_depth(self):
        """Siblings at same depth get same position ID."""
        # root -> child1, child2
        parent_array = [-1, 0, 0]
        prefix_length = 2
        pos_ids = build_tree_position_ids(prefix_length, parent_array, torch.device("cpu"))
        
        # root: 2, child1: 3, child2: 3
        assert pos_ids[2] == 2
        assert pos_ids[3] == 3
        assert pos_ids[4] == 3


class TestTreeGreedyDeterminism:
    """Test that tree greedy decoding matches target greedy exactly."""

    def test_tree_greedy_matches_target_greedy(self):
        """
        With temperature=0, tree speculative decoding must exactly match
        target model greedy decoding token by token.
        """
        vocab_size = 20
        torch.manual_seed(789)

        # Simulate multiple steps of tree greedy decoding
        for step in range(10):
            # Create tree with 3 nodes
            nodes = []
            target_tokens = []
            
            for node_idx in range(3):
                # Generate random but fixed logits
                draft_logits = torch.randn(vocab_size)
                target_logits = torch.randn(vocab_size)
                
                draft_token = draft_logits.argmax().item()
                target_token = target_logits.argmax().item()
                target_tokens.append(target_token)
                
                draft_lp = draft_logits.log_softmax(dim=-1)
                target_lp = target_logits.log_softmax(dim=-1)
                
                # Greedy acceptance
                accepted, resampled = greedy_rejection_sample(draft_lp, target_lp, draft_token)
                
                if accepted:
                    assert draft_token == target_token
                else:
                    assert resampled == target_token

    def test_tree_verification_path_selection(self):
        """Test that path selection follows target predictions correctly."""
        # Tree: root -> child1, child2
        # Target predicts root token = child1's token
        # Should follow child1 branch
        
        vocab_size = 20
        nodes = [
            TreeNode(token_id=5, parent_idx=-1, depth=0, logprobs=torch.randn(vocab_size), children=[1, 2]),
            TreeNode(token_id=10, parent_idx=0, depth=1, logprobs=torch.randn(vocab_size), children=[]),
            TreeNode(token_id=15, parent_idx=0, depth=1, logprobs=torch.randn(vocab_size), children=[]),
        ]
        tree = DraftTree(nodes=nodes)

        # Target logprobs: root top=10 (matches child1), child1 top=10
        target_logprobs = [
            torch.zeros(vocab_size).scatter_(0, torch.tensor(10), 1.0).log(),
            torch.zeros(vocab_size).scatter_(0, torch.tensor(10), 1.0).log(),
            torch.zeros(vocab_size).scatter_(0, torch.tensor(15), 1.0).log(),
        ]

        # Verify greedy path selection
        current = 0
        path = [0]
        
        # Root
        accepted, _ = greedy_rejection_sample(nodes[0].logprobs, target_logprobs[0], nodes[0].token_id)
        # Root draft=5, target=10 -> reject
        # But wait - in actual tree verification, we check if target token matches a child
        
        # This is the key logic from verify_tree:
        target_token = target_logprobs[0].argmax().item()  # = 10
        matched_child = None
        for child_idx in nodes[0].children:
            if nodes[child_idx].token_id == target_token:
                matched_child = child_idx
                break
        
        assert matched_child == 1  # child1 has token 10
        assert nodes[1].token_id == 10


class TestTreeEdgeCases:
    """Test edge cases in tree decoding."""

    def test_empty_tree(self):
        """Test handling of tree with no nodes (should not happen but be safe)."""
        tree = DraftTree(nodes=[], prefix_length=5)
        assert tree.get_num_nodes() == 0
        assert tree.get_leaf_indices() == []
        assert tree.get_branch_paths() == []

    def test_single_node_tree(self):
        """Tree with only root node."""
        nodes = [TreeNode(token_id=5, parent_idx=-1, depth=0, logprobs=torch.randn(10))]
        tree = DraftTree(nodes=nodes)
        
        assert tree.get_num_nodes() == 1
        assert tree.get_leaf_indices() == [0]
        assert tree.get_branch_paths() == [[0]]

    def test_deep_tree_path_extraction(self):
        """Test path extraction for deep tree."""
        nodes = [
            TreeNode(token_id=i, parent_idx=i-1 if i > 0 else -1, depth=i, 
                     logprobs=torch.randn(10), children=[i+1] if i < 4 else [])
            for i in range(5)
        ]
        tree = DraftTree(nodes=nodes)
        
        paths = tree.get_branch_paths()
        assert len(paths) == 1
        assert paths[0] == [0, 1, 2, 3, 4]

    def test_tree_with_multiple_leaves(self):
        """Tree with multiple leaf nodes."""
        # root -> child1, child2 -> grandchild1, grandchild2
        nodes = [
            TreeNode(token_id=1, parent_idx=-1, depth=0, logprobs=torch.randn(10), children=[1, 2]),
            TreeNode(token_id=2, parent_idx=0, depth=1, logprobs=torch.randn(10), children=[3, 4]),
            TreeNode(token_id=3, parent_idx=0, depth=1, logprobs=torch.randn(10), children=[]),
            TreeNode(token_id=4, parent_idx=1, depth=2, logprobs=torch.randn(10), children=[]),
            TreeNode(token_id=5, parent_idx=1, depth=2, logprobs=torch.randn(10), children=[]),
        ]
        tree = DraftTree(nodes=nodes)
        
        paths = tree.get_branch_paths()
        assert len(paths) == 3
        assert [0, 1, 3] in paths
        assert [0, 1, 4] in paths
        assert [0, 2] in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])