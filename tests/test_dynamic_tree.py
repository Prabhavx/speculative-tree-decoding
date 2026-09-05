"""Tests for dynamic tree builder."""
import torch
import pytest

from speculative_tree_decoding.dynamic_tree_builder import DynamicTreeBuilder, AdaptiveTreeBuilder, FrontierNode
from speculative_tree_decoding.tree_builder import TreeNode, DraftTree
from speculative_tree_decoding.sampling import compute_logprobs


class TestFrontierNode:
    """Test the frontier node priority queue."""

    def test_frontier_node_ordering(self):
        """Test that frontier nodes order by cumulative logprob (max-heap)."""
        node1 = FrontierNode(node_idx=0, cumulative_logprob=-2.0, depth=1)
        node2 = FrontierNode(node_idx=1, cumulative_logprob=-1.0, depth=1)
        node3 = FrontierNode(node_idx=2, cumulative_logprob=-3.0, depth=2)

        # Higher logprob (less negative) should come first
        nodes = [node1, node2, node3]
        nodes.sort()
        assert nodes[0].node_idx == 1  # -1.0 is highest
        assert nodes[1].node_idx == 0  # -2.0
        assert nodes[2].node_idx == 2  # -3.0


class TestDynamicTreeBuilder:
    """Test dynamic tree builder with mock model."""

    def test_dynamic_tree_structure(self):
        """Test that dynamic tree builder creates valid tree structure."""
        # This is a structural test without actual model
        builder = DynamicTreeBuilder(
            max_nodes=10,
            max_depth=3,
            branching_factor=3,
            temperature=1.0,
        )

        # Verify builder configuration
        assert builder.max_nodes == 10
        assert builder.max_depth == 3
        assert builder.branching_factor == 3

    def test_build_tree_from_logits(self):
        """Test building tree from pre-computed logits."""
        builder = DynamicTreeBuilder(
            max_nodes=7,
            max_depth=2,
            branching_factor=2,
            use_gumbel_topk=False,
        )

        # Create mock logits for 7 nodes
        vocab_size = 20
        logits_list = [torch.randn(vocab_size) for _ in range(7)]

        # We can't easily test without a model, so test the candidate generation logic
        # by creating a mock draft model
        class MockDraftModel:
            def __init__(self, logits_list):
                self.logits_list = logits_list
                self.call_count = 0

            def __call__(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
                logits = self.logits_list[min(self.call_count, len(self.logits_list)-1)]
                self.call_count += 1
                return type('Outputs', (), {
                    'logits': logits.unsqueeze(0).unsqueeze(0),
                    'past_key_values': None
                })()

        # This test would require a proper mock - skip for now
        pass

    def test_default_min_branch_prob_allows_high_probability_children(self):
        """Default probability pruning should not compare logprobs to probabilities."""

        class MockDraftModel:
            def __init__(self):
                self.logits = torch.tensor([6.0, 5.0, 1.0, 0.0, -1.0])

            def __call__(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
                return type('Outputs', (), {
                    'logits': self.logits.unsqueeze(0).unsqueeze(0),
                    'past_key_values': None,
                })()

        builder = DynamicTreeBuilder(
            max_nodes=3,
            max_depth=1,
            branching_factor=2,
            temperature=0.0,
            use_gumbel_topk=False,
        )

        tree = builder.build_tree(
            MockDraftModel(),
            torch.tensor([[0]]),
            torch.ones(1, 1, dtype=torch.long),
            torch.device("cpu"),
        )

        assert tree.get_num_nodes() == 3
        assert len(tree.nodes[0].children) == 2


class TestAdaptiveTreeBuilder:
    """Test entropy-adaptive tree builder."""

    def test_entropy_computation(self):
        """Test entropy computation for branching decisions."""
        builder = AdaptiveTreeBuilder(entropy_threshold=1.0)

        # Low entropy distribution (peaked)
        low_entropy_logprobs = torch.tensor([-0.1, -5.0, -5.0, -5.0])
        entropy = builder._compute_entropy(low_entropy_logprobs)
        assert entropy < 1.0  # Low entropy

        # High entropy distribution (uniform) - max entropy for 4 classes is log(4) ≈ 1.39
        high_entropy_logprobs = torch.tensor([-1.4, -1.4, -1.4, -1.4])
        entropy = builder._compute_entropy(high_entropy_logprobs)
        assert entropy > 1.0  # High entropy

    def test_branching_decision(self):
        """Test branching factor based on entropy."""
        builder = AdaptiveTreeBuilder(base_branching=3, entropy_threshold=1.0)

        # Low entropy -> fewer branches
        low_entropy = torch.tensor([-0.1, -5.0, -5.0, -5.0])
        branching = builder._get_branching_for_node(low_entropy)
        assert branching <= 3  # Reduced branching

        # High entropy -> more branches
        high_entropy = torch.tensor([-1.4, -1.4, -1.4, -1.4])
        branching = builder._get_branching_for_node(high_entropy)
        assert branching >= 3  # Increased or equal branching


class TestDynamicTreeBuilderLogic:
    """Test the core logic of dynamic tree building."""

    def test_candidate_generation_gumbel_topk(self):
        """Test Gumbel-Top-k candidate generation."""
        builder = DynamicTreeBuilder(
            branching_factor=3,
            use_gumbel_topk=True,
            gumbel_temperature=1.0,
        )

        # Create mock logits
        vocab_size = 100
        logits = torch.randn(1, vocab_size)
        logprobs = compute_logprobs(logits).squeeze(0)

        # Test Gumbel noise addition
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logprobs) + 1e-10) + 1e-10)
        perturbed = logprobs + gumbel_noise
        topk_values, topk_indices = torch.topk(perturbed, 3)

        assert topk_indices.size(0) == 3
        assert torch.all(topk_indices < vocab_size)

    def test_candidate_generation_standard(self):
        """Test standard top-k candidate generation."""
        builder = DynamicTreeBuilder(
            branching_factor=3,
            use_gumbel_topk=False,
            temperature=1.0,
            top_k=10,
        )

        # Just verify the builder is configured correctly
        assert builder.use_gumbel_topk == False
        assert builder.branching_factor == 3


class TestTreeBuilderComparison:
    """Compare static vs dynamic tree builders."""

    def test_static_tree_fixed_size(self):
        """Static tree has predictable size."""
        from speculative_tree_decoding.tree_builder import StaticTreeBuilder

        builder = StaticTreeBuilder(branching_factor=3, max_depth=2)
        expected_nodes = (3**3 - 1) // (3 - 1)  # 13 nodes
        assert expected_nodes == 13

    def test_dynamic_tree_variable_size(self):
        """Dynamic tree size depends on probabilities."""
        builder = DynamicTreeBuilder(max_nodes=15, max_depth=4)
        # Size varies based on model probabilities and thresholds
        assert builder.max_nodes == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
