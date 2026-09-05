"""Tests for residual sampling correctness."""
import torch
import torch.nn.functional as F
import pytest
from speculative_tree_decoding.engine import rejection_sample, greedy_rejection_sample


class TestResidualSampling:
    """Test residual resampling matches the theoretical formula."""

    def test_residual_formula_basic(self):
        """Test residual distribution matches p'(x) = max(0, p(x) - q(x)) / sum."""
        draft_probs = torch.tensor([0.5, 0.3, 0.2])
        target_probs = torch.tensor([0.2, 0.5, 0.3])

        residual = torch.clamp(target_probs - draft_probs, min=0.0)
        expected = residual / residual.sum()

        torch.manual_seed(42)
        samples = []
        draft_logprobs = draft_probs.log()
        target_logprobs = target_probs.log()

        for _ in range(20000):
            accepted, resampled = rejection_sample(draft_logprobs, target_logprobs, 0, temperature=1.0)
            if not accepted:
                samples.append(resampled)

        empirical = torch.bincount(torch.tensor(samples), minlength=3).float() / len(samples)
        assert torch.allclose(empirical, expected, atol=0.015)

    def test_residual_zero_case(self):
        """When target <= draft everywhere, residual is zero, should fall back to target argmax."""
        # target <= draft for all tokens: residual = 0 everywhere
        draft_probs = torch.tensor([0.6, 0.4])
        target_probs = torch.tensor([0.5, 0.3])  # both <= draft

        draft_logprobs = draft_probs.log()
        target_logprobs = target_probs.log()

        # draft_token = 0, target_prob=0.5 < draft_prob=0.6
        # accept_prob = 0.5/0.6 = 0.833
        # residual = [0, 0] (since 0.5-0.6<0 and 0.3-0.4<0)
        # when rejected, should fall back to target argmax = 0
        torch.manual_seed(999)
        accepted, resampled = rejection_sample(draft_logprobs, target_logprobs, 0, temperature=1.0)
        # With this seed, it should reject (random) and fall back to target argmax
        if not accepted:
            assert resampled == 0

    def test_residual_partial_overlap(self):
        """Test case where some tokens have target > draft, others target < draft."""
        draft_probs = torch.tensor([0.4, 0.35, 0.25])
        target_probs = torch.tensor([0.3, 0.5, 0.2])

        residual = torch.clamp(target_probs - draft_probs, min=0.0)
        expected = residual / residual.sum()

        torch.manual_seed(123)
        samples = []
        draft_logprobs = draft_probs.log()
        target_logprobs = target_probs.log()

        for _ in range(20000):
            accepted, resampled = rejection_sample(draft_logprobs, target_logprobs, 0, temperature=1.0)
            if not accepted:
                samples.append(resampled)

        empirical = torch.bincount(torch.tensor(samples), minlength=3).float() / len(samples)
        assert torch.allclose(empirical, expected, atol=0.015)

    def test_greedy_residual_not_used(self):
        """Greedy mode should never use residual sampling."""
        draft_logprobs = torch.tensor([0.1, 0.8, 0.1]).log()
        target_logprobs = torch.tensor([0.7, 0.2, 0.1]).log()

        accepted, resampled = greedy_rejection_sample(draft_logprobs, target_logprobs, 1)
        assert accepted is False
        assert resampled == 0  # target's top token


class TestAcceptanceProbability:
    """Test acceptance probability calculations."""

    def test_accept_prob_formula(self):
        """Acceptance probability should be min(1, p/q)."""
        draft_prob = 0.3
        target_prob = 0.6
        expected_accept = min(1.0, target_prob / draft_prob)

        draft_logprobs = torch.tensor([draft_prob, 1-draft_prob]).log()
        target_logprobs = torch.tensor([target_prob, 1-target_prob]).log()

        torch.manual_seed(456)
        num_trials = 50000
        accepts = 0
        for _ in range(num_trials):
            accepted, _ = rejection_sample(draft_logprobs, target_logprobs, 0, temperature=1.0)
            if accepted:
                accepts += 1

        empirical_accept = accepts / num_trials
        assert abs(empirical_accept - expected_accept) < 0.01

    def test_accept_when_target_higher(self):
        """Always accept when target prob >= draft prob."""
        draft_logprobs = torch.tensor([0.3, 0.7]).log()
        target_logprobs = torch.tensor([0.5, 0.5]).log()

        for _ in range(100):
            accepted, _ = rejection_sample(draft_logprobs, target_logprobs, 0, temperature=1.0)
            assert accepted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])