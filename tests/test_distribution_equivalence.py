"""Tests for distributional equivalence of speculative decoding."""
import torch
import torch.nn.functional as F
import pytest
from collections import Counter
import numpy as np

from speculative_tree_decoding.engine import (
    rejection_sample,
    greedy_rejection_sample,
    compute_logprobs,
    sample_from_logits,
    SpeculativeDecodingEngine,
    autoregressive_generate,
)
from speculative_tree_decoding.models import load_models


class TestRejectionSampling:
    """Test the core rejection sampling logic."""

    def test_greedy_accept_match(self):
        """Greedy: accept when draft and target agree on top token."""
        draft_logprobs = torch.tensor([0.1, 0.8, 0.1]).log()
        target_logprobs = torch.tensor([0.2, 0.7, 0.1]).log()
        accepted, resampled = greedy_rejection_sample(draft_logprobs, target_logprobs, 1)
        assert accepted is True
        assert resampled is None

    def test_greedy_reject_mismatch(self):
        """Greedy: reject when draft and target disagree on top token."""
        draft_logprobs = torch.tensor([0.1, 0.8, 0.1]).log()
        target_logprobs = torch.tensor([0.7, 0.2, 0.1]).log()
        accepted, resampled = greedy_rejection_sample(draft_logprobs, target_logprobs, 1)
        assert accepted is False
        assert resampled == 0  # target's top token

    def test_rejection_sample_accept_higher_target(self):
        """Accept when target prob >= draft prob."""
        draft_logprobs = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        target_logprobs = torch.log(torch.tensor([0.1, 0.4, 0.5]))
        accepted, resampled = rejection_sample(draft_logprobs, target_logprobs, 2, temperature=1.0)
        assert accepted is True
        assert resampled is None

    def test_rejection_sample_residual_distribution(self):
        """Test residual resampling matches expected distribution."""
        torch.manual_seed(42)
        draft_probs = torch.tensor([0.6, 0.3, 0.1])
        target_probs = torch.tensor([0.2, 0.5, 0.3])
        draft_logprobs = draft_probs.log()
        target_logprobs = target_probs.log()

        residual = torch.clamp(target_probs - draft_probs, min=0.0)
        expected_residual = residual / residual.sum()

        samples = []
        for _ in range(10000):
            accepted, resampled = rejection_sample(draft_logprobs, target_logprobs, 0, temperature=1.0)
            if not accepted:
                samples.append(resampled)

        if len(samples) > 100:
            empirical = torch.bincount(torch.tensor(samples), minlength=3).float() / len(samples)
            assert torch.allclose(empirical, expected_residual, atol=0.02)


class TestDistributionalEquivalence:
    """Test that speculative decoding matches target-only distribution."""

    def test_toy_distribution_equivalence(self):
        """
        Test on a toy model where we can compute exact distributions.

        We create a synthetic scenario with known probabilities.
        """
        vocab_size = 10
        torch.manual_seed(123)

        draft_logits = torch.randn(1, vocab_size)
        target_logits = torch.randn(1, vocab_size)

        draft_probs = F.softmax(draft_logits, dim=-1).squeeze(0)
        target_probs = F.softmax(target_logits, dim=-1).squeeze(0)

        num_samples = 5000
        speculative_samples = []
        target_samples = []

        for _ in range(num_samples):
            draft_token = torch.multinomial(draft_probs, 1).item()
            draft_logprobs = draft_logits.log_softmax(dim=-1).squeeze(0)
            target_logprobs = target_logits.log_softmax(dim=-1).squeeze(0)

            accepted, resampled = rejection_sample(draft_logprobs, target_logprobs, draft_token, temperature=1.0)
            speculative_samples.append(draft_token if accepted else resampled)

            target_token = torch.multinomial(target_probs, 1).item()
            target_samples.append(target_token)

        spec_counts = Counter(speculative_samples)
        target_counts = Counter(target_samples)

        for token in range(vocab_size):
            spec_freq = spec_counts.get(token, 0) / num_samples
            target_freq = target_counts.get(token, 0) / num_samples
            assert abs(spec_freq - target_freq) < 0.03, f"Token {token}: spec={spec_freq:.4f}, target={target_freq:.4f}"

    def test_greedy_determinism(self):
        """With temperature=0, output must exactly match greedy target decoding."""
        vocab_size = 20
        torch.manual_seed(456)

        draft_logits = torch.randn(1, vocab_size)
        target_logits = torch.randn(1, vocab_size)

        for _ in range(100):
            draft_token = draft_logits.argmax().item()
            draft_logprobs = draft_logits.log_softmax(dim=-1).squeeze(0)
            target_logprobs = target_logits.log_softmax(dim=-1).squeeze(0)

            accepted, resampled = greedy_rejection_sample(draft_logprobs, target_logprobs, draft_token)
            target_token = target_logits.argmax().item()

            if accepted:
                assert draft_token == target_token
            else:
                assert resampled == target_token


class TestSamplingUtilities:
    """Test sampling utility functions."""

    def test_sample_from_logits_greedy(self):
        logits = torch.tensor([0.1, 0.8, 0.1])
        token = sample_from_logits(logits, temperature=0.0)
        assert token.item() == 1

    def test_sample_from_logits_temperature(self):
        torch.manual_seed(789)
        logits = torch.tensor([1.0, 0.0, 0.0])
        tokens = [sample_from_logits(logits, temperature=1.0).item() for _ in range(1000)]
        assert 0 in tokens
        assert len(set(tokens)) > 1

    def test_sample_from_logits_top_k(self):
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        token = sample_from_logits(logits, temperature=1.0, top_k=2)
        assert token.item() in [3, 4]

    def test_sample_from_logits_top_p(self):
        logits = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0])
        token = sample_from_logits(logits, temperature=1.0, top_p=0.5)
        assert token.item() == 0

    def test_compute_logprobs(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        logprobs = compute_logprobs(logits)
        probs = logprobs.exp()
        assert torch.allclose(probs.sum(), torch.tensor(1.0))
        assert torch.all(probs >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])