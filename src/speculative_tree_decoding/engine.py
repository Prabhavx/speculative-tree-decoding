"""Core speculative decoding engine: chain-based and tree-based drafting with rejection sampling."""
import torch
from typing import List, Tuple, Optional
import logging

from .kv_cache import DraftKVCache, TargetKVCache
from .tree_builder import StaticTreeBuilder
from .dynamic_tree_builder import DynamicTreeBuilder, AdaptiveTreeBuilder
from .tree_kv_cache import TreeKVCache
from .verify import (
    verify_tree,
    verify_tree_greedy,
    verify_tree_sampling,
    verify_tree_with_kv_cache,
    extract_accepted_path_tokens,
)
from .sampling import (
    sample_from_logits,
    compute_logprobs,
    rejection_sample,
    greedy_rejection_sample,
)

logger = logging.getLogger(__name__)


def _trim_at_stop(tokens: List[int], stop_tokens: List[int]) -> List[int]:
    """Return tokens up to and including the first stop token."""
    for idx, token in enumerate(tokens):
        if token in stop_tokens:
            return tokens[: idx + 1]
    return tokens


@torch.no_grad()
def _sample_target_next_token(
    target_model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    """Sample one fallback token from the target model."""
    outputs = target_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    next_logits = outputs.logits[:, -1, :]
    return sample_from_logits(
        next_logits.squeeze(0),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    ).item()


class SpeculativeDecodingEngine:
    """
    Chain-based speculative decoding engine with KV-cache optimization.

    A draft model proposes a chain of gamma tokens, then the target model
    verifies them all in one forward pass using rejection sampling.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        gamma: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        device: Optional[torch.device] = None,
        max_cache_length: int = 4096,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device = device or next(draft_model.parameters()).device

        self.draft_model.eval()
        self.target_model.eval()

        self.draft_cache = DraftKVCache(
            draft_model.config, batch_size=1, max_length=max_cache_length,
            device=self.device, dtype=next(draft_model.parameters()).dtype
        )
        self.target_cache = TargetKVCache(
            target_model.config, batch_size=1, max_length=max_cache_length,
            device=self.device, dtype=next(target_model.parameters()).dtype
        )

    def reset_caches(self):
        """Reset both draft and target KV caches."""
        self.draft_cache.reset()
        self.target_cache.reset()

    @torch.no_grad()
    def generate_draft_chain(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Generate a chain of draft tokens autoregressively using KV cache.

        Returns:
            draft_tokens: List of generated token ids
            draft_logprobs: List of log probability tensors for each position
        """
        gamma = max_new_tokens or self.gamma
        draft_tokens = []
        draft_logprobs = []

        if gamma <= 0:
            return draft_tokens, draft_logprobs

        outputs = self.draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None

        for step in range(gamma):
            next_logprobs = compute_logprobs(next_logits)

            next_token = sample_from_logits(
                next_logits.squeeze(0),
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

            draft_tokens.append(next_token.item())
            draft_logprobs.append(next_logprobs.squeeze(0))

            if step == gamma - 1:
                break

            next_input = next_token.reshape(1, 1)
            if current_attention_mask is not None:
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones_like(current_attention_mask[:, :1])],
                    dim=1,
                )

            outputs = self.draft_model(
                input_ids=next_input,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        if past_key_values is not None:
            self.draft_cache._store_past_key_values(past_key_values)

        return draft_tokens, draft_logprobs

    @torch.no_grad()
    def verify_chain(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Verify draft chain with target model in single forward pass using KV cache.

        Returns:
            target_logprobs: List of target model logprobs at each draft position
            target_tokens: Target model's own predictions at each position
        """
        target_logprobs, target_tokens = self.target_cache.verify_chain(
            self.target_model, input_ids, draft_tokens, attention_mask
        )
        return target_logprobs, target_tokens

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        stop_tokens: Optional[List[int]] = None,
    ) -> str:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            stop_tokens: Optional list of token ids to stop at

        Returns:
            Generated text string
        """
        self.reset_caches()

        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_tokens = []
        total_generated = 0

        while total_generated < max_new_tokens:
            remaining = max_new_tokens - total_generated
            gamma = min(self.gamma, remaining)

            draft_tokens, draft_logprobs = self.generate_draft_chain(
                input_ids, attention_mask, max_new_tokens=gamma
            )

            target_logprobs, target_tokens = self.verify_chain(
                input_ids, draft_tokens, attention_mask
            )

            step_tokens: List[int] = []
            accepted_count = 0
            rejected = False
            bonus_added = False
            for i, (draft_token, d_logprobs, t_logprobs) in enumerate(
                zip(draft_tokens, draft_logprobs, target_logprobs)
            ):
                if self.temperature == 0.0:
                    accepted, resampled = greedy_rejection_sample(d_logprobs, t_logprobs, draft_token)
                else:
                    accepted, resampled = rejection_sample(d_logprobs, t_logprobs, draft_token, self.temperature)

                if accepted:
                    step_tokens.append(draft_token)
                    accepted_count += 1
                else:
                    step_tokens.append(resampled)
                    rejected = True
                    break

            if (
                accepted_count == len(draft_tokens)
                and total_generated + len(step_tokens) < max_new_tokens
            ):
                bonus_idx = len(draft_tokens)
                if self.temperature == 0.0:
                    bonus_token = target_tokens[bonus_idx]
                    step_tokens.append(bonus_token)
                    bonus_added = True
                else:
                    bonus_token = sample_from_logits(
                        target_logprobs[bonus_idx],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                    ).item()
                    step_tokens.append(bonus_token)
                    bonus_added = True

            step_tokens = _trim_at_stop(step_tokens, stop_tokens)

            tokens_added = len(step_tokens)
            if tokens_added == 0:
                break

            generated_tokens.extend(step_tokens)
            new_ids = torch.tensor(step_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
            input_ids = torch.cat([input_ids, new_ids], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, tokens_added, device=self.device, dtype=attention_mask.dtype)
            ], dim=1)

            if rejected or bonus_added or tokens_added != len(draft_tokens):
                self.target_cache.reset()
            else:
                self.target_cache.set_prefix_length(input_ids.size(1))

            total_generated += tokens_added

            if any(tok in stop_tokens for tok in step_tokens):
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def autoregressive_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    stop_tokens: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    Standard autoregressive generation for baseline comparison.
    """
    device = device or next(model.parameters()).device
    stop_tokens = stop_tokens or [tokenizer.eos_token_id]

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            next_logits = outputs.logits[:, -1, :]

            next_token = sample_from_logits(
                next_logits.squeeze(0), temperature, top_p, top_k
            )

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1)

            if next_token.item() in stop_tokens:
                break

    return tokenizer.decode(generated, skip_special_tokens=True)


class TreeSpeculativeDecodingEngine:
    """
    Tree-based speculative decoding engine with KV-cache optimization.

    A draft model proposes a tree of candidate continuations, then the target model
    verifies the entire tree in one parallel forward pass using a custom tree attention mask.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        branching_factor: int = 3,
        max_depth: int = 2,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        device: Optional[torch.device] = None,
        max_cache_length: int = 4096,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device = device or next(draft_model.parameters()).device

        self.draft_model.eval()
        self.target_model.eval()

        self.tree_builder = StaticTreeBuilder(
            branching_factor=branching_factor,
            max_depth=max_depth,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Tree-aware KV cache for target model
        self.target_cache = TreeKVCache(
            target_model.config, batch_size=1, max_length=max_cache_length,
            device=self.device, dtype=next(target_model.parameters()).dtype
        )

    def reset_cache(self):
        """Reset the target KV cache."""
        self.target_cache.reset()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        stop_tokens: Optional[List[int]] = None,
    ) -> str:
        """
        Generate text using tree-based speculative decoding with KV cache.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            stop_tokens: Optional list of token ids to stop at

        Returns:
            Generated text string
        """
        self.reset_cache()

        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_tokens = []
        total_generated = 0

        while total_generated < max_new_tokens:
            remaining = max_new_tokens - total_generated
            # Build draft tree
            tree = self.tree_builder.build_tree(
                self.draft_model, input_ids, attention_mask, self.device
            )

            # Verify tree with target model using KV cache
            if self.temperature == 0.0:
                result = verify_tree_with_kv_cache(
                    self.target_model, tree, input_ids, self.target_cache,
                    attention_mask, temperature=0.0, top_p=1.0, top_k=0, device=self.device
                )
            else:
                result = verify_tree_with_kv_cache(
                    self.target_model, tree, input_ids, self.target_cache,
                    attention_mask, self.temperature, self.top_p, self.top_k, self.device
                )

            # Extract accepted tokens
            accepted_tokens = extract_accepted_path_tokens(tree, result)
            num_accepted = len(accepted_tokens)

            if num_accepted == 0:
                next_token = _sample_target_next_token(
                    self.target_model,
                    input_ids,
                    attention_mask,
                    self.temperature,
                    self.top_p,
                    self.top_k,
                )
                accepted_tokens = [next_token]

            accepted_tokens = _trim_at_stop(accepted_tokens[:remaining], stop_tokens)
            num_accepted = len(accepted_tokens)
            if num_accepted == 0:
                break

            generated_tokens.extend(accepted_tokens)

            # Update input_ids and attention_mask
            new_ids = torch.tensor(accepted_tokens, device=self.device).unsqueeze(0)
            input_ids = torch.cat([input_ids, new_ids], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, num_accepted, device=self.device, dtype=attention_mask.dtype)
            ], dim=1)

            self.target_cache.reset()
            total_generated += num_accepted

            if any(tok in stop_tokens for tok in accepted_tokens):
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


class DynamicTreeSpeculativeDecodingEngine:
    """
    Dynamic/adaptive tree-based speculative decoding engine.

    Uses Gumbel-Top-k or stochastic beam search to build the draft tree adaptively,
    expanding high-probability branches first within a compute budget.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        max_nodes: int = 15,
        max_depth: int = 4,
        branching_factor: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        cum_prob_threshold: float = 0.0,
        use_gumbel_topk: bool = True,
        gumbel_temperature: float = 1.0,
        min_branch_prob: float = 0.01,
        device: Optional[torch.device] = None,
        max_cache_length: int = 4096,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device = device or next(draft_model.parameters()).device

        self.draft_model.eval()
        self.target_model.eval()

        self.tree_builder = DynamicTreeBuilder(
            max_nodes=max_nodes,
            max_depth=max_depth,
            branching_factor=branching_factor,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cum_prob_threshold=cum_prob_threshold,
            use_gumbel_topk=use_gumbel_topk,
            gumbel_temperature=gumbel_temperature,
            min_branch_prob=min_branch_prob,
        )

        # Tree-aware KV cache for target model
        self.target_cache = TreeKVCache(
            target_model.config, batch_size=1, max_length=max_cache_length,
            device=self.device, dtype=next(target_model.parameters()).dtype
        )

    def reset_cache(self):
        """Reset the target KV cache."""
        self.target_cache.reset()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        stop_tokens: Optional[List[int]] = None,
    ) -> str:
        """
        Generate text using dynamic tree-based speculative decoding with KV cache.
        """
        self.reset_cache()

        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_tokens = []
        total_generated = 0

        while total_generated < max_new_tokens:
            remaining = max_new_tokens - total_generated
            # Build adaptive draft tree
            tree = self.tree_builder.build_tree(
                self.draft_model, input_ids, attention_mask, self.device
            )

            # Verify tree with target model using KV cache
            if self.temperature == 0.0:
                result = verify_tree_with_kv_cache(
                    self.target_model, tree, input_ids, self.target_cache,
                    attention_mask, temperature=0.0, top_p=1.0, top_k=0, device=self.device
                )
            else:
                result = verify_tree_with_kv_cache(
                    self.target_model, tree, input_ids, self.target_cache,
                    attention_mask, self.temperature, self.top_p, self.top_k, self.device
                )

            # Extract accepted tokens
            accepted_tokens = extract_accepted_path_tokens(tree, result)
            num_accepted = len(accepted_tokens)

            if num_accepted == 0:
                next_token = _sample_target_next_token(
                    self.target_model,
                    input_ids,
                    attention_mask,
                    self.temperature,
                    self.top_p,
                    self.top_k,
                )
                accepted_tokens = [next_token]

            accepted_tokens = _trim_at_stop(accepted_tokens[:remaining], stop_tokens)
            num_accepted = len(accepted_tokens)
            if num_accepted == 0:
                break

            generated_tokens.extend(accepted_tokens)

            new_ids = torch.tensor(accepted_tokens, device=self.device).unsqueeze(0)
            input_ids = torch.cat([input_ids, new_ids], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, num_accepted, device=self.device, dtype=attention_mask.dtype)
            ], dim=1)

            self.target_cache.reset()
            total_generated += num_accepted

            if any(tok in stop_tokens for tok in accepted_tokens):
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


class AdaptiveTreeSpeculativeDecodingEngine:
    """
    Entropy-adaptive tree-based speculative decoding engine.

    Adjusts branching factor based on draft model confidence (entropy):
    - High entropy -> more branches (exploration)
    - Low entropy -> deeper branches (exploitation)
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        max_nodes: int = 20,
        max_depth: int = 4,
        base_branching: int = 3,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        entropy_threshold: float = 2.0,
        device: Optional[torch.device] = None,
        max_cache_length: int = 4096,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.base_branching = base_branching
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device = device or next(draft_model.parameters()).device

        self.draft_model.eval()
        self.target_model.eval()

        self.tree_builder = AdaptiveTreeBuilder(
            max_nodes=max_nodes,
            max_depth=max_depth,
            base_branching=base_branching,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            entropy_threshold=entropy_threshold,
        )

        # Tree-aware KV cache for target model
        self.target_cache = TreeKVCache(
            target_model.config, batch_size=1, max_length=max_cache_length,
            device=self.device, dtype=next(target_model.parameters()).dtype
        )

    def reset_cache(self):
        """Reset the target KV cache."""
        self.target_cache.reset()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        stop_tokens: Optional[List[int]] = None,
    ) -> str:
        """Generate text using adaptive tree-based speculative decoding with KV cache."""
        self.reset_cache()

        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_tokens = []
        total_generated = 0

        while total_generated < max_new_tokens:
            remaining = max_new_tokens - total_generated
            tree = self.tree_builder.build_tree(
                self.draft_model, input_ids, attention_mask, self.device
            )

            if self.temperature == 0.0:
                result = verify_tree_with_kv_cache(
                    self.target_model, tree, input_ids, self.target_cache,
                    attention_mask, temperature=0.0, top_p=1.0, top_k=0, device=self.device
                )
            else:
                result = verify_tree_with_kv_cache(
                    self.target_model, tree, input_ids, self.target_cache,
                    attention_mask, self.temperature, self.top_p, self.top_k, self.device
                )

            accepted_tokens = extract_accepted_path_tokens(tree, result)
            num_accepted = len(accepted_tokens)

            if num_accepted == 0:
                next_token = _sample_target_next_token(
                    self.target_model,
                    input_ids,
                    attention_mask,
                    self.temperature,
                    self.top_p,
                    self.top_k,
                )
                accepted_tokens = [next_token]

            accepted_tokens = _trim_at_stop(accepted_tokens[:remaining], stop_tokens)
            num_accepted = len(accepted_tokens)
            if num_accepted == 0:
                break

            generated_tokens.extend(accepted_tokens)

            new_ids = torch.tensor(accepted_tokens, device=self.device).unsqueeze(0)
            input_ids = torch.cat([input_ids, new_ids], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, num_accepted, device=self.device, dtype=attention_mask.dtype)
            ], dim=1)

            self.target_cache.reset()
            total_generated += num_accepted

            if any(tok in stop_tokens for tok in accepted_tokens):
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
