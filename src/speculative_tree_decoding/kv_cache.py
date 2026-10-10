"""KV-cache management for speculative decoding."""
import torch
from typing import List, Tuple, Optional
from transformers import PretrainedConfig


class KVCache:
    """
    Manages key-value cache for transformer models during generation.

    Supports both standard (single sequence) and speculative (draft + verify) modes.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = 1,
        max_length: int = 2048,
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

        self.current_length = 0

    def get_seq_length(self) -> int:
        return self.current_length

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer.

        Args:
            layer_idx: Layer index
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]

        Returns:
            Updated key and value tensors for attention computation
        """
        seq_len = key.size(2)
        end_pos = self.current_length + seq_len

        if end_pos > self.max_length:
            raise ValueError(f"Sequence length {end_pos} exceeds max_length {self.max_length}")

        self.k_cache[layer_idx, :, :, self.current_length:end_pos, :] = key
        self.v_cache[layer_idx, :, :, self.current_length:end_pos, :] = value

        return (
            self.k_cache[layer_idx, :, :, :end_pos, :],
            self.v_cache[layer_idx, :, :, :end_pos, :],
        )

    def store_layer(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> int:
        """
        Replace a layer's cached key/value tensors with an exact model cache.

        HuggingFace models return the full cache length for each layer. This
        method writes that cache at position zero; callers set current_length
        once after all layers have been stored.
        """
        seq_len = key.size(2)
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}")

        self.k_cache[layer_idx, :, :, :seq_len, :] = key
        self.v_cache[layer_idx, :, :, :seq_len, :] = value
        return seq_len

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached key/value up to current length."""
        return (
            self.k_cache[layer_idx, :, :, :self.current_length, :],
            self.v_cache[layer_idx, :, :, :self.current_length, :],
        )

    def set_length(self, length: int):
        """Set the current sequence length (for rollback on rejection)."""
        if length < 0 or length > self.max_length:
            raise ValueError(f"Invalid cache length {length}; max_length={self.max_length}")
        self.current_length = length

    def advance(self, steps: int):
        """Advance the current length by steps."""
        self.current_length += steps

    def reset(self):
        """Reset cache to empty state."""
        self.current_length = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    def slice(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a slice of the cache (for tree verification)."""
        return (
            self.k_cache[:, :, :, start:end, :],
            self.v_cache[:, :, :, start:end, :],
        )


class DraftKVCache:
    """
    KV-cache for draft model during chain generation.
    Lightweight version that only stores what's needed for draft generation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = 1,
        max_length: int = 2048,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
    ):
        self.cache = KVCache(config, batch_size, max_length, device, dtype)

    def generate_step(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single generation step using cached KV.

        Returns:
            logits: Next token logits [batch, vocab]
            past_key_values: Updated cache for next step
        """
        seq_len = input_ids.size(1)
        if self.cache.current_length == 0:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            self._store_past_key_values(past_key_values)
            return outputs.logits[:, -1, :], past_key_values

        last_token = input_ids[:, -1:]
        last_mask = attention_mask[:, -1:] if attention_mask is not None else None

        outputs = model(
            input_ids=last_token,
            attention_mask=last_mask,
            past_key_values=self._get_past_key_values(),
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        self._store_past_key_values(past_key_values)
        return outputs.logits[:, -1, :], past_key_values

    def _store_past_key_values(self, past_key_values):
        """Store HF past_key_values format into our cache.
        
        Supports both legacy tuple format and new DynamicCache format.
        """
        # Handle DynamicCache format (has .layers attribute)
        if hasattr(past_key_values, 'layers'):
            layers = past_key_values.layers
        else:
            # Legacy tuple format
            layers = past_key_values
        
        cache_length = 0
        for layer_idx, layer_kv in enumerate(layers):
            # Handle different formats: DynamicLayer (has .keys, .values), 
            # legacy tuple (k, v), or tuple with extra elements
            if hasattr(layer_kv, 'keys') and hasattr(layer_kv, 'values'):
                # DynamicLayer format
                k, v = layer_kv.keys, layer_kv.values
            elif isinstance(layer_kv, (tuple, list)):
                k, v = layer_kv[0], layer_kv[1]
            else:
                raise ValueError(f"Unknown past_key_values format at layer {layer_idx}: {type(layer_kv)}")
            cache_length = max(cache_length, self.cache.store_layer(layer_idx, k, v))
        self.cache.set_length(cache_length)

    def _get_past_key_values(self):
        """Retrieve past_key_values as DynamicCache for modern HF models."""
        from transformers import DynamicCache
        dynamic_cache = DynamicCache()
        for layer_idx in range(self.cache.num_layers):
            k, v = self.cache.get(layer_idx)
            dynamic_cache.update(k, v, layer_idx)
        return dynamic_cache

    def reset(self):
        self.cache.reset()

    def get_length(self) -> int:
        return self.cache.get_seq_length()


class TargetKVCache:
    """
    KV-cache for target model during verification.
    Supports efficient verification of draft chains.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = 1,
        max_length: int = 4096,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
    ):
        self.cache = KVCache(config, batch_size, max_length, device, dtype)
        self.device = device
        self.last_logits: Optional[torch.Tensor] = None

    def verify_chain(
        self,
        model,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Verify a chain of draft tokens in one forward pass.

        Returns:
            target_logprobs: Log probs at each draft position
            target_tokens: Target model's own predictions
        """
        if not draft_tokens:
            return [], []

        draft_ids = torch.tensor(draft_tokens, device=self.device, dtype=torch.long).unsqueeze(0)

        if attention_mask is not None:
            full_mask = torch.cat([
                attention_mask,
                torch.ones(1, len(draft_tokens), device=self.device, dtype=attention_mask.dtype)
            ], dim=1)
        else:
            full_mask = None

        prefix_len = input_ids.size(1)

        if self.cache.current_length == prefix_len and self.last_logits is not None:
            prefix_next_logits = self.last_logits
            outputs = model(
                input_ids=draft_ids,
                attention_mask=full_mask,
                past_key_values=self._get_past_key_values(),
                use_cache=True,
            )
            distribution_logits = [prefix_next_logits]
            distribution_logits.extend(outputs.logits[:, i, :] for i in range(len(draft_tokens)))
        else:
            full_ids = torch.cat([input_ids, draft_ids], dim=1)
            outputs = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                use_cache=True,
            )
            distribution_logits = [
                outputs.logits[:, prefix_len - 1 + i, :]
                for i in range(len(draft_tokens) + 1)
            ]

        target_logprobs = []
        target_tokens = []

        for logits in distribution_logits:
            logprobs = torch.log_softmax(logits, dim=-1).squeeze(0)
            target_logprobs.append(logprobs)
            target_tokens.append(logits.argmax(dim=-1).item())

        self._store_past_key_values(outputs.past_key_values)
        self.last_logits = distribution_logits[-1].detach()
        return target_logprobs, target_tokens

    def _store_past_key_values(self, past_key_values):
        """Store HF past_key_values format into our cache."""
        if hasattr(past_key_values, 'layers'):
            layers = past_key_values.layers
        else:
            layers = past_key_values
        
        cache_length = 0
        for layer_idx, layer_kv in enumerate(layers):
            if hasattr(layer_kv, 'keys') and hasattr(layer_kv, 'values'):
                k, v = layer_kv.keys, layer_kv.values
            elif isinstance(layer_kv, (tuple, list)):
                k, v = layer_kv[0], layer_kv[1]
            else:
                raise ValueError(f"Unknown past_key_values format at layer {layer_idx}: {type(layer_kv)}")
            cache_length = max(cache_length, self.cache.store_layer(layer_idx, k, v))
        self.cache.set_length(cache_length)

    def _get_past_key_values(self):
        from transformers import DynamicCache
        dynamic_cache = DynamicCache()
        for layer_idx in range(self.cache.num_layers):
            k, v = self.cache.get(layer_idx)
            dynamic_cache.update(k, v, layer_idx)
        return dynamic_cache

    def reset(self):
        self.cache.reset()
        self.last_logits = None

    def set_prefix_length(self, length: int, last_logits: Optional[torch.Tensor] = None):
        """Set cache length to prefix length (after accepting tokens)."""
        old_length = self.cache.get_seq_length()
        self.cache.set_length(length)
        if last_logits is not None:
            self.last_logits = last_logits.detach()
        elif length != old_length:
            self.last_logits = None

    def get_length(self) -> int:
        return self.cache.get_seq_length()
