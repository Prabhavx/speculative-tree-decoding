"""Regression tests for engine/cache integration paths."""
import torch
from transformers import PretrainedConfig

from speculative_tree_decoding.engine import SpeculativeDecodingEngine
from speculative_tree_decoding.kv_cache import DraftKVCache, TargetKVCache
from speculative_tree_decoding.tree_builder import DraftTree, TreeNode
from speculative_tree_decoding.verify import build_tree_attention_mask_4d, verify_tree


def one_hot_logprobs(token_id: int, vocab_size: int = 32) -> torch.Tensor:
    logits = torch.full((vocab_size,), -100.0)
    logits[token_id] = 100.0
    return torch.log_softmax(logits, dim=-1)


def past_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "layers"):
        if not past_key_values.layers:
            return 0
        return past_key_values.layers[0].keys.size(2)
    return past_key_values[0][0].size(2)


class PositionalToyModel(torch.nn.Module):
    """Tiny causal-LM stand-in whose argmax token is based on absolute position."""

    def __init__(self, vocab_size: int = 32, base_token: int = 10):
        super().__init__()
        self.vocab_size = vocab_size
        self.base_token = base_token
        self.config = PretrainedConfig(
            num_hidden_layers=1,
            num_attention_heads=1,
            hidden_size=4,
        )
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
    ):
        past_len = past_length(past_key_values)
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -100.0)

        for pos in range(seq_len):
            token_id = (self.base_token + past_len + pos) % self.vocab_size
            logits[:, pos, token_id] = 100.0

        total_len = past_len + seq_len
        key = torch.zeros(batch_size, 1, total_len, 4)
        value = torch.zeros(batch_size, 1, total_len, 4)
        return type("Outputs", (), {"logits": logits, "past_key_values": ((key, value),)})()


class FakeTokenizer:
    eos_token_id = -1

    def __call__(self, prompt, return_tensors="pt"):
        return FakeInputs(torch.tensor([[99]]), torch.ones(1, 1, dtype=torch.long))

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens)


class FakeInputs:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class FakeCache:
    def reset(self):
        pass

    def set_prefix_length(self, length):
        pass


def make_stub_engine():
    engine = SpeculativeDecodingEngine.__new__(SpeculativeDecodingEngine)
    engine.gamma = 3
    engine.temperature = 0.0
    engine.top_p = 1.0
    engine.top_k = 0
    engine.device = torch.device("cpu")
    engine.tokenizer = FakeTokenizer()
    engine.target_cache = FakeCache()
    engine.reset_caches = lambda: None
    return engine


def test_past_key_value_store_updates_cache_length():
    model = PositionalToyModel()
    cache = DraftKVCache(model.config, batch_size=1, max_length=16, dtype=torch.float32)
    key = torch.zeros(1, 1, 5, 4)
    value = torch.zeros(1, 1, 5, 4)

    cache._store_past_key_values(((key, value),))

    assert cache.get_length() == 5
    assert cache._get_past_key_values().layers[0].keys.shape == (1, 1, 5, 4)


def test_target_verify_chain_uses_prefix_and_returns_bonus_distribution():
    model = PositionalToyModel()
    cache = TargetKVCache(model.config, batch_size=1, max_length=16, dtype=torch.float32)
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    _, target_tokens = cache.verify_chain(model, input_ids, [12, 13], attention_mask)

    assert target_tokens == [12, 13, 14]
    assert cache.get_length() == 5

    cache.set_prefix_length(5)
    next_input_ids = torch.tensor([[1, 2, 3, 12, 13]])
    next_attention_mask = torch.ones_like(next_input_ids)
    _, next_target_tokens = cache.verify_chain(model, next_input_ids, [14], next_attention_mask)

    assert next_target_tokens == [14, 15]


def test_chain_generation_does_not_add_bonus_past_max_tokens():
    engine = make_stub_engine()

    engine.generate_draft_chain = lambda input_ids, attention_mask, max_new_tokens=None: (
        [1, 2, 3],
        [one_hot_logprobs(1), one_hot_logprobs(2), one_hot_logprobs(3)],
    )
    engine.verify_chain = lambda input_ids, draft_tokens, attention_mask=None: (
        [one_hot_logprobs(1), one_hot_logprobs(2), one_hot_logprobs(3), one_hot_logprobs(4)],
        [1, 2, 3, 4],
    )

    assert engine.generate("prompt", max_new_tokens=3) == "1 2 3"


def test_chain_generation_keeps_replacement_token_in_next_prefix():
    engine = make_stub_engine()
    seen_prefixes = []

    def generate_draft_chain(input_ids, attention_mask, max_new_tokens=None):
        seen_prefixes.append(input_ids.squeeze(0).tolist())
        if len(seen_prefixes) == 1:
            return [1, 2, 3], [
                one_hot_logprobs(1),
                one_hot_logprobs(2),
                one_hot_logprobs(3),
            ]
        return [8], [one_hot_logprobs(8)]

    def verify_chain(input_ids, draft_tokens, attention_mask=None):
        if len(draft_tokens) == 3:
            return [
                one_hot_logprobs(1),
                one_hot_logprobs(2),
                one_hot_logprobs(9),
                one_hot_logprobs(10),
            ], [1, 2, 9, 10]
        return [one_hot_logprobs(8), one_hot_logprobs(11)], [8, 11]

    engine.generate_draft_chain = generate_draft_chain
    engine.verify_chain = verify_chain

    assert engine.generate("prompt", max_new_tokens=4) == "1 2 9 8"
    assert seen_prefixes[1] == [99, 1, 2, 9]


def test_tree_verification_uses_next_token_distribution_for_branch_and_bonus():
    model = PositionalToyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    tree = DraftTree(
        nodes=[
            TreeNode(token_id=12, parent_idx=-1, depth=0, logprobs=one_hot_logprobs(12), children=[1]),
            TreeNode(token_id=13, parent_idx=0, depth=1, logprobs=one_hot_logprobs(13), children=[]),
        ],
        prefix_length=3,
    )

    result = verify_tree(
        model,
        tree,
        input_ids,
        attention_mask=attention_mask,
        temperature=0.0,
        device=torch.device("cpu"),
    )

    assert result.accepted_node_indices == [0, 1]
    assert result.accepted_tokens == [12, 13, 14]


def test_tree_4d_attention_mask_keeps_ancestor_edges_with_prefix_mask():
    additive_mask = build_tree_attention_mask_4d(
        parent_array=[-1, 0],
        prefix_length=2,
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    allowed = additive_mask == 0

    # Sequence positions: prefix 0,1; root 2; child 3.
    assert allowed[0, 0, 3, 0]
    assert allowed[0, 0, 3, 1]
    assert allowed[0, 0, 3, 2]
    assert allowed[0, 0, 3, 3]
