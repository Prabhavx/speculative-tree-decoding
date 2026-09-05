"""Speculative Tree Decoding - A tree-based speculative decoding engine for LLM inference."""
from .models import load_models, get_model_device, prepare_inputs
from .engine import (
    SpeculativeDecodingEngine,
    TreeSpeculativeDecodingEngine,
    DynamicTreeSpeculativeDecodingEngine,
    AdaptiveTreeSpeculativeDecodingEngine,
    autoregressive_generate,
)
from .kv_cache import KVCache, DraftKVCache, TargetKVCache
from .tree_builder import StaticTreeBuilder, DraftTree, TreeNode
from .dynamic_tree_builder import DynamicTreeBuilder, AdaptiveTreeBuilder
from .tree_kv_cache import TreeKVCache, DraftTreeKVCache, build_tree_position_ids as build_tree_pos_ids, build_tree_attention_mask as build_tree_attn_mask
from .tree_attention import (
    build_tree_attention_mask,
    build_tree_position_ids,
    flatten_tree_for_verification,
    TreeAttentionMask,
    verify_tree_mask_correctness,
)
from .verify import (
    verify_tree,
    verify_tree_greedy,
    verify_tree_sampling,
    verify_tree_with_kv_cache,
    TreeVerificationResult,
)
from .sampling import (
    sample_from_logits,
    compute_logprobs,
    rejection_sample,
    greedy_rejection_sample,
)

__version__ = "0.1.0"
__all__ = [
    "load_models",
    "get_model_device",
    "prepare_inputs",
    "SpeculativeDecodingEngine",
    "TreeSpeculativeDecodingEngine",
    "DynamicTreeSpeculativeDecodingEngine",
    "AdaptiveTreeSpeculativeDecodingEngine",
    "autoregressive_generate",
    "rejection_sample",
    "greedy_rejection_sample",
    "sample_from_logits",
    "compute_logprobs",
    "KVCache",
    "DraftKVCache",
    "TargetKVCache",
    "TreeKVCache",
    "DraftTreeKVCache",
    "StaticTreeBuilder",
    "DynamicTreeBuilder",
    "AdaptiveTreeBuilder",
    "DraftTree",
    "TreeNode",
    "build_tree_attention_mask",
    "build_tree_position_ids",
    "build_tree_pos_ids",
    "build_tree_attn_mask",
    "flatten_tree_for_verification",
    "TreeAttentionMask",
    "verify_tree_mask_correctness",
    "verify_tree",
    "verify_tree_greedy",
    "verify_tree_sampling",
    "verify_tree_with_kv_cache",
    "TreeVerificationResult",
]