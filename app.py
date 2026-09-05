"""
Streamlit demo frontend for Speculative Tree Decoding.

Run with: streamlit run app.py
"""
import streamlit as st
import torch
import time
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speculative_tree_decoding.models import load_models
from speculative_tree_decoding.engine import (
    SpeculativeDecodingEngine,
    TreeSpeculativeDecodingEngine,
    DynamicTreeSpeculativeDecodingEngine,
    AdaptiveTreeSpeculativeDecodingEngine,
    autoregressive_generate,
)
from speculative_tree_decoding.tree_builder import DraftTree, TreeNode
from speculative_tree_decoding.tree_attention import flatten_tree_for_verification, build_tree_attention_mask
from speculative_tree_decoding.sampling import sample_from_logits, compute_logprobs


# Page configuration
st.set_page_config(
    page_title="Speculative Tree Decoding Demo",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_models_cached(draft_name, target_name, device, dtype):
    """Load and cache models."""
    with st.spinner(f"Loading models on {device}..."):
        draft_model, target_model, tokenizer = load_models(
            draft_name, target_name, device=device, dtype=dtype
        )
    return draft_model, target_model, tokenizer


def create_engine(engine_type, draft_model, target_model, tokenizer, params, device):
    """Create the appropriate engine instance."""
    if engine_type == "Chain (Phase 0)":
        return SpeculativeDecodingEngine(
            draft_model, target_model, tokenizer,
            gamma=params.get("gamma", 4),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 0),
            device=device,
        )
    elif engine_type == "Static Tree (Phase 1)":
        return TreeSpeculativeDecodingEngine(
            draft_model, target_model, tokenizer,
            branching_factor=params.get("branching_factor", 3),
            max_depth=params.get("max_depth", 2),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 0),
            device=device,
        )
    elif engine_type == "Dynamic Tree (Phase 2)":
        return DynamicTreeSpeculativeDecodingEngine(
            draft_model, target_model, tokenizer,
            max_nodes=params.get("max_nodes", 15),
            max_depth=params.get("max_depth", 4),
            branching_factor=params.get("branching_factor", 4),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 0),
            cum_prob_threshold=params.get("cum_prob_threshold", 0.0),
            use_gumbel_topk=params.get("use_gumbel_topk", True),
            gumbel_temperature=params.get("gumbel_temperature", 1.0),
            min_branch_prob=params.get("min_branch_prob", 0.01),
            device=device,
        )
    elif engine_type == "Adaptive Tree (Phase 2)":
        return AdaptiveTreeSpeculativeDecodingEngine(
            draft_model, target_model, tokenizer,
            max_nodes=params.get("max_nodes", 20),
            max_depth=params.get("max_depth", 4),
            base_branching=params.get("base_branching", 3),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 0),
            entropy_threshold=params.get("entropy_threshold", 2.0),
            device=device,
        )
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


def run_generation(engine, prompt, max_new_tokens):
    """Run generation and measure time."""
    start = time.time()
    output = engine.generate(prompt, max_new_tokens=max_new_tokens)
    elapsed = time.time() - start
    return output, elapsed


def run_autoregressive(target_model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k, device):
    """Run autoregressive generation."""
    start = time.time()
    output = autoregressive_generate(
        target_model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        device=device,
    )
    elapsed = time.time() - start
    return output, elapsed


def visualize_tree(tree, result, tokenizer):
    """Create a Graphviz visualization of the draft tree with acceptance results."""
    try:
        import graphviz
    except ImportError:
        st.warning("Graphviz not installed. Install with `pip install graphviz` to see tree visualization.")
        return None

    dot = graphviz.Digraph(comment='Draft Tree', format='svg')
    dot.attr(rankdir='TB', size='8,5')
    dot.attr('node', shape='box', style='filled', fontname='Arial')
    dot.attr('edge', fontname='Arial')

    # Color mapping
    ACCEPTED_COLOR = '#90EE90'      # Light green
    REJECTED_COLOR = '#FFB6C1'      # Light red
    RESAMPLED_COLOR = '#FFD700'     # Gold
    PENDING_COLOR = '#E0E0E0'       # Light gray

    accepted_indices = set(result.accepted_node_indices)
    rejected_at = result.rejected_at

    for i, node in enumerate(tree.nodes):
        token_text = tokenizer.decode([node.token_id], skip_special_tokens=True)
        label = f"Node {i}\nToken: '{token_text}'\nID: {node.token_id}"

        if i in accepted_indices:
            color = ACCEPTED_COLOR
            label += "\n✓ ACCEPTED"
        elif i == rejected_at:
            color = REJECTED_COLOR
            label += f"\n✗ REJECTED\nResampled: {result.resampled_token}"
        else:
            color = PENDING_COLOR
            label += "\n⟳ NOT VERIFIED"

        dot.node(str(i), label, fillcolor=color)

        # Add edges
        if node.parent_idx >= 0:
            dot.edge(str(node.parent_idx), str(i))

    return dot


def build_tree_from_result(tree, result, tokenizer):
    """Extract tree info for display."""
    nodes_info = []
    for i, node in enumerate(tree.nodes):
        token_text = tokenizer.decode([node.token_id], skip_special_tokens=True)
        status = "pending"
        if i in result.accepted_node_indices:
            status = "accepted"
        elif i == result.rejected_at:
            status = f"rejected (resampled: {result.resampled_token})"
        
        nodes_info.append({
            "index": i,
            "token": token_text,
            "token_id": node.token_id,
            "parent": node.parent_idx,
            "depth": node.depth,
            "children": node.children,
            "status": status,
        })
    return nodes_info


def main():
    st.title("🌳 Speculative Tree Decoding Demo")
    st.markdown("""
    This demo compares **speculative decoding** engines against standard autoregressive generation.
    
    **What is Speculative Decoding?**
    A small draft model proposes multiple tokens, a large target model verifies them in parallel.
    If the draft matches the target, we get multiple tokens for the price of one forward pass!
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Model selection
    model_pairs = {
        "Qwen2.5-0.5B + Qwen2.5-1.5B (Recommended)": (
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        "SmolLM2-135M + SmolLM2-1.7B (CPU-friendly)": (
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        ),
    }
    selected_pair = st.sidebar.selectbox("Model Pair", list(model_pairs.keys()))
    draft_name, target_name = model_pairs[selected_pair]

    # Device
    device_options = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    if not torch.cuda.is_available():
        st.sidebar.warning("⚠️ No CUDA device detected. Running on CPU will be slow.")
    device = st.sidebar.selectbox("Device", device_options)
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Engine selection
    engine_types = [
        "Chain (Phase 0)",
        "Static Tree (Phase 1)",
        "Dynamic Tree (Phase 2)",
        "Adaptive Tree (Phase 2)",
    ]
    engine_type = st.sidebar.selectbox("Engine Type", engine_types)

    # Engine-specific parameters
    st.sidebar.subheader("Engine Parameters")
    if engine_type == "Chain (Phase 0)":
        params = {
            "gamma": st.sidebar.slider("Gamma (draft length)", 1, 8, 4),
            "temperature": st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1),
            "top_p": st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05),
            "top_k": st.sidebar.slider("Top-k (0=disabled)", 0, 100, 0),
        }
    elif engine_type == "Static Tree (Phase 1)":
        params = {
            "branching_factor": st.sidebar.slider("Branching Factor", 1, 6, 3),
            "max_depth": st.sidebar.slider("Max Depth", 1, 5, 2),
            "temperature": st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1),
            "top_p": st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05),
            "top_k": st.sidebar.slider("Top-k (0=disabled)", 0, 100, 0),
        }
    elif engine_type == "Dynamic Tree (Phase 2)":
        params = {
            "max_nodes": st.sidebar.slider("Max Nodes", 5, 30, 15),
            "max_depth": st.sidebar.slider("Max Depth", 1, 6, 4),
            "branching_factor": st.sidebar.slider("Branching Factor", 1, 8, 4),
            "temperature": st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1),
            "top_p": st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05),
            "top_k": st.sidebar.slider("Top-k (0=disabled)", 0, 100, 0),
            "cum_prob_threshold": st.sidebar.slider("Cumulative Prob Threshold", 0.0, 0.5, 0.0, 0.01),
            "use_gumbel_topk": st.sidebar.checkbox("Use Gumbel-Top-k", True),
            "gumbel_temperature": st.sidebar.slider("Gumbel Temperature", 0.1, 2.0, 1.0, 0.1),
            "min_branch_prob": st.sidebar.slider("Min Branch Prob", 0.0, 0.1, 0.01, 0.001),
        }
    else:  # Adaptive Tree
        params = {
            "max_nodes": st.sidebar.slider("Max Nodes", 5, 40, 20),
            "max_depth": st.sidebar.slider("Max Depth", 1, 6, 4),
            "base_branching": st.sidebar.slider("Base Branching", 1, 6, 3),
            "temperature": st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1),
            "top_p": st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05),
            "top_k": st.sidebar.slider("Top-k (0=disabled)", 0, 100, 0),
            "entropy_threshold": st.sidebar.slider("Entropy Threshold", 0.5, 4.0, 2.0, 0.1),
        }

    # Generation parameters
    st.sidebar.subheader("Generation")
    prompt = st.sidebar.text_area("Prompt", "Write a short poem about coding:", height=100)
    max_new_tokens = st.sidebar.slider("Max New Tokens", 10, 100, 40)
    run_button = st.sidebar.button("🚀 Generate", type="primary")

    # Device warning
    if device == "cpu":
        st.warning("⚠️ Running on CPU - generation will be slow. For best results, use a GPU.")

    # Load models
    with st.spinner("Loading models..."):
        draft_model, target_model, tokenizer = load_models_cached(
            draft_name, target_name, device, dtype
        )

    # Create engine
    engine = create_engine(engine_type, draft_model, target_model, tokenizer, params, torch.device(device))

    # Main area
    col1, col2 = st.columns(2)

    if run_button:
        if not prompt.strip():
            st.error("Please enter a prompt!")
            return

        with st.spinner("Generating..."):
            # Run speculative decoding
            spec_output, spec_time = run_generation(engine, prompt, max_new_tokens)

            # Run autoregressive baseline
            ar_output, ar_time = run_autoregressive(
                target_model, tokenizer, prompt, max_new_tokens,
                params["temperature"], params["top_p"], params.get("top_k", 0),
                torch.device(device)
            )

        # Display results
        with col1:
            st.subheader("🚀 Speculative Decoding")
            st.code(spec_output)
            st.metric("Time", f"{spec_time:.3f}s")
            st.metric("Tokens/sec", f"{max_new_tokens/spec_time:.1f}")

        with col2:
            st.subheader("🐢 Autoregressive (Baseline)")
            st.code(ar_output)
            st.metric("Time", f"{ar_time:.3f}s")
            st.metric("Tokens/sec", f"{max_new_tokens/ar_time:.1f}")

        # Comparison metrics
        speedup = ar_time / spec_time if spec_time > 0 else 0
        st.subheader("📊 Comparison")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Speedup", f"{speedup:.2f}x")
        metric_col2.metric("Time Saved", f"{ar_time - spec_time:.3f}s")
        metric_col3.metric("Relative Speed", f"{(speedup-1)*100:.0f}% faster")

        # Tree visualization (for tree-based engines)
        if engine_type != "Chain (Phase 0)":
            st.subheader("🌳 Draft Tree Visualization")
            st.info("The tree shows the draft model's proposed continuations. Green = accepted, Red = rejected, Gold = resampled.")

            # Build a tree from the last generation step for visualization
            # We need to re-run the tree builder to get the tree structure
            # This is a simplified version - in practice, you'd store the tree from the last step
            
            # For demo purposes, let's build a tree to show the structure
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask

                if engine_type == "Static Tree (Phase 1)":
                    tree = engine.tree_builder.build_tree(
                        draft_model, input_ids, attention_mask, device
                    )
                elif engine_type == "Dynamic Tree (Phase 2)":
                    tree = engine.tree_builder.build_tree(
                        draft_model, input_ids, attention_mask, device
                    )
                else:  # Adaptive Tree
                    tree = engine.tree_builder.build_tree(
                        draft_model, input_ids, attention_mask, device
                    )

                # Run verification to get results
                if params["temperature"] == 0.0:
                    from speculative_tree_decoding.verify import verify_tree_with_kv_cache
                    from speculative_tree_decoding.tree_kv_cache import TreeKVCache
                    target_cache = TreeKVCache(
                        target_model.config, batch_size=1, max_length=4096,
                        device=device, dtype=dtype
                    )
                    result = verify_tree_with_kv_cache(
                        target_model, tree, input_ids, target_cache,
                        attention_mask, temperature=params["temperature"],
                        top_p=params["top_p"], top_k=params.get("top_k", 0),
                        device=device
                    )
                else:
                    from speculative_tree_decoding.verify import verify_tree_with_kv_cache
                    from speculative_tree_decoding.tree_kv_cache import TreeKVCache
                    target_cache = TreeKVCache(
                        target_model.config, batch_size=1, max_length=4096,
                        device=device, dtype=dtype
                    )
                    result = verify_tree_with_kv_cache(
                        target_model, tree, input_ids, target_cache,
                        attention_mask, params["temperature"], params["top_p"],
                        params.get("top_k", 0), device
                    )

                # Visualize
                dot = visualize_tree(tree, result, tokenizer)
                if dot:
                    st.graphviz_chart(dot)

                # Show tree details
                with st.expander("📋 Tree Details"):
                    nodes_info = build_tree_from_result(tree, result, tokenizer)
                    for node in nodes_info:
                        status_emoji = {
                            "accepted": "✅",
                            "rejected": "❌",
                            "pending": "⏳"
                        }.get(node["status"].split()[0], "❓")
                        st.write(f"{status_emoji} **Node {node['index']}** (depth {node['depth']}, parent {node['parent']}): "
                                 f"Token='{node['token']}' (ID: {node['token_id']}) - {node['status']}")

    else:
        st.info("👈 Configure settings in the sidebar and click **Generate** to see the comparison!")

    # Footer
    st.markdown("---")
    st.markdown("""
    ### How it works
    
    **Chain (Phase 0)**: Draft model proposes a linear chain of γ tokens. Target model verifies all in one pass.
    
    **Static Tree (Phase 1)**: Draft model builds a fixed-shape tree. Target verifies all branches in parallel with custom attention mask.
    
    **Dynamic Tree (Phase 2)**: Draft model adaptively expands high-probability branches using Gumbel-Top-k sampling.
    
    **Adaptive Tree (Phase 2)**: Branching factor adjusts based on draft model's entropy (confidence).
    
    ### Key Metrics
    - **Acceptance Length**: Average tokens accepted per verification step
    - **Speedup**: Ratio of autoregressive time to speculative time
    - **Tokens/sec**: Throughput for each method
    """)


if __name__ == "__main__":
    main()