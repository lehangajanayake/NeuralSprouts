"""Streamlit app to visualize Model V5 architecture (triple-branch fusion).

Run:
    streamlit run visualize_model_arch.py

Controls let you tweak branch_dim, fc_hidden, and input resolution; the app
renders an ASCII-style diagram plus a per-branch shape table derived from a
dummy forward pass (no training data needed).
"""

import textwrap
import streamlit as st
import torch
import plotly.graph_objects as go

from model import PlantV5TripleBranch

st.set_page_config(page_title="Model V5 Architecture", layout="wide")
st.title("Model V5 Architecture Viewer")
st.caption("Triple-branch fusion: RGB, RGBD, Depth → Fusion FC → Dry weight")

# Sidebar controls
st.sidebar.header("Model Hyperparameters")
branch_dim = st.sidebar.slider("branch_dim (features per branch)", 64, 512, 256, step=32)
fc_hidden = st.sidebar.slider("Fusion hidden size", 64, 512, 256, step=32)
dropout = st.sidebar.slider("Dropout", 0.0, 0.6, 0.2, step=0.05)
input_size = st.sidebar.selectbox("Input resolution", [128, 96, 64], index=0)
device = "cpu"  # keep small and CPU-safe for visualization

# Instantiate model
model = PlantV5TripleBranch(branch_dim=branch_dim, fc_hidden=fc_hidden, dropout=dropout).to(device)
model.eval()

# Dummy tensors to probe shapes
with torch.no_grad():
    x_rgb = torch.zeros(1, 3, input_size, input_size, device=device)
    x_rgbd = torch.zeros(1, 4, input_size, input_size, device=device)
    x_d = torch.zeros(1, 1, input_size, input_size, device=device)
    rgb_feat = model.rgb_branch(x_rgb)
    rgbd_feat = model.rgbd_branch(x_rgbd)
    d_feat = model.depth_branch(x_d)
    fused = torch.cat([rgb_feat, rgbd_feat, d_feat], dim=1)
    out = model.fusion_fc(fused)


def render_graph():
    nodes = [
        {"name": f"RGB\n3x{input_size}x{input_size}", "x": 0, "y": 2.4, "color": "#4e79a7"},
        {"name": f"RGBD\n4x{input_size}x{input_size}", "x": 0, "y": 1.6, "color": "#59a14f"},
        {"name": f"Depth\n1x{input_size}x{input_size}", "x": 0, "y": 0.8, "color": "#9c755f"},
        {"name": f"RGB Branch\nfeat {branch_dim}", "x": 1.5, "y": 2.4, "color": "#4e79a7"},
        {"name": f"RGBD Branch\nfeat {branch_dim}", "x": 1.5, "y": 1.6, "color": "#59a14f"},
        {"name": f"Depth Branch\nfeat {branch_dim}", "x": 1.5, "y": 0.8, "color": "#9c755f"},
        {"name": f"Concat\n{branch_dim*3}", "x": 3, "y": 1.6, "color": "#f28e2b"},
        {"name": f"Fusion FC\n{branch_dim*3}→{fc_hidden}", "x": 4.5, "y": 1.6, "color": "#edc948"},
        {"name": f"Hidden\n{fc_hidden}→{fc_hidden//2}", "x": 6, "y": 1.6, "color": "#bab0ab"},
        {"name": "Output\n1", "x": 7.5, "y": 1.6, "color": "#af7aa1"},
    ]

    edges = [
        (0, 3, f"Conv stack → {branch_dim}"),
        (1, 4, f"Conv stack → {branch_dim}"),
        (2, 5, f"Conv stack → {branch_dim}"),
        (3, 6, f"{branch_dim}"),
        (4, 6, f"{branch_dim}"),
        (5, 6, f"{branch_dim}"),
        (6, 7, f"{branch_dim*3} → {fc_hidden}"),
        (7, 8, f"{fc_hidden} → {fc_hidden//2}"),
        (8, 9, f"{fc_hidden//2} → 1"),
    ]

    edge_traces = []
    label_x, label_y, label_text = [], [], []
    for src, dst, label in edges:
        x0, y0 = nodes[src]["x"], nodes[src]["y"]
        x1, y1 = nodes[dst]["x"], nodes[dst]["y"]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#b0b0b0", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        label_x.append((x0 + x1) / 2)
        label_y.append((y0 + y1) / 2 + 0.08)
        label_text.append(label)

    node_trace = go.Scatter(
        x=[n["x"] for n in nodes],
        y=[n["y"] for n in nodes],
        mode="markers+text",
        marker=dict(size=36, color=[n["color"] for n in nodes], line=dict(width=1, color="#2f2f2f")),
        text=[n["name"] for n in nodes],
        textposition="bottom center",
        hoverinfo="text",
        showlegend=False,
    )

    label_trace = go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=label_text,
        textfont=dict(size=11, color="#444"),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(edge_traces + [node_trace, label_trace])
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
    )
    st.subheader("Interactive Graph (updates with sliders)")
    st.plotly_chart(fig, use_container_width=True)


render_graph()

# ASCII diagram
ascii_diagram = f"""
RGB (3x{input_size}x{input_size})        RGBD (4x{input_size}x{input_size})        Depth (1x{input_size}x{input_size})
        │                                  │                                      │
        ▼                                  ▼                                      ▼
   RGB Branch                         RGBD Branch                           Depth Branch
   Conv stacks                        Conv stacks                           Conv stacks
   → feat: {branch_dim}                    → feat: {branch_dim}                          → feat: {branch_dim}
        \                                  |                                      /
         \                                 |                                     /
          \                                |                                    /
           --------- Concatenate (dim={branch_dim*3}) ---------
                                  │
                                  ▼
                            Fusion FC (hidden={fc_hidden})
                                  │
                                  ▼
                            Output: 1 scalar (dry weight)
"""

st.subheader("Architecture Diagram")
st.code(textwrap.dedent(ascii_diagram), language="text")

# Shapes table
st.subheader("Tensor Shapes (dummy forward)")
st.table({
    "Tensor": ["rgb", "rgbd", "depth", "rgb_feat", "rgbd_feat", "depth_feat", "fused", "output"],
    "Shape": [
        tuple(x_rgb.shape),
        tuple(x_rgbd.shape),
        tuple(x_d.shape),
        tuple(rgb_feat.shape),
        tuple(rgbd_feat.shape),
        tuple(d_feat.shape),
        tuple(fused.shape),
        tuple(out.shape),
    ],
})

# Layer summary (lightweight textual)
st.subheader("Layer Summary (per branch)")
st.markdown(
    f"- **RGB Branch**: ConvBlock x4 → Linear(256*{input_size//16}*{input_size//16} → {branch_dim})"
)
st.markdown(
    f"- **RGBD Branch**: ConvBlock x4 → Linear(256*{input_size//16}*{input_size//16} → {branch_dim})"
)
st.markdown(
    f"- **Depth Branch**: ConvBlock x4 → Linear(256*{input_size//16}*{input_size//16} → {branch_dim})"
)
st.markdown(
    f"- **Fusion FC**: Linear({branch_dim*3} → {fc_hidden}) → Linear({fc_hidden} → {fc_hidden//2}) → Linear({fc_hidden//2} → 1)"
)

st.info("Shapes are computed with a dummy forward pass on CPU; no real data needed.")
