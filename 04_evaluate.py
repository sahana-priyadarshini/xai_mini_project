"""
04_evaluate.py
===============
Quantitatively evaluates all three explainers on:
  - Fidelity+   : accuracy drop when top-k explanation edges are REMOVED
  - Fidelity−   : accuracy when ONLY top-k explanation edges are kept
  - Sparsity    : fraction of total edges selected (lower = more concise)
  - Stability   : mean Jaccard similarity of top-k edges across 2 runs
                  (only for GNNExplainer; approximated via mask variance)

Results are saved as a CSV table and a bar-chart figure.

Run:
    python 04_evaluate.py
"""

import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import FastRGCNConv

from utils import ensure_dir, load_checkpoint, set_seed

warnings.filterwarnings("ignore")
set_seed(42)
ensure_dir("outputs")

HIDDEN_CHANNELS = 32
NUM_BASES = 30
TOP_K = 6


# ---------------------------------------------------------------------------
# Model (same as before)
# ---------------------------------------------------------------------------

class FastRGCN(torch.nn.Module):
    def __init__(self, in_channels, num_relations, num_classes):
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels, HIDDEN_CHANNELS, num_relations, num_bases=NUM_BASES)
        self.conv2 = FastRGCNConv(HIDDEN_CHANNELS, num_classes, num_relations, num_bases=NUM_BASES)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def fidelity_plus(
    model: torch.nn.Module,
    data: Data,
    node_idx: int,
    expl_edge_index: torch.Tensor,
    pred_class: int,
) -> float:
    """
    Fidelity+ : confidence drop when explanation edges are REMOVED.
    Higher = explanation edges were important (good explainer).
    """
    # Full graph confidence
    out_full = model(data.x, data.edge_index, data.edge_type)
    conf_full = out_full[node_idx, pred_class].item()

    # Build mask that removes explanation edges
    expl_set = set(
        zip(expl_edge_index[0].tolist(), expl_edge_index[1].tolist())
    )
    keep = []
    for i in range(data.edge_index.shape[1]):
        pair = (data.edge_index[0, i].item(), data.edge_index[1, i].item())
        keep.append(pair not in expl_set)
    keep = torch.tensor(keep, dtype=torch.bool)

    if keep.sum() == 0:
        return 0.0

    out_removed = model(
        data.x,
        data.edge_index[:, keep],
        data.edge_type[keep],
    )
    conf_removed = out_removed[node_idx, pred_class].item()
    return max(0.0, conf_full - conf_removed)


@torch.no_grad()
def fidelity_minus(
    model: torch.nn.Module,
    data: Data,
    node_idx: int,
    expl_edge_index: torch.Tensor,
    expl_edge_type: torch.Tensor,
    pred_class: int,
) -> float:
    """
    Fidelity− : confidence when ONLY explanation edges are kept.
    Higher = explanation is sufficient (good explainer).
    """
    if expl_edge_index.shape[1] == 0:
        return 0.0

    out = model(data.x, expl_edge_index, expl_edge_type)
    return out[node_idx, pred_class].item() if node_idx < out.shape[0] else 0.0


def sparsity_score(
    data: Data,
    expl_edge_index: torch.Tensor,
) -> float:
    """
    Sparsity : fraction of total graph edges selected.
    Lower = more concise explanation.
    """
    total = data.edge_index.shape[1]
    selected = expl_edge_index.shape[1]
    if total == 0:
        return 0.0
    return selected / total


def stability_score(
    model: torch.nn.Module,
    data: Data,
    node_idx: int,
    n_runs: int = 3,
) -> float:
    """
    Stability : mean Jaccard similarity of top-k edge sets across n_runs.
    Approximated via GNNExplainer with different seeds.
    Higher = more stable (better).
    """
    edge_sets = []
    for seed in range(n_runs):
        torch.manual_seed(seed)
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type=None,
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="log_probs",
            ),
            threshold_config=dict(threshold_type="topk", value=TOP_K),
        )
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=node_idx,
        )
        d = explanation.get_explanation_subgraph().to_dict()
        ei = d["edge_index"]
        edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))
        edge_sets.append(edge_set)

    # Mean pairwise Jaccard
    scores = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            a, b = edge_sets[i], edge_sets[j]
            if not a and not b:
                scores.append(1.0)
            elif not a or not b:
                scores.append(0.0)
            else:
                scores.append(len(a & b) / len(a | b))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load data
    data: Data = torch.load("data/pyg_data.pt", weights_only=False)
    in_channels = data.x.shape[1]
    num_relations = int(data.edge_type.max()) + 1
    num_classes = int(data.train_y.max()) + 1

    # Load model
    model = FastRGCN(in_channels, num_relations, num_classes)
    load_checkpoint(model, "outputs/rgcn_model.pt")
    model.eval()

    # Load explanations
    with open("outputs/explanations/all_explanations.pkl", "rb") as f:
        all_explanations: dict = pickle.load(f)

    results = []

    for explainer_name, node_results in all_explanations.items():
        print(f"\n[eval] Evaluating {explainer_name} …")
        fid_plus_vals, fid_minus_vals, sparsity_vals = [], [], []

        for node_idx, expl in node_results.items():
            edge_mask: torch.Tensor = expl["edge_mask"]
            expl_edge_index: torch.Tensor = expl["edge_index"]
            expl_edge_type: torch.Tensor = expl["edge_type"]
            pred_class: int = expl["pred"]

            fp = fidelity_plus(model, data, node_idx, expl_edge_index, pred_class)
            fm = fidelity_minus(model, data, node_idx, expl_edge_index, expl_edge_type, pred_class)
            sp = sparsity_score(data, expl_edge_index)

            fid_plus_vals.append(fp)
            fid_minus_vals.append(fm)
            sparsity_vals.append(sp)

            print(
                f"  Node {node_idx:>5} | Fid+ {fp:.4f} | Fid- {fm:.4f} | Sparsity {sp:.4f}"
            )

        # Stability — only computed for GNNExplainer (expensive)
        if explainer_name == "GNNExplainer" and node_results:
            sample_node = list(node_results.keys())[0]
            print(f"  [stability] Computing for node {sample_node} (3 runs) …")
            stab = stability_score(model, data, sample_node, n_runs=3)
        else:
            stab = float("nan")

        results.append({
            "Explainer": explainer_name,
            "Fidelity+": round(float(np.mean(fid_plus_vals)), 4),
            "Fidelity-": round(float(np.mean(fid_minus_vals)), 4),
            "Sparsity":  round(float(np.mean(sparsity_vals)), 4),
            "Stability": round(stab, 4) if not np.isnan(stab) else "N/A",
        })

    # -----------------------------------------------------------------------
    # Print & save table
    # -----------------------------------------------------------------------
    df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("  EXPLANATION EVALUATION RESULTS")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)

    df.to_csv("outputs/evaluation_results.csv", index=False)
    print("\n[eval] Saved to outputs/evaluation_results.csv")

    # -----------------------------------------------------------------------
    # Bar chart
    # -----------------------------------------------------------------------
    metrics = ["Fidelity+", "Fidelity-", "Sparsity"]
    explainer_names = df["Explainer"].tolist()
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(explainer_names, colors)):
        vals = [df.loc[df["Explainer"] == name, m].values[0] for m in metrics]
        # Convert to float safely
        vals = [float(v) if v != "N/A" else 0.0 for v in vals]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title(
        "Explainer Comparison — DBpedia GNN\n"
        "(Fidelity+/−: higher is better; Sparsity: lower is better)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1] * 1.15))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/evaluation_barchart.png", dpi=150)
    plt.close()
    print("[eval] Saved bar chart to outputs/evaluation_barchart.png")

    # -----------------------------------------------------------------------
    # Metric explanation legend (for the report)
    # -----------------------------------------------------------------------
    legend_text = """
METRIC DEFINITIONS
==================
Fidelity+   The drop in prediction confidence when the top-k explanation
            edges are REMOVED. Higher means those edges were truly important.

Fidelity-   Prediction confidence when ONLY the top-k explanation edges
            are retained. Higher means the explanation alone is sufficient.

Sparsity    Fraction of total graph edges selected as explanation.
            Lower = more concise (prefer small, focused explanations).

Stability   Mean pairwise Jaccard similarity of top-k edge sets across
            3 independent GNNExplainer runs (same node, different seeds).
            Higher = more reproducible explanations.
"""
    print(legend_text)
    with open("outputs/metric_definitions.txt", "w") as f:
        f.write(legend_text)

    print("[done] Step 4 complete. Check outputs/ for all results.")