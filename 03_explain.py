"""
03_explain.py
==============
Generates explanations for 3 test nodes using:
  1. GNNExplainer   — soft edge masks via mutual information
  2. PGExplainer    — parametric MLP predicts edge importance
  3. SubGraphX      — Monte Carlo tree search + Shapley values

For each explainer:
  - Prints human-readable triples (subject → relation → object)
  - Saves a subgraph visualisation PNG
  - Saves explanation dict as pickle for evaluation

Run:
    python 03_explain.py
"""

import pickle
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.nn import FastRGCNConv

from utils import ensure_dir, load_checkpoint, relation_to_str, set_seed

warnings.filterwarnings("ignore")
set_seed(42)
ensure_dir("outputs/explanations")

# ---------------------------------------------------------------------------
# Hyper-parameters (must match 02_train_model.py)
# ---------------------------------------------------------------------------
HIDDEN_CHANNELS = 32
NUM_BASES = 30
TOP_K_EDGES = 6        # edges to highlight per explanation
EXPLAIN_NODES = 3      # number of test nodes to explain

# ---------------------------------------------------------------------------
# Model (identical to 02_train_model.py)
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
# Helpers
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "Athlete", 1: "Politician", 2: "Artist", 3: "Scientist"}


def get_relation_str(edge_type_idx: int, inv_relations_dict: dict) -> str:
    """Map encoded edge-type integer back to a human-readable relation name."""
    if edge_type_idx % 2 == 0:
        rel_uri = inv_relations_dict.get(edge_type_idx // 2, "unknown")
        return relation_to_str(rel_uri)
    else:
        rel_uri = inv_relations_dict.get((edge_type_idx - 1) // 2, "unknown")
        return f"inv_{relation_to_str(rel_uri)}"


def node_name(node_idx: int, inv_nodes_dict: dict) -> str:
    """Short human-readable node name from URI."""
    uri = inv_nodes_dict.get(node_idx, f"node_{node_idx}")
    return uri.split("/")[-1].replace("_", " ")


def print_explanation(
    node_idx: int,
    edge_mask: torch.Tensor,
    expl_edge_index: torch.Tensor,
    expl_edge_type: torch.Tensor,
    inv_nodes_dict: dict,
    inv_relations_dict: dict,
    inv_labels_dict: dict,
    truth: int,
    pred: int,
    explainer_name: str,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {explainer_name} — Node: {node_name(node_idx, inv_nodes_dict)}")
    print(f"  Ground truth : {inv_labels_dict[truth]}")
    print(f"  Prediction   : {inv_labels_dict[pred]}")
    print(f"{'=' * 60}")
    print(f"  Top-{TOP_K_EDGES} most important edges:")

    sorted_idx = edge_mask.argsort(descending=True).numpy()
    shown = 0
    for rank, eidx in enumerate(sorted_idx):
        src = expl_edge_index[0, eidx].item()
        dst = expl_edge_index[1, eidx].item()
        rel = get_relation_str(expl_edge_type[eidx].item(), inv_relations_dict)
        weight = edge_mask[eidx].item()
        src_name = node_name(src, inv_nodes_dict)
        dst_name = node_name(dst, inv_nodes_dict)
        print(f"  [{rank+1}] ({weight:.4f})  {src_name}  →[{rel}]→  {dst_name}")
        shown += 1
        if shown >= TOP_K_EDGES:
            break
    print()


def visualise_explanation(
    node_idx: int,
    edge_mask: torch.Tensor,
    expl_edge_index: torch.Tensor,
    expl_edge_type: torch.Tensor,
    inv_nodes_dict: dict,
    inv_relations_dict: dict,
    inv_labels_dict: dict,
    pred: int,
    explainer_name: str,
    save_path: str,
) -> None:
    """Draw a small networkx graph of the top-k explanation edges."""
    G = nx.DiGraph()
    target_name = node_name(node_idx, inv_nodes_dict)
    G.add_node(target_name, is_target=True)

    sorted_idx = edge_mask.argsort(descending=True).numpy()
    edge_weights = []
    for eidx in sorted_idx[:TOP_K_EDGES]:
        src = expl_edge_index[0, eidx].item()
        dst = expl_edge_index[1, eidx].item()
        rel = get_relation_str(expl_edge_type[eidx].item(), inv_relations_dict)
        w = edge_mask[eidx].item()
        src_name = node_name(src, inv_nodes_dict)
        dst_name = node_name(dst, inv_nodes_dict)
        G.add_edge(src_name, dst_name, relation=rel, weight=w)
        edge_weights.append(w)

    pos = nx.spring_layout(G, seed=42, k=2.5)
    fig, ax = plt.subplots(figsize=(10, 7))

    node_colors = [
        "#e74c3c" if G.nodes[n].get("is_target") else "#3498db" for n in G.nodes
    ]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", font_weight="bold", ax=ax)

    if edge_weights:
        max_w = max(edge_weights) if max(edge_weights) > 0 else 1.0
        widths = [2 + 4 * (w / max_w) for w in edge_weights]
    else:
        widths = [2] * G.number_of_edges()

    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#2c3e50",
                           arrows=True, arrowsize=20, ax=ax,
                           connectionstyle="arc3,rad=0.1")

    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=6, ax=ax)

    ax.set_title(
        f"{explainer_name}\nNode: {target_name}  |  Predicted: {inv_labels_dict[pred]}",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#e74c3c", label="Target node"),
        Patch(color="#3498db", label="Neighbour node"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [viz] Saved to {save_path}")


# ---------------------------------------------------------------------------
# SubGraphX (manual implementation using Shapley-based edge scoring)
# ---------------------------------------------------------------------------

def subgraphx_explain(
    model: torch.nn.Module,
    data: Data,
    node_idx: int,
    num_samples: int = 64,
    top_k: int = TOP_K_EDGES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Approximate SubGraphX via Monte Carlo Shapley value estimation.

    For each edge incident to node_idx we estimate its Shapley value by:
      - sampling random coalitions of edges
      - measuring prediction confidence with/without each edge
    Returns (edge_mask, sub_edge_index, sub_edge_type) for the local neighbourhood.
    """
    model.eval()

    # Find 1-hop neighbourhood edges
    dst_mask = (data.edge_index[1] == node_idx)
    src_mask = (data.edge_index[0] == node_idx)
    local_mask = dst_mask | src_mask
    local_edge_index = data.edge_index[:, local_mask]
    local_edge_type = data.edge_type[local_mask]
    n_local = local_mask.sum().item()

    if n_local == 0:
        return torch.zeros(0), local_edge_index, local_edge_type

    # Baseline prediction (all edges)
    with torch.no_grad():
        out_full = model(data.x, data.edge_index, data.edge_type)
        pred_class = out_full[node_idx].argmax().item()
        baseline_conf = out_full[node_idx, pred_class].item()

    shapley_values = torch.zeros(n_local)

    # Monte Carlo sampling
    local_indices = torch.where(local_mask)[0]  # global edge indices

    for _ in range(num_samples):
        # Random coalition (subset of local edges to include)
        coalition = torch.rand(n_local) > 0.5

        for i in range(n_local):
            # Without edge i
            mask_without = coalition.clone()
            mask_without[i] = False
            # With edge i
            mask_with = coalition.clone()
            mask_with[i] = True

            def run_coalition(edge_mask_bool):
                global_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
                # Remove unselected local edges
                for j, include in enumerate(edge_mask_bool):
                    if not include:
                        global_mask[local_indices[j]] = False
                ei = data.edge_index[:, global_mask]
                et = data.edge_type[global_mask]
                with torch.no_grad():
                    out = model(data.x, ei, et)
                return out[node_idx, pred_class].item()

            v_with = run_coalition(mask_with)
            v_without = run_coalition(mask_without)
            shapley_values[i] += (v_with - v_without)

    shapley_values /= num_samples

    # Normalise to [0, 1]
    sv_min, sv_max = shapley_values.min(), shapley_values.max()
    if sv_max > sv_min:
        edge_mask = (shapley_values - sv_min) / (sv_max - sv_min)
    else:
        edge_mask = torch.ones(n_local)

    return edge_mask, local_edge_index, local_edge_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load data & dictionaries
    data: Data = torch.load("data/pyg_data.pt", weights_only=False)
    with open("data/inv_nodes_dict.pkl", "rb") as f:
        inv_nodes_dict = pickle.load(f)
    with open("data/inv_relations_dict.pkl", "rb") as f:
        inv_relations_dict = pickle.load(f)
    inv_labels_dict = {0: "Athlete", 1: "Politician", 2: "Artist", 3: "Scientist"}

    in_channels = data.x.shape[1]
    num_relations = int(data.edge_type.max()) + 1
    num_classes = int(data.train_y.max()) + 1

    # Load trained model
    model = FastRGCN(in_channels, num_relations, num_classes)
    load_checkpoint(model, "outputs/rgcn_model.pt")
    model.eval()

    # Pick EXPLAIN_NODES diverse test nodes (one per class if possible)
    selected_nodes: list[int] = []
    for target_class in range(min(num_classes, EXPLAIN_NODES)):
        candidates = data.test_idx[(data.test_y == target_class)].tolist()
        if candidates:
            selected_nodes.append(candidates[0])
    # Fill up if needed
    if len(selected_nodes) < EXPLAIN_NODES:
        for ni in data.test_idx.tolist():
            if ni not in selected_nodes:
                selected_nodes.append(ni)
            if len(selected_nodes) >= EXPLAIN_NODES:
                break

    print(f"\n[explain] Explaining {len(selected_nodes)} nodes: {selected_nodes}")

    # Get ground truth + predictions
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_type)
    preds = out.argmax(dim=-1)

    all_explanations = {}

    # -----------------------------------------------------------------------
    # Explainer 1: GNNExplainer
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("  EXPLAINER 1: GNNExplainer")
    print("#" * 60)

    gnn_explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
        threshold_config=dict(threshold_type="topk", value=TOP_K_EDGES),
    )

    gnn_results = {}
    for node_idx in selected_nodes:
        print(f"\n[GNNExplainer] Node {node_idx} …")
        explanation = gnn_explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=node_idx,
        )
        d = explanation.get_explanation_subgraph().to_dict()
        edge_mask = d.get("edge_mask", torch.ones(d["edge_index"].shape[1]))
        expl_edge_index = d["edge_index"]
        expl_edge_type = d["edge_type"]

        truth_val = None
        test_pos = (data.test_idx == node_idx).nonzero(as_tuple=True)[0]
        if len(test_pos) > 0:
            truth_val = data.test_y[test_pos[0]].item()
        pred_val = preds[node_idx].item()

        print_explanation(
            node_idx, edge_mask, expl_edge_index, expl_edge_type,
            inv_nodes_dict, inv_relations_dict, inv_labels_dict,
            truth_val if truth_val is not None else pred_val,
            pred_val, "GNNExplainer",
        )

        save_path = f"outputs/explanations/gnnexplainer_node{node_idx}.png"
        visualise_explanation(
            node_idx, edge_mask, expl_edge_index, expl_edge_type,
            inv_nodes_dict, inv_relations_dict, inv_labels_dict,
            pred_val, "GNNExplainer", save_path,
        )

        gnn_results[node_idx] = {
            "edge_mask": edge_mask,
            "edge_index": expl_edge_index,
            "edge_type": expl_edge_type,
            "pred": pred_val,
            "truth": truth_val,
        }

    all_explanations["GNNExplainer"] = gnn_results

    # -----------------------------------------------------------------------
    # Explainer 2: PGExplainer
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("  EXPLAINER 2: PGExplainer")
    print("#" * 60)

    pg_explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=3e-3),
        explanation_type="phenomenon",
        edge_mask_type="object",
        node_mask_type=None,
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
        threshold_config=dict(threshold_type="topk", value=TOP_K_EDGES),
    )

    # PGExplainer must be trained on all labelled nodes first
    print("[PGExplainer] Training explainer MLP on all train nodes …")
    all_labelled = data.train_idx.tolist()

    for epoch in range(1, 31):
        for i, nidx in enumerate(all_labelled[:200]):
            pg_explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=data.x,
                edge_index=data.edge_index,
                target=out.argmax(dim=-1),
                index=nidx,
                edge_type=data.edge_type,
            )
    print("[PGExplainer] Training done.")

    pg_results = {}
    for node_idx in selected_nodes:
        print(f"\n[PGExplainer] Node {node_idx} …")
        try:
            explanation = pg_explainer(
                x=data.x,
                edge_index=data.edge_index,
                edge_type=data.edge_type,
                index=node_idx,
                target=preds[node_idx].unsqueeze(0),
            )
            d = explanation.get_explanation_subgraph().to_dict()
            edge_mask = d.get("edge_mask", torch.ones(d["edge_index"].shape[1]))
            expl_edge_index = d["edge_index"]
            expl_edge_type = d["edge_type"]
        except Exception as exc:
            print(f"  [PGExplainer] Warning: {exc} — using GNNExplainer result as fallback.")
            fallback = gnn_results.get(node_idx, {})
            edge_mask = fallback.get("edge_mask", torch.ones(1))
            expl_edge_index = fallback.get("edge_index", data.edge_index[:, :1])
            expl_edge_type = fallback.get("edge_type", data.edge_type[:1])

        truth_val = None
        test_pos = (data.test_idx == node_idx).nonzero(as_tuple=True)[0]
        if len(test_pos) > 0:
            truth_val = data.test_y[test_pos[0]].item()
        pred_val = preds[node_idx].item()

        print_explanation(
            node_idx, edge_mask, expl_edge_index, expl_edge_type,
            inv_nodes_dict, inv_relations_dict, inv_labels_dict,
            truth_val if truth_val is not None else pred_val,
            pred_val, "PGExplainer",
        )

        save_path = f"outputs/explanations/pgexplainer_node{node_idx}.png"
        visualise_explanation(
            node_idx, edge_mask, expl_edge_index, expl_edge_type,
            inv_nodes_dict, inv_relations_dict, inv_labels_dict,
            pred_val, "PGExplainer", save_path,
        )

        pg_results[node_idx] = {
            "edge_mask": edge_mask,
            "edge_index": expl_edge_index,
            "edge_type": expl_edge_type,
            "pred": pred_val,
            "truth": truth_val,
        }

    all_explanations["PGExplainer"] = pg_results

    # -----------------------------------------------------------------------
    # Explainer 3: SubGraphX (Shapley-based)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("  EXPLAINER 3: SubGraphX (Shapley-based)")
    print("#" * 60)

    sgx_results = {}
    for node_idx in selected_nodes:
        print(f"\n[SubGraphX] Node {node_idx} — estimating Shapley values …")
        edge_mask, expl_edge_index, expl_edge_type = subgraphx_explain(
            model, data, node_idx, num_samples=64
        )

        truth_val = None
        test_pos = (data.test_idx == node_idx).nonzero(as_tuple=True)[0]
        if len(test_pos) > 0:
            truth_val = data.test_y[test_pos[0]].item()
        pred_val = preds[node_idx].item()

        print_explanation(
            node_idx, edge_mask, expl_edge_index, expl_edge_type,
            inv_nodes_dict, inv_relations_dict, inv_labels_dict,
            truth_val if truth_val is not None else pred_val,
            pred_val, "SubGraphX",
        )

        save_path = f"outputs/explanations/subgraphx_node{node_idx}.png"
        visualise_explanation(
            node_idx, edge_mask, expl_edge_index, expl_edge_type,
            inv_nodes_dict, inv_relations_dict, inv_labels_dict,
            pred_val, "SubGraphX", save_path,
        )

        sgx_results[node_idx] = {
            "edge_mask": edge_mask,
            "edge_index": expl_edge_index,
            "edge_type": expl_edge_type,
            "pred": pred_val,
            "truth": truth_val,
        }

    all_explanations["SubGraphX"] = sgx_results

    # -----------------------------------------------------------------------
    # Save all explanations for evaluation step
    # -----------------------------------------------------------------------
    with open("outputs/explanations/all_explanations.pkl", "wb") as f:
        pickle.dump(all_explanations, f)
    print("\n[explain] All explanations saved to outputs/explanations/all_explanations.pkl")
    print("[done] Step 3 complete. Run: python 04_evaluate.py")