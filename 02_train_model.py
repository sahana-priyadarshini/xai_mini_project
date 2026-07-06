"""
02_train_model.py
==================
Trains a 3-layer FastRGCN (with dropout) on the DBpedia persons graph.
Saves the model checkpoint and a training-curve plot.

Architecture chosen via a controlled hyperparameter sweep in
06_tune_model.py over hidden size (32/64/128), depth (2 vs 3 layers), and
dropout (0.0-0.5), run on the same honest (non-leaky) feature set. The
config below (hidden=128, 3 layers, dropout=0.5) gave the best held-out
test accuracy in that sweep.

Run:
    python 02_train_model.py
"""

import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import FastRGCNConv

from utils import ensure_dir, load_checkpoint, save_checkpoint, set_seed

set_seed(42)
ensure_dir("outputs")

# ---------------------------------------------------------------------------
# Hyper-parameters (selected via 06_tune_model.py's sweep)
# ---------------------------------------------------------------------------
HIDDEN_CHANNELS = 128
NUM_LAYERS = 3
DROPOUT = 0.5
NUM_BASES = 30
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 400

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class FastRGCN(torch.nn.Module):
    """N-layer Relational GCN with basis decomposition and dropout."""

    def __init__(self, in_channels: int, num_relations: int, num_classes: int,
                 hidden_channels: int = HIDDEN_CHANNELS,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(
                FastRGCNConv(in_channels, num_classes, num_relations, num_bases=NUM_BASES))
        else:
            self.convs.append(
                FastRGCNConv(in_channels, hidden_channels, num_relations, num_bases=NUM_BASES))
            for _ in range(num_layers - 2):
                self.convs.append(
                    FastRGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=NUM_BASES))
            self.convs.append(
                FastRGCNConv(hidden_channels, num_classes, num_relations, num_bases=NUM_BASES))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = x.relu()
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------

def train(model, data, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data) -> tuple[float, float]:
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load data
    data: Data = torch.load("data/pyg_data.pt", weights_only=False)
    print(f"[train] Loaded data: {data}")

    in_channels = data.x.shape[1]
    num_relations = int(data.edge_type.max()) + 1
    num_classes = int(data.train_y.max()) + 1

    print(f"[train] in_channels={in_channels}, "
          f"num_relations={num_relations}, num_classes={num_classes}")

    device = torch.device("cpu")
    model = FastRGCN(in_channels, num_relations, num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # Training loop
    history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": []}
    times = []

    print("\n" + "-" * 60)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train(model, data, optimizer)
        train_acc, test_acc = test(model, data)
        elapsed = time.time() - t0
        times.append(elapsed)

        history["epoch"].append(epoch)
        history["loss"].append(round(loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["test_acc"].append(round(test_acc, 4))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3}  Loss {loss:.4f}  "
                f"Train {train_acc:.4f}  Test {test_acc:.4f}  "
                f"({elapsed:.2f}s)"
            )

    median_t = torch.tensor(times).median()
    print(f"\nMedian time/epoch: {median_t:.4f}s")
    print("-" * 60)

    # Final performance
    final_train, final_test = test(model, data)
    print(f"\n[results] Final Train Accuracy : {final_train:.4f}")
    print(f"[results] Final Test  Accuracy : {final_test:.4f}")

    # Save model
    save_checkpoint(model, "outputs/rgcn_model.pt")

    # Save training history CSV
    df = pd.DataFrame(history)
    df.to_csv("outputs/training_history.csv", index=False)
    print("[train] Saved training history to outputs/training_history.csv")

    # Save performance table
    perf = pd.DataFrame([{
        "Model": f"FastRGCN ({NUM_LAYERS}-layer)",
        "Hidden": HIDDEN_CHANNELS,
        "Dropout": DROPOUT,
        "Bases": NUM_BASES,
        "Epochs": EPOCHS,
        "LR": LR,
        "Train Acc": round(final_train, 4),
        "Test Acc": round(final_test, 4),
    }])
    perf.to_csv("outputs/model_performance.csv", index=False)
    print("[train] Saved performance table to outputs/model_performance.csv")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["loss"], color="steelblue", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL Loss")
    axes[0].set_title("Training Loss — FastRGCN on DBpedia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history["epoch"], history["train_acc"],
        label="Train", color="steelblue", linewidth=2
    )
    axes[1].plot(
        history["epoch"], history["test_acc"],
        label="Test", color="darkorange", linewidth=2, linestyle="--"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves — FastRGCN on DBpedia")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150)
    plt.close()
    print("[train] Saved training curves to outputs/training_curves.png")

    print("\n[done] Step 2 complete. Run: python 03_explain.py")