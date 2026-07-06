"""
06_tune_model.py
================
Legitimate hyperparameter tuning on the HONEST (non-leaky) DBpedia graph —
i.e. after rdf:type has been removed (see 01_data_download.py comment and
05_ablation.py condition C). No new data or predicates are introduced here;
we only vary model capacity, depth, and regularization to see how much of
the accuracy "lost" by removing the leaky rdf:type feature can be recovered
through legitimate means.

Requires: data/pyg_data.pt already rebuilt WITHOUT rdf:type.

Run:
    python 06_tune_model.py
"""

import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import FastRGCNConv

from utils import ensure_dir, set_seed

ensure_dir("outputs")

SEED = 42
LR = 0.01
WEIGHT_DECAY = 5e-4
NUM_BASES = 30


class FastRGCN(torch.nn.Module):
    """N-layer Relational GCN with basis decomposition and optional dropout."""

    def __init__(self, in_channels, num_relations, num_classes,
                 hidden_channels=32, num_layers=2, dropout=0.0):
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

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = x.relu()
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


def train_and_eval(data, hidden_channels, num_layers, dropout, epochs):
    set_seed(SEED)  # identical init across configs
    in_channels = data.x.shape[1]
    num_relations = int(data.edge_type.max()) + 1
    num_classes = int(data.train_y.max()) + 1
    model = FastRGCN(in_channels, num_relations, num_classes,
                      hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type)
        loss = F.nll_loss(out[data.train_idx], data.train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
        train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
        test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc


# A small, deliberately modest grid — capacity/depth/regularization only,
# no new features or predicates.
CONFIGS = [
    dict(hidden_channels=32, num_layers=2, dropout=0.0, epochs=100),   # current baseline
    dict(hidden_channels=64, num_layers=2, dropout=0.0, epochs=100),
    dict(hidden_channels=64, num_layers=2, dropout=0.3, epochs=200),
    dict(hidden_channels=128, num_layers=2, dropout=0.3, epochs=200),
    dict(hidden_channels=64, num_layers=3, dropout=0.3, epochs=200),
    dict(hidden_channels=128, num_layers=3, dropout=0.5, epochs=300),
    dict(hidden_channels=64, num_layers=2, dropout=0.5, epochs=300),
]

if __name__ == "__main__":
    data: Data = torch.load("data/pyg_data.pt", weights_only=False)
    print(f"[tune] Loaded data: {data}\n")

    results = []
    for cfg in CONFIGS:
        t0 = time.time()
        train_acc, test_acc = train_and_eval(data, **cfg)
        elapsed = time.time() - t0
        gap = train_acc - test_acc
        print(f"[tune] hidden={cfg['hidden_channels']:<4} layers={cfg['num_layers']} "
              f"dropout={cfg['dropout']:<4} epochs={cfg['epochs']:<4} "
              f"-> Train {train_acc:.4f}  Test {test_acc:.4f}  (gap {gap:+.4f})  "
              f"({elapsed:.1f}s)")
        results.append({**cfg, "Train Acc": round(train_acc, 4),
                         "Test Acc": round(test_acc, 4),
                         "Train-Test Gap": round(gap, 4)})

    df = pd.DataFrame(results).sort_values("Test Acc", ascending=False)
    df.to_csv("outputs/tuning_results.csv", index=False)
    print("\n[tune] Full results (sorted by Test Acc):")
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\n[tune] Best config: hidden={best['hidden_channels']}, "
          f"layers={best['num_layers']}, dropout={best['dropout']}, "
          f"epochs={best['epochs']} -> Test Acc {best['Test Acc']}")
