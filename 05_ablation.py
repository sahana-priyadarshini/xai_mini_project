"""
05_ablation.py
================
Ablation study: what is actually driving the R-GCN's near-perfect accuracy?

Round 1 tested whether "class-exclusive" predicates (party/sport/team/
genre/associatedBand/field — each present almost only on one occupation
class in DBpedia's schema) were responsible. Result: removing them did NOT
reduce accuracy (it went up slightly), so that hypothesis was wrong.

Round 2 (this version) tests the two next most likely suspects:
    (C) ALL rdf:type triples removed — not just the exact 4 class URIs
        filtered in the leakage fix, but also DBpedia's other synonymous
        types (e.g. YAGO/WordNet subtypes like yago:Politician110451863),
        which are near-duplicates of the label even though they're
        different URIs from the ones we already excluded.
    (D) The `occupation` predicate removed — dbo:occupation often points
        to a specific occupation-title resource that is itself basically
        synonymous with the class (e.g. "Basketball player" for Athlete).
    (E) Both (C) and (D) removed together.

Conditions A and B are kept for continuity with the Round 1 run.

We rebuild each graph from the SAME cached triples (no new DBpedia query
needed). Every condition uses the same random seed for the train/test
split and model initialization, so the ONLY difference between conditions
is which predicates are present.

Requires: data/persons.pkl and data/triples.pkl already present.

Run:
    python 05_ablation.py
"""

import importlib.util
import pickle
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv

from utils import ensure_dir, set_seed

ensure_dir("outputs")

spec = importlib.util.spec_from_file_location("data_download", "01_data_download.py")
dd = importlib.util.module_from_spec(spec)
sys.modules["data_download"] = dd
spec.loader.exec_module(dd)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OCCUPATION_PRED = "http://dbpedia.org/ontology/occupation"

CLASS_EXCLUSIVE_PREDICATES = {
    "http://dbpedia.org/ontology/party",
    "http://dbpedia.org/ontology/sport",
    "http://dbpedia.org/ontology/team",
    "http://dbpedia.org/ontology/genre",
    "http://dbpedia.org/ontology/associatedBand",
    "http://dbpedia.org/ontology/field",
}

# Each condition: (label, set of predicates to remove entirely)
CONDITIONS = [
    ("A: baseline (all predicates)", set()),
    ("B: minus class-exclusive predicates (party/sport/team/genre/associatedBand/field)",
     CLASS_EXCLUSIVE_PREDICATES),
    ("C: minus ALL rdf:type triples", {RDF_TYPE}),
    ("D: minus occupation predicate", {OCCUPATION_PRED}),
    ("E: minus rdf:type AND occupation", {RDF_TYPE, OCCUPATION_PRED}),
]

HIDDEN_CHANNELS = 32
NUM_BASES = 30
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 100
SEED = 42


class FastRGCN(torch.nn.Module):
    def __init__(self, in_channels, num_relations, num_classes):
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels, HIDDEN_CHANNELS, num_relations, num_bases=NUM_BASES)
        self.conv2 = FastRGCNConv(HIDDEN_CHANNELS, num_classes, num_relations, num_bases=NUM_BASES)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


def train_and_eval(data, epochs: int = EPOCHS) -> tuple[float, float]:
    """Train a fresh FastRGCN on `data` and return (train_acc, test_acc)."""
    set_seed(SEED)  # identical initial weights across conditions
    in_channels = data.x.shape[1]
    num_relations = int(data.edge_type.max()) + 1
    num_classes = int(data.train_y.max()) + 1
    model = FastRGCN(in_channels, num_relations, num_classes)
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


if __name__ == "__main__":
    with open("data/persons.pkl", "rb") as f:
        persons = pickle.load(f)
    with open("data/triples.pkl", "rb") as f:
        triples = pickle.load(f)

    print(f"[ablation] Loaded {len(persons)} persons, {len(triples)} cached triples "
          f"(already leakage-fixed).")

    results = []
    for label, remove_preds in CONDITIONS:
        cond_triples = (
            triples if not remove_preds
            else [t for t in triples if t[1] not in remove_preds]
        )
        n_removed = len(triples) - len(cond_triples)
        print(f"\n[ablation] ({label}) — removing {n_removed} triples …")

        set_seed(SEED)  # identical train/test split across conditions
        data_cond, *_ = dd.build_pyg_data(persons, cond_triples)

        print(f"[ablation] ({label}) Training ({EPOCHS} epochs) …")
        t0 = time.time()
        train_acc, test_acc = train_and_eval(data_cond)
        elapsed = time.time() - t0
        print(f"[ablation] ({label}) Train {train_acc:.4f}  "
              f"Test {test_acc:.4f}  ({elapsed:.1f}s)")

        results.append({
            "Condition": label,
            "Triples removed": n_removed,
            "Train Acc": round(train_acc, 4),
            "Test Acc": round(test_acc, 4),
        })

    df = pd.DataFrame(results)
    df.to_csv("outputs/ablation_results.csv", index=False)
    print("\n[ablation] Saved comparison to outputs/ablation_results.csv")
    print(df.to_string(index=False))

    baseline_test = results[0]["Test Acc"]
    print("\n[ablation] Test-accuracy drop vs. baseline:")
    for r in results[1:]:
        drop = baseline_test - r["Test Acc"]
        print(f"  {r['Condition']}: {drop:+.4f} ({drop*100:+.2f} pp)")
