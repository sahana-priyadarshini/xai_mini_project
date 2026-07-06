"""
01_data_download.py
====================
Downloads a subset of DBpedia persons via SPARQL, builds an RDF graph,
converts it to a PyG Data object, and saves everything to disk.

Occupation categories (4 classes):
    0 — Athlete      (dbo:Athlete)
    1 — Politician   (dbo:Politician)
    2 — Artist       (dbo:Artist)
    3 — Scientist    (dbo:Scientist)

Run:
    python 01_data_download.py
"""

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import requests
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import index_sort

from utils import ensure_dir, set_seed

set_seed(42)
ensure_dir("data")
ensure_dir("outputs")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
N_PER_CLASS = 400          # persons fetched per occupation class
OCCUPATION_CLASSES = {
    "http://dbpedia.org/ontology/Athlete":    0,
    "http://dbpedia.org/ontology/Politician": 1,
    "http://dbpedia.org/ontology/Artist":     2,
    "http://dbpedia.org/ontology/Scientist":  3,
}
# NOTE — rdf:type is deliberately EXCLUDED here.
# We originally included it (filtering out only the exact 4 occupation
# class URIs used as labels) but a controlled 5-condition ablation
# (see 05_ablation.py) showed test accuracy drops from 99.79% to 62.92%
# when ALL rdf:type triples are removed, vs. an essentially unchanged
# accuracy when the 6 "class-exclusive" predicates below are removed.
# This means DBpedia's OTHER rdf:type triples (YAGO/WordNet/Wikidata
# subtypes such as yago:Politician110451863) were still near-duplicates
# of the label even after we excluded the 4 exact class URIs — rdf:type
# as a whole is simply too redundant with occupation to use as a feature
# here. Dropping it entirely gives a lower but far more honest, non-leaky
# accuracy and makes the resulting explanations meaningfully non-trivial.
RELATION_PREDICATES = [
    "http://dbpedia.org/ontology/birthPlace",
    "http://dbpedia.org/ontology/nationality",
    "http://dbpedia.org/ontology/occupation",
    "http://dbpedia.org/ontology/field",
    "http://dbpedia.org/ontology/knownFor",
    "http://dbpedia.org/ontology/genre",
    "http://dbpedia.org/ontology/sport",
    "http://dbpedia.org/ontology/party",
    "http://dbpedia.org/ontology/team",
    "http://dbpedia.org/ontology/influenced",
    "http://dbpedia.org/ontology/associatedBand",
    # Added after the ablation study to legitimately add more relational
    # signal without being tautologically tied to the occupation label
    # (unlike rdf:type/occupation, these aren't defined per-class in
    # DBpedia's schema; see 07_expand_predicates.py for the supplemental
    # fetch used when these were added to an already-cached dataset).
    "http://dbpedia.org/ontology/almaMater",
    "http://dbpedia.org/ontology/award",
    "http://dbpedia.org/ontology/spouse",
    "http://dbpedia.org/ontology/child",
    "http://dbpedia.org/ontology/parent",
    "http://dbpedia.org/ontology/deathPlace",
    "http://dbpedia.org/ontology/religion",
    "http://dbpedia.org/ontology/restingPlace",
    "http://dbpedia.org/ontology/residence",
    "http://dbpedia.org/ontology/notableWork",
]
HEADERS = {"Accept": "application/sparql-results+json"}
TIMEOUT = 30  # seconds per request


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------
def sparql_query(query: str) -> list[dict]:
    """Execute a SPARQL SELECT query and return list of binding dicts."""
    params = {"query": query, "format": "json"}
    for attempt in range(3):
        try:
            r = requests.get(
                SPARQL_ENDPOINT, params=params, headers=HEADERS, timeout=TIMEOUT
            )
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as exc:
            print(f"  [SPARQL] Attempt {attempt + 1} failed: {exc}")
            time.sleep(2 * (attempt + 1))
    return []


# ---------------------------------------------------------------------------
# Step 1 — Fetch person URIs per class
# ---------------------------------------------------------------------------
CACHE_PERSONS = "data/persons_raw.pkl"
CACHE_TRIPLES = "data/triples_raw.pkl"


def fetch_persons() -> dict[str, int]:
    """Return {person_uri: class_idx} dict."""
    if os.path.exists(CACHE_PERSONS):
        print("[data] Loading cached person URIs …")
        with open(CACHE_PERSONS, "rb") as f:
            return pickle.load(f)

    persons: dict[str, int] = {}
    for class_uri, class_idx in OCCUPATION_CLASSES.items():
        class_name = class_uri.split("/")[-1]
        print(f"[data] Fetching {N_PER_CLASS} {class_name}s …")
        query = f"""
        SELECT DISTINCT ?person WHERE {{
            ?person a <{class_uri}> .
            ?person <http://www.w3.org/2000/01/rdf-schema#label> ?label .
            FILTER (lang(?label) = 'en')
        }}
        LIMIT {N_PER_CLASS}
        """
        results = sparql_query(query)
        for row in results:
            uri = row["person"]["value"]
            if uri not in persons:
                persons[uri] = class_idx
        print(f"  → {len([v for v in persons.values() if v == class_idx])} persons")

    with open(CACHE_PERSONS, "wb") as f:
        pickle.dump(persons, f)
    print(f"[data] Total persons: {len(persons)}")
    return persons


# ---------------------------------------------------------------------------
# Step 2 — Fetch triples for all persons
# ---------------------------------------------------------------------------
def fetch_triples(persons: dict[str, int]) -> list[tuple[str, str, str]]:
    """Return list of (subject, predicate, object) string triples.

    IMPORTANT — avoiding label leakage:
    fetch_persons() assigns each person's class label using the exact triple
    (person, rdf:type, <occupation class URI>), e.g. (dbr:LeBron_James,
    rdf:type, dbo:Athlete). RELATION_PREDICATES also includes rdf:type
    because a person's *other* types (e.g. dbo:Person, more specific
    sub-types) are legitimately useful graph context. If we did not exclude
    it, this exact labelling triple would be re-fetched here and added to
    the graph as an edge — handing the model the answer directly instead of
    making it learn from surrounding context (occupation, field, teams,
    etc.). We therefore explicitly filter out (rdf:type, <label class URI>)
    triples while keeping all other rdf:type triples.
    """
    if os.path.exists(CACHE_TRIPLES):
        print("[data] Loading cached triples …")
        with open(CACHE_TRIPLES, "rb") as f:
            return pickle.load(f)

    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    label_class_uris = set(OCCUPATION_CLASSES.keys())

    uris = list(persons.keys())
    triples: list[tuple[str, str, str]] = []
    dropped_leaky = 0
    batch_size = 50

    for i in range(0, len(uris), batch_size):
        batch = uris[i: i + batch_size]
        values = " ".join(f"<{u}>" for u in batch)
        preds = " ".join(f"<{p}>" for p in RELATION_PREDICATES)
        query = f"""
        SELECT ?s ?p ?o WHERE {{
            VALUES ?s {{ {values} }}
            VALUES ?p {{ {preds} }}
            ?s ?p ?o .
            FILTER (!isLiteral(?o))
        }}
        """
        results = sparql_query(query)
        for row in results:
            s = row["s"]["value"]
            p = row["p"]["value"]
            o = row["o"]["value"]
            # Drop the exact triple that defines the label (see docstring).
            if p == RDF_TYPE and o in label_class_uris:
                dropped_leaky += 1
                continue
            triples.append((s, p, o))

        if (i // batch_size) % 5 == 0:
            pct = min(100, int(100 * i / len(uris)))
            print(f"  [triples] {pct}% — {len(triples)} triples so far …")

    if dropped_leaky:
        print(f"[data] Dropped {dropped_leaky} label-defining rdf:type triples "
              f"to prevent leakage.")

    with open(CACHE_TRIPLES, "wb") as f:
        pickle.dump(triples, f)
    print(f"[data] Total triples: {len(triples)}")
    return triples


# ---------------------------------------------------------------------------
# Fallback: build synthetic DBpedia-like graph if SPARQL fails
# ---------------------------------------------------------------------------
def build_fallback_graph(n_persons: int = 800) -> tuple[dict, list]:
    """Build a small synthetic graph matching DBpedia schema for offline use."""
    print("[data] SPARQL unreachable — building synthetic fallback graph …")
    np.random.seed(42)
    n_classes = 4
    n_per_class = n_persons // n_classes

    class_names = ["Athlete", "Politician", "Artist", "Scientist"]
    relations = [
        "birthPlace", "nationality", "occupation",
        "field", "knownFor", "genre", "sport", "party", "team",
    ]
    # Shared object nodes per class (locations, fields, etc.)
    n_obj_nodes = 200
    obj_uris = [f"http://dbpedia.org/resource/Object_{i}" for i in range(n_obj_nodes)]

    persons: dict[str, int] = {}
    for c, cname in enumerate(class_names):
        for j in range(n_per_class):
            uri = f"http://dbpedia.org/resource/{cname}_{j}"
            persons[uri] = c

    person_uris = list(persons.keys())
    triples: list[tuple[str, str, str]] = []

    # Each person connects to 3–8 object nodes via random relations
    for p_uri, c_idx in persons.items():
        # Class-biased objects (so classes are separable)
        base = c_idx * (n_obj_nodes // n_classes)
        n_edges = np.random.randint(3, 9)
        chosen_objs = np.random.choice(
            range(base, base + n_obj_nodes // n_classes), size=n_edges, replace=False
        )
        chosen_rels = np.random.choice(relations, size=n_edges)
        for obj_i, rel in zip(chosen_objs, chosen_rels):
            triples.append((
                p_uri,
                f"http://dbpedia.org/ontology/{rel}",
                obj_uris[obj_i],
            ))

    # Some person–person edges
    for _ in range(200):
        i, j = np.random.choice(len(person_uris), 2, replace=False)
        triples.append((person_uris[i], "http://dbpedia.org/ontology/influenced", person_uris[j]))

    print(f"[fallback] persons={len(persons)}, triples={len(triples)}")
    return persons, triples


# ---------------------------------------------------------------------------
# Step 3 — Build PyG Data object
# ---------------------------------------------------------------------------
def build_pyg_data(
    persons: dict[str, int], triples: list[tuple[str, str, str]]
) -> tuple[Data, dict, dict, dict, list]:
    """Convert raw persons + triples into a PyG heterogeneous-edge Data object."""

    # Collect all unique nodes
    all_nodes_set: set[str] = set(persons.keys())
    for s, p, o in triples:
        all_nodes_set.add(s)
        all_nodes_set.add(o)
    nodes = sorted(all_nodes_set)
    nodes_dict = {uri: i for i, uri in enumerate(nodes)}

    # Collect unique relations; add inverse
    rel_set: set[str] = set()
    for _, p, _ in triples:
        rel_set.add(p)
    relations = sorted(rel_set)
    relations_dict = {r: i for i, r in enumerate(relations)}
    N = len(nodes)
    R = 2 * len(relations)

    # Build edges (forward + inverse)
    edges = []
    for s, p, o in triples:
        src, dst = nodes_dict[s], nodes_dict[o]
        rel = relations_dict[p]
        edges.append([src, dst, 2 * rel])        # forward
        edges.append([dst, src, 2 * rel + 1])    # inverse

    if not edges:
        raise ValueError("No edges found — check SPARQL results or fallback.")

    edge = torch.tensor(edges, dtype=torch.long).t().contiguous()
    _, perm = index_sort(N * R * edge[0] + R * edge[1] + edge[2])
    edge = edge[:, perm]
    edge_index, edge_type = edge[:2], edge[2]

    # Node features: one-hot degree.
    # IMPORTANT: max_degree must NOT be assumed to equal N - 1. Because we
    # add both a forward and an inverse edge per triple, a single "hub"
    # object node (e.g. a common nationality/occupation-title resource
    # referenced by hundreds of persons) can end up with an out-degree well
    # above N - 1 via its inverse edges. Using N - 1 as the one-hot size
    # works by chance on some graphs and crashes with an out-of-bounds
    # index on others (e.g. after removing predicates in an ablation study,
    # which shrinks N but not necessarily any individual hub's degree). We
    # instead size the one-hot encoding to the ACTUAL maximum out-degree
    # present in this graph (still capped at 4999 to bound feature width).
    from torch_geometric.utils import degree as _degree
    actual_max_degree = int(_degree(edge_index[0], num_nodes=N).max().item())
    max_degree = min(4999, actual_max_degree)
    transform = T.OneHotDegree(max_degree=max_degree, cat=False)
    tmp = Data(edge_index=edge_index, num_nodes=N)
    tmp = transform(tmp)
    X = tmp.x

    # Remove zero-variance features
    mask = X.abs().sum(dim=0) > 0
    X = X[:, mask]

    # Build train/test split (only labelled person nodes)
    labelled = [(nodes_dict[uri], label) for uri, label in persons.items()]
    np.random.shuffle(labelled)
    split = int(0.7 * len(labelled))
    train_items = labelled[:split]
    test_items = labelled[split:]

    train_idx = torch.tensor([x[0] for x in train_items], dtype=torch.long)
    train_y = torch.tensor([x[1] for x in train_items], dtype=torch.long)
    test_idx = torch.tensor([x[0] for x in test_items], dtype=torch.long)
    test_y = torch.tensor([x[1] for x in test_items], dtype=torch.long)

    data = Data(
        x=X,
        edge_index=edge_index,
        edge_type=edge_type,
        train_idx=train_idx,
        train_y=train_y,
        test_idx=test_idx,
        test_y=test_y,
        num_nodes=N,
    )

    inv_nodes_dict = {v: k for k, v in nodes_dict.items()}
    inv_relations_dict = {v: k for k, v in relations_dict.items()}
    inv_labels_dict = {v: k for k, v in {
        "http://dbpedia.org/ontology/Athlete": 0,
        "http://dbpedia.org/ontology/Politician": 1,
        "http://dbpedia.org/ontology/Artist": 2,
        "http://dbpedia.org/ontology/Scientist": 3,
    }.items()}
    inv_labels_dict = {v: k for k, v in inv_labels_dict.items()}

    return data, nodes_dict, inv_nodes_dict, inv_relations_dict, relations


# ---------------------------------------------------------------------------
# Step 4 — Dataset statistics
# ---------------------------------------------------------------------------
def print_statistics(
    data: Data,
    persons: dict,
    relations: list,
    inv_labels_dict: dict,
) -> None:
    label_names = {0: "Athlete", 1: "Politician", 2: "Artist", 3: "Scientist"}
    print("\n" + "=" * 55)
    print("  DBpedia Dataset Statistics")
    print("=" * 55)
    print(f"  Total nodes       : {data.num_nodes}")
    print(f"  Labelled persons  : {len(persons)}")
    print(f"  Total edges       : {data.edge_index.shape[1]}")
    print(f"  Relation types    : {len(relations)} (+{len(relations)} inverse = {2*len(relations)})")
    print(f"  Node feature dim  : {data.x.shape[1]}")
    print(f"  Training nodes    : {len(data.train_idx)}")
    print(f"  Test nodes        : {len(data.test_idx)}")
    print()
    print("  Class distribution:")
    for c_idx, c_name in label_names.items():
        n_train = (data.train_y == c_idx).sum().item()
        n_test = (data.test_y == c_idx).sum().item()
        print(f"    {c_name:<12} train={n_train:>3}  test={n_test:>3}")
    print("=" * 55 + "\n")

    # Save stats to CSV
    rows = []
    for c_idx, c_name in label_names.items():
        rows.append({
            "Class": c_name,
            "Train": (data.train_y == c_idx).sum().item(),
            "Test": (data.test_y == c_idx).sum().item(),
        })
    df = pd.DataFrame(rows)
    df.to_csv("outputs/dataset_statistics.csv", index=False)
    print("[stats] Saved to outputs/dataset_statistics.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Try SPARQL; fall back to synthetic graph
    try:
        print("[data] Attempting DBpedia SPARQL download …")
        persons = fetch_persons()
        if len(persons) < 100:
            raise ValueError("Too few persons fetched — using fallback.")
        triples = fetch_triples(persons)
        if len(triples) < 200:
            raise ValueError("Too few triples — using fallback.")
    except Exception as exc:
        print(f"[data] SPARQL failed ({exc}) — switching to synthetic fallback.")
        persons, triples = build_fallback_graph(n_persons=800)

    # Save raw
    with open("data/persons.pkl", "wb") as f:
        pickle.dump(persons, f)
    with open("data/triples.pkl", "wb") as f:
        pickle.dump(triples, f)

    # Build PyG object
    print("[data] Building PyG Data object …")
    data, nodes_dict, inv_nodes_dict, inv_relations_dict, relations = build_pyg_data(
        persons, triples
    )
    print(f"[data] Data object: {data}")

    # Save all artefacts
    torch.save(data, "data/pyg_data.pt")
    with open("data/nodes_dict.pkl", "wb") as f:
        pickle.dump(nodes_dict, f)
    with open("data/inv_nodes_dict.pkl", "wb") as f:
        pickle.dump(inv_nodes_dict, f)
    with open("data/inv_relations_dict.pkl", "wb") as f:
        pickle.dump(inv_relations_dict, f)
    with open("data/relations.pkl", "wb") as f:
        pickle.dump(relations, f)

    inv_labels_dict = {0: "Athlete", 1: "Politician", 2: "Artist", 3: "Scientist"}
    with open("data/inv_labels_dict.pkl", "wb") as f:
        pickle.dump(inv_labels_dict, f)

    print_statistics(data, persons, relations, inv_labels_dict)
    print("[done] Step 1 complete. Run: python 02_train_model.py")