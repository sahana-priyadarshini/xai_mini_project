"""
07_expand_predicates.py
========================
Legitimate accuracy-improvement attempt: fetch ADDITIONAL real DBpedia
predicates for the same 1600 cached persons, then merge them into the
graph and rebuild data/pyg_data.pt — without needing to re-download
everything from scratch.

New predicates fetched here:
    almaMater    - university/school attended
    award        - awards/honors received
    spouse       - marital relationships
    child        - children
    parent       - parents
    deathPlace   - place of death
    religion     - religious affiliation
    restingPlace - place of burial
    residence    - place of residence
    notableWork  - notable works/achievements

None of these are defined per-occupation-class in DBpedia's schema (unlike
rdf:type/occupation, which the ablation study proved WAS a near-duplicate
of the label, or party/sport/genre/field, which were tested and ruled out
as the driver). They may correlate with occupation somewhat, but they are
not circular by definition — a legitimate way to add relational signal.

Run:
    python 07_expand_predicates.py
Then:
    python 02_train_model.py
"""

import os
import pickle
import time

import requests
import torch

from utils import ensure_dir, set_seed

ensure_dir("data")

SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
HEADERS = {"Accept": "application/sparql-results+json"}
TIMEOUT = 30
SEED = 42

ADDITIONAL_PREDICATES = [
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

CACHE_EXTRA = "data/triples_extra.pkl"


def sparql_query(query: str) -> list[dict]:
    params = {"query": query, "format": "json"}
    for attempt in range(3):
        try:
            r = requests.get(SPARQL_ENDPOINT, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as exc:
            print(f"  [SPARQL] Attempt {attempt + 1} failed: {exc}")
            time.sleep(2 * (attempt + 1))
    return []


def fetch_extra_triples(persons: dict) -> list[tuple[str, str, str]]:
    if os.path.exists(CACHE_EXTRA):
        print("[extra] Loading cached extra triples …")
        with open(CACHE_EXTRA, "rb") as f:
            return pickle.load(f)

    uris = list(persons.keys())
    triples: list[tuple[str, str, str]] = []
    batch_size = 50

    for i in range(0, len(uris), batch_size):
        batch = uris[i: i + batch_size]
        values = " ".join(f"<{u}>" for u in batch)
        preds = " ".join(f"<{p}>" for p in ADDITIONAL_PREDICATES)
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
            triples.append((row["s"]["value"], row["p"]["value"], row["o"]["value"]))

        if (i // batch_size) % 5 == 0:
            pct = min(100, int(100 * i / len(uris)))
            print(f"  [extra] {pct}% — {len(triples)} extra triples so far …")

    with open(CACHE_EXTRA, "wb") as f:
        pickle.dump(triples, f)
    print(f"[extra] Total extra triples fetched: {len(triples)}")
    return triples


if __name__ == "__main__":
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("dd", "01_data_download.py")
    dd = importlib.util.module_from_spec(spec)
    sys.modules["dd"] = dd
    spec.loader.exec_module(dd)

    with open("data/persons.pkl", "rb") as f:
        persons = pickle.load(f)
    with open("data/triples.pkl", "rb") as f:
        base_triples = pickle.load(f)

    print(f"[extra] Base: {len(persons)} persons, {len(base_triples)} existing (honest) triples.")
    print("[extra] Fetching additional non-tautological predicates from DBpedia …")
    extra_triples = fetch_extra_triples(persons)

    # De-dupe in case of any overlap, then merge.
    merged = list(set(base_triples) | set(extra_triples))
    print(f"\n[extra] Merged triples: {len(base_triples)} base + "
          f"{len(extra_triples)} extra -> {len(merged)} unique total")

    print("[extra] Rebuilding graph with merged triples …")
    set_seed(SEED)
    data, nodes_dict, inv_nodes_dict, inv_relations_dict, relations = dd.build_pyg_data(
        persons, merged
    )
    torch.save(data, "data/pyg_data.pt")
    with open("data/nodes_dict.pkl", "wb") as f:
        pickle.dump(nodes_dict, f)
    with open("data/inv_nodes_dict.pkl", "wb") as f:
        pickle.dump(inv_nodes_dict, f)
    with open("data/inv_relations_dict.pkl", "wb") as f:
        pickle.dump(inv_relations_dict, f)
    with open("data/relations.pkl", "wb") as f:
        pickle.dump(relations, f)
    with open("data/triples.pkl", "wb") as f:
        pickle.dump(merged, f)

    print(f"\n[extra] Rebuilt data/pyg_data.pt: {data}")
    print("[extra] Done. Now run: python 02_train_model.py")
