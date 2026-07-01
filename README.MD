# XAI Mini Project — DBpedia GNN Explainability
**Course:** Explainable Artificial Intelligence — Dr. Stefan Heindorf  
**Strategy:** Strategy 1 — GNN Explainers (GNNExplainer, PGExplainer, SubGraphX)  
**Dataset:** DBpedia Persons subset (occupation-based node classification)

---

## Task
Given a person node in the DBpedia knowledge graph, predict their **occupation category**
(e.g., Athlete, Politician, Artist, Scientist). Then explain *why* the GNN made that prediction.

---

## Setup (Windows + Anaconda)

```bash
# 1. Create and activate conda environment
conda create -n xai26-mini python=3.12
conda activate xai26-mini

# 2. Install PyTorch (CPU — works on any Windows machine)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyTorch Geometric
pip install torch-geometric

# 4. Install remaining dependencies
pip install rdflib pandas numpy matplotlib scikit-learn requests SPARQLWrapper captum tqdm ipykernel jupyter

# 5. Register kernel
python -m ipykernel install --user --name xai26-mini --display-name "Python (xai26-mini)"
```

---

## Run Order

```bash
conda activate xai26-mini

# Step 1 — Download & analyze DBpedia data
python 01_data_download.py

# Step 2 — Train R-GCN model
python 02_train_model.py

# Step 3 — Generate explanations (all 3 explainers)
python 03_explain.py

# Step 4 — Evaluate explanations quantitatively
python 04_evaluate.py
```

All outputs (plots, tables, explanation subgraphs) are saved in `outputs/`.

---

## Project Structure
```
xai_mini_project/
├── 01_data_download.py      # DBpedia SPARQL download + RDF graph building
├── 02_train_model.py        # R-GCN training + evaluation
├── 03_explain.py            # GNNExplainer, PGExplainer, SubGraphX
├── 04_evaluate.py           # Fidelity, sparsity, stability metrics
├── utils.py                 # Shared helpers
├── data/                    # Raw + processed graph files
├── outputs/                 # Figures, tables, model checkpoints
├── requirements.txt
└── README.md
```

---

## Dataset — DBpedia Persons
- **Source:** DBpedia SPARQL endpoint (live.dbpedia.org)
- **Nodes:** ~2,000–4,000 person entities
- **Edges:** Relations such as `occupation`, `birthPlace`, `field`, `nationality`, `knownFor`
- **Task:** 4-class node classification (Athlete / Politician / Artist / Scientist)
- **Features:** One-hot degree encoding (same approach as AIFB example)

---

## Model — R-GCN
Relational Graph Convolutional Network (Schlichtkrull et al., 2018)  
- 2 layers: input → 32 hidden → 4 classes  
- Basis decomposition (num_bases=30) to handle many relation types  
- Optimizer: Adam, lr=0.01, weight_decay=5e-4  
- Epochs: 100

---

## Explainers
| Explainer | Type | Scope | Key idea |
|-----------|------|-------|----------|
| GNNExplainer | Gradient/mask | Local | Learns soft edge masks via mutual information |
| PGExplainer | Parametric | Local | Trains a separate MLP to predict edge importance |
| SubGraphX | Shapley | Local | Uses Monte Carlo tree search + Shapley values |

---

## Evaluation Metrics
- **Fidelity+**: Accuracy drop when explanation edges are removed
- **Fidelity−**: Accuracy drop when only explanation edges are kept  
- **Sparsity**: Fraction of edges selected (lower = more concise)
- **Stability**: Explanation consistency across random seeds