"""utils.py — Shared helpers for XAI Mini Project."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model state dict."""
    torch.save(model.state_dict(), path)
    print(f"[checkpoint] Saved model to {path}")


def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load model state dict in-place."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    print(f"[checkpoint] Loaded model from {path}")
    return model


def node_label_to_str(label_idx: int, inv_labels_dict: dict) -> str:
    """Convert integer label back to human-readable string."""
    raw = inv_labels_dict.get(label_idx, "Unknown")
    # Extract last path segment for readability
    return raw.split("/")[-1].split("#")[-1]


def relation_to_str(relation_uri) -> str:
    """Convert a relation URI to a short human-readable string."""
    s = str(relation_uri)
    return s.split("/")[-1].split("#")[-1]