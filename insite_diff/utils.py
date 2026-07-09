"""Deterministic seeding and device selection."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # not expected on Apple Silicon, harmless otherwise
        torch.cuda.manual_seed_all(seed)


def select_device(pref: str = "auto") -> torch.device:
    """Resolve a device string.

    "auto" -> MPS if available (Apple Silicon), else CPU. Explicit "cpu"/"mps"/
    "cuda" are honored as given. We never assume CUDA on this machine.
    """
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)
