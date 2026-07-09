"""Import-time compatibility shim for e3nn 0.4.4 under torch >= 2.6.

e3nn 0.4.4 loads its Wigner constants with ``torch.load``; torch 2.6 flipped the
default to ``weights_only=True``, which rejects the ``slice`` object pickled in
that file. Allowlisting ``slice`` before e3nn is imported restores the load
without editing site-packages. Import this module before importing ``e3nn``.
"""
from __future__ import annotations

import torch

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
