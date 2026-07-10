"""Config-driven training loop: eps-MSE, EMA, checkpointing, logging.

Batches are lists of variable-size structures. Logging uses tensorboard when
available and otherwise degrades to a JSONL metrics file.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import Config, config_to_dict
from ..data.dataset import build_datasets
from ..data.graph import build_graph
from ..diffusion.schedule import VPSchedule
from ..model.denoiser import E3Denoiser
from ..utils import seed_everything, select_device
from .losses import batch_eps_loss


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class Logger:
    """Tensorboard if installed, else append-only JSONL at log_dir/metrics.jsonl."""

    def __init__(self, log_dir: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir)
        except Exception:
            self.jsonl = open(os.path.join(log_dir, "metrics.jsonl"), "a")
            print(f"[logger] tensorboard unavailable; logging JSONL to {log_dir}/metrics.jsonl")

    def log(self, step: int, **scalars):
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, step)
        else:
            self.jsonl.write(json.dumps({"step": step, **scalars}) + "\n")
            self.jsonl.flush()

    def close(self):
        if self.tb is not None:
            self.tb.close()
        else:
            self.jsonl.close()


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict()) if decay > 0 else None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        if self.shadow is None:
            return
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone()


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def estimate_avg_neighbors(dataset, cutoff: float, max_neighbors: int, n: int = 4) -> float:
    degs = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        g = build_graph(item["positions"].numpy(), item["cell"].numpy(), cutoff,
                        max_neighbors=max_neighbors)
        degs.append(g.edge_index.shape[1] / item["n_atoms"])
    return float(sum(degs) / len(degs)) if degs else 20.0


def build_model(cfg: Config, avg_neighbors: float) -> E3Denoiser:
    m = cfg.model
    return E3Denoiser(
        n_species=len(cfg.data.species), hidden_irreps=m.hidden_irreps, sh_lmax=m.sh_lmax,
        n_layers=m.n_layers, radial_basis=m.radial_basis, sigma_embed_dim=m.sigma_embed_dim,
        cutoff=cfg.graph.cutoff, avg_neighbors=avg_neighbors,
    )


def load_model(path: str, device="cpu", use_ema: bool = True) -> tuple[E3Denoiser, Config, float]:
    """Rebuild the denoiser from a checkpoint. Prefers EMA weights when present."""
    from ..config import _from_dict  # local import to avoid cycles
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # strict=False so a checkpoint saved under an older config schema (e.g. the
    # pre-1b mask/sampling keys) still loads — the denoiser architecture is what
    # matters here and it is unchanged.
    cfg = _from_dict(Config, ckpt["config"], strict=False)
    avg_neighbors = ckpt["avg_neighbors"]
    model = build_model(cfg, avg_neighbors).to(device)
    state = ckpt["ema"] if (use_ema and ckpt.get("ema") is not None) else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    return model, cfg, avg_neighbors


def save_checkpoint(path: str, step: int, model, ema: EMA, optim, cfg: Config,
                    avg_neighbors: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema.shadow,
        "optim": optim.state_dict(),
        "config": config_to_dict(cfg),
        "species": cfg.data.species,
        "avg_neighbors": avg_neighbors,
    }, path)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: Config) -> str:
    # Stream logs live to a redirected file (SLURM block-buffers stdout otherwise).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    print(f"[train] device={device}", flush=True)

    train_ds, val_ds, split = build_datasets(cfg)
    print(f"[train] train={len(train_ds)} frames/{len(split.train_traj_ids)} trajs, "
          f"val={len(val_ds)} frames/{len(split.val_traj_ids)} trajs")

    schedule = VPSchedule(cfg.diffusion.timesteps, cfg.diffusion.beta_schedule).to(device)
    avg_neighbors = estimate_avg_neighbors(train_ds, cfg.graph.cutoff, cfg.graph.max_neighbors)
    model = build_model(cfg, avg_neighbors).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params={n_params:,} avg_neighbors={avg_neighbors:.1f}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    ema = EMA(model, cfg.train.ema_decay)
    logger = Logger(cfg.train.log_dir)

    loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                        collate_fn=list, num_workers=cfg.train.num_workers)
    data_iter = itertools.cycle(loader)

    model.train()
    for step in range(1, cfg.train.max_steps + 1):
        batch = next(data_iter)
        loss = batch_eps_loss(model, schedule, batch, cfg.graph.cutoff,
                              cfg.graph.max_neighbors, device)
        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update(model)

        if step % cfg.train.log_every == 0 or step == 1:
            logger.log(step, train_loss=loss.item())
            print(f"[train] step {step}/{cfg.train.max_steps} loss {loss.item():.4f}", flush=True)

        if len(val_ds) and step % cfg.train.val_every == 0:
            model.eval()
            with torch.no_grad():
                vbatch = [val_ds[i] for i in range(min(len(val_ds), cfg.train.batch_size))]
                vloss = batch_eps_loss(model, schedule, vbatch, cfg.graph.cutoff,
                                       cfg.graph.max_neighbors, device)
            logger.log(step, val_loss=vloss.item())
            print(f"[train] step {step} val_loss {vloss.item():.4f}", flush=True)
            model.train()

        if step % cfg.train.ckpt_every == 0:
            save_checkpoint(os.path.join(cfg.train.checkpoint_dir, f"step_{step}.pt"),
                            step, model, ema, optim, cfg, avg_neighbors)

    final = os.path.join(cfg.train.checkpoint_dir, "final.pt")
    save_checkpoint(final, cfg.train.max_steps, model, ema, optim, cfg, avg_neighbors)
    logger.close()
    print(f"[train] done. final checkpoint: {final}")
    return final
