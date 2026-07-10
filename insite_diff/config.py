"""Typed configuration schema and YAML loader.

ALL external paths (data files, MACE model, output dirs) flow through this
module. Nothing else in the codebase hard-codes a path.

Config is a tree of frozen-ish dataclasses so downstream code gets attribute
access and mistyped keys fail loudly at load time instead of silently at use.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class SourceConfig:
    """One extxyz file plus its role in train/val.

    role:
      - "train_val":  eligible for both training and the held-out val split.
      - "train_only": used for training but never placed in validation
                      (e.g. the small 80-atom cells, too small for a clean
                      interior mask — see milestone-one scope).
    """

    path: str
    role: str = "train_val"

    def __post_init__(self) -> None:
        if self.role not in ("train_val", "train_only"):
            raise ValueError(f"source role must be train_val|train_only, got {self.role!r}")


@dataclass
class DataConfig:
    sources: list[SourceConfig] = field(default_factory=list)
    # Trajectory id lives in the extxyz comment as `config_type=...`; ASE
    # mis-parses that comment, so we recover the id with this regex instead.
    traj_id_regex: str = r"config_type=(\S+)"
    species: list[str] = field(default_factory=lambda: ["In", "O"])
    stride: int = 1          # thinning within a trajectory (frames)
    dedup_rmsd: float = 0.0  # drop frames closer than this RMSD (0 disables)

    def __post_init__(self) -> None:
        self.sources = [SourceConfig(**s) if isinstance(s, dict) else s for s in self.sources]


@dataclass
class SplitConfig:
    by: str = "trajectory"                 # only trajectory-level split is allowed
    val_fraction: float = 0.2
    val_source_roles: list[str] = field(default_factory=lambda: ["train_val"])
    seed: int = 0

    def __post_init__(self) -> None:
        if self.by != "trajectory":
            raise ValueError("split.by must be 'trajectory' (frame splits leak across thermal rattles)")


@dataclass
class GraphConfig:
    cutoff: float = 4.0
    max_neighbors: int = 32  # 0 => no cap
    # NOTE: cutoff * model.n_layers must stay < box/2 on the validation set so
    # the masked interior keeps a receptive-field-thick context buffer.


@dataclass
class MaskConfig:
    """Which atoms to regenerate, parameterized by ATOM fraction (not cube edge).

    mask_frac is the fraction of atoms made mobile; the region is grown to that
    exact count in the chosen geometry. mask_frac=1.0 is pure unconditional
    generation (no context).
    """

    mask_frac: float = 0.10         # fraction of ATOMS to regenerate
    geometry: str = "sphere"        # sphere | cube | slab
    center: str = "box_center"      # box_center | random  (sphere/cube)
    slab_axis: int = 2              # 0/1/2 -> x/y/z (slab)
    slab_side: str = "top"          # top | bottom (slab)

    def __post_init__(self) -> None:
        if self.geometry not in ("sphere", "cube", "slab"):
            raise ValueError(f"mask.geometry must be sphere|cube|slab, got {self.geometry!r}")
        if self.center not in ("box_center", "random"):
            raise ValueError(f"mask.center must be box_center|random, got {self.center!r}")
        if self.slab_side not in ("top", "bottom"):
            raise ValueError(f"mask.slab_side must be top|bottom, got {self.slab_side!r}")


@dataclass
class DiffusionConfig:
    type: str = "vp"                # variance-preserving (DDPM); documented choice
    timesteps: int = 1000
    beta_schedule: str = "cosine"   # cosine | linear
    com_free: bool = True
    pbc_aware: bool = True

    def __post_init__(self) -> None:
        if self.type != "vp":
            raise ValueError("milestone one uses VP diffusion only")


@dataclass
class ModelConfig:
    name: str = "e3nn_denoiser"
    n_layers: int = 2
    hidden_irreps: str = "32x0e + 16x1o"
    sh_lmax: int = 2
    radial_basis: int = 8
    sigma_embed_dim: int = 32


@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 1e-3
    max_steps: int = 200_000
    ema_decay: float = 0.999
    loss: str = "eps_mse"
    checkpoint_dir: str = "checkpoints"
    ckpt_every: int = 5000
    log_dir: str = "runs"
    log_every: int = 50
    val_every: int = 1000
    num_workers: int = 0


@dataclass
class SamplingConfig:
    num_steps: int = 1000
    n_resample: int = 8             # RePaint resample count (U) per step: harmonizes the seam
    n_samples: int = 8


@dataclass
class MaceConfig:
    model_path: str = ""
    relax_fmax: float = 0.05
    relax_steps: int = 200


@dataclass
class ValidationTolerances:
    rdf_peak_A: float = 0.1
    mean_bond_A: float = 0.05


@dataclass
class ValidationConfig:
    rdf_rmax: float = 6.0
    rdf_bins: int = 240
    cn_cutoff_ino: float = 2.7      # In-O coordination cutoff (~first RDF minimum)
    tolerances: ValidationTolerances = field(default_factory=ValidationTolerances)


@dataclass
class Config:
    seed: int = 0
    device: str = "auto"            # "auto" -> cuda, else mps, else cpu
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    mace: MaceConfig = field(default_factory=MaceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _from_dict(cls: type, data: Any, strict: bool = True) -> Any:
    """Recursively build a dataclass tree from nested dicts.

    ``strict=True`` (config files) rejects unknown keys — mistyped config fails
    loudly. ``strict=False`` (loading a config embedded in an old checkpoint)
    drops keys no longer in the schema, so a checkpoint saved under an earlier
    config schema still loads (fields that changed fall back to their defaults).
    """
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    field_map = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(field_map)
    if unknown:
        if strict:
            raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
        data = {k: v for k, v in data.items() if k in field_map}
    # Resolve string annotations (PEP 563 / `from __future__ import annotations`)
    # back to real types so nested dataclasses are detected.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name)
        # list[SourceConfig] and the like are handled by each dataclass __post_init__,
        # so only recurse into nested single-dataclass *fields* here.
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value, strict=strict)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a validated Config tree."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(Config, raw)


def config_to_dict(cfg: Config) -> dict:
    """Round-trip a Config back to a plain dict (for logging / checkpoint metadata)."""
    return dataclasses.asdict(cfg)
