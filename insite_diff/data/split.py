"""Trajectory-level train/val split.

The split is by ``traj_id`` (config_type), never by frame: frames within a
trajectory are near-duplicate thermal rattles, so a frame split would leak.
Only trajectories whose source role is in ``val_source_roles`` are eligible for
validation (e.g. the 640-atom cells); everything else — including all
``train_only`` sources like the 80-atom cells — goes to training.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import SplitConfig
from .extxyz import FrameRecord


@dataclass
class DataSplit:
    train: list[FrameRecord]
    val: list[FrameRecord]
    val_traj_ids: set[str]
    train_traj_ids: set[str]


def _roles_by_traj(records: list[FrameRecord]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for r in records:
        prev = roles.setdefault(r.traj_id, r.role)
        if prev != r.role:
            raise ValueError(
                f"trajectory {r.traj_id!r} appears with conflicting roles {prev!r} vs {r.role!r}"
            )
    return roles


def trajectory_split(records: list[FrameRecord], cfg: SplitConfig) -> DataSplit:
    """Partition records into train/val by trajectory id.

    Guarantees train and val share no trajectory id. Validation trajectories are
    sampled only from roles in ``cfg.val_source_roles``.
    """
    roles = _roles_by_traj(records)
    eligible = sorted(t for t, role in roles.items() if role in cfg.val_source_roles)
    rng = random.Random(cfg.seed)
    rng.shuffle(eligible)

    n_val = round(len(eligible) * cfg.val_fraction)
    val_traj_ids = set(eligible[:n_val])
    train_traj_ids = set(roles) - val_traj_ids

    train = [r for r in records if r.traj_id in train_traj_ids]
    val = [r for r in records if r.traj_id in val_traj_ids]

    # Invariants the tests also assert.
    assert not (train_traj_ids & val_traj_ids), "train/val share a trajectory id"
    assert all(roles[t] in cfg.val_source_roles for t in val_traj_ids), \
        "a validation trajectory came from a non-val role"
    return DataSplit(train=train, val=val, val_traj_ids=val_traj_ids, train_traj_ids=train_traj_ids)
