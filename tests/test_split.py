"""Trajectory-level split must not leak: train/val share zero trajectory ids,
and validation is drawn only from eligible source roles."""
import numpy as np

from insite_diff.config import SplitConfig
from insite_diff.data.extxyz import FrameRecord
from insite_diff.data.split import trajectory_split


def _make_records():
    records = []
    # 10 train_val trajectories, 3 near-duplicate frames each
    for t in range(10):
        for f in range(3):
            records.append(FrameRecord(
                positions=np.zeros((4, 3)), numbers=np.array([49, 49, 8, 8]),
                cell=np.eye(3) * 10.0, traj_id=f"tv-{t}", source_path="640.extxyz",
                role="train_val", frame_index=f))
    # 5 train_only trajectories
    for t in range(5):
        for f in range(3):
            records.append(FrameRecord(
                positions=np.zeros((4, 3)), numbers=np.array([49, 49, 8, 8]),
                cell=np.eye(3) * 10.0, traj_id=f"to-{t}", source_path="80.extxyz",
                role="train_only", frame_index=f))
    return records


def test_zero_shared_trajectory():
    cfg = SplitConfig(val_fraction=0.2, val_source_roles=["train_val"], seed=0)
    split = trajectory_split(_make_records(), cfg)
    assert not (split.train_traj_ids & split.val_traj_ids)
    # frame-level check too: no traj_id appears in both frame lists
    train_ids = {r.traj_id for r in split.train}
    val_ids = {r.traj_id for r in split.val}
    assert train_ids.isdisjoint(val_ids)


def test_val_only_from_eligible_roles():
    cfg = SplitConfig(val_fraction=0.2, val_source_roles=["train_val"], seed=0)
    split = trajectory_split(_make_records(), cfg)
    assert all(t.startswith("tv-") for t in split.val_traj_ids)          # no train_only in val
    assert all(r.role == "train_val" for r in split.val)
    # all train_only trajectories must be present in train
    assert {f"to-{t}" for t in range(5)} <= split.train_traj_ids


def test_val_fraction_counts():
    cfg = SplitConfig(val_fraction=0.2, val_source_roles=["train_val"], seed=0)
    split = trajectory_split(_make_records(), cfg)
    assert len(split.val_traj_ids) == round(10 * 0.2)  # 2 of 10 eligible trajectories


def test_split_is_deterministic():
    cfg = SplitConfig(val_fraction=0.2, val_source_roles=["train_val"], seed=0)
    a = trajectory_split(_make_records(), cfg).val_traj_ids
    b = trajectory_split(_make_records(), cfg).val_traj_ids
    assert a == b
