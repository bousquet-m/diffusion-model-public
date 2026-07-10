"""Config loading: strict for yaml, lenient for old-schema checkpoints."""
import pytest

from insite_diff.config import Config, _from_dict

# A pre-1b checkpoint's embedded config (old mask + sampling schema).
OLD_CKPT_CONFIG = {
    "seed": 0, "device": "cuda",
    "data": {"sources": [{"path": "/x/640.extxyz", "role": "train_val"}],
             "species": ["In", "O"], "stride": 1, "dedup_rmsd": 0.05},
    "mask": {"mode": "interior", "interior": {"region_frac": 0.25, "center": "box_center"},
             "random_subset": {"mobile_frac": 0.15}},
    "sampling": {"num_steps": 1000, "repaint_jumps": 1, "n_samples": 8},
    "model": {"n_layers": 2, "hidden_irreps": "32x0e + 16x1o", "sh_lmax": 2,
              "radial_basis": 8, "sigma_embed_dim": 32},
}


def test_strict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        _from_dict(Config, OLD_CKPT_CONFIG, strict=True)


def test_lenient_loads_old_checkpoint_config():
    cfg = _from_dict(Config, OLD_CKPT_CONFIG, strict=False)
    # model config (what rebuilds the denoiser) is preserved intact
    assert cfg.model.hidden_irreps == "32x0e + 16x1o"
    assert cfg.model.n_layers == 2
    assert cfg.data.species == ["In", "O"]
    # fields whose schema changed fall back to current defaults
    assert cfg.mask.geometry == "sphere"
    assert cfg.sampling.n_resample == 8
