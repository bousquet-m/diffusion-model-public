# insite-diffusion

E(3)-equivariant diffusion model for atomic-structure **inpainting** of amorphous
In₂O₃, integrated with a MACE interatomic potential.

**Milestone one (this repo's current scope):** build and validate the core machinery
on **bulk** In₂O₃, where we have ground truth. Mask an interior region of a held-out
structure, regenerate it, and check it reproduces reference structural statistics
(RDFs, coordination, bond lengths). Surface/vacuum handling, force-guided sampling,
and the DFT/active-learning loop are later milestones.

## Approach (summary)

- **Diffusion:** variance-preserving (VP / DDPM), ε-prediction. Gaussian noise on
  Cartesian positions in a **center-of-mass-free** frame, **PBC-aware** (minimum
  image). The system always has periodic boundary conditions.
- **Inpainting:** RePaint-style — context atoms are clamped to known coordinates and,
  at each reverse step, overwritten with a re-noised copy of their known positions.
  The fixed block's CoM is realigned after each substitution to preserve the
  equivariant CoM-free frame.
- **Denoiser:** E(3)-equivariant message-passing network (e3nn), conditioned on the
  noise level, emitting a per-atom equivariant vector for mobile atoms.

## Data

External, read-only, not tracked in git (see `.gitignore`):
`/Users/matt/Desktop/data/in2o3/` — `640.extxyz`, `80.extxyz`, `scan_v3_swa.model`.

Train on **80 + 640 combined**; run interior-mask **validation on held-out 640 only**
(the 80-atom box is smaller than the denoiser receptive field, so it has no clean
context buffer). Train/val split is **by trajectory** (`config_type`), never by frame.

## Environment

Uses the `insite-diff` conda env (torch, ase, e3nn, mace-torch, dscribe). Develop on
CPU; uses Apple MPS if available. `PYTORCH_ENABLE_MPS_FALLBACK=1` recommended.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Step-0 data report (frames, densities, split, stride justification)
python scripts/inspect_data.py --config configs/base.yaml

# Fast end-to-end smoke (~seconds on CPU): trains, samples, validates plumbing
python scripts/train.py    --config configs/smoke.yaml
python scripts/sample.py   --config configs/smoke.yaml --checkpoint checkpoints/smoke/final.pt --n 2
python scripts/validate.py --config configs/smoke.yaml --checkpoint checkpoints/smoke/final.pt --n 4

# Real training / validation
python scripts/train.py    --config configs/base.yaml
python scripts/validate.py --config configs/base.yaml --checkpoint checkpoints/final.pt --n 8
python scripts/validate.py --config configs/base.yaml --checkpoint checkpoints/final.pt --n 4 --mace  # + energies
```

Validation writes `rdf_comparison.png`, `coordination_comparison.png`, an optional
`energy_comparison.png`, and `summary.json` (with PASS/FAIL vs tolerances) to the output dir.

## Environment notes

- **e3nn 0.4.4 + torch 2.11:** e3nn needs `torch.serialization.add_safe_globals([slice])`
  before import; handled automatically by `insite_diff/e3nn_compat.py`.
- **tensorboard** is optional — if not installed, training logs to `runs/*/metrics.jsonl`.
- **MACE** (`scan_v3_swa.model`) was serialized on CUDA; `analysis/mace_relax.py` patches
  `torch.jit.load` to load it on CPU. A single-point energy is ~1.4 s for 80 atoms but
  ~40 s for 640 atoms on CPU, so MACE energy/relaxation is **opt-in** (`--mace`) and best
  kept to small `--n`; full LBFGS relaxation of 640-atom cells is impractical on CPU.

## Layout

```
insite_diff/{data,diffusion,model,sampling,analysis,training}/   # package
configs/                                                         # base.yaml, smoke.yaml
scripts/                                                         # train/sample/validate/inspect
tests/                                                           # CoM+PBC, inpaint clamp, equivariance, split
```

## Tests

```bash
pytest -q
```
