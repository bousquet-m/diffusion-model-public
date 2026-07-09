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
python scripts/inspect_data.py --config configs/base.yaml   # Step-0 data report
python scripts/train.py       --config configs/smoke.yaml   # fast end-to-end smoke
python scripts/train.py       --config configs/base.yaml    # real training
python scripts/sample.py      --config configs/base.yaml    # inpaint a held-out structure
python scripts/validate.py    --config configs/base.yaml    # RDF/CN/energy comparison
```

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
