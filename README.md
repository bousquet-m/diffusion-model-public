# insite-diffusion

E(3)-equivariant diffusion model for atomic-structure **inpainting** of amorphous
In₂O₃, integrated with a MACE interatomic potential.

**Milestone one (this repo's current scope):** build and validate the core machinery
on **bulk** In₂O₃, where we have ground truth. Mask an interior region of a held-out
structure, regenerate it, and check it reproduces reference structural statistics
(RDFs, coordination, bond lengths). Surface/vacuum handling, force-guided sampling,
and the DFT/active-learning loop are later milestones.

## Approach (summary)

- **Training is unconditional.** The denoiser noises **all** atoms and predicts the
  noise on all of them — there is no context/conditioning channel. This means one
  training run supports **arbitrary** masks at sampling time (RePaint), including
  pure unconditional generation (`mask_frac = 1.0`).
- **Diffusion:** variance-preserving (VP / DDPM), ε-prediction. Gaussian noise on
  Cartesian positions in a **center-of-mass-free** frame, **PBC-aware** (minimum
  image). The system always has periodic boundary conditions.
- **Inpainting (sampling time only):** RePaint-style — context atoms are clamped to
  known coordinates and, at each reverse step, overwritten with a re-noised copy of
  their known positions, with the fixed block's CoM realigned after each substitution
  to preserve the equivariant CoM-free frame. A resampling loop (`n_resample`)
  harmonizes the seam between generated and frozen regions.
- **Denoiser:** E(3)-equivariant message-passing network (e3nn), conditioned on the
  noise level, emitting one equivariant vector per atom (the predicted CoM-free noise).

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

## GPU / cluster

- Set `device: auto` (picks CUDA when available) or `device: cuda` explicitly. RNG is
  decoupled from the compute device, so CPU-seeded sampling reproducibly drives CUDA compute.
- MACE runs on GPU when the compute device is CUDA (validation plumbs it through), which
  removes the 640-atom energy bottleneck seen on CPU.
- **Known throughput limit (not a correctness issue):** the neighbor graph is rebuilt every
  diffusion step via ASE on host memory (`.cpu().numpy()`), so each step incurs a host sync
  and a CPU neighbor-list build. This caps GPU utilization. For long GPU runs, replace
  `data/graph.py` with a GPU-native periodic neighbor list (e.g. a cell-list / the MACE or
  matscipy neighbor routines) — this is the single highest-impact optimization.
- Structures are processed one at a time (variable atom count); there is no batched-graph
  path or multi-GPU/distributed training yet. Fine for single-GPU testing.

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
