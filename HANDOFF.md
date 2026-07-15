# HANDOFF

## Objective
Build an E(3)-equivariant diffusion model that generates amorphous In₂O₃ atomic
structures by **inpainting** (freeze "context" atoms, generate "mobile" ones),
validated against ground-truth MD data and a MACE potential. Milestone 1 = get the
core machinery working on **bulk** In₂O₃; the end goal is **surface reconstructions**
(freeze bulk, regenerate a surface region).

## Current State
- **Two branches.** `milestone-one` = the original VP/DDPM pipeline (gen1–gen4).
  `amorphous-score-diffusion` (**current, checked out**) = the VE rewrite (gen5), branched
  off `milestone-one`.
- **VP pipeline is complete but the model does not work.** It runs end-to-end
  (train → inpaint → validate/sweep) and all machinery is correct, but every VP model
  generates physically-invalid structures (atoms overlapping, catastrophic MACE energies).
  See Dead Ends.
- **VE rewrite is complete and de-risked** (branch `amorphous-score-diffusion`, ~5 commits).
  Adopts the amorphous-materials recipe (npj Comput. Mater. 2025, arXiv:2507.05024):
  variance-exploding, annealed-Langevin score model, **uniform-in-cell prior**, small
  physical noise (σ ≤ 0.75 Å). Config-selectable via `diffusion.type: vp|ve`. All eval
  paths (train, inpaint, validate, sweep, diagnose, oracle) support VE. **65 tests pass.**
- **`configs/gen5.yaml` is launch-ready** (VE recipe, 640-only, ~1.08M params). The user is
  launching gen5 training on the cluster (Midway3 A100). **No gen5 results yet.**
- **gen4** (VP, 640-only "confinement ablation") is running on the cluster but is now
  **superseded** — the oracle test proved the failure is the VP *formulation*, not
  confinement. Let it finish only as a data point.
- **Untouched:** surface/slab geometry (vacuum handling), force-guided sampling, DFT/
  active-learning loop, MACE-MD post-refinement.

## Key Decisions & Rationale
- **VP → VE, but stay Cartesian.** The `oracle_test.py` (model-free) showed the VP
  sampler/formulation is *self-consistent* (reconstructs a real structure to 0.002 Å with
  a perfect denoiser), so the formulation isn't "broken" — but with a real (imperfect)
  model the **Gaussian prior collapses atoms to a central blob**. The amorphous paper is
  **Cartesian** (like us) and works with a uniform prior + small noise, so we adopted that
  recipe rather than fractional coordinates. VE's de-risk (`oracle_test.py --type ve`)
  confirms: oracle reconstructs (0.000 Å) and a no-skill model stays box-filled
  (spread ≈ 1) instead of collapsing — the collapse mechanism is gone before training.
- **Data volume is NOT the bottleneck.** We have ~200 independent structures (see below);
  the amorphous paper succeeds with ~24. This killed the "fine-tune MatterGen to beat data
  volume" idea.
- **Rejected MatterGen / DiffCSP.** Both are crystalline (≤20-atom cells, symmetry),
  generate whole structures (no inpainting), and their pretraining wouldn't transfer to
  amorphous 640-atom cells. We import the *method* (VE + uniform prior) not their code.
- **Keep the NequIP-style e3nn denoiser** — it's the same architecture family the working
  amorphous paper uses; the problem was never the network.
- **Data facts (verified in Step 0, do not rediscover):** trajectory id is the extxyz
  `config_type` field (ASE mangles the rest of the comment line); split MUST be by
  trajectory (frames within a trajectory are near-duplicate thermal rattles). `640.extxyz`
  = 40 independent structures (constant 20.25 Å box). `80.extxyz` = 160 structures, 3
  densities (box 10.08–10.21 Å). In-O RDF first peak ~2.15 Å.
- **Receptive-field rule:** keep `graph.cutoff × model.n_layers < min(box)/2`. For 640,
  box/2 = 10.13 Å. For 80, box/2 = 5.04 Å (so RF wraps on 80 — that's why gen5 is 640-only
  for a clean first VE test; surfaces will be in-plane-confined like 80, so this must be
  revisited, not designed around).
- **Model is unconditional; inpainting is sampling-time only** (RePaint). One trained model
  serves any mask, including `mask_frac=1.0` (pure generation).
- **Validation must be MOBILE-atom-restricted.** All-atom stats are dominated by frozen
  context and cannot fail — this was a real bug we fixed in 1b.

## Dead Ends (do not repeat)
- **VP/DDPM with Gaussian prior + unit-variance coordinate normalization + cosine schedule.**
  Fails by collapsing to a Gaussian blob → overlapping atoms. Confirmed across gen2
  (7.5k-param model) and gen3 (**8M params + min-SNR**) — *identical* failure.
- **Scaling capacity** (gen3, 1000× bigger) — no effect; the dead zone (see `diagnose.py`)
  at mid/high noise is unchanged.
- **min-SNR-γ loss weighting** (gen3) — did not fill the dead zone.
- **Fractional coordinates / MatterGen / DiffCSP** — investigated and rejected (oracle test
  refuted "formulation broken"; those tools are crystalline/whole-structure).
- **Blaming confinement (small 80-atom box)** — the oracle test says the fix is the
  formulation, not confinement; MatterGen handles small cells fine. gen4 tests this anyway.

## Next Steps (ordered)
1. **Watch gen5 training** (cluster). At `checkpoints/gen5/step_20000.pt`, run
   `scripts/diagnose.py` (auto-detects VE). **Success = `cos` high and `x0_RMSD` small at
   ALL σ (no dead zone).** This is the fast verdict — independent of Langevin sampling params.
2. **If diagnose is good, run `scripts/sweep.py`** on the gen5 checkpoint (add `--mace`) for
   the physical verdict: mobile In-O RDF/coordination/bond and before/after-relax energies.
   Compare to the gen2/gen3 sweep failure (bond ~2.02 vs ref 2.19, energies ~−10⁹).
3. **If sampling under/over-shoots** (structure right but energies off), tune the Langevin
   knobs in `gen5.yaml` `diffusion:` — `langevin_step_lr`, `langevin_steps`, `refine_steps`
   (NCSN defaults; `annealed_langevin` in `insite_diff/diffusion/sampler.py`). Diagnose
   (prediction quality) is separate from these.
4. **Add MACE-MD post-refinement** (the paper uses NVT+NPT to clean rare bad atoms). New
   helper in `insite_diff/analysis/mace_relax.py`; call it after sampling in `sweep.py`.
5. **Then surfaces:** `mask.geometry: slab` is already implemented/tested; needs vacuum-cell
   handling. Re-check the density-normalization assumption note in
   `insite_diff/diffusion/noising.py` (VP-only; VE doesn't normalize, so likely fine). Add
   multi-size / 80-cell training once bulk VE works.

## Relevant Files
- `README.md` — project overview, usage, environment gotchas, GPU/cluster notes.
- `configs/gen5.yaml` — **the current VE run** (type ve, 640-only, σ 0.01–0.75, ~1M params).
- `configs/gen4.yaml` / `gen3.yaml` / `base.yaml` — VP configs (history; gen3 = the 8M-param
  dead end). `configs/smoke_ve.yaml`, `configs/smoke.yaml` — fast CPU plumbing checks.
- `insite_diff/config.py` — `DiffusionConfig` (`type: vp|ve` + VE σ/Langevin params); strict
  YAML loader, lenient checkpoint loader (`_from_dict(..., strict=False)`).
- `insite_diff/diffusion/schedule.py` — `VPSchedule`, `VESchedule` (σ ladder + `cond()`).
- `insite_diff/diffusion/noising.py` — VP `normalize`/`q_sample`/`add_noise` **and** VE
  `ve_add_noise` (additive, physical scale) + `uniform_init` (box-filling prior). `min_image`.
- `insite_diff/diffusion/sampler.py` — VP `sample()` (DDPM) + `annealed_langevin()` (VE).
- `insite_diff/sampling/inpaint.py` — `inpaint` (VP RePaint w/ CoM realign), `ve_inpaint`
  (RePaint-in-Langevin), `inpaint_dispatch` (picks vp|ve; used by all eval scripts).
- `insite_diff/training/losses.py` — `batch_eps_loss` (VP ε-MSE), `batch_ve_loss` (VE DSM).
- `insite_diff/training/trainer.py` — `train()` (auto vp|ve switch), `load_model` (rebuilds
  denoiser from checkpoint, EMA-first, lenient config), checkpoint save.
- `insite_diff/model/denoiser.py` — `E3Denoiser` (unconditional; noise level = t/T for VP or
  normalized log-σ for VE, passed as a scalar to `NoiseEmbedding`).
- `insite_diff/data/` — `extxyz.py` (config_type traj id), `split.py` (trajectory-level),
  `graph.py` (`build_graph_torch` = on-device min-image neighbor list; ASE `build_graph` =
  reference), `mask.py` (sphere/cube/slab by **atom fraction**), `dataset.py`
  (`check_receptive_field` warning).
- `insite_diff/analysis/` — `rdf.py`/`coordination.py` (mobile-restricted metrics),
  `similarity.py` (permutation-RMSD spread + SOAP memorization), `mace_relax.py` (loads
  `scan_v3_swa.model`; patches `torch.jit.load` for CUDA→CPU), `plots.py`.
- `scripts/oracle_test.py` — **the pivotal model-free diagnostic** (`--type vp|ve`); proved
  VP self-consistent-but-collapsing and VE sound.
- `scripts/diagnose.py` — one-step dead-zone check (auto vp|ve). `scripts/train.py`,
  `sample.py`, `validate.py`, `sweep.py`, `inspect_data.py`.
- `insite_diff/e3nn_compat.py` — `add_safe_globals([slice])` so e3nn 0.4.4 imports under
  torch ≥2.6 (import before any e3nn use).

## Environment / logistics
- Python env (Mac dev): `/opt/anaconda3/envs/insite-diff/bin/python`; run with `PYTHONPATH=.`
  and `export PYTORCH_ENABLE_MPS_FALLBACK=1`. MPS unavailable → CPU on the Mac.
- Data (read-only): Mac `/Users/matt/Desktop/data/in2o3/{640,80}.extxyz`,
  `scan_v3_swa.model`; cluster `/scratch/midway3/bousquet/diffusion/data/`. Configs ship
  with Mac paths — `sed` them to the cluster path (this recurs on every clone; a `DATA_DIR`
  env-var override was proposed but not built).
- Cluster: Midway3, SLURM, partition `gagalli-gpu`, `--constraint=A100`, env
  `/project2/gagalli/bousquet/envs/insite-diff`, modules
  `python/miniforge-25.3.0 cuda/12.2 cudnn mkl/2024.2`. Use `--ntasks-per-node=1` (no srun),
  `mkdir -p logs` before submit. gen{N} runs live under `/scratch/.../diffusion/gen{N}/`.
- Git: two GitHub repos — `bousquet-m/diffusion-model` (private `origin`) and
  `diffusion-model-public` (public; the **cluster pulls from public**). A PAT is embedded in
  `.git/config` remote URLs so pushes work from this environment; push both remotes.
- MACE 640-atom single-point energy ≈ 41 s on CPU (fast on GPU) → keep `--mace` runs small
  or on GPU.
