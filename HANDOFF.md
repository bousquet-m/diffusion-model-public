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
- **SWEEP FOOTGUN — `--config` must match the checkpoint's diffusion type.** `sweep.py:171`
  does `model, _, _ = load_model(...)`: the *architecture* is rebuilt from the checkpoint
  (correct), but the **checkpoint's config is discarded**, and `inpaint_dispatch(cfg, ...)`
  reads `diffusion.type` from the `--config` file. Passing a `vp` config with a `ve`
  checkpoint therefore runs the **DDPM sampler on a VE model** (t/T conditioning into a
  network trained on normalized log-σ) — silent garbage, no error. Always sweep gen5 with
  `--config configs/gen5.yaml`, gen6 with `configs/gen6.yaml`. Worth hardening: prefer the
  checkpoint's `diffusion.type` and error on mismatch.
- **Validation must be MOBILE-atom-restricted.** All-atom stats are dominated by frozen
  context and cannot fail — this was a real bug we fixed in 1b.
- **`model.tp_mode: fc|uvu` — the message tensor product.** gen1–gen5 used
  `FullyConnectedTensorProduct(..., shared_weights=False)`, so the radial net emitted a full
  uvw block **per edge**: 16,064 weights/edge → ~1.6 GB of weights materialized per layer per
  forward, and `Linear(32→16064)` alone was 531k of the 1.08M params. Despite the docstring,
  **that is not what NequIP does** — NequIP uses `uvu` (per-path weights, 352/edge, + a
  Linear mix). Measured at gen5 irreps, CPU, 640 atoms: **fwd 14080→250 ms (56×), fwd+bwd
  76769→734 ms (105×)**. This is the cause of both the 20 h train and the 4 h sweep.
  **Default is `fc` so pre-gen6 checkpoints (config predates the key) still load** — `uvu`
  changes the architecture, so gen5 checkpoints are NOT loadable as uvu. New configs opt in.

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

## gen5 RESULTS (training done, 150k steps, 20 h on A100)
- **VE learned — the VP dead zone is gone.** `diagnose.py` on `final.pt` (640 frame):
  `cos` peaks 0.886 at σ=0.218 and `x0_RMSD` is small at EVERY σ (1.105 Å at σ_max down to
  0.017 Å at σ_min). Contrast gen2/gen3: `cos≈0`, `MSE≈1`, `ratio≈0`. **The reformulation
  worked.** Physical verdict (sweep) still open.
- **Low-σ falloff is benign, not failure.** At σ≤0.03 the injected noise is below the MD
  thermal rattle, so the true score really is ~0 (`MSE→0.966` = correctly predicting
  nothing); `x0_RMSD` there is 0.017 Å. Do not "fix" this.
- **σ_max is the weak end**: `cos` 0.53, ratio 0.546 (under-predicts ~2×) — and diagnose is
  *optimistic* there, since it feeds `real structure + 0.75 Å noise` while the sampler
  starts from `uniform_init` (much further off-manifold). Watch this if sampling fails.
- **Converged by ~10-20k of 150k steps.** diagnose on step_10000/20000/45000/final is flat
  to within 4-draw noise at every σ (σ_max cos: 0.499/0.530/0.537/0.530 — step_45000 is
  *marginally above* final). ~14 h of the 20 h run bought nothing.

## Next Steps (ordered)
1. **Train gen6** (`configs/gen6.yaml`): gen5's recipe + the `tp_mode: uvu` fix + 30k steps.
   Then `diagnose.py` on it and compare to the gen5 table above — same irreps, so the
   comparison isolates the tp_mode effect. **Note gen6 is 74,656 params vs gen5's 1,079,520**
   (the FC radial net was 531k of pure weight-emission overhead). If diagnose regresses vs
   gen5, spend the freed budget on `hidden_irreps`/`n_layers` — that is the moment to do it,
   not before.
2. **Run `scripts/sweep.py`** (add `--mace`) for the physical verdict: mobile In-O
   RDF/coordination/bond and before/after-relax energies. Compare to the gen2/gen3 sweep
   failure (bond ~2.02 vs ref 2.19, energies ~−10⁹). NB the sweep is 7 fractions × 3
   structures × 10 seeds × 2600 forwards = **546k sequential single-structure forwards** —
   that, not MACE, is why it takes ~4 h. gen5 checkpoints are `fc` and get no speedup;
   only a gen6 (uvu) model does.
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

## Key Papers & References
- **A Generative Diffusion Model for Amorphous Materials** — arXiv:2507.05024, npj Comput.
  Mater. 2025. **THE recipe we are now implementing (gen5/VE).** Variance-exploding,
  annealed-Langevin score model, **uniform-in-cell prior**, small physical noise
  (σ ≤ 0.75 Å), customized **NequIP** denoiser; validated on silica glass with ~24 training
  structures. Key tricks: external noise during denoising is "essential"; short MACE/MD
  post-refinement. Their code builds on **LLNL/graphite** (github.com/LLNL/graphite).
- **NCSN — "Generative Modeling by Estimating Gradients of the Data Distribution"** (Song &
  Ermon, NeurIPS 2019). The score-matching + **annealed Langevin** method our VE sampler
  (`annealed_langevin` in `sampler.py`) follows; step size ∝ σ².
- **DDPM — "Denoising Diffusion Probabilistic Models"** (Ho et al., 2020). The VP framework
  behind gen1–gen4 (now a dead end for this problem).
- **RePaint** (Lugmayr et al., CVPR 2022, arXiv:2201.09865). The inpainting-by-masking scheme
  our `inpaint`/`ve_inpaint` implement (clamp context, re-noise each reverse step).
- **CHGGen** — host-guided inpainting; the original project brief modeled our context
  clamping on it.
- **NequIP** (Batzner et al., Nat. Commun. 2022) + the **e3nn** library — the E(3)-equivariant
  message-passing architecture family for `E3Denoiser`.
- **min-SNR-γ** (Hang et al., 2023, arXiv:2303.09556) — loss weighting tried in gen3 (dead end).
- **DiffCSP** (Jiao et al., NeurIPS 2023, github.com/jiaor17/DiffCSP) and **MatterGen**
  (Microsoft, github.com/microsoft/mattergen) — fractional-coordinate / crystalline diffusion.
  **Assessed and rejected** (crystalline ≤20-atom cells, whole-structure, no inpainting,
  pretraining won't transfer to amorphous). See Dead Ends.
- **Crystal structure prediction with host-guided inpainting + foundation potentials** —
  arXiv:2504.16893 (came up in the lit search on inpainting-style generation).

## Environment / logistics
- Python env (Mac dev): `/opt/anaconda3/envs/insite-diff/bin/python`; run with `PYTHONPATH=.`
  and `export PYTORCH_ENABLE_MPS_FALLBACK=1`. MPS unavailable → CPU on the Mac.
- Data (read-only): Mac `/Users/matt/Desktop/data/in2o3/{640,80}.extxyz`,
  `scan_v3_swa.model`; cluster `/scratch/midway3/bousquet/diffusion/data/`.
- **Config path convention (changed — no more sed-on-every-clone):** the real run configs
  (`base`, `gen3`, `gen4`, `gen5`, `gen6`) now carry **cluster** paths, since that is the only
  place they are ever run. The smoke configs (`smoke.yaml`, `smoke_ve.yaml`) keep **Mac**
  paths — they are the local CPU plumbing checks. A `DATA_DIR` env-var override is still the
  proper fix if this ever splits again; proposed, not built.
- **`80.extxyz` on the cluster is UNVERIFIED.** `base.yaml`/`gen3.yaml` reference it, but only
  `640.extxyz` + `scan_v3_swa.model` are confirmed present (gen5 trained off them). Check
  before running a config with two sources.
- **`.job` files are NOT in the repo** — they exist only on the cluster. `sweep.job` had three
  bugs as of the gen5 pull: it `cd`s to the **gen2** tree, passes `--config configs/base.yaml`
  (Mac paths → the `FileNotFoundError` in `slurm-52283054`), and — worse — `base.yaml` is
  `type: vp`, so it would run the **VP sampler on the VE gen5 checkpoint**. See the sweep
  footgun under Key Decisions.
- Cluster: Midway3, SLURM, partition `gagalli-gpu`, `--constraint=A100`, env
  `/project2/gagalli/bousquet/envs/insite-diff`, modules
  `python/miniforge-25.3.0 cuda/12.2 cudnn mkl/2024.2`. Use `--ntasks-per-node=1` (no srun),
  `mkdir -p logs` before submit. gen{N} runs live under `/scratch/.../diffusion/gen{N}/`.
- Git: two GitHub repos — `bousquet-m/diffusion-model` (private `origin`) and
  `diffusion-model-public` (public; the **cluster pulls from public**). A PAT is embedded in
  `.git/config` remote URLs so pushes work from this environment; push both remotes.
- MACE 640-atom single-point energy ≈ 41 s on CPU (fast on GPU) → keep `--mace` runs small
  or on GPU.
