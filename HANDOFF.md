# HANDOFF

## amorphous-carbon branch (a-C) — NEW MATERIAL on the gen6 machinery
Branch `amorphous-carbon`, off `amorphous-score-diffusion`. The method is unchanged from
gen6 (VE + annealed Langevin + uniform-in-cell prior + uvu E(3) denoiser); only the
material-specific pieces are new. Goal: train on 10 AIMD a-C snapshots and GENERATE more
structures from the same distribution — the finetuned MLIP cannot reproduce this
distribution via MD, and the diffusion model samples the structure distribution directly
(no potential), so it can populate what MD can't.

**Data (Mac `/Users/matt/Desktop/data/aC`, cluster `/scratch/midway3/bousquet/diffusion/aC`).**
10 independent AIMD final snapshots, 216 atoms each, single density 3.25 g/cc, boxes
10.86–10.97 Å. This is the ENTIRE training set. `coordination_numbers.py` (rcut=1.95 Å,
cn 2/3/4 = sp/sp²/sp³ as % of atoms) is THE metric.

**Fixes applied in preprocessing (`scripts/preprocess_ac.py`, verified — do not rediscover):**
- Species mislabel: raw extxyz column is "H" but the atoms are carbon (216 atoms in a
  10.88 Å box = ~3.3 g/cc = a-C, not 0.28 g/cc for H). Relabelled H→C. Verified rho on the
  10 snapshots = 3.26–3.36 g/cc.
- Trajectory IDs: all 10 raw files share `config_type=ac_325` (one trajectory → no split).
  Each snapshot gets a unique id `ac_325_1 … ac_325_10`. With only 10, val_fraction=0.2 →
  2 held out (ac_325_9, ac_325_10 under seed 0).
- Output is one combined `aC.extxyz` (10 frames), mirroring how 640.extxyz packs In2O3.
  STEP 0 of `train_ac.job` runs the relabel on the cluster (idempotent).

**Reference sp³% distribution (the target), computed on the 10 relabelled snapshots:**
**sp³% = 91.9 ± 3.2 (range 84.3–95.4), sp²% = 8.0, mean C-coordination = 3.92.** High-sp³
(tetrahedral "ta-C"). SUCCESS = generated structures reproduce this distribution's MEAN AND
SPREAD at the right density, are DIVERSE (multi-seed spread ≫ 0, low SOAP self-similarity),
hold under DFT single-points, and give sane MACE energies. sp²/sp³ is the PRIMARY verdict;
MACE energy is SECONDARY (same finetuned MLIP that fails dynamically).

**Receptive field (the a-C analog of the In2O3 rule):** box ≈ 10.88 Å → box/2 = 5.43 Å, so
`graph.cutoff × n_layers` must stay under it. `ac.yaml` uses cutoff 2.5 × n_layers 2 = 5.0 Å
(gen6's 5.0×2=10 would wrap this small a cell). Verified avg_neighbors @2.5 Å = **9.1** (sane),
no receptive-field warning at train time. This graph/message cutoff is SEPARATE from the
metric's C-C rcut = 1.95 Å.

**What was built (all tests pass: 79):**
- `scripts/preprocess_ac.py` — the H→C + unique-id relabel pass.
- `insite_diff/analysis/carbon.py` — sp/sp²/sp³ metric (mobile-restricted), per-frame
  distribution + mean/spread. Reproduces the reference (91.9 ± 3.2) exactly. Tests added to
  `tests/test_analysis.py`.
- `configs/ac.yaml` — type ve, tp_mode uvu, species [C], cutoff 2.5, **sigma_max 0.60**
  (scaled below gen6's 0.75 for the shorter ~1.5 Å C-C bond: 0.60/1.5 ≈ 0.75/2.15),
  cluster aC data + `mace-aC-finetune.model`. `configs/smoke_ac.yaml` = fast CPU plumbing check.
- `scripts/sweep.py` is now material-aware (`_is_carbon` = species==["C"]): C-C RDF, the sp³%
  metric gen/ref, SOAP species=("C",), sp³% panels, and a once-computed reference distribution
  in `sweep.json`. In-O path untouched. New `--sdedit-sigma` override.
- **SDEdit augmentation** (`ve_inpaint(..., start_sigma=)`, plumbed via `diffusion.sdedit_sigma`
  / `sweep --sdedit-sigma`): start the anneal from a real structure + intermediate-σ noise
  instead of the uniform prior, iterating only the ladder ≤ start_sigma. Similarity↔diversity
  dial; recommended for augmentation given only 10 training points. Tested (stays closer to the
  seed than full generation; raises below sigma_min).
- Job scripts (repo, not hand-edited on cluster): `train_ac.job` (relabel → VE oracle canary →
  train), `diagnose_ac.job` (data-sufficiency verdict, held-out vs training snapshot),
  `sweep_ac.job` (sp³% verdict + SDEdit line-search). Run tree:
  `/scratch/midway3/bousquet/diffusion/amorphous-carbon/diffusion-model-public`.
- Config: added `validation.cn_cutoff_cc` (1.95) and `diffusion.sdedit_sigma` (0.0).

**De-risked locally (CPU):** VE oracle on an a-C snapshot reconstructs 0.000 Å; zero control
stays box-filled (spread 0.99, no collapse) — formulation sound before training. Smoke
train/diagnose/sweep run clean (split 8/2, avg_neighbors 9.1, carbon metric + SDEdit sweep
end-to-end).

**Next on the cluster (de-risk cheaply, exactly as gen6):** ① `train_ac.job` (oracle canary
then train). ② `diagnose_ac.job` — cos high / x0_RMSD small at all σ = the data-sufficiency
verdict for 10 structures; a mid-σ dead zone = too few structures/too broad → get more data.
Watch the held-out-vs-training gap for overfitting the 10. ③ `sweep_ac.job` — sp³% gen vs the
91.9 ± 3.2 reference, multi-seed spread + SOAP memorization (with 10 points the real risk is
memorizing them — want NEW structures at the right sp³%, not copies), MACE energy secondary.
Retune `langevin_step_lr` if sp³% under/overshoots; use SDEdit for the actual augmentation runs.
DFT is the arbiter for any structures fed back into MLIP training.

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

## gen6 RESULTS — MILESTONE 1 PASSED (bulk In₂O₃, uvu model)
- **uvu costs nothing.** 32-repeat diagnose, gen6 (74,656 params) vs gen5 (1,079,520),
  same run/structure: cos within ~0.01 at every σ (σ_max 0.520 vs 0.528; peak 0.877 vs
  0.887), x0_RMSD identical to 3 dp. The FC radial net's 531k params were pure overhead.
  **Do NOT spend the freed budget on capacity — the gap is below the noise floor.**
- **gen6 physically works — full ladder, 30 samples/fraction, `outputs/sweep_gen6/`.**
  mean In-O bond tracks ref within a few mÅ across all mask_frac; energies +28..+102
  meV/atom above MD (slightly under-relaxed, ~thermal). vs gen2/gen3 VP failure (bond ~2.02,
  E ~−10⁹) this is a different planet.
- **Diversity is real (NOT memorization).** multi-seed spread at mask_frac=1.0 = 1.57 Å RMSD
  across 30 seeds → different valid structures, not one memorized cell. The `memNN` ≈ 0.9999
  is the SATURATED metric ceiling (a real held-out structure scores 0.99997 against the same
  bank; random gas scores 0.962). Read the spread panel, not memNN. TODO: report the held-out
  ref baseline alongside memNN so that panel stops reading as a false alarm.
- **One coherent flaw → drives step 3.** Under-coordination, worst at low mask_frac:
  d_coord (gen−ref) = −0.54 @ frac 0.016 → −0.03 @ 1.0. Low frac = a small generated region
  in fixed context = the SURFACE geometry, so this is the number to watch toward the end goal.

## Next Steps (ordered)
1. **[DONE] Langevin tuning** (`tune_langevin_gen6.job`, `outputs/tune_gen6/`). Verdict:
   **`langevin_step_lr` is the lever; `refine_steps` is not.** Bumping step_lr 2e-5 → **1e-4
   (now the gen6.yaml default)** closed ~66% of the low-frac coordination gap (d_coord −0.48
   → −0.16 @ frac 0.1) and ~72% of the energy error (+96 → +27 meV), with diversity intact
   (spread 1.57 unchanged @ 1.0). refine_steps {500,2000} did ~nothing → under-coordination
   is set during the ANNEAL, not the final quench. 1e-4 is near the safe ceiling: step ~
   lr·(σ/σ_min)² ≈ 0.56 Å @ σ_max; 2e-4 (~1.1 Å) risks instability, so don't crank further.
   **Residual: low-frac still −0.16 / +27 meV** — that is step 2's (NVT) job.
2. **[BUILT, not yet run] MACE NVT post-refinement** (`nvt_refine_gen6.job`) — the paper's
   cleanup for the under-relaxation. `mace_relax.nvt_refine` = short Langevin NVT MD + 0 K
   quench. **NVT ONLY (no NPT):** scan_v3_swa is unreliable for stress/cell dynamics, so the
   generated box is held fixed — the thermostat never touches the cell (a test asserts this).
   `sweep.py --nvt-steps/--nvt-temp/--nvt-timestep` (or `mace.nvt_*` in config) refines each
   generated frame and reports post-NVT coordination/bond/energy (`*.gen_nvt`,
   `energy_per_atom.gen_after_nvt`). `collect_langevin_tuning.py` shows the extra columns.
   Job does a T line-search {300,500,800 K}. Works if `d_coord_nvt`→0 and `E_nvt-ref` <
   `E gen-ref`; too-high T restructures. Run only if step 1's knobs leave a gap — but it's
   ready either way. NVT is MACE-heavy (nvt_steps+relax_steps force evals per frame).
3. **Surfaces** (the end goal) — `mask.geometry: slab` implemented/tested; needs vacuum-cell
   handling. The low-frac tuning above is the closest proxy already in hand.

### For re-running the passing verdict (reference)
`scripts/sweep.py` + `--mace`: 7 fractions × 3 × 10 × 2600 forwards = 546k sequential
single-structure forwards (that, not MACE, is the cost). gen5 (`fc`) gets no uvu speedup;
gen6 does. Always `--config` matching the checkpoint's diffusion type (see SWEEP FOOTGUN).
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
