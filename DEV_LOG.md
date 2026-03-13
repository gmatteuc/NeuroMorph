# NeuroMorph Development Log

## Session 1 — 2026-03-07: Project Review & Roadmap

### Project Understanding

**NeuroMorph** is a computational companion to the MUSIC ERC Starting Grant (PI: Giulio Matteucci).
It implements a convolutional denoising autoencoder with a contextual input to simulate how
unsupervised exposure to cross-modal statistical regularities reshapes cortical representations
and biases perception — a simplified *in silico* analogue of the audiovisual learning paradigm
proposed in the grant.

**Core idea**: A denoising autoencoder learns to reconstruct MNIST digits from noisy inputs.
A secondary "context" vector (representing another sensory modality) is paired with a specific
digit (e.g., "3") during an association training phase. After training, the context biases
reconstructions, shifts latent representations toward the associated digit, and shifts the
psychometric curve of a downstream binary classifier — mirroring predictions for behavior
(WP1), neural activity (WP2), and synaptic plasticity (WP3) in the grant.

### Current Codebase State

| File | Description | Status |
|------|-------------|--------|
| `simple_autoencoder.ipynb` | Vanilla AE (no context, no denoising). Reconstruction, interpolation, t-SNE/RDM analysis. | Complete baseline |
| `contextual_autoencoder.ipynb` | Contextual denoising AE. Context concatenated with conv features & latent. Baseline + association training. Reconstruction comparison, latent sampling. | Working prototype |
| `deep_contextual_autoencoder.ipynb` | Improved version: uses `src/models.py` (additive context projection via `fc_context`), `src/utils.py`, correlated+uncorrelated noise, freeze-all-except-context during association, t-SNE flow fields, RDMs, psychometric curves via logistic discriminator on interpolated stimuli. | **Most advanced** — main notebook |
| `src/models.py` | Encoder (additive context projection), Decoder (context unused beyond signature), Autoencoder, `freeze_except_context` | Needs attention: Decoder receives context arg but doesn't use it |
| `src/utils.py` | `generate_context`, `add_noise` (correlated/uncorrelated), `visualize_noisy_images` | Clean utilities |

### Key Architectural Observations

1. **Three notebooks with significant code duplication** — model definitions, training loops, noise functions,
   visualization helpers are redefined inline in earlier notebooks rather than imported.
2. **`src/models.py` inconsistency**: `Decoder.forward(latent, context)` accepts `context` but never uses it.
   In `contextual_autoencoder.ipynb`, context is concatenated in both encoder and decoder (different arch).
   In `deep_contextual_autoencoder.ipynb`, context is only used via additive projection in encoder.
3. **No config system** — hyperparameters scattered across notebook cells.
4. **No experiment reproducibility** — no seeds set, no structured experiment logging.
5. **No tests** — pure notebook exploration.

### TODO Items from README (with assessment)

#### Behavior (Partially Done)
- [x] Choose two categories ("3" and "4") ✓ (deep notebook)
- [x] Interpolate in latent space ✓ (deep notebook)
- [x] Build logistic regression discriminator ✓ (deep notebook)
- [x] Construct psychometric curves ✓ (deep notebook)
- [ ] **Quantify psychometric shift properly** — current implementation needs cleanup;
      shift is visible but not rigorously quantified (e.g., PSE shift, threshold change)

#### Physiology (Not Started)
- [ ] Sparse noise experiment (both modalities, alone and together)
- [ ] Congruent/incongruent tuning & multisensory enhancement
- [ ] Most Exciting Input (MEI) via optimization

#### Bayesian Analysis (Not Started)
- [ ] Bimodal prior (50% "3", 50% "4")
- [ ] Latent sampling with/without context → category histograms
- [ ] Bimodal vs. blurred average analysis

### Refactoring Plan (Priority Order)

1. **Project structure**: proper Python package layout with config, training scripts, analysis modules
2. **Consolidate models**: single source of truth in `src/models.py`, fix decoder context usage
3. **Config system**: YAML/dataclass-based experiment configs
4. **Training pipeline**: standalone training script (not just notebooks)
5. **Analysis modules**: extract analysis functions (t-SNE, RDM, psychometric curves) into reusable modules
6. **Notebooks as consumers**: thin notebooks that import and call, not define
7. **Reproducibility**: seed management, experiment logging, model versioning

### Scientific Directions to Explore

1. **Psychometric analysis rigor**: fit sigmoid/Weibull to psychometric data, extract PSE & threshold,
   bootstrap confidence intervals, properly quantify context-induced shift
2. **Sparse noise / tuning curves**: systematic mapping of latent units' responses to visual and
   contextual features — congruent vs incongruent, multisensory enhancement index
3. **MEI (Most Exciting Input)**: gradient-based optimization to find inputs that maximally activate
   specific latent units, with and without context
4. **Bayesian/bimodal prior**: train with 50/50 association, sample, analyze categorical distributions
5. **Prior sampling in darkness**: feed only context (no visual input / pure noise) and decode what
   the network "imagines" — direct analogue of WP2.2
6. **Synaptic substrate analogue**: analyze weight matrices of `fc_context` to find clustering of
   congruent feature detectors — analogue of WP3
7. **Multiple association paradigm**: associate different contexts with different digits, test
   generalization and interference
8. **Variational extension**: VAE version for better-behaved latent space and principled sampling
9. **Orientation-based stimuli**: move beyond MNIST to oriented gratings (closer to grant's
   actual visual stimuli) with tone-frequency context

---

## Session 2 — 2026-03-07: Package Refactoring & Scientific Extensions

### Changes Made

#### Phase 1: Refactoring into `neuromorph` Package

1. **Package scaffolding**: Created `neuromorph/` package with submodules: `configs`, `models`, `data`, `training`, `analysis`, `visualization`. Added `pyproject.toml` for `pip install -e .`.

2. **Config system** (`neuromorph/configs/experiment.py`): Frozen dataclasses — `ModelConfig`, `NoiseConfig`, `ContextConfig`, `TrainingConfig`, `DiscriminatorConfig`, `PsychometricConfig`, `BimodalContextConfig`, `ExperimentConfig`. Defaults match deep notebook values exactly.

3. **Decoder dead argument fixed**: `Decoder.forward(self, latent)` no longer accepts unused `context` arg. `Decoder.__init__` no longer takes `context_dim`. `Autoencoder.forward` calls `self.decoder(latent)`. Old `.pth` weights remain compatible.

4. **Models extracted**: `Encoder`, `Decoder`, `Autoencoder` in separate files. Added `Discriminator`, `MNISTClassifier`, `freeze_except_context`, `unfreeze_all`.

5. **Data utilities**: `get_mnist_dataloaders`, `add_noise`, `generate_context`, `collect_digit_examples`, `collect_all_digit_examples`.

6. **Training pipeline**: `set_global_seed`, `AutoencoderTrainer` (baseline + association), `train_discriminator`, `evaluate_discriminator`.

7. **Analysis modules**: `capture_activations`, `compute_joint_tsne`, `compute_distance_change`, `compute_balanced_rdm`, `generate_interpolated_images_batch`, psychometric fitting, `BayesianSampler`.

8. **Visualization**: Central style module (dark_background, plasma cmap, yellow/white condition colors). `show_image`, `save_high_res_image`, `plot_average_images`, `visualize_noisy_images`, `plot_tsne_and_rdm`, `plot_tsne_flow_field`, `plot_rdm_comparison`, `plot_psychometric_curve`, `plot_pse_shift_summary`, `plot_category_histogram`.

9. **Backward compatibility**: `src/models.py` and `src/utils.py` converted to re-export shims from `neuromorph.*`.

10. **Training CLI**: `scripts/train.py` with argparse (--retrain, --seed, --epochs-baseline, --epochs-association, --pretrained-dir, --output-dir).

11. **Refactored notebooks**: `notebooks/01_association_analysis.ipynb`, `02_behavioral_analysis.ipynb`, `03_bayesian_analysis.ipynb` — thin consumers importing from `neuromorph`.

#### Phase 2: Rigorous Behavioral Analysis

- 4-parameter psychometric sigmoid: `p(x) = γ + (δ - γ) · σ(β(x - α))`
- `fit_psychometric_curve` → `FitResult` with PSE, JND, R²
- `bootstrap_pse_ci` for percentile confidence intervals
- `compute_pse_shift` with CI overlap significance test
- `run_full_psychometric_pipeline` for multi-pair analysis (default: 3vs4, 3vs5, 3vs8)
- `plot_psychometric_curve` with fitted sigmoid, PSE lines, CI shading
- `plot_pse_shift_summary` bar plot across digit pairs

#### Phase 3: Bayesian Analysis

- `BimodalContextConfig` for bimodal training (50% digit_a, 50% digit_b)
- `MNISTClassifier` (CNN) for labeling decoded samples
- `BayesianSampler`: estimate latent distribution → sample with/without context → decode → classify → histogram
- `plot_category_histogram` grouped bars by condition
- Bimodal training loop in notebook 03

### File Summary

| New | Count |
|-----|-------|
| `neuromorph/` package files | 26 |
| `scripts/train.py` | 1 |
| `notebooks/` | 3 |
| `pyproject.toml` | 1 |

| Modified | |
|----------|---|
| `src/models.py` | Backward compat shim |
| `src/utils.py` | Backward compat shim |
| `.gitignore` | Added `*.egg-info/` |
| `DEV_LOG.md` | This entry |

Old notebooks preserved in place (not deleted).

---
*Next steps: Run full verification — train from scratch, run all 3 notebooks, verify outputs match.*
