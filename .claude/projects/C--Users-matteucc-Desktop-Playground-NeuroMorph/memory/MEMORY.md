# NeuroMorph Project Memory

## Project Overview
- Computational companion to MUSIC ERC Starting Grant (PI: Giulio Matteucci, CHUV)
- Convolutional denoising autoencoder with contextual input simulating multisensory association learning
- MNIST digits as visual modality, context vector as auditory modality analogue
- Grant has 3 work packages: WP1 (behavior/psychometrics), WP2 (neural activity), WP3 (synaptic substrates)

## Key Files
- `deep_contextual_autoencoder.ipynb` — most advanced notebook (psychometric curves, t-SNE, RDM)
- `src/models.py` — Encoder (additive context via fc_context), Decoder (context arg unused!), freeze_except_context
- `src/utils.py` — generate_context, add_noise (correlated+uncorrelated), visualize_noisy_images
- `DEV_LOG.md` — development log tracking progress and brainstorming

## Architecture Details
- Encoder: Conv(1→c)→Conv(c→2c)→FC(2c*7*7→latent) + FC_context(context_dim→latent) additive
- Decoder: FC(latent→2c*7*7)→ConvT→ConvT→tanh (context passed but NOT used in deep version)
- Training: baseline phase (denoising, no context signal) then association phase (context paired with target digit, all weights frozen except fc_context)
- Default params: capacity=16, latent_dims=32, context_dim=10, target_digit=3

## Known Issues
- Decoder context arg is dead code in src/models.py
- Heavy code duplication across 3 notebooks
- No config system, seeds, or experiment logging
- Psychometric shift not rigorously quantified (no PSE/threshold extraction)

## User Preferences
- Giulio is a systems neuroscientist (physics background), comfortable with Python/PyTorch
- Values: scientific rigor, clean code, good software engineering practices
- Dark background for matplotlib plots (plt.style.use('dark_background'))
