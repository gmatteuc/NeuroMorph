"""Prior variation experiment: train bimodal models with different P(a|ctx) priors.

For each (digit_pair, prior_a) combination:
1. Train a bimodal model where context co-occurs with digit_a at rate prior_a
2. Run corruption test (with and without context)
3. Compute no-context sigmoid, with-context sigmoid, NP Bayesian at that prior
4. Save results incrementally to JSON

Results are saved to output/prior_variation/ with load-if-exists guards,
so the script can be stopped and resumed, or extended with new pairs/priors.

Usage:
    python scripts/run_prior_variation.py
    python scripts/run_prior_variation.py --pairs 3,4 0,1 5,8
    python scripts/run_prior_variation.py --priors 0.5 0.7 0.9
    python scripts/run_prior_variation.py --retrain
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from neuromorph.configs import ExperimentConfig
from neuromorph.training import set_global_seed
from neuromorph.data import get_mnist_dataloaders, generate_context, add_noise
from neuromorph.models import Autoencoder, MNISTClassifier, freeze_except_context
from neuromorph.analysis.bayesian import (
    compute_digit_centroids, run_corruption_test, compute_sigmoid_data,
    fit_sigmoid, compute_nonparametric_bayesian,
)

# --- Default experiment grid ---
DEFAULT_PAIRS = [(3, 4), (1, 7), (0, 6), (5, 8), (4, 9)]
DEFAULT_PRIORS = [0.5, 0.6, 0.7, 0.8, 0.9]
CONTEXT_ID = 5
NOISE_LEVEL = 1.0


def train_pair_model_with_prior(baseline_path, digit_a, digit_b, prior_a,
                                 context_id, config, train_dl, device):
    """Train a bimodal model with asymmetric prior P(a|ctx) = prior_a.

    Achieves the target prior by subsampling context assignments:
    - If prior_a >= 0.5: all digit_a images get context, fraction f of digit_b do
      where f = (1 - prior_a) / prior_a
    - If prior_a < 0.5: symmetric (subsample digit_a instead)
    """
    mc = config.model
    nc = config.noise
    bc = config.bimodal

    ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(device)
    ae.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
    freeze_except_context(ae)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, ae.parameters()), lr=config.training.lr
    )

    # Context assignment probabilities
    if prior_a >= 0.5:
        p_ctx_a = 1.0
        p_ctx_b = (1 - prior_a) / prior_a
    else:
        p_ctx_a = prior_a / (1 - prior_a)
        p_ctx_b = 1.0

    for epoch in range(bc.epochs_association):
        for image_batch, label_batch in train_dl:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            # Start with noise-only context for all images
            context_batch = generate_context(
                image_batch.size(0), signal_level=None,
                noise_level=bc.noise_level, category_id=0,
                context_dim=mc.context_dim, device=device
            )

            # Assign context to digit_a images with probability p_ctx_a
            for target, p_ctx in [(digit_a, p_ctx_a), (digit_b, p_ctx_b)]:
                idx = (label_batch == target).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    if p_ctx >= 1.0:
                        selected = idx
                    else:
                        mask = torch.rand(len(idx), device=device) < p_ctx
                        selected = idx[mask]
                    if len(selected) > 0:
                        context_batch[selected] = generate_context(
                            len(selected), signal_level=bc.signal_level,
                            noise_level=bc.noise_level, category_id=context_id,
                            context_dim=mc.context_dim, device=device
                        )

            noisy_batch = add_noise(
                image_batch, noise_level=nc.training_noise_level,
                correlated=nc.correlated, uncorrelated=nc.uncorrelated,
                kernel_size=nc.kernel_size, sigma=nc.sigma
            )
            reconstructed = ae(noisy_batch, context_batch)
            loss = torch.nn.functional.mse_loss(reconstructed, image_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ae.eval()
    return ae


def analyze_model(ae, classifier, test_dl, digit_a, digit_b, prior_a,
                  context_id, config, device):
    """Compute three curves for one (model, prior) condition."""
    mc = config.model

    # Centroids from this model's encoder
    centroids = compute_digit_centroids(ae.encoder, test_dl, config, device)

    # Run corruption test (returns both with-context and no-context predictions)
    exp = run_corruption_test(
        ae, classifier, test_dl, noise_level=NOISE_LEVEL,
        context_signal_level=config.context.eval_signal_level,
        context_id=context_id, config=config, device=device
    )

    # --- Curve 1: Network WITH context ---
    ctx_bin_centers, ctx_p_a, ctx_bin_counts, _ = compute_sigmoid_data(
        exp['noisy_latents'], exp['preds_ctx'], centroids,
        digit_a, digit_b, n_bins=25
    )
    ctx_params, ctx_curve = fit_sigmoid(ctx_bin_centers, ctx_p_a, ctx_bin_counts)

    # --- Curve 2: Network WITHOUT context ---
    nc_bin_centers, nc_p_a, nc_bin_counts, _ = compute_sigmoid_data(
        exp['noisy_latents'], exp['preds_no_ctx'], centroids,
        digit_a, digit_b, n_bins=25
    )
    nc_params, nc_curve = fit_sigmoid(nc_bin_centers, nc_p_a, nc_bin_counts)

    # --- Curve 3: NP Bayesian at this prior ---
    np_bayes = compute_nonparametric_bayesian(
        exp['noisy_latents'], exp['orig_labels'], centroids,
        digit_a, digit_b, prior_a=prior_a
    )

    # Package results
    result = {
        'digit_a': digit_a,
        'digit_b': digit_b,
        'prior_a': prior_a,
        # With-context network
        'ctx_fitted_midpoint': float(ctx_params[0]) if ctx_params is not None else None,
        'ctx_fitted_slope': float(ctx_params[1]) if ctx_params is not None else None,
        'ctx_fitted_gamma': float(ctx_params[2]) if ctx_params is not None else None,
        'ctx_fitted_delta': float(ctx_params[3]) if ctx_params is not None else None,
        # No-context network
        'nc_fitted_midpoint': float(nc_params[0]) if nc_params is not None else None,
        'nc_fitted_slope': float(nc_params[1]) if nc_params is not None else None,
        'nc_fitted_gamma': float(nc_params[2]) if nc_params is not None else None,
        'nc_fitted_delta': float(nc_params[3]) if nc_params is not None else None,
        # NP Bayesian
        'bayesian_midpoint': float(np_bayes['bayesian_midpoint']),
        'bayesian_slope': float(np_bayes['bayesian_slope_at_midpoint']),
        'delta_mu': float(np_bayes['delta_mu']),
        'sigma_a': float(np_bayes['sigma_a']),
        'sigma_b': float(np_bayes['sigma_b']),
        # Category histogram stats
        'mass_a_ctx': float(exp['hist_ctx'][digit_a]),
        'mass_b_ctx': float(exp['hist_ctx'][digit_b]),
        'mass_a_no_ctx': float(exp['hist_no_ctx'][digit_a]),
        'mass_b_no_ctx': float(exp['hist_no_ctx'][digit_b]),
    }

    # Derived metrics
    if ctx_params is not None:
        result['ctx_slope_ratio'] = float(ctx_params[1] / np_bayes['bayesian_slope_at_midpoint']) \
            if np_bayes['bayesian_slope_at_midpoint'] != 0 else None
        result['ctx_midpoint_error'] = float(ctx_params[0] - np_bayes['bayesian_midpoint'])
    if nc_params is not None:
        result['nc_slope_ratio'] = float(nc_params[1] / np_bayes['bayesian_slope_at_midpoint']) \
            if np_bayes['bayesian_slope_at_midpoint'] != 0 else None

    return result


def load_existing_results(results_path):
    """Load existing results, keyed by (digit_a, digit_b, prior_a)."""
    if os.path.isfile(results_path):
        with open(results_path) as f:
            data = json.load(f)
        return {(r['digit_a'], r['digit_b'], r['prior_a']): r for r in data}
    return {}


def main():
    parser = argparse.ArgumentParser(description='Prior variation experiment')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='Digit pairs as "a,b" (e.g., 3,4 1,7). Default: 5 representative pairs')
    parser.add_argument('--priors', nargs='+', type=float, default=None,
                        help='Prior values for P(a|ctx). Default: 0.5 0.6 0.7 0.8 0.9')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain even if model exists')
    args = parser.parse_args()

    pairs = DEFAULT_PAIRS
    if args.pairs:
        pairs = [tuple(int(x) for x in p.split(',')) for p in args.pairs]
    priors = args.priors or DEFAULT_PRIORS

    config = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_global_seed(config.training.seed)

    train_dl, test_dl = get_mnist_dataloaders(batch_size=config.training.batch_size)
    mc = config.model

    # Load classifier
    classifier_path = os.path.join(config.pretrained_dir, 'mnist_classifier.pth')
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    classifier.eval()

    baseline_path = os.path.join(config.pretrained_dir, 'baseline_deep_autoencoder.pth')

    # Output paths
    out_dir = os.path.join(config.output_dir, 'prior_variation')
    models_dir = os.path.join(config.pretrained_dir, 'prior_variation')
    results_path = os.path.join(out_dir, 'results.json')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Load existing results (for incremental runs)
    existing = load_existing_results(results_path)
    all_results = list(existing.values())

    total = len(pairs) * len(priors)
    done = 0
    print(f"\nExperiment grid: {len(pairs)} pairs x {len(priors)} priors = {total} conditions")
    print(f"Already computed: {len(existing)}")

    for da, db in pairs:
        pair_name = f"{da}v{db}"
        for prior_a in priors:
            key = (da, db, prior_a)
            if key in existing and not args.retrain:
                done += 1
                continue

            t0 = time.time()
            prior_str = f"p{prior_a:.2f}".replace('.', '')
            model_name = f"{pair_name}_{prior_str}"
            model_path = os.path.join(models_dir, f'{model_name}.pth')

            # Train or load model
            if os.path.isfile(model_path) and not args.retrain:
                ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(device)
                ae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                ae.eval()
            else:
                ae = train_pair_model_with_prior(
                    baseline_path, da, db, prior_a, CONTEXT_ID,
                    config, train_dl, device
                )
                torch.save(ae.state_dict(), model_path)

            # Analyze
            result = analyze_model(
                ae, classifier, test_dl, da, db, prior_a,
                CONTEXT_ID, config, device
            )

            # Update results
            if key in existing:
                all_results = [r for r in all_results if (r['digit_a'], r['digit_b'], r['prior_a']) != key]
            all_results.append(result)
            existing[key] = result
            done += 1

            dt = time.time() - t0
            ctx_mp = f"{result['ctx_fitted_midpoint']:.2f}" if result.get('ctx_fitted_midpoint') is not None else "N/A"
            bay_mp = f"{result['bayesian_midpoint']:.2f}"
            print(f"  [{done:2d}/{total}] {pair_name} prior={prior_a:.1f}: "
                  f"ctx_midpoint={ctx_mp} bayes_midpoint={bay_mp} ({dt:.1f}s)")

            # Save incrementally
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_path}")
    print(f"Models saved to {models_dir}/")


if __name__ == '__main__':
    main()
